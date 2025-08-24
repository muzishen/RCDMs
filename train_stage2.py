import argparse
import logging
import os
from typing import Iterable, Optional
import json
import time
import random
from einops import rearrange
import itertools
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel,ControlNetModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection,CLIPVisionModel
from resampler import Resampler
from torch import nn
from transformers import Dinov2Model
from configs.stage1_config import  args
logger = get_logger(__name__)
from src.models.unet import UNet3DConditionModel
from omegaconf import OmegaConf
from mydatasets.flintstones import FlintDataset,Collate_fn
from mydatasets.pororosv import PororosvDataset


def mask2list_label(mask_label, imgs_embeds, imgs_proj, text_embeds): # bcfhw
    label_list = []
    for i in range (mask_label.size(2)):
        mask_label_i = mask_label[:,:,i,:,:].squeeze()
        if torch.all(mask_label_i==0):
            label_list.append(0)
        elif torch.all(mask_label_i==1):
            label_list.append(1)
        else:
            raise ValueError('please check mask label')

    label_list_tensor = torch.tensor(label_list)
    mask_1 = (label_list_tensor == 1)
    mask_0 = (label_list_tensor == 0)

    imgs_embeds = imgs_embeds[mask_1]
    text_1 = text_embeds[mask_1]

    imgs_proj = imgs_proj[mask_0]
    text_0 = text_embeds[mask_0]

    return imgs_embeds, text_1, imgs_proj, text_0




def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
    print(checkpoint_state_dict.keys())
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict
    return (epoch, last_global_step)


def count_model_params(model):
    return sum([p.numel() for p in model.parameters()]) / 1e6




class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x

class semantic_stack(nn.Module):
    def __init__(self, text_dim, vis_dim, hidden_dim=768, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.vis_fc = nn.Linear(vis_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, vis_f, text_f):
        query = (self.text_fc(text_f)).transpose(0, 1)
        key_value = (self.vis_fc(vis_f)).transpose(0, 1)
        attn_output, attn_output_weights = self.multihead_attn(query, key_value, key_value)
        out = attn_output.transpose(0, 1)
        return out



class fine_stack(nn.Module):
    def __init__(self, text_dim, vis_dim, hidden_dim=768, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.vis_fc = nn.Linear(vis_dim, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads) # seq, bs, dim

    def forward(self, vis_f, text_f):
        query = (self.text_fc(text_f)).transpose(0, 1)
        key_value = (self.vis_fc(vis_f)).transpose(0, 1)
        attn_output, attn_output_weights = self.multihead_attn(query, key_value, key_value)
        out = attn_output.transpose(0, 1)
        return out

class SDModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, unet) -> None:
        super().__init__()
        # self.mlp= mlp
        self.unet = unet
        self.seen_module =  fine_stack(text_dim=768, vis_dim=1664)
        self.unseen_module = semantic_stack(text_dim=768, vis_dim=1280)


    def forward(self, noisy_latents, timesteps, imgs_embeds, text_embeds, imgs_proj, text_proj):

        feature_1 = self.seen_module(imgs_embeds, text_embeds)
        feature_0 = self.unseen_module(imgs_proj, text_proj)

        new_encoder_hidden_states = torch.cat([feature_1, feature_0], dim=0)

        pred_noise = self.unet(noisy_latents, timesteps, new_encoder_hidden_states)
        return pred_noise





def main(unet_additional_kwargs: Dict = {},):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    token_number = {
        'flintstones': [91, 49412],
        'pororosv':  [85, 49416],
    }

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    accelerator = Accelerator(
        mixed_precision = 'fp16',
        log_with=args.report_to,
        project_dir=logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained('./weights/prior_diffuser/kandinsky-2-2-prior', subfolder="image_encoder")
    unet = UNet3DConditionModel.from_pretrained_2d(
        args.pretrained_model_name_or_path, subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")


    if args.unet_init_ckpt is not None:
        unet.load_state_dict(torch.load(args.unet_init_ckpt, map_location="cpu")["module"])
        accelerator.print(f"UNet resumed from checkpoint: {args.unet_init_ckpt}")

    max_lengths = token_number[args.dataset][0]
    text_encoder.resize_token_embeddings(token_number[args.dataset][1])
    old_embeddings = text_encoder.text_model.embeddings.position_embedding
    new_embeddings = text_encoder._get_resized_embeddings(old_embeddings, max_lengths)
    text_encoder.text_model.embeddings.position_embedding = new_embeddings
    text_encoder.config.max_position_embeddings = max_lengths
    text_encoder.max_position_embeddings = max_lengths
    text_encoder.text_model.embeddings.position_ids = torch.arange(max_lengths).expand((1, -1))

    # Freeze vae and text_encoder
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # unet.train()

    sd_model = SDModel(unet=unet)


    # accelerator.print("The trained Model parameters: {:.2f}M, {:.2f}M,  {:.2f}M".format(
    #     count_model_params(sd_model.image_proj), count_model_params(sd_model.ip_layers), count_model_params(sd_model.ref_unet)
    # ))

    if args.gradient_checkpointing:
        sd_model.unet.enable_gradient_checkpointing()

    params_to_opt = itertools.chain(sd_model.unet.parameters(), sd_model.unseen_module.parameters(),
                                    sd_model.seen_module.parameters())


    if (
            accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    else:
        # use deepspeed config
        optimizer = DummyOptim(
            params_to_opt,
            lr=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"],
            weight_decay=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["weight_decay"]
        )

    # TODO (patil-suraj): load scheduler using args
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    if args.dataset=='flintstones':
        dataset = FlintDataset(
            sr=args.sr,
            text_encoder_path='/mnt/aigc_cq/private/feishen/weights/prior_diffuser/kandinsky-2-2-prior',
        )

    elif  args.dataset=='pororosv':
        dataset = PororosvDataset(
            sr=args.sr,
            text_encoder_path=args.pretrained_model_name_or_path,
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler, collate_fn=Collate_fn, batch_size=args.train_batch_size, num_workers=4,
    )

    if accelerator.state.deepspeed_plugin is not None:
        # here we use agrs.gradient_accumulation_steps
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        # use deepspeed scheduler
        lr_scheduler = DummyScheduler(
            optimizer,
            warmup_num_steps=accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_num_steps"]
        )

    if (
            accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto"
    ):
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    sd_model, optimizer, lr_scheduler = accelerator.prepare(sd_model, optimizer, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin is None:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        if accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]:
            weight_dtype = torch.float16
        elif accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]:
            weight_dtype = torch.bfloat16
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)




    checkpointing_steps = args.checkpointing_steps
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        last_epoch, last_global_step = load_training_checkpoint(
            sd_model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        starting_epoch = last_epoch
        global_steps = last_global_step

    for epoch in range(starting_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        step = 0
        begin = time.perf_counter()
        for batch in train_dataloader:
            load_data_time = time.perf_counter() - begin
            # Convert images to latent space
            with torch.no_grad():

                # Convert target images to latent space
                target_image = batch["target_image"]  # b, f, c, h, w
                target_image = rearrange(target_image, "b f c h w -> (b f) c h w")
                latents = vae.encode(target_image.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=5)
                latents = latents * 0.18215

                # Convert source images to latent space
                source_image = batch["source_image"]  # b, f, c, h, w
                source_image = rearrange(source_image, "b f c h w -> (b f) c h w")
                masked_latents = vae.encode(
                    source_image.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                masked_latents = rearrange(masked_latents, "(b f) c h w -> b c f h w", f=5)
                masked_latents = masked_latents * 0.18215

                # Get the masked label
                masked_label = batch["masked_label"].to(accelerator.device, dtype=weight_dtype)
                masked_label = rearrange(masked_label, "b f c h w -> b c f h w", f=5)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)

            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], latents.shape[2], 1, 1), device=latents.device
                )
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=latents.device, )
            timesteps = timesteps.long()
            # print(latents.shape, noise.shape, timesteps.shape)
            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


            # Get the text embedding for conditioning
            with torch.no_grad():

                # encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]  # bs,length,1024

                text_encoder_output = text_encoder(batch["input_ids"].to(accelerator.device))
                text_embeds = text_encoder_output.last_hidden_state  # (b f) 91, 1280


                ref_image = batch["reference_image"]
                ref_image = rearrange(ref_image, "b f c h w -> (b f) c h w")

                output = image_encoder(ref_image.to(accelerator.device, dtype=weight_dtype),output_hidden_states=True)
                imgs_embeds = output.last_hidden_state  # [b, 257, 1280]
                imgs_proj = output.image_embeds.unsqueeze(1)  # [b, 1, 1280]


                imgs_embeds, text_embeds, imgs_proj, text_proj = mask2list_label(masked_label, imgs_embeds, imgs_proj,text_embeds)



            noisy_latents = torch.cat([noisy_latents, masked_label, masked_latents], dim=1)  # b 9 f h w


            # Predict the noise residual
            noise_pred = sd_model(noisy_latents, timesteps, imgs_embeds, text_embeds, imgs_proj, text_proj)

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item()

            # Backpropagate
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # do nothing
                lr_scheduler.step()  # only for not deepspeed lr_scheduler
                optimizer.zero_grad()  # do nothing

                if accelerator.sync_gradients:
                    accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_steps)
                    train_loss = 0.0


            if accelerator.is_main_process:
                logging.info(
                    "Epoch {}, step {},  step_loss: {}, lr: {}, time: {}, data_time: {}".format(
                        epoch, global_steps, loss.detach().item(), lr_scheduler.get_lr()[0],
                        time.perf_counter() - begin, load_data_time)
                )
            global_steps += 1
            step += 1

            # checkpoint
            if isinstance(checkpointing_steps, int):
                if global_steps % checkpointing_steps == 0:

                    checkpoint_model(args.output_dir, global_steps, sd_model, epoch, global_steps)

            # stop training
            if global_steps >= args.max_train_steps:
                break
            begin = time.perf_counter()

    accelerator.wait_for_everyone()
    # Save last model
    checkpoint_model(args.output_dir, global_steps, sd_model, epoch, global_steps)

    accelerator.end_training()


if __name__ == "__main__":
    config = OmegaConf.load(args.config)
    main(**config)
