import argparse
import logging
import os
from typing import Iterable, Optional
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import time
import random
from packaging import version

import torch.nn.functional as F
from einops import rearrange
import torch.utils.checkpoint
import transformers
import mydatasets,diffusers
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from src.models.myprior_transformer import MyPriorTransformer
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version, is_wandb_available
from mydatasets.flintstones import FlintDataset,Collate_fn
from mydatasets.pororosv import PororosvDataset
from configs.stage1_config import args
from omegaconf import OmegaConf
logger = get_logger(__name__)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")



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
    checkpoint_state_dict= torch.load(load_dir, map_location="cpu")


    print(checkpoint_state_dict.keys())
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    # TODO optimizer lr, and loss state

    weight_dict = checkpoint_state_dict["module"]
    new_weight_dict = {f"module.{key}": value for key, value in weight_dict.items()}
    model.load_state_dict(new_weight_dict)
    del checkpoint_state_dict

    return model, epoch, last_global_step


def count_model_params(model):
    return sum([p.numel() for p in model.parameters()]) / 1e6


def main(unet_additional_kwargs: Dict = {},):
    token_number = {
        'flintstones': [91, 49412],
        'pororosv':  [85, 49416],
    }

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    # prior_model = MyPriorTransformer.from_pretrained(args.pretrained_model_name_or_path, subfolder="prior", num_embeddings=2, embedding_dim=1024, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    prior_model = MyPriorTransformer.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="prior",
                                                        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs))
    # print(prior_model)

    max_lengths = token_number[args.dataset][0]
    text_encoder.resize_token_embeddings(token_number[args.dataset][1])
    old_embeddings = text_encoder.text_model.embeddings.position_embedding
    new_embeddings = text_encoder._get_resized_embeddings(old_embeddings, max_lengths)
    text_encoder.text_model.embeddings.position_embedding = new_embeddings
    text_encoder.config.max_position_embeddings = max_lengths
    text_encoder.max_position_embeddings = max_lengths
    text_encoder.text_model.embeddings.position_ids = torch.arange(max_lengths).expand((1, -1))

    # Freeze vae and text_encoder
    image_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)


    clip_mean = prior_model.clip_mean.clone()
    clip_std = prior_model.clip_std.clone()

    # clip_mean = torch.tensor(-0.016)
    # clip_std = torch.tensor(0.415)
    #
    # prior_model.clip_mean = None
    # prior_model.clip_std = None

    accelerator.print("The Model parameters: Prior_model {:.2f}M, Image: {:.2f}M".format(
        count_model_params(prior_model), count_model_params(image_encoder)
    ))

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            prior_model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.gradient_checkpointing:
        prior_model.enable_gradient_checkpointing()

    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    if (accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        optimizer = torch.optim.AdamW(prior_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        # use deepspeed config
        optimizer = DummyOptim(
            prior_model.parameters(),
            lr=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"],
            weight_decay=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["weight_decay"]
        )

    # TODO (patil-suraj): load scheduler using args
    noise_scheduler = DDPMScheduler(beta_schedule='squaredcos_cap_v2', prediction_type='sample')


    if args.dataset=='flintstones':
        dataset = FlintDataset(
            sr=args.sr,
            text_encoder_path=args.pretrained_model_name_or_path,
        )

    elif  args.dataset=='pororosv':
        dataset = PororosvDataset(
            sr=args.sr,
            text_encoder_path=args.pretrained_model_name_or_path,
        )


    train_dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=Collate_fn, batch_size=args.train_batch_size, num_workers=8,
        pin_memory=True,
    )

    if accelerator.state.deepspeed_plugin is not None:
        # here we use agrs.gradient_accumulation_steps
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (accelerator.state.deepspeed_plugin is None or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        lr_scheduler = get_scheduler(name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,)
    else:
        # use deepspeed scheduler
        lr_scheduler = DummyScheduler(optimizer,
        warmup_num_steps=accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"]["warmup_num_steps"])

    if (accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto"):
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size

    prior_model, optimizer, lr_scheduler = accelerator.prepare(prior_model, optimizer, lr_scheduler)

    # weight_dtype = torch.float32
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
    # pose_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_mean = clip_mean.to(weight_dtype).to(accelerator.device)
    clip_std = clip_std.to(weight_dtype).to(accelerator.device)



    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

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




    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        prior_model, last_epoch, last_global_step = load_training_checkpoint(
            prior_model,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}, global step: {last_global_step}")
        starting_epoch = last_epoch
        global_steps = last_global_step
        prior_model = prior_model
    else:
        global_steps = 0
        starting_epoch = 0
        prior_model = prior_model
    progress_bar = tqdm(range(global_steps, args.max_train_steps), initial=global_steps, desc="Steps",
                        # Only show the progress bar once on each machine.
                        disable=not accelerator.is_local_main_process, )

    for epoch in range(starting_epoch, args.num_train_epochs):
        prior_model.train()
        train_loss = 0.0


        for step, batch in enumerate(train_dataloader):
            #TODO resume model
            with torch.no_grad():

                # Convert target images to latent space
                target_image = batch["reference_image"]  # b, f, 3, 224, 224
                target_image = rearrange(target_image, "b f c h w -> (b f) c h w")
                image_embeds = (image_encoder(target_image.to(accelerator.device, dtype=weight_dtype)).image_embeds).unsqueeze(1) #(b f) 1, 1024
                # image_embeds = rearrange(image_embeds, "(b f) c hw -> b c f hw", f=5) # b, 1, 5, 1024


                # Convert source images to latent space
                source_image = batch["source_clip_image"]  # b, f, c, h, w
                source_image = rearrange(source_image, "b f c h w -> (b f) c h w")

                masked_label = batch["masked_label_clip"].to(accelerator.device, dtype=weight_dtype)
                masked_label = rearrange(masked_label, "b f c h w -> (b f) c h w" )

                # imgs_encoder_hidden_states = image_encoder(source_image.to(accelerator.device, dtype=weight_dtype),
                #                             output_hidden_states=True).last_hidden_state  # [b, 257, 1280]
                imgs_proj_embeds = image_encoder(source_image.to(accelerator.device, dtype=weight_dtype),
                                            output_hidden_states=True).image_embeds.unsqueeze(1)  # [b, 1, 1280]
                masked_label_embeds = image_encoder(masked_label.to(accelerator.device, dtype=weight_dtype),
                                            output_hidden_states=True).image_embeds.unsqueeze(1)  # [b, 1, 1280]

                text_mask = batch["text_mask"].to(accelerator.device)
                text_encoder_output = text_encoder(batch["input_ids"].to(accelerator.device))
                prompt_embeds = text_encoder_output.text_embeds    # (b f),  1, 1280
                text_encoder_hidden_states = text_encoder_output.last_hidden_state #(b f) 91, 1280

                noise = torch.randn_like(image_embeds)

                # 添加 offset

                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((image_embeds.shape[0], image_embeds.shape[1], 1), device=image_embeds.device)

                # Sample a random timestep for each image
                bsz = image_embeds.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=image_embeds.device)
                timesteps = timesteps.long()

                # 归一化
                image_embeds = (image_embeds - clip_mean) / clip_std

                # 加噪
                noisy_x = noise_scheduler.add_noise(image_embeds, noise, timesteps)
                #Groud Truth
                target = image_embeds.squeeze(1)



            with accelerator.accumulate(prior_model):
                model_pred = prior_model(
                    noisy_x,
                    timestep=timesteps,
                    proj_embedding=prompt_embeds,
                    encoder_hidden_states=text_encoder_hidden_states,
                    proj_embedding1=imgs_proj_embeds,
                    mask_label = masked_label_embeds,
                    # encoder_hidden_states1=imgs_encoder_hidden_states,
                    # attention_mask=None,
                    attention_mask=text_mask,
                ).predicted_image_embedding


                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prior_model.parameters(), args.max_grad_norm)

                optimizer.step()  # do nothing
                lr_scheduler.step()  # only for not deepspeed lr_scheduler
                optimizer.zero_grad()  # do nothing

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_steps += 1
                accelerator.log({"train_loss": train_loss}, step=global_steps)
                train_loss = 0.0

                if global_steps % args.checkpointing_steps == 0:
                    checkpoint_model(
                        args.output_dir, global_steps, prior_model, epoch, global_steps
                    )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            accelerator.log(logs, step=global_steps)

            if global_steps >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    # Save last model
    checkpoint_model(args.output_dir, global_steps, prior_model, epoch, global_steps)
    accelerator.end_training()


if __name__ == "__main__":
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    config = OmegaConf.load(args.config)
    main(**config)
