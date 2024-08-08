import os
import json
import cv2
import torch
from torch import nn
from PIL import Image
import numpy as np
from src.models.unet import UNet3DConditionModel
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel,DDIMScheduler
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPVisionModel,
    CLIPImageProcessor,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
import argparse
from omegaconf import OmegaConf
from typing import Any, Dict, List, Optional, Tuple, Union
from skimage.metrics import structural_similarity as compare_ssim
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import json
import time
from src.pipelines.RCDMs_pipeline import RCDMsPipeline
import h5py
import random



dataset_config = {
    'flintstones': [91, 49412, ["fred", "barney", "wilma", "betty", "pebbles", "dino", "slate"]],
    'pororosv':  [85, 49416, ["pororo", "loopy", "eddy", "harry", "poby", "tongtong", "crong", "rody", "petty"]],
}

def decode_image(h5_img):
    save_list=[]
    for img in h5_img:
        im = cv2.imdecode(img, cv2.IMREAD_COLOR)
        save_list.append(im)
    return save_list


def decode_text(h5_text):
    save_list=[]
    for txt in h5_text:
        text = txt.decode('utf-8').split('|')
        save_list.append(text)
    return save_list



def split_list(n, m):
    quotient = n // m
    remainder = n % m
    result = []
    start = 0
    for i in range(m):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient
        result.append(list(range(start, end)))
        start = end
    return result


def tensor2list(bs):
    reshaped_tensor = bs.reshape(-1, bs.size(-1))
    splitted_tensors = torch.split(reshaped_tensor, bs.size(1) * bs.size(2), dim=0)
    x_list = [t.view(bs.size(1), bs.size(2), bs.size(-1)) for t in splitted_tensors]
    return x_list

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    new_imgs =[]
    for img in imgs:
        # print(img.shape)
        image = np.array(img)
        image = Image.fromarray((image * 255).astype(np.uint8))
        new_imgs.append(image)

    w, h = (new_imgs[0]).size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(new_imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

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

class semantic_stack(ModelMixin, ConfigMixin):
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

class fine_stack(ModelMixin, ConfigMixin):
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



def inference(args, select_test_datas, rank, indexs, unet_additional_kwargs,noise_scheduler_kwargs):

    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)



    save_dir = "./stage2/real_{}/{}_guidancescale{}_seed{}_reg/".format(
        args.exp_name, args.weights_number, args.guidance_scale, args.seed_number)
    save_dir_metric = "./stage2/real_{}/metric_{}_guidancescale{}_seed{}_reg/".format(
        args.exp_name, args.weights_number, args.guidance_scale, args.seed_number)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir_metric):
        os.makedirs(save_dir_metric, exist_ok=True)

    clip_image_processor = CLIPImageProcessor()
    vis_augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([args.img_height, args.img_weigh]),
        transforms.ToTensor(),

    ])

    reg_augment = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


    img_augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([args.img_height, args.img_weigh]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    mask_augment = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    local_module = fine_stack(text_dim=768, vis_dim=1664)
    global_module = semantic_stack(text_dim=768, vis_dim=1280)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained('./weights/prior_diffuser/kandinsky-2-2-prior',subfolder="image_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    clip_tokenizer = CLIPTokenizer.from_pretrained('./weights/prior_diffuser/kandinsky-2-2-prior',subfolder="tokenizer")
    msg = clip_tokenizer.add_tokens(dataset_config[args.dataset_name][2])
    print("clip add {} new tokens".format(msg))


    text_encoder = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.resize_token_embeddings(dataset_config[args.dataset_name][1])
    # resize_position_embeddings for clip
    max_lengths = dataset_config[args.dataset_name][0]
    old_embeddings = text_encoder.text_model.embeddings.position_embedding
    new_embeddings = text_encoder._get_resized_embeddings(old_embeddings, max_lengths)
    text_encoder.text_model.embeddings.position_embedding = new_embeddings
    text_encoder.config.max_position_embeddings = max_lengths
    text_encoder.max_position_embeddings = max_lengths
    text_encoder.text_model.embeddings.position_ids = torch.arange(max_lengths).expand((1, -1))


    # print(OmegaConf.to_container(unet_additional_kwargs))
    unet = UNet3DConditionModel.from_pretrained_2d(
         args.pretrained_model_name_or_path, subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
    )

    model_ckpt = "./stage2/{}/{}/mp_rank_00_model_states.pt".format(args.exp_name, args.weights_number)
    model_sd = torch.load(model_ckpt, map_location="cpu")["module"]
    seen_module_dict = {}
    unseen_module_dict = {}
    unet_dict = {}

    for k in model_sd.keys():
        if k.startswith("seen_module"):
            seen_module_dict[k.replace("seen_module.", "")] = model_sd[k]
        elif k.startswith("unseen_module"):
            unseen_module_dict[k.replace("unseen_module.", "")] = model_sd[k]
        elif k.startswith("unet"):
            unet_dict[k.replace("unet.", "")] = model_sd[k]
        else:
            print(k)

    local_module.load_state_dict(seen_module_dict)
    global_module.load_state_dict(unseen_module_dict)
    unet.load_state_dict(unet_dict)


    pipe = AnimationPipeline(vae=vae, text_encoder=text_encoder, tokenizer=clip_tokenizer, unet=unet, local_module =local_module, global_module=global_module,
                             scheduler=DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))).to(device)


    print('====================== dataset: {}, model load finish ==================='.format((args.dataset_name).split('/')[-1]))


    start_time = time.time()
    for index in indexs:
        images = list() # len:5
        if args.sr:
            data_root = args.dataset_sr_path
            for i in range(5):
                im_path = data_root + '/' + '{}_{}.png'.format(index, i)
                images.append(np.array(Image.open(im_path).convert("RGB")))
        else:
            for i in range(5):
                im =select_test_datas['image{}'.format(i)][index]
                idx = random.randint(0, 4)
                images.append(im[idx * 128: (idx + 1) * 128])

        image0 = img_augment(images[0])

        black_img = mask_augment(Image.new("RGB", (args.img_weigh, args.img_height), (0, 0, 0)))

        reference_image0=  (clip_image_processor(images=images[0], return_tensors="pt").pixel_values)

        reference_image1=  (clip_image_processor(images=images[1], return_tensors="pt").pixel_values)
        reference_image2=  (clip_image_processor(images=images[2], return_tensors="pt").pixel_values)
        reference_image3=  (clip_image_processor(images=images[3], return_tensors="pt").pixel_values)
        reference_image4=  (clip_image_processor(images=images[4], return_tensors="pt").pixel_values)

        # setting mask label
        black0 = torch.zeros((1, int(args.img_height / 8), int(args.img_weigh / 8)))
        white1 = torch.ones((1, int(args.img_height / 8), int(args.img_weigh / 8)))

        if args.mode == 'visualization':
            source = torch.stack((black_img, black_img, black_img, black_img, black_img), dim=0)
            mask_label = torch.stack((white1, black0, black0, black0, black0), dim=0)

        elif args.mode ==  'continue':
            source = torch.stack((image0, black_img, black_img, black_img, black_img), dim=0)
            mask_label = torch.stack((white1, black0, black0, black0, black0), dim=0)
            with torch.no_grad():
                reference_image0_embed = (image_encoder(reference_image0.to(device)).last_hidden_state)
                reference_image1_embed = torch.tensor(np.load('{}/{}_{}.npy'.format(args.target_embed_path, index, str(1)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                reference_image2_embed = torch.tensor(np.load('{}/{}_{}.npy'.format(args.target_embed_path, index, str(2)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                reference_image3_embed = torch.tensor(np.load('{}/{}_{}.npy'.format(args.target_embed_path, index, str(3)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                reference_image4_embed = torch.tensor(np.load('{}/{}_{}.npy'.format(args.target_embed_path, index, str(4)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                image_embeds_1 = reference_image0_embed
                proj_embeds_0 = torch.cat([reference_image1_embed, reference_image2_embed, reference_image3_embed, reference_image4_embed], dim=0)
        else:
            raise ValueError("check mode")

        texts = select_test_datas['text'][index] # list length 5
        for i in range(len(texts)):
            texts[i] = texts[i].lower()

        if args.autoreg:
            for i in range(5):
                if i == 1:
                    image1 = Image.open('{}{}_0.png'.format(save_dir_metric, index)).convert("RGB")
                    image1 = reg_augment(image1)

                    source = torch.stack((image1, black_img, black_img, black_img, black_img), dim=0)
                    mask_label = torch.stack((white1, black0, black0, black0, black0), dim=0)

                if i == 2:
                    image1 = Image.open('{}{}_0.png'.format(save_dir_metric, index)).convert("RGB")
                    image1 = reg_augment(image1)
                    image2 = Image.open('{}{}_1.png'.format(save_dir_metric, index)).convert("RGB")
                    image2 = reg_augment(image2)

                    source = torch.stack((image1, image2, black_img, black_img, black_img), dim=0)
                    mask_label = torch.stack((white1, white1, black0, black0, black0), dim=0)

                elif i == 3:
                    image1 = Image.open('{}{}_0.png'.format(save_dir_metric, index)).convert("RGB")
                    image1 = reg_augment(image1)
                    image2 = Image.open('{}{}_1.png'.format(save_dir_metric, index)).convert("RGB")
                    image2 = reg_augment(image2)
                    image3 = Image.open('{}{}_2.png'.format(save_dir_metric, index)).convert("RGB")
                    image3 = reg_augment(image3)

                    source = torch.stack((image1, image2, image3, black_img, black_img), dim=0)
                    mask_label = torch.stack((white1, white1, white1, black0, black0), dim=0)

                elif i == 4:
                    image1 = Image.open('{}{}_0.png'.format(save_dir_metric, index)).convert("RGB")
                    image1 = reg_augment(image1)
                    image2 = Image.open('{}{}_1.png'.format(save_dir_metric, index)).convert("RGB")
                    image2 = reg_augment(image2)
                    image3 = Image.open('{}{}_2.png'.format(save_dir_metric, index)).convert("RGB")
                    image3 = reg_augment(image3)
                    image4 = Image.open('{}{}_3.png'.format(save_dir_metric, index)).convert("RGB")
                    image4 = reg_augment(image4)
                    source = torch.stack((image1, image2, image3, image4, black_img), dim=0)
                    mask_label = torch.stack((white1, white1, white1, white1, black0), dim=0)

                output = pipe(
                        prompt= texts,
                        source_img=source,
                        mask_label=mask_label,
                        video_length=5,
                        height=args.img_height,
                        width=args.img_weigh,
                        guidance_rescale=0.0,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        num_inference_steps=args.num_inference_steps,
                    )

                result = output.videos.permute(0, 2, 3, 4, 1).squeeze()
                x_list = tensor2list(result)

                gen_grid = image_grid([x_list[i]], 1, 1)
                gen_grid.save('{}{}_{}.png'.format(save_dir_metric, index, i))
        else:
            output = pipe(
                prompt=texts,
                source_img=source,
                image_embeds_1 = image_embeds_1,
                proj_embeds_0 = proj_embeds_0,
                mask_label=mask_label,
                video_length=5,
                height=args.img_height,
                width=args.img_weigh,
                guidance_scale=args.guidance_scale,
                generator=generator,
                num_inference_steps=args.num_inference_steps,
            )

            result = output.videos.permute(0, 2, 3, 4, 1).squeeze()

            x_list = tensor2list(result)

            for i in range(5):
                gen_grid = image_grid([x_list[i]], 1, 1)
                gen_grid.save('{}{}_{}.png'.format(save_dir_metric, index, i))




        ## target process
        image0v = vis_augment(images[0])
        image1v = vis_augment(images[1])
        image2v = vis_augment(images[2])
        image3v = vis_augment(images[3])
        image4v = vis_augment(images[4])
        target = torch.stack((image0v, image1v, image2v, image3v, image4v), dim=0)
        target_list = tensor2list(target.permute(0, 2, 3,1))


        all_list = x_list + target_list
        compare_grid = image_grid(all_list, 2, 5)
        compare_grid.save('{}{}.png'.format(save_dir, index))


    end_time =time.time()
    print(end_time-start_time)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple example of a prior model of stage1 script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="/mnt/aigc_cq/private/huye/model_weights/stable-diffusion-v1-5",
                        help="Path to pretrained model or model identifier from huggingface.co/models.", )
    parser.add_argument("--mode", type=str,default="continue",help="[visualization, continue]", )
    parser.add_argument("--dataset_name", type=str,default="pororosv",help="dataset name", )
    parser.add_argument("--dataset_h5", type=str,default="./datasets/ARLDM/pororo.h5",help="dataset path")
    parser.add_argument("--dataset_sr_path", type=str,default="./datasets/ARLDM/pororosv_data/test_data_sr",help="dataset path")
    parser.add_argument("--target_embed_path", type=str,
                        default="./stage1/pororosv/metric_100000_guidancescale2.0_seed42_reg",
                        help="t_img_embed path", )
    parser.add_argument('--autoreg', action='store_true', help='test use autoreg')
    parser.add_argument("--sr", action='store_true', help='super resolution data', )
    parser.add_argument("--guidance_scale", type=int, default=2.0, help="guidance_scale", )
    parser.add_argument("--seed_number", type=int, default=42, help="seed_number", )
    parser.add_argument("--num_inference_steps", type=int, default=20, help="num_inference_steps", )
    parser.add_argument("--img_weigh", type=int, default=512, help="img_weigh", )
    parser.add_argument("--img_height", type=int, default=512, help="img_height", )
    parser.add_argument("--exp_name", type=str, default="pororosv", help="exp_name", )
    parser.add_argument("--weights_number", type=int, default=310000, help="weights number", )
    args = parser.parse_args()
    print(args)

    num_devices = torch.cuda.device_count()
    print("using {} num_processes inference".format(num_devices))

    config = OmegaConf.load('./configs/testing.yaml')

    h5 = h5py.File(args.dataset_h5, "r")
    test_data = h5['test']
    select_test_datas = test_data
    print(len(select_test_datas['image0']),  len(select_test_datas['image1']), len(select_test_datas['image2']),
          len(select_test_datas['image3']), len(select_test_datas['image4']),len(select_test_datas['text']))

    image0 = decode_image(select_test_datas['image0'])
    image1 = decode_image(select_test_datas['image1'])
    image2 = decode_image(select_test_datas['image2'])
    image3 = decode_image(select_test_datas['image3'])
    image4 = decode_image(select_test_datas['image4'])
    text = decode_text(select_test_datas['text'])

    dataset_dict = {'image0':image0,'image1':image1,'image2':image2,'image3':image3,'image4':image4, 'text':text }



    mp.set_start_method("spawn")
    data_list = split_list(len(text), num_devices)
    print('=====')
    print(config)
    processes = []
    for rank in range(num_devices):
        p = mp.Process(target=inference, args=(args, dataset_dict, rank, data_list[rank], config['unet_additional_kwargs'], config['noise_scheduler_kwargs']))
        processes.append(p)
        p.start()

    for rank, p in enumerate(processes):
        p.join()



