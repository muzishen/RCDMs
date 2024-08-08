import os
import json
import cv2
import torch
from torch import nn
from PIL import Image
import numpy as np
from src.models.myprior_transformer import MyPriorTransformer
from src.pipelines.prior_pipeline import Seq_Inpaint_Prior_Pipeline
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPTokenizer,
    CLIPImageProcessor,
)
import argparse
import numpy as np
# from mytransformers import Dinov2Model
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.multiprocessing as mp
import json
import time
from omegaconf import OmegaConf
import h5py
import random
from diffusers import UnCLIPScheduler


dataset_config = {
    'flintstones': [91, 49412, ["fred", "barney", "wilma", "betty", "pebbles", "dino", "slate"]],
    'pororosv':  [85, 49416, [ "pororo", "loopy", "eddy", "harry", "poby", "tongtong", "crong", "rody", "petty" ]],
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

def inference(args, select_test_datas, rank, indexs, unet_additional_kwargs):


    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(args.seed_number)

    # 保存路径
    save_dir = "./stage1/{}/{}_guidancescale{}_seed{}_numsteps{}/".format(
        args.exp_name, args.weights_number, args.guidance_scale, args.seed_number, args.num_inference_steps)

    save_dir_metric = "./stage1/{}/metric_{}_guidancescale{}_seed{}_reg/".format(
        args.exp_name, args.weights_number, args.guidance_scale, args.seed_number)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir_metric):
        os.makedirs(save_dir_metric, exist_ok=True)

    # prepare data aug
    clip_image_processor = CLIPImageProcessor()


    # prepare model
    model_ckpt = "./logs/{}/{}/mp_rank_00_model_states.pt".format(
        args.exp_name, args.weights_number)



    prior= MyPriorTransformer.from_pretrained_2d(args.pretrained_model_name_or_path, subfolder="prior",
                                                        unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs))
    scheduler = UnCLIPScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    prior_dict = torch.load(model_ckpt, map_location="cpu")["module"]
    prior.load_state_dict(prior_dict)
    # pipe.enable_xformers_memory_efficient_attention()

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path,subfolder="image_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    msg = tokenizer.add_tokens(dataset_config[args.dataset_name][2])
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

    pipe = Seq_Inpaint_Prior_Pipeline(prior=prior, image_encoder=image_encoder, text_encoder=text_encoder, tokenizer= tokenizer, scheduler=scheduler).to(device)
    print('====================== dataset: {}, model load finish ==================='.format((args.dataset_name).split('/')[-1]))

    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # start test
    number = 0
    sum_simm = 0
    start_time = time.time()
    for number, index in enumerate(indexs):
        number += 1
        images = list() # len:5
        if args.sr:
            data_root = args.dataset_sr_path
            for i in range(5):
                im_path = data_root + '/' + '{}_{}.png'.format(index, i)
               # print(im_path)
                images.append(np.array(Image.open(im_path).convert("RGB")))
        else:
            for i in range(5):
                im =select_test_datas['image{}'.format(i)][index]
                idx = random.randint(0, 4)
                images.append(im[idx * 128: (idx + 1) * 128])

        # target
        reference_image0 = (clip_image_processor(images=images[0], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image1 = (clip_image_processor(images=images[1], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image2 = (clip_image_processor(images=images[2], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image3 = (clip_image_processor(images=images[3], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image4 = (clip_image_processor(images=images[4], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image = torch.stack(
            [reference_image0, reference_image1, reference_image2, reference_image3, reference_image4], dim=0).to(
            memory_format=torch.contiguous_format).float()
        with torch.no_grad():

            target_embed = (image_encoder(reference_image.to(device)).image_embeds)


        black_img_clip = (clip_image_processor(images=Image.new("RGB", (args.img_width, args.img_height), (0, 0, 0)),
                                                   return_tensors="pt").pixel_values).squeeze(dim=0)
        white_img_clip = (clip_image_processor(images=Image.new("RGB", (args.img_width, args.img_height), (255, 255, 255)),
                                                   return_tensors="pt").pixel_values).squeeze(dim=0)
        if args.mode == 'visualization':
            source_clip = torch.stack([black_img_clip, black_img_clip, black_img_clip, black_img_clip, black_img_clip],dim=0)
            mask_label_clip = torch.stack([black_img_clip, black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)
            with torch.no_grad():

                imgs_proj_embeds = image_encoder(source_clip.to(device),output_hidden_states=True).image_embeds.unsqueeze(1)  # [b, 1, 1280]
                mask_label_embeds = image_encoder(mask_label_clip.to(device), output_hidden_states=True).image_embeds.unsqueeze(1)

        elif args.mode ==  'continue':
            source_clip = torch.stack([reference_image0, black_img_clip, black_img_clip, black_img_clip, black_img_clip],dim=0)
            mask_label_clip = torch.stack([white_img_clip, black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)
            with torch.no_grad():

                imgs_proj_embeds = image_encoder(source_clip.to(device),output_hidden_states=True).image_embeds.unsqueeze(1)  # [b, 1, 1280]
                mask_label_embeds = image_encoder(mask_label_clip.to(device),output_hidden_states=True).image_embeds.unsqueeze(1)
        else:
            raise ValueError("check mode")

        texts = select_test_datas['text'][index] # list length 5
        for i in range(len(texts)):
            texts[i] = texts[i].lower()

        if args.autoreg:
            for i in range(5):
                if i ==0:
                    image_bemds = torch.empty(0, 1, 1280).to(device)
                elif i == 1:
                    image1 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(0)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image_bemds = image1
                    source_clip = torch.stack((black_img_clip, black_img_clip, black_img_clip, black_img_clip), dim=0)
                    mask_label_clip = torch.stack([white_img_clip, black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)

                elif i == 2:
                    image1 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(0)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image2 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(1)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image_bemds = torch.cat([image1, image2], dim=0)
                    source_clip = torch.stack((black_img_clip, black_img_clip, black_img_clip), dim=0)
                    mask_label_clip = torch.stack([white_img_clip, white_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)

                elif i == 3:
                    image1 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(0)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image2 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(1)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image3 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(2)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image_bemds = torch.cat([image1, image2, image3], dim=0)
                    source_clip = torch.stack((black_img_clip, black_img_clip), dim=0)
                    mask_label_clip = torch.stack([white_img_clip, white_img_clip, white_img_clip, black_img_clip, black_img_clip], dim=0)

                elif i == 4:
                    image1 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(0)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image2 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(1)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image3 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(2)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image4 = torch.tensor(np.load('{}-{}_{}.npy'.format(save_dir_metric, index, str(3)))).unsqueeze(0).unsqueeze(0).to(device)  # 1280
                    image_bemds = torch.cat([image1, image2, image3, image4], dim=0)
                    source_clip = black_img_clip.unsqueeze(0)
                    mask_label_clip = torch.stack([white_img_clip, white_img_clip, white_img_clip, white_img_clip, black_img_clip], dim=0)

                with torch.no_grad():

                    imgs_proj_embeds = image_encoder(source_clip.to(device),output_hidden_states=True).image_embeds.unsqueeze(1)  # [b, 1, 1280]
                    imgs_proj_embeds = torch.cat([image_bemds, imgs_proj_embeds], dim=0)
                    mask_label_embeds = image_encoder(mask_label_clip.to(device),output_hidden_states=True).image_embeds.unsqueeze(1)  # [b, 1, 1280]

                output = pipe(
                        prompt= texts,
                        imgs_proj_embeds1= imgs_proj_embeds,
                        mask_label=mask_label_embeds,
                        video_length=5,
                        height=args.img_height,
                        width=args.img_width,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        num_inference_steps=args.num_inference_steps,
                    )

                feature = output[0][i].cpu().detach().numpy()
                cosine_similarities = F.cosine_similarity(output[0][i], target_embed[i:i+1, :].squeeze(1))
                print("{}-{}:  {}".format(index, i, cosine_similarities))
                np.save('{}-{}_{}.npy'.format(save_dir_metric, index, i), feature)
                sum_simm += cosine_similarities.item()
        else:
            output = pipe(
                        prompt= texts,
                        imgs_proj_embeds1= imgs_proj_embeds,
                        mask_label=mask_label_embeds,
                        video_length=5,
                        height=args.img_height,
                        width=args.img_width,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                        num_inference_steps=args.num_inference_steps,
                    )
            for j in range(5):
                feature = output[0][j].cpu().detach().numpy()

                cosine_similarities = F.cosine_similarity(output[0][j], target_embed[j:j+1, :].squeeze(1))
                print("{}-{}:  {}".format(index, j, cosine_similarities))
                np.save('{}{}_{}.npy'.format(save_dir_metric, index, str(j)), feature)
                sum_simm += cosine_similarities.item()
        # save features
        feature = output[0].cpu().detach().numpy()
        np.save('{}{}.npy'.format(save_dir, index),  feature)



    end_time =time.time()
    print(end_time-start_time)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a prior model of stage1 script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default="./weights/prior_diffuser/kandinsky-2-2-prior",
        help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--mode", type=str,default="continue",help="[visualization, continue]", )
    parser.add_argument("--dataset_name", type=str,default="flintstones",help="dataset name", )
    parser.add_argument("--dataset_h5", type=str,default="./datasets/ARLDM/flintstones.h5",help="dataset path")
    parser.add_argument("--dataset_sr_path", type=str,default="./datasets/ARLDM/flintstones_data/train_data_sr",help="dataset path")
    parser.add_argument('--autoreg', action='store_true', help='test use autoreg')
    parser.add_argument("--sr", action='store_true', help='super resolution data', )
    parser.add_argument("--guidance_scale",type=int,default=2.0,help="guidance_scale",)
    parser.add_argument("--seed_number",type=int,default=42,help="seed number",)
    parser.add_argument("--num_inference_steps",type=int,default=20,help="num_inference_steps",)
    parser.add_argument("--img_width",type=int,default=512,help="img_width",)
    parser.add_argument("--img_height",type=int,default=512,help="img_height",)
    parser.add_argument("--exp_name",type=str,default="stage1/FlintstonesSV",help="exp_name",)
    parser.add_argument("--weights_number",type=int,default=100000,help="weights number",)

    args = parser.parse_args()
    print(args)

    # 设置进程和 GPU 数量
    num_devices = torch.cuda.device_count()
    # num_devices = 1
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
        p = mp.Process(target=inference, args=(args, dataset_dict, rank, data_list[rank], config['unet_additional_kwargs']))
        processes.append(p)
        p.start()


    for rank, p in enumerate(processes):
        p.join()





