import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer,CLIPImageProcessor
from PIL import Image
from src.blip_override.blip import init_tokenizer
from transformers import (CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPVisionModel)

def Collate_fn(data):


    reference_image = torch.stack([example["reference_image"] for example in data]).to(memory_format=torch.contiguous_format).float()
    source_clip_image = torch.stack([example["source_clip"] for example in data]).to(memory_format=torch.contiguous_format).float()
    mask_label_image = torch.stack([example["mask_label_clip"] for example in data]).to(memory_format=torch.contiguous_format).float()



    source_image = torch.stack([example["source"] for example in data])
    source_image = source_image.to(memory_format=torch.contiguous_format).float()

    ###target image
    target_image = torch.stack([example["target"] for example in data])
    target_image = target_image.to(memory_format=torch.contiguous_format).float() # shape: bs,1,1280

    ###mask label
    masked_label = torch.stack([example["mask_label"] for example in data])
    masked_label = masked_label.to(memory_format=torch.contiguous_format).float() # shape: bs,1,1280

    input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_mask = torch.cat([example["text_mask"] for example in data], dim=0)

    return {

        "source_clip_image": source_clip_image,
        "reference_image": reference_image,
        "source_image": source_image,
        "target_image": target_image,
        "masked_label":masked_label,
        "masked_label_clip":mask_label_image,
        "input_ids": input_ids,
        "text_mask":text_mask,
        # "id_list": id_list,
    }


class FlintDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self,
                 sr,
                 text_encoder_path,
                 subset='train',
                 h5_file='./datasets/ARLDM/flintstones.h5',
                 size=512,
                 text_drop_rate=0.1,
                 length=91,

                 ):
        super(FlintDataset, self).__init__()

        self.sr = sr
        self.text_encoder_path = text_encoder_path
        self.subset = subset
        self.h5_file = h5_file
        self.size=size
        self.text_drop_rate = text_drop_rate
        self.max_length = length


        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([self.size, self.size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.mask_augment = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.clip_image_processor = CLIPImageProcessor()
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder_path, subfolder="tokenizer")
        msg = self.clip_tokenizer.add_tokens(list([ "fred", "barney", "wilma", "betty", "pebbles", "dino", "slate"]))

        print("clip add {} new tokens".format(msg))




    def open_h5(self):
        h5 = h5py.File(self.h5_file, "r")
        self.h5 = h5[self.subset]

    def __getitem__(self, index):
        """
        self.h5.keys:['image0', 'image1', 'image2', 'image3', 'image4', 'text']
        """
        if not hasattr(self, 'h5'):
            self.open_h5()


        images = list() # len:5

        # use super resolution  dataset
        if self.sr:
            data_root = './datasets/ARLDM/flintstones_data/train_data_sr'

            for i in range(5):
                im_path = data_root + '/' + '{}_{}.png'.format(index, i)
               # print(im_path)
                images.append(np.array(Image.open(im_path).convert("RGB")))

        else:
            for i in range(5):
                im = self.h5['image{}'.format(i)][index]
                im = cv2.imdecode(im, cv2.IMREAD_COLOR)   #im: [640,128,3]   5x128

                idx = random.randint(0, 4)
                images.append(im[idx * 128: (idx + 1) * 128])


        reference_image0=  (self.clip_image_processor(images=images[0], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image1=  (self.clip_image_processor(images=images[1], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image2=  (self.clip_image_processor(images=images[2], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image3=  (self.clip_image_processor(images=images[3], return_tensors="pt").pixel_values).squeeze(dim=0)
        reference_image4=  (self.clip_image_processor(images=images[4], return_tensors="pt").pixel_values).squeeze(dim=0)
        black_img_clip = (self.clip_image_processor(images=Image.new("RGB", (self.size, self.size), (0, 0, 0)),
                                                   return_tensors="pt").pixel_values).squeeze(dim=0)
        white_img_clip = (self.clip_image_processor(images=Image.new("RGB", (self.size, self.size), (255, 255, 255)),
                                                   return_tensors="pt").pixel_values).squeeze(dim=0)


        reference_image = torch.stack([reference_image0, reference_image1, reference_image2, reference_image3, reference_image4], dim=0).to(memory_format=torch.contiguous_format).float()


        image0 = self.augment(images[0])
        image1 = self.augment(images[1])
        image2 = self.augment(images[2])
        image3 = self.augment(images[3])
        image4 = self.augment(images[4])
        black_img = self.mask_augment(Image.new("RGB", (self.size,self.size), (0, 0, 0)))


        # setting mask label
        black0 = torch.zeros(1, int(self.size / 8), int(self.size / 8))
        white1 = torch.ones(1, int(self.size / 8), int(self.size / 8))



        length = random.randint(0, 4)

        if length == 0:
            source   =  torch.stack([black_img, black_img, black_img, black_img, black_img], dim=0)
            source_clip = torch.stack([black_img_clip, black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)
            mask_label = torch.stack([black0, black0, black0, black0, black0], dim=0)
            mask_label_clip = torch.stack([black_img_clip, black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)


        elif length == 1:
            source   =  torch.stack([image0, black_img, black_img, black_img, black_img], dim=0)
            source_clip = torch.stack([reference_image0, black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)
            mask_label = torch.stack([white1, black0, black0, black0, black0], dim=0)
            mask_label_clip = torch.stack([white_img_clip, black_img_clip, black_img_clip, black_img_clip, black_img_clip], dim=0)



        elif length == 2:
            source   =  torch.stack([image0, image1, black_img, black_img, black_img], dim=0)
            source_clip = torch.stack(
                [reference_image0, reference_image1, black_img_clip, black_img_clip, black_img_clip], dim=0)
            mask_label = torch.stack([white1, white1, black0, black0, black0], dim=0)
            mask_label_clip = torch.stack([white_img_clip, white_img_clip, black_img_clip, black_img_clip, black_img_clip],
                                      dim=0)


        elif length == 3:
            source   =  torch.stack([image0, image1, image2, black_img, black_img], dim=0)
            source_clip = torch.stack(
                [reference_image0, reference_image1, reference_image2, black_img_clip, black_img_clip], dim=0)
            mask_label = torch.stack([white1, white1, white1, black0, black0], dim=0)
            mask_label_clip = torch.stack(
                [white_img_clip, white_img_clip, white_img_clip, black_img_clip, black_img_clip],
                dim=0)

        elif length == 4:
            source   =  torch.stack([image0, image1, image2, image3, black_img], dim=0)
            source_clip = torch.stack(
                [reference_image0, reference_image1, reference_image2, reference_image3, black_img_clip], dim=0)
            mask_label = torch.stack([white1, white1, white1, white1, black0], dim=0)
            mask_label_clip = torch.stack(
                [white_img_clip, white_img_clip, white_img_clip, white_img_clip, black_img_clip],
                dim=0)


        target = torch.stack([image0, image1, image2, image3, image4], dim=0)



        texts = self.h5['text'][index].decode('utf-8').split('|')  # list length 5


        for i in range(len(texts)):


            texts[i] = texts[i].lower()
            if random.random() < self.text_drop_rate:
                texts[i]=""



        text_inputs = self.clip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_mask = text_inputs.attention_mask.bool()



        return {

            "source": source,
            "source_clip": source_clip,
            "target": target,
            "reference_image": reference_image,
            "mask_label": mask_label,
            "mask_label_clip": mask_label_clip,
            "text_mask":text_mask,
            "text_input_ids": text_input_ids,

        }



    def __len__(self):
        if not hasattr(self, 'h5'):
            self.open_h5()
        return len(self.h5['text'])


if __name__ == "__main__":
    dataset = FlintDataset(
    )