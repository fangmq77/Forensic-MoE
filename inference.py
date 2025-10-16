from __future__ import print_function
import os
from torch.utils import data
import torch
import numpy as np
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
import sys
import argparse
from pathlib import Path


from sklearn.metrics import accuracy_score

import random
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import csv
import glob
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, ToTensor, Normalize


from models.moe_adapter_clip import MoE_Adapter_CLIP



def read_txt(file_path):
    path_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            path_list.append(line.strip())  # 使用strip()去除行尾的换行符
    return path_list

def _convert_image_to_rgb(image):
    return image.convert("RGB")

preprocess = Compose([
            CenterCrop((224,224)),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

resize_preprocess = Compose([
            Resize(256),
            CenterCrop((224,224)),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

def read_and_preprocess_image(image_path):
    temp_img = Image.open(image_path)
    try:
        image = preprocess(temp_img)
    except (ValueError, RuntimeError):
        print('size can not crop to 224x224 in {}'.format(image_path))
        image = resize_preprocess(temp_img)
    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch DeepFake Detection')
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--image-path', type=str)
    args = parser.parse_args()


    model = MoE_Adapter_CLIP()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.cuda()

    sigmoid = torch.nn.Sigmoid()

    

    image = read_and_preprocess_image(args.image_path).unsqueeze(0)
    with torch.no_grad():
        pred = sigmoid(model(image.cuda()))[0][0].data.cpu().numpy()

            
    if pred > 0.5:
        print('This image is FAKE with fakeness score {}'.format(pred))
    else:
        print('This image is REAL with fakeness score {}'.format(pred))


