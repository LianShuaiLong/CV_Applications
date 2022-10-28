import torch
import cv2
import requests
import argparse

from torchvision import transforms,datasets
from PIL import Image
from model import UNETModel
from noise_schedule import *
from data import CustomData

import numpy as np
import pdb
import os

def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',type=str,default='./data/tmp')
    parser.add_argument('--model_path',type=str,default='./model_autohome')
    parser.add_argument('--schedule',type=str,default='linear')
    args = parser.parse_args()
    return args


def generate(model_path=None,schedule='linear'):
    gaussian_diffusion=GaussianDiffusion(timesteps=1000,beta_schedule=schedule)
    unet = UNETModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1,2,2),
        attention_resolutions=[]
    )
    unet.load_state_dict(torch.load(model_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet.to(device)
    generated_images = gaussian_diffusion.sample(unet,112,batch_size=64,channels=3)
        
    return generated_images[-1]

if __name__=='__main__':
    args = parse_parser()
    model_path = args.model_path
    save_path = args.save_path
    schedule = args.schedule
    os.makedirs(save_path,exist_ok=True)
    
    generated_images=generate(model_path=model_path,schedule=schedule)
    generated_images = generated_images.reshape(8,8,3,112,112)
    for n_row in range(8):
        for n_col in range(8):
            cv2.imwrite(f'{save_path}/{n_row}_{n_col}.jpg',np.transpose((generated_images[n_row][n_col]*0.5+0.5)*255,(1,2,0))) 





