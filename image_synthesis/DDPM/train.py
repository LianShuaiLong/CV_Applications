import torch
import cv2
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
    parser.add_argument('--infer_only',action='store_true',default=False)
    parser.add_argument('--img_path',type=str,default='./data/data.autohome.1.res')
    parser.add_argument('--model_path',type=str,default='./model_autohome')
    parser.add_argument('--model_name',type=str,default='DDPM_UNET.pth')
    args = parser.parse_args()
    return args


def train(model_path='./',img_path='./data/data.autohome.1.res'):
    batch_size = 64
    timesteps = 1000


    transform = transforms.Compose([
        # transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5])
    ])

    # using MNIST dataset

    # dataset = datasets.MNIST('./data',train=True,download=True,transform=transform)
    dataset = CustomData(img_path = img_path,transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    model = UNETModel(
        in_channels=3,
        model_channels=96,
        out_channels=3,
        channel_mult=(1,2,2),
        attention_resolutions=[]
    )

    model.to(device)

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps,beta_schedule='linear')
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)

    epochs = 10

    for epoch in range(epochs):
        for step,(images,labels) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = images.shape[0]
            images = images.to(device)

            # sample t uniformally for every example in the batch
            t =  torch.randint(0,timesteps,(batch_size,),device=device).long()

            loss = gaussian_diffusion.train_losses(model,images,t)

            if step %10==0:
                print(f'[{epoch}-{step}]Loss:',loss.item())
            if step %500==0:
                torch.save(model.state_dict(),f'{model_path}/EPOCH{epoch}_STEP{step}_DDPM_UNET.pth')
            
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(),f'{model_path}/DDPM_UNET.pth')
    return model

def generate(model=None,model_path=None,model_name=None):
    gaussian_diffusion=GaussianDiffusion(timesteps=1000,beta_schedule='linear')
    if model:
        generated_images = gaussian_diffusion.sample(model,112,batch_size=16,channels=3)
    elif model_path and model_name:
        unet = UNETModel(
            in_channels=3,
            model_channels=96,
            out_channels=3,
            channel_mult=(1,2,2),
            attention_resolutions=[]
        )
        unet.load_state_dict(torch.load(os.path.join(model_path,model_name)))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        unet.to(device)
        generated_images = gaussian_diffusion.sample(unet,112,batch_size=16,channels=3)
    else:
        raise ValueError('Invalid training model object||trained model path')
        
    return generated_images[-1]

if __name__=='__main__':
    args = parse_parser()
    img_path = args.img_path
    model_path = args.model_path
    model_name = args.model_name
    os.makedirs(model_path,exist_ok=True)
    if not args.infer_only:
        model = train(model_path = model_path,img_path =img_path)
        model_path = None
        model_name = None
    else:
        model = None
        model_path = model_path
        model_name = model_name
    
    generated_images=generate(model=model,model_path=model_path,model_name=model_name)
    generated_images = generated_images.reshape(4,4,3,112,112)
    for n_row in range(4):
        for n_col in range(4):
            # 需要先反标准化:(y*std+mean)*255,然后采用transpose进行维度交换(不能采用reshape,reshape只是改变形状)
            cv2.imwrite(f'{n_row}_{n_col}.jpg',np.transpose((generated_images[n_row][n_col]*0.5+0.5)*255,(1,2,0))) 





