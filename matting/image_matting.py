import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import numpy as np

import argparse
import os
import sys

from MODNet.src.models.modnet import *

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',type=str,default='./test_images',help='Test images` root_dir')
    parser.add_argument('--output_path',type=str,default='./test_results',help='Result images dir')
    parser.add_argument('--ckpt_path',type=str,default='./model/modnet_photographic_portrait_matting.ckpt',help='Pretrained model path')
    parser.add_argument('--image_type',type=str,default='jpg,jpeg,png',help='Test images suffix')
    parser.add_argument('--input_size',type=int,default=512,help='Input image size')
    args=parser.parse_args()
    return args

def save_fg(img_data,matte_data,filename):
    w,h = img_data.width,img_data.height
    img_data = np.asarray(img_data)
    if len(img_data.shape) ==2:
       img_data = img_data[:,:,None]
    if img_data.shape[2]==1:
       img_data = np.repeat(img_data,3,axis=2)
    elif img_data.shape[2] ==4:
       img_data = img_data[:,:,0:3]
    matte = np.repeat(matte_data[:,:,None],3,axis=2)
    fg_black = matte*img_data#+(1-matte)*np.full(img_data.shape,0)
    fg_white = matte*img_data+(1-matte)*np.full(img_data.shape,255)
    combined = np.concatenate((img_data,matte*255,fg_black,fg_white),axis=1)
    combined = Image.fromarray(np.uint8(combined))
    combined.save(filename)
    return
    

def infer(images,input_size,ckpt_path,output_path):
    ref_size=input_size
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    modnet=MODNet(backbone_pretrained=False)
    modnet=nn.DataParallel(modnet).cuda()
    modnet.load_state_dict(torch.load(ckpt_path))
    modnet.eval()
    
    for image in images:
        print('Process image:{}'.format(image))
        img_data = Image.open(image)
        img = np.asarray(img_data)
    
        if len(img.shape)==2:
           img = img[:,:,None]
        if img.shape[2]==1:
           img = np.repeat(img,3,axis=2)
        if img.shape[2]==4:
           img = img[:,:,0:3]
        img = Image.fromarray(img)
        img = transform(img)
        img = img[None,:,:,:]
    
        img_b,img_c,img_h,img_w = img.shape
        if max(img_h,img_w)<ref_size or min(img_h,img_w)>ref_size:
           if img_h>=img_w:
              img_rw = ref_size
              img_rh = int(img_h/img_w*ref_size)
           if img_h<=img_w:
              img_rh = ref_size
              img_rw = int(img_w/img_h*ref_size)
        else:
           img_rw = img_w
           img_rh = img_h
    
        img_rw = img_rw - img_rw%32
        img_rh = img_rh - img_rh%32
        im = F.interpolate(img,size=(img_rh,img_rw),mode ='area')
    
        _,_,matte = modnet(im.cuda(),True)
   
        matte = F.interpolate(matte,size=(img_h,img_w),mode ='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = image.split('/')[-1].split('.')[0] + '_alpha.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(output_path, matte_name))
        combined_name = os.path.join(output_path,image.split('/')[-1].split('.')[0]+'_combined.png')
        save_fg(img_data,matte,combined_name)

if __name__=='__main__':

    args = parse_parser()
    input_path = args.input_path
    if not os.path.isdir(input_path):
        print("Please configure test images input path correctly")
        sys.exit(0)
    ckpt_path = args.ckpt_path
    if not os.path.isfile(ckpt_path):
        print("Please configure ckpt-path correctly")
        sys.exit(0)
    input_images = []
    image_type = set(args.image_type.split(','))
    for root,dirs,files in os.walk(input_path):
        for file in files:
            if file.split('.')[-1].lower() in image_type:
                input_images.append(os.path.join(root,file))
    print('Total test images:{}'.format(len(input_images)))
    os.makedirs(args.output_path,exist_ok=True)
    output_path = args.output_path
    input_size = args.input_size
    infer(input_images,input_size,ckpt_path,output_path)

