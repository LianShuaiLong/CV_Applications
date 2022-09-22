import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import cv2
import imageio

import argparse
import os
import sys
from tqdm import tqdm

from MODNet.src.models.modnet import *
import pdb

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path',type=str,default='./test_videos/demo.mp4',help='Test video')
    parser.add_argument('--output_path',type=str,default='./video_results',help='Result videos dir')
    parser.add_argument('--model_path',type=str,default='./model/modnet_webcam_portrait_matting.ckpt',help='Pretrained model path')
    parser.add_argument('--fps',type=int,default=20)
    parser.add_argument('--result-type', type=str, default='fg', choices=['fg', 'matte'],help='matte - save the alpha matte; fg - save the foreground')
    parser.add_argument('--save_gif',type=bool,default=False,help='Save result video in GIF format')
    args=parser.parse_args()
    return args

def combine_frames(src_video_frames,res_video_frames,rw,rh):
    if rw>=rh:
       rh_ = 256
       rw_ = int(rw/rh*256)
    else:
       rw_ = 256
       rh_ = int(rh/rw*256)
    src_res_video_frames=[]
    for i,frame in enumerate(src_video_frames):
        item1 = cv2.resize(frame,(rw_,rh_))
        item2 = cv2.resize(res_video_frames[i],(rw_,rh_))
        src_res_video_frames.append(np.concatenate((item1,item2),axis=1))
    return src_res_video_frames
       
    
def frame_2_gif(frames_list,gif_name):
    imageio.mimsave(gif_name,frames_list,'GIF') 
    print(f'save {gif_name}') 
    return  

def matting(video,result,alpah_matte=False,fps=30,GIF=True):
    src_video_frames=[]
    res_video_frames=[]
    Cap = cv2.VideoCapture(video)
    if Cap.isOpened():
       retval,frame = Cap.read()
    else:
       print(f'Cannot open video:{video}')
       sys.exit(0)
    frame_num = int(Cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(Cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(Cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = Cap.get(cv2.CAP_PROP_FPS)
    if frame_width>=frame_height:
       rh = 512
       rw = int(frame_width/frame_height*512)
    else:
       rw = 512
       rh = int(frame_height/frame_width*512)
    rw = rw-rw%32
    rh = rh-rh%32
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #pdb.set_trace()
    video_writer = cv2.VideoWriter(result,fourcc,fps,(frame_width,frame_height))
    print(f'start matting {video}...')
    with tqdm(range(frame_num)) as t:
         for c in t:
             frame_np = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
             frame_np = cv2.resize(frame_np,(rw,rh),cv2.INTER_AREA)
             if GIF:
                src_video_frames.append(frame_np)
             frame_PIL = Image.fromarray(frame_np)
             frame_tensor = transform(frame_PIL)
             frame_tensor = frame_tensor[None,:,:,:]
             if GPU:
                frame_tensor.to('cuda')
             with torch.no_grad():
                _,_,matte_tensor = modnet(frame_tensor,True)
             matte_tensor = matte_tensor.repeat(1,3,1,1)#(B,C,H,W),repeat 3 times in dim Channel
             matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
             if alpha_matte:
                view_np_ = matte_np * np.full(frame_np.shape, 255.0)
             else:
                view_np_ = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
             view_np = cv2.cvtColor(view_np_.astype(np.uint8), cv2.COLOR_RGB2BGR)
             view_np = cv2.resize(view_np, (frame_width,frame_height))
             video_writer.write(view_np)
             if GIF:
                res_video_frames.append(cv2.resize(view_np_.astype(np.uint8),(rw,rh),cv2.INTER_AREA))
             retval, frame = Cap.read()
             #c += 1

    video_writer.release()
    if GIF:
       src_video_frames = combine_frames(src_video_frames,res_video_frames,rw,rh)
       src_gif = video.rsplit('.',1)[0]+'.gif'
       res_gif = result.rsplit('.',1)[0]+'.gif'
       src_res_gif = result.rsplit('.',1)[0]+'_res.gif'
       frame_2_gif(src_video_frames,src_gif)
       frame_2_gif(res_video_frames,res_gif)
       frame_2_gif(src_video_frames,src_res_gif)
       
    print('Save the result video to {0}'.format(result))

            
if __name__=='__main__':

    args = parse_parser()
    model_path = args.model_path
    os.makedirs(args.output_path,exist_ok=True)
    if not os.path.isfile(model_path):
       print(f'Cannot find pretrained model:{model_path}')
       sys.exit(0)
    video_path = args.video_path
    if not os.path.isfile(video_path):
       print(f'Cannot find video:{video_path}')
       sys.exit(0)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    modnet = MODNet(backbone_pretrained=False)
    modnet=nn.DataParallel(modnet)
   
    GPU = True if torch.cuda.device_count()>0 else False    
    if GPU:
        print('matting with GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(model_path))
    else:
        print('matting with CPU...')
        modnet.load_state_dict(torch.load(model_path))
    modnet.eval()
    result = os.path.join(args.output_path,video_path.split('/')[-1].split('.')[0]+'.mp4')
    alpha_matte = True if args.result_type == 'matte' else False
    GIF = True if args.save_gif else False
    matting(video_path,result,alpha_matte,args.fps,GIF)
    
        


   
