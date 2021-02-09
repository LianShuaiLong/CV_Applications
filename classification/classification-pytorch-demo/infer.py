import torch
import torch.nn as nn
from net import Net
from PIL import Image
from preprocess import transform

import argparse
import os
import pdb
import glob

os.environ['CUDA_VISIBILE_DEVICES']='0'

class Get_Img(object):
    def __init__(self,img_folder):
        self.img_folder=img_folder
        self.imgs_list=[]
    def _search_img(self):
        for root,dirs,files in os.walk(self.img_folder):
            for file in files:
                fp = os.path.join(root,file)
                self.imgs_list.append(fp)
        return self.imgs_list
        

def infer(img_list,model_path):
    model = Net(3,10)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    idx_to_class = torch.load(model_path)['idx_to_class']
    model = model.cuda()
    model.eval()
    for img_name in img_list:
        img = Image.open(img_name)
        img = transform(img)[None,:,:,:].cuda()
        class_idx = model(img)
        pred = torch.argmax(class_idx,1).item()
        class_name = idx_to_class[pred]
        print('file:{},label:{},pred:{}'.format(img_name,img_name.split('/')[-2],class_name))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder',type=str,default='/workspace/pytorch_demo/dataset/cifar10/test/')
    parser.add_argument('--model_path',type=str,default='/workspace/pytorch_demo/checkpoint/model_20000.pth')
    args = parser.parse_args()
    img_obj = Get_Img(args.img_folder)
    img_list = img_obj._search_img()
    infer(img_list,args.model_path)


