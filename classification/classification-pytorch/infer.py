import torch
import torch.nn as nn
import torchvision.utils as vutil

from backbones.scratch_net import Net,Basic_Conv
from backbones.mobilenetv2 import Mobilenet_v2
from backbones.vgg19 import vgg19
from PIL import Image
from preprocess import transform
import numpy as np

import argparse
import os
import pdb
import glob

IMG_NAME = '' 

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
        

def infer(img_list,model_path,flag):
    model = Net(3,10)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    state_dict = torch.load(model_path,map_location='cpu')
    model.load_state_dict(state_dict['model_state_dict'])
    idx_to_class = state_dict['idx_to_class']
    model = model.cuda()#.to('cuda:1')
    model.eval()
    modules_for_plot = (Basic_Conv,torch.nn.AdaptiveAvgPool2d)
    if flag:
       for name, module in model.named_modules():
           #pdb.set_trace()
           if isinstance(module, modules_for_plot):
              module.register_forward_hook(hook_func)
    with torch.no_grad():
        for img_name in img_list:
            global IMG_NAME
            IMG_NAME = img_name
            img = Image.open(img_name)
            img = transform(img)[None,:,:,:].cuda()#.to('cuda:1')
            class_idx = model(img)
            pred = torch.argmax(class_idx,1).item()
            class_name = idx_to_class[pred]
            print('file:{},label:{},pred:{}'.format(img_name,img_name.split('/')[-2],class_name))

def hook_func(module,input,output):
    '''
    param input:
    param output:
    param module:
    '''
    image_name = get_image_name_func(module)
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)#(n,c,h,w)->(c,n,,h,w)
    vutil.save_image(data, image_name, pad_value=0.5)

def get_image_name_func(module):
    module_str = str(module)
    #pdb.set_trace()
    global IMG_NAME
    class_name = '.'.join(IMG_NAME.rsplit('.',1)[0].rsplit('/',2)[-2:])
    image_name = module_str.rsplit(':',1)[0].split('(')[-1].rsplit('_',1)[0]
    image_name = class_name+'.'+image_name+'.png'
    return image_name

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder',type=str,default='/workspace/classification/classification-pytorch/dataset/cifar10/test/')
    parser.add_argument('--model_path',type=str,default='/workspace/classification/classification-pytorch/checkpoint/model_10000.pth')
    parser.add_argument('--vision',type=bool,default=True)
    parser.add_argument('--GPU',type=int,default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.GPU)
    img_obj = Get_Img(args.img_folder)
    img_list = img_obj._search_img()
    flag = args.vision
    infer(img_list,args.model_path,flag)


