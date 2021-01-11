import cv2
import argparse
import os
import sys
import glob
import numpy as np
from PIL import Image

def parse_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_image_folder',type=str,default='./test_images',help='')
    parser.add_argument('--matting_folder',type=str,default='./test_result',help='')
    parser.add_argument('--matting_result_folder',type=str,default='./matting_results',help='')
    parser.add_argument('--image_type',type=str,default='jpg,jpeg,png')
    args=parser.parse_args()
    return args

    
def matting(image,matte,matting_result_folder):
   print(f'{image} begin to process...')
   img = Image.open(image)
   w,h = img.width,img.height 
   img = np.asarray(img)
   matte = Image.open(matte)
   if len(img.shape) == 2:
      img = img[:,:,None]
   if img.shape[2] == 1:
      img = np.repeat(img,3,axis=2)
   elif img.shape[2] == 4:
      img = img[:,:,0:3]
   matte = np.repeat(np.asarray(matte)[:,:,None],3,axis=2)/255
   fg = matte*img+(1-matte)*np.full(img.shape,255)
   combined = np.concatenate((img, fg, matte * 255), axis=1)
   combined = Image.fromarray(np.uint8(combined))
   combined.save(os.path.join(matting_result_folder,image.split('/')[-1]))
   print(f'{image} finish')
  

if __name__=='__main__':
   args = parse_parser()
   ori_image_folder = args.ori_image_folder
   matting_folder = args.matting_folder
   matting_result_folder = args.matting_result_folder
   image_type = set(args.image_type.split(','))
   if not os.path.isdir(ori_image_folder):
      print('Cannot find original images folder')
      sys.exit(0)
   if not os.path.isdir(matting_folder):
      print('Cannot find matting folder')
      sys.exit(0)
   os.makedirs(matting_result_folder,exist_ok=True)
   ori_images=[]
   for img_type in image_type:
       ori_images.extend(glob.glob(os.path.join(ori_image_folder,f'*.{img_type}')))
   for image in ori_images:
       image_name = image.split('/')[-1].split('.')[0]+'.png'
       if not os.path.isfile(os.path.join(matting_folder,image_name)):
          print('Cannot find matting of {}'.format(image))
          continue
       matting(image,os.path.join(matting_folder,image_name),matting_result_folder)
       
