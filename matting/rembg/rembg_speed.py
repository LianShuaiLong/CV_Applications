# based on remove_bg_v2.py
import sys
import cv2
import os
import sys
import argparse
import pdb
import numpy as np

import onnxruntime as ort 
from PIL.Image import Image as PILImage
from PIL import Image
from enum import Enum
from typing import List, Optional, Union
from tqdm import tqdm

from utils import(
    preprocess_img,
    postprocess_mask,
    adjust_mask,
    naive_cutout,
    alpha_matting_cutout
)



class ReturnType(Enum):
    BYTES = 0
    PILLOW = 1
    NDARRAY = 2


def naive_remove(
    onnx_session,
    data: Union[bytes, PILImage, np.ndarray],
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    only_mask: bool = True,
    post_process_mask: bool = True,
   ):
    if isinstance(data, PILImage):
        return_type = ReturnType.PILLOW
        img = data
    elif isinstance(data, bytes):
        return_type = ReturnType.BYTES
        img = Image.open(io.BytesIO(data))
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = Image.fromarray(data)
    else:
        raise ValueError("Input type {} is not supported.".format(type(data)))

    tmpImg,t_c = preprocess_img(img)
    #print(f'pre:{t_c}')
    ort_outs = onnx_session.run(None,{onnx_session.get_inputs()[0].name:np.expand_dims(tmpImg,0).astype(np.float32)})
    pred = ort_outs[0][:,0,:,:]
    mask,t_c = postprocess_mask(img,pred)
    #print(f'mask:{t_c}')
    if ReturnType.NDARRAY == return_type:
        mask_tmp = np.asarray(mask)
        cv2.imwrite('mask.jpg',mask_tmp)


    if post_process_mask:
        mask,t_c = adjust_mask(np.array(mask))
        mask= Image.fromarray(mask)
        #print(f'adjust:{t_c}')

    if only_mask:
        cutout = mask
    elif alpha_matting:
        try:
            cutout,t_c = alpha_matting_cutout(
                img,
                mask,
                alpha_matting_foreground_threshold,
                alpha_matting_background_threshold,
                alpha_matting_erode_size,
            )
            #print(f'alpha:{t_c}')
        except ValueError:
            cutout = naive_cutout(img, mask)
    else:
        cutout = naive_cutout(img, mask)


    if ReturnType.PILLOW == return_type:
        return cutout

    if ReturnType.NDARRAY == return_type:
        return np.asarray(cutout)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()


def remove_video_bg(video_name,onnx_session,output_dir):
    cap = cv2.VideoCapture(video_name)

    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name,video_suffix = video_name.rsplit('/',1)[-1].split('.')[:]
    video_name = f'{video_name}_res.{video_suffix}'
    result = os.path.join(output_dir,video_name)
    video_writer = cv2.VideoWriter(result,fourcc,fps,(frame_width,frame_height))

    for i in tqdm(range(frame_num)):
        ret,frame = cap.read()
        if ret:
            frame_rb = naive_remove(data=frame,onnx_session=onnx_session)
            frame_rb = cv2.cvtColor(frame_rb,cv2.COLOR_BGRA2BGR)#?
            video_writer.write(frame_rb)
        else:
            print(f'Reading error occurs on frame index:{i}') 
            break
    video_writer.release()


def remove_image_bg(image_name,onnx_session,output_dir):
    img = Image.open(image_name)
    img_rb = naive_remove(data=img,onnx_session=onnx_session)
    img_rb.save(os.path.join(output_dir,f'{image_name.split("/")[-1].split(".")[0]}.png'))
    
def remove_bg(filename,onnx_session,output_dir):
    video_type=set(['mp4','avi'])
    image_type = set(['jpg','jpeg','png'])
    
    suffix = filename.rsplit('.',1)[-1]

    if suffix in video_type:
        remove_video_bg(video_name=filename,onnx_session=onnx_session,output_dir = output_dir)
    elif suffix in image_type:
        remove_image_bg(image_name=filename,onnx_session=onnx_session,output_dir = output_dir)
    else:
        print(f'Invalid file type:{filename}')
        return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input','-i',type=str,default='./test_data')
    parser.add_argument('--output','-o',type=str,default='./rembg_out')
    args = parser.parse_args()
    providers=['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider']
    onnx_session = ort.InferenceSession('./u2net.onnx',providers=providers)
    output_dir = args.output
    os.makedirs(output_dir,exist_ok=True)
    if os.path.isdir(args.input):
        fnames = os.listdir(args.input)
        pbar = tqdm(fnames)
        for fname in pbar:
            pbar.set_description(f'{fname}')
            fpath = os.path.join(args.input,fname)
            #print(fpath)
            remove_bg(fpath,onnx_session,output_dir)
    elif os.path.isfile(args.input):
        remove_bg(args.input,onnx_session,output_dir)
    else:
        #print('Invalid input path!')
        sys.exit(0)
        

