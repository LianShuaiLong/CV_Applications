from rembg import remove
from rembg_speed import naive_remove
from tqdm import tqdm

import cv2
import os
import sys
import argparse

import onnxruntime as ort

def remove_image_bg(image_name,output_dir,onnx_session=None):
    # img = Image.open(image_name)
    img = cv2.imread(image_name)
    img_rb = naive_remove(data=img,onnx_session=onnx_session) if onnx_session else remove(img)
    # img_rb.save(os.path.join(output_dir,f'{image_name.split("/")[-1].split(".")[0]}.png'))
    img_name = f'{image_name.split("/")[-1].split(".")[0]}_speed.png' if onnx_session else f'{image_name.split("/")[-1].split(".")[0]}_acc.png'
    cv2.imwrite(os.path.join(output_dir,img_name),img_rb)

def remove_video_bg(video_name,output_dir,onnx_session=None):
    cap = cv2.VideoCapture(video_name)

    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name,video_suffix = video_name.rsplit('/',1)[-1].split('.')[:]
    video_name = f'{video_name}_speed_res.{video_suffix}' if onnx_session else f'{video_name}_acc_res.{video_suffix}'
    result = os.path.join(output_dir,video_name)
    video_writer = cv2.VideoWriter(result,fourcc,fps,(frame_width,frame_height))

    for i in tqdm(range(frame_num)):
        ret,frame = cap.read()
        if ret:
            frame_rb = naive_remove(data=frame,onnx_session=onnx_session) if onnx_session else remove(frame)
            frame_rb = cv2.cvtColor(frame_rb,cv2.COLOR_BGRA2BGR)#能不能将有透明度通道的图片保存为视频
            video_writer.write(frame_rb)
        else:
            print(f'Reading error occurs on frame index:{i}')
            break
    video_writer.release()


def remove_bg(filename,output_dir,onnx_session=None):
    video_type=set(['mp4','avi'])
    image_type = set(['jpg','jpeg','png'])

    suffix = filename.rsplit('.',1)[-1].lower()

    if suffix in video_type:
        remove_video_bg(video_name=filename,output_dir = output_dir,onnx_session=onnx_session)
    elif suffix in image_type:
        remove_image_bg(image_name=filename,output_dir = output_dir,onnx_session=onnx_session)
    else:
        print(f'Invalid file type:{filename}')
        return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed',action='store_true',default=False,help='rembg:speed version')
    parser.add_argument('--input','-i',type=str,default='./test_data')
    parser.add_argument('--output','-o',type=str,default='./rembg_out')
    args = parser.parse_args()
    output_dir = args.output
    os.makedirs(output_dir,exist_ok=True)
    if args.speed:
        print('You are using SPEED version(around 0.6s per image)...')
        # providers=['TensorrtExecutionProvider','CUDAExecutionProvider', 'CPUExecutionProvider']
        providers=['CPUExecutionProvider']# bugs: onnxruntime version matters, upgrade to CUDA, speed gets lower
        onnx_path = os.path.expanduser('~/.u2net/u2net.onnx')
        onnx_session = ort.InferenceSession(onnx_path,providers=providers)
    else:
        print('You are using ACC version(around 1.6s per image)...')
        onnx_session = None
    if os.path.isdir(args.input):
        fnames = os.listdir(args.input)
        pbar = tqdm(fnames)
        for fname in pbar:
            pbar.set_description(f'{fname}')
            fpath = os.path.join(args.input,fname)
            remove_bg(fpath,output_dir,onnx_session) 
    elif os.path.isfile(args.input):
        remove_bg(args.input,output_dir,onnx_session)
    else:
        print(f'Invalid input path:{args.input}!')
        sys.exit(0)



