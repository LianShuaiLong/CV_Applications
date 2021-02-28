import pyinotify
import torch
from preprocess import transform
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Dataset
#from backbones.scratch_net import Net as Net
from backbones.mobilenetv2 import Mobilenet_v2 as Net

import argparse
import logging
import os
import glob

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(name)s-%(message)s')
logger = logging.getLogger(os.path.basename(__file__))

parser=argparse.ArgumentParser()
parser.add_argument('--test_folder',type=str,default='/workspace/pytorch_demo/dataset/cifar10/test')
parser.add_argument('--log_step',type=int,default=100)
parser.add_argument('--checkpoint_dir',type=str,default='/workspace/pytorch_demo/checkpoint')
parser.add_argument('--GPU',type=str,default='0')
args=parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.GPU
test_dataset = ImageFolder(args.test_folder,transform=transform)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)


def evaluate(model_path):
    model = Net(3,10)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model = model.cuda()
    model.eval()
    correct_num=0
    total_num=0
    for idx,(img,label) in enumerate(test_loader):
            img,label = img.cuda(),label.cuda()
            pred = model(img)
            correct_num += (torch.argmax(pred,1)==label).sum().cpu().data.numpy()
            total_num += 64
            if idx % args.log_step==0:
                batch_acc= (torch.argmax(pred,1)==label).sum().cpu().data.numpy()/64
                print('step:[{}],batch_acc:{}'.format(idx,batch_acc))
    logger.info('model:{},acc:{}'.format(model_path,correct_num/total_num))

class CREATE_EventHandler(pyinotify.ProcessEvent):
    def process_IN_CLOSE_WRITE(self,event):
        file_dir = args.checkpoint_dir
        file_time = {}
        file_list = glob.glob(os.path.join(file_dir,'*.pth'))
        for file in file_list:
            c_time = os.stat(file).st_ctime
            file_time[c_time] = file
        newest_file = file_time[max(file_time.keys())]
        print('begin to evalue:{}'.format(newest_file))
        evaluate(newest_file)

notify_event = pyinotify.IN_CLOSE_WRITE
wm = pyinotify.WatchManager()
handler = CREATE_EventHandler()
notifier = pyinotify.Notifier(wm,handler)
wm.add_watch(args.checkpoint_dir,notify_event)
notifier.loop()

