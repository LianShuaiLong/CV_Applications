import argparse
import glob
from tqdm import tqdm
import logging
import os
import pdb

from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
import torch.optim as optim
from preprocess import transform
import torch
import torch.nn as nn

from net import Net

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(name)s-%(message)s')
logger = logging.getLogger(os.path.basename(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data_type',type=str,default='pic_folder',help='pic_label or pic_folder')
parser.add_argument('--img_path',type=str,default='/workspace/dataset/train_imgs')
parser.add_argument('--label_path',type=str,default='/workspace/dataset/train_imgs/label.txt')
parser.add_argument('--train_folder',type=str,default='/workspace/pytorch_demo/dataset/cifar10/train')
parser.add_argument('--class_num',type=int,default=10)
parser.add_argument('--resume',type=bool,default=False)
parser.add_argument('--pretrained_model',type=str,default='/workspace/pytorch_demo/pretrained/model.pth')
parser.add_argument('--epoch',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--log_step',type=int,default=10)
parser.add_argument('--save_step',type=int,default=100)
parser.add_argument('--checkpoint_dir',type=str,default='/workspace/pytorch_demo/checkpoint/')
parser.add_argument('--GPU',type=str,default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.GPU
class Custom_Dataset(Dataset):
    def __init__(self,img_path,label_path,transform):
        super(Custom_Dataset,self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.file_list = open(label_path,'r').readlines()
        self.transform = transform
    def __getitem__(self,index):
        img,label = self.file_list[index].strip()
        image = Image.open(os.path.join(self.img_path,img))
        image_tensor = self.transform(image)
        return image_tensor,label
    def __len__(self):
        return len(self.file_list)


def prepare_data(opt):
    class_to_idx={}
    if opt['data_type'] == 'pic_label':
        train_dataset = Custom_Dataset(opt['img_path'],opt['label_path'],transform=transform)
    elif opt['data_type'] == 'pic_folder':
        train_dataset = ImageFolder(opt['train_folder'],transform=transform)
        class_to_idx = train_dataset.class_to_idx
    
    train_loader = DataLoader(train_dataset,batch_size=opt['batch_size'],shuffle=True)
    idx_to_class = dict(zip(class_to_idx.values(),class_to_idx.keys()))

    return train_loader,idx_to_class

def train(opt):
    # data
    train_loader,idx_to_class = prepare_data(opt)
    # model
    model = Net(3,opt['class_num'])
    if opt['resume'] and os.path.isfile(opt['pretrained_model']):
        model.load_state_dict(torch.load(opt['pretrained_model'])['model_state_dict'])
        print('load pretrained model from:{}....'.format(opt['pretrained_model']))
    else:
        print('trained from scratch...')
    #pdb.set_trace()
    model = nn.DataParallel(model).cuda()
    # loss
    train_loss = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.SGD(model.parameters(),lr=opt['lr'],momentum=0.9)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(train_loader)*opt['epoch'])
    # train
    iteration = 0
    total_steps = len(train_loader)*opt['epoch'] 
    logger.info('total steps:{}'.format(total_steps))
    for epoch in range(opt['epoch']):
        for idx,(img,label) in enumerate(train_loader):
            img,label = img.cuda(),label.cuda()
            model.train()
            pred = model(img)
            loss = train_loss(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            if iteration%opt['log_step']==0:
                correct_num = (torch.argmax(pred,1)==label).sum().cpu().data.numpy()
                batch_acc = correct_num/opt['batch_size']
                logger.info('step:{},lr:{},loss:{},batch_acc:{}'.format(idx+epoch*len(train_loader),optimizer.state_dict()['param_groups'][0]['lr'],loss,batch_acc))
            if iteration%opt['save_step']==0 or iteration==total_steps:
                save_dict={
                        'model_state_dict':model.module.state_dict(),
                        'learning_rate':optimizer.state_dict()['param_groups'][0]['lr'],
                        'train_loss':loss,
                        'train_acc':batch_acc,
                        'iter':idx+epoch*len(train_loader),
                        'idx_to_class':idx_to_class
                        }
                os.makedirs(opt['checkpoint_dir'],exist_ok=True)
                torch.save(save_dict,os.path.join(opt['checkpoint_dir'],'model_%d.pth'%iteration))
            iteration +=1

if __name__=='__main__':
    opt = vars(args)
    train(opt)
