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

# distribute train
import torch.multiprocessing as mp
import torch.distributed as dist

from backbones.scratch_net import Net
from backbones.mobilenetv2 import Mobilenet_v2 
from backbones.vgg19 import vgg19

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(name)s-%(message)s')
logger = logging.getLogger(os.path.basename(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--data_type',type=str,default='pic_folder',help='pic_label or pic_folder')
parser.add_argument('--img_path',type=str,default='/workspace/dataset/train_imgs')
parser.add_argument('--label_path',type=str,default='/workspace/dataset/train_imgs/label.txt')
parser.add_argument('--train_folder',type=str,default='/workspace/classification/classification-pytorch/dataset/cifar10/train')
parser.add_argument('--class_num',type=int,default=10)
parser.add_argument('--resume',type=bool,default=False)
parser.add_argument('--pretrained_model',type=str,default='/workspace/classification/classification-pytorch/pretrained/model.pth')
parser.add_argument('--epoch',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--log_step',type=int,default=10)
parser.add_argument('--save_step',type=int,default=100)
parser.add_argument('--checkpoint_dir',type=str,default='/workspace/classification/classification-pytorch/checkpoint/')
parser.add_argument('--backbone',type=str,default='scratch_net',help='scratch_net,vgg19,mobilenetv2')
###########################distribute train######################################
parser.add_argument('--nodes',type=int,default=1)                               #
parser.add_argument('--gpus',type=int,default=1,help='num gpus per node')       #
parser.add_argument('--nr',type=int,default=0,help='ranking within the nodes')  #
#################################################################################


args = parser.parse_args()

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


def prepare_data(args):
    class_to_idx={}
    if args.data_type == 'pic_label':
        train_dataset = Custom_Dataset(
                args.img_path,
                args.label_path,
                transform=transform
                )
    elif args.data_type == 'pic_folder':
        train_dataset = ImageFolder(
                args.train_folder,
                transform=transform
                )
        class_to_idx = train_dataset.class_to_idx

    #############################distribution################################
    train_sampler=torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank
            )

    train_loader = DataLoader(
             train_dataset,
             batch_size=args.batch_size,
             shuffle=False,
             num_workers=0,
             pin_memory=True,
             sampler=train_sampler
             )
    idx_to_class = dict(zip(class_to_idx.values(),class_to_idx.keys()))

    return train_sampler,train_loader,idx_to_class

def train(gpu,args):
    args.rank = args.nr*args.gpus+gpu
    dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank)
    # data
    train_sampler,train_loader,idx_to_class = prepare_data(args)
    # model
    if args.backbone == 'scratch_net':
        print('trained with scratch_net...')
        model = Net(3,args.class_num)
    elif args.backbone == 'mobilenetv2':
        print('trained with mobilenetv2...')
        model = Mobilenet_v2(3,args.class_num)
    elif args.backbone == 'vgg19':
        print('trained with vgg19...')
        model = vgg19(3,args.class_num,bn=False)
    if args.resume and os.path.isfile(args.pretrained_model):
        model.load_state_dict(torch.load(args.pretrained_model)['model_state_dict'])
        print('load pretrained model from:{}....'.format(args.pretrained_model))
    else:
        print('trained from scratch...')
    #model = nn.DataParallel(model).cuda()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model= nn.parallel.DistributedDataParallel(
            model,
            device_ids=[gpu]
            )
    # loss
    train_loss = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(train_loader)*args.epoch)
    # train
    iteration = 0
    total_steps = len(train_loader)*args.epoch 
    logger.info('total steps:{}'.format(total_steps))
    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)
        for idx,(img,label) in enumerate(train_loader):
            img,label = img.cuda(),label.cuda()
            model.train()
            pred = model(img)
            loss = train_loss(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            if iteration%args.log_step==0:
                correct_num = (torch.argmax(pred,1)==label).sum().cpu().data.numpy()
                batch_acc = correct_num/args.batch_size
                logger.info('step:{},lr:{},loss:{},batch_acc:{}'.format(idx+epoch*len(train_loader),optimizer.state_dict()['param_groups'][0]['lr'],loss,batch_acc))
            if iteration%args.save_step==0 or iteration==total_steps:
                save_dict={
                        'model_state_dict':model.module.state_dict(),
                        'learning_rate':optimizer.state_dict()['param_groups'][0]['lr'],
                        'train_loss':loss,
                        'train_acc':batch_acc,
                        'iter':idx+epoch*len(train_loader),
                        'idx_to_class':idx_to_class
                        }
                os.makedirs(args.checkpoint_dir,exist_ok=True)
                torch.save(save_dict,os.path.join(args.checkpoint_dir,'model_%d.pth'%iteration))
            iteration +=1

if __name__=='__main__':
    
    args.world_size=args.nodes*args.gpus
    args.lr = args.lr*args.nodes*args.gpus #//scale learning rate
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='8888'
    mp.spawn(train,nprocs=args.gpus,args=(args,),join=True)
