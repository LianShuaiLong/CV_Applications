import argparse
import logging
import os

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

from backbones.mobilenetv2 import Mobilenet_v2 

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(name)s-%(message)s')
logger = logging.getLogger(os.path.basename(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--train_folder',type=str,default='/workspace/classification/classification-pytorch/dataset/cifar10/train')
parser.add_argument('--class_num',type=int,default=10)
parser.add_argument('--epoch',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--log_step',type=int,default=10)
parser.add_argument('--save_step',type=int,default=100)
parser.add_argument('--checkpoint_dir',type=str,default='/workspace/classification/classification-pytorch/checkpoint/')
###########################distribute train######################################
parser.add_argument('--nodes',type=int,default=1)                               #
parser.add_argument('--gpus',type=int,default=1,help='num gpus per node')       #
parser.add_argument('--nr',type=int,default=0,help='ranking within the nodes')  #
#################################################################################


args = parser.parse_args()

def prepare_data(args):
    train_dataset = ImageFolder(args.train_folder,transform=transform)
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
    torch.manual_seed(100)
    args.rank = args.nr*args.gpus+gpu
    dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank)
    # data
    train_sampler,train_loader,idx_to_class = prepare_data(args)
    # model
    model = Mobilenet_v2(3,args.class_num)

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
