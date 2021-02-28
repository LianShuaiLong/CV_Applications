import torch
import torch.nn as nn
import pdb

class residual_block(nn.Module):
    def __init__(self,in_channels):
        super(residual_block,self).__init__()
        self.op = nn.Sequential(
                nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,padding=padding),
                nn.ReLU(inpalce=True),
                nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,padding=padding)
                )
    def forward(self,x):
        residual = self.op(x)
        out = x+residual
        out = nn.ReLU(out)
        return out


class resnet(nn.Module):
    def __init__(self,in_channels):
        super(resnet,self).__init__()
        
#https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py        
