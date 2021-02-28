import torch
import torch.nn as nn
import pdb

from backbones.Blocks import Channel_Attention

class Basic_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,layer_name):
        super(Basic_Conv,self).__init__()
        self.op = nn.Sequential()
        self.op.add_module(layer_name+'_conv',nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding))
        self.op.add_module(layer_name+'_bn',nn.BatchNorm2d(out_channels))
        self.op.add_module(layer_name+'_relu',nn.ReLU(inplace=True))
    def forward(self,x):
        return self.op(x)


class Net(nn.Module):
    def __init__(self,in_channels,class_num):
        super(Net,self).__init__()
        self.in_channels = in_channels
        self.class_num = class_num
        self.layers=[
                Basic_Conv(in_channels,32,kernel_size=3,stride=1,padding=1,layer_name='Basic_Conv1'),
                Basic_Conv(32,64,kernel_size=3,stride=1,padding=1,layer_name='Basic_Conv2'),
                Basic_Conv(64,128,kernel_size=3,stride=1,padding=1,layer_name='Basic_Conv3'),
                Basic_Conv(128,256,kernel_size=3,stride=2,padding=0,layer_name='Basic_Conv4'),
                Channel_Attention(256,1)
                ]
        self.net = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Linear(256,class_num,bias=True)

        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                self._init_conv(layer)
            if isinstance(layer,nn.BatchNorm2d) or isinstance(layer,nn.InstanceNorm2d):
                self._init_norm(layer)
            if isinstance(layer,nn.Linear):
                self._init_fc(layer)
        self._show_net()

    def _show_net(self):
        for idx,m in enumerate(self.named_children()):
            print(idx,'-->',m)

    def forward(self,x):
        #pdb.set_trace()
        b,_,_,_ = x.size()
        x = self.net(x)
        x_global = self.pool(x).view(b,256)
        out = self.fc(x_global)
        return out
    def _init_conv(self,conv):
        nn.init.kaiming_uniform_(conv.weight,a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias,0)
    def _init_norm(self,norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight,1)
            nn.init.constant_(norm.bias,0)
    def _init_fc(self,fc):
        nn.init.kaiming_uniform_(fc.weight,a=0,mode='fan_in',nonlinearity='leaky_relu')

