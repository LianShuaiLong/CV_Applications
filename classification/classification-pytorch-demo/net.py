import torch
import torch.nn as nn
import pdb

class Basic_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Basic_Conv,self).__init__()
        self.op = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                )
    def forward(self,x):
        return self.op(x)

class Channel_Attention(nn.Module):
    '''
    Squeeze-and-Excitation method
    '''
    def __init__(self,in_channels,ratio):
        super(Channel_Attention,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.op = nn.Sequential(
                nn.Linear(in_channels,in_channels//ratio,bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels//ratio,in_channels,bias=False),
                nn.Sigmoid()
                )
    def forward(self,x):
        b,c,h,w = x.size()
        w = self.pool(x).view(b,c)
        w = self.op(w).view(b,c,1,1)
        out = x*w.expand_as(x)
        return out

class Spatial_Attention(nn.Module):
    def __init__(self,kernel_size):
        super(Spatial_Attention,self).__init__()
        padding = kernel_size//2
        self.op = nn.Sequential(
                nn.Conv2d(2,1,kernel_size=kernel_size,stride=1,padding=padding,bias=False),
                nn.Sigmoid()
                )
    def foward(self,x):
        x_mean = torch.mean(x,dim=1,keep_dim=True)
        x_max,x_max_id = torch.max(x,dim=1,keep_dim=True)
        w = torch.cat((x_mean,x_max),1)
        w = self.op(w)
        return x*w.expand_as(x)

        
#class Self_Attention(nn.Module):
#    def __init__(self,in_channels):
#        super(Self_Attention).__init__()
#    def forward(self,x):
#        return x

class Net(nn.Module):
    def __init__(self,in_channels,class_num):
        super(Net,self).__init__()
        self.in_channels = in_channels
        self.class_num = class_num
        self.layers=[
                Basic_Conv(in_channels,32,kernel_size=3,stride=1,padding=1),
                Basic_Conv(32,64,kernel_size=3,stride=1,padding=1),
                Basic_Conv(64,128,kernel_size=3,stride=1,padding=1),
                Basic_Conv(128,256,kernel_size=3,stride=2,padding=0),
                Channel_Attention(256,1)
                ]
        self.pool = nn.AdaptiveMaxPool2d(1) 
        self.fc = nn.Linear(256,class_num,bias=True)
                
        self.net = nn.Sequential(*self.layers)
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                self._init_conv(layer)
            if isinstance(layer,nn.BatchNorm2d) or isinstance(layer,nn.InstanceNorm2d):
                self._init_norm(layer)
            if isinstance(layer,nn.Linear):
                self._init_fc(layer)
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

