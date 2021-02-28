import torch 
import torch.nn as nn

class Channel_Attention(nn.Module):
   '''
        method of Sequeeze-and-Excitation https://arxiv.org/pdf/1709.01507.pdf

   '''
   def __init__(self,in_channels,ratio):
       super(Channel_Attention,self).__init__()
       self.pool=nn.AdaptiveAvgPool2d(1)
       self.op=nn.Sequential(
               nn.Linear(in_channels,in_channels//ratio,bias=False),
               nn.ReLU(inplace=True),
               nn.Linear(in_channels//ratio,in_channels,bias=False),
               nn.Sigmoid()
               )
   def forward(self,x):
       b,c,_,_ = x.size()
       w = self.pool(x)
       w = w.view(b,c)
       w = self.op(w)
       w = w.view(b,c,1,1)
       out = x*w.expand_as(x)
       return out

class Spatial_Attention(nn.Module):
    def __init__(self,kernel_size):
        super(Spatial_Attention,self).__init__()
        self.op=nn.Sequential(
                nn.Conv2d(2,1,kernel_size=kernel_size,padding=1,stride=1,bias=False),
                nn.Sigmoid()
                )
    def forward(self,x):
        _,c,_,_ = x.size()
        mean_f = torch.mean(x,dim=1,keep_dim=True)
        max_f,max_idx = torch.max(x,dim=1,keep_dim=True)
        w = self.op(torch.cat((mean_f,max_f),1))
        #w = w.repeat(1,c,1,1)????
        w = w.expand_as(x)
        return x*w
	
class MB_Conv(nn.Module):
      '''
      Implement of mobilenetv2 : https://arxiv.org/pdf/1801.04381.pdf

      PointWise-DepthWise-PointWise
      '''
      def __init__(self,in_channels,out_channels,expansion,stride,padding):
         super(MB_Conv,self).__init__()
         self.stride = stride
         self.in_channels = in_channels
         self.out_channels = out_channels
         self.op = nn.Sequential(
                               nn.Conv2d(in_channels,in_channels*expansion,kernel_size=1,stride=1,padding=0,bias=False),
                               nn.BatchNorm2d(in_channels*expansion),
                               nn.ReLU6(inplace=True),
                               # o = (w+2p-f)/s+1
                               nn.Conv2d(in_channels*expansion,in_channels*expansion,kernel_size=3,stride=self.stride,padding=padding,groups=in_channels*expansion),
                               nn.BatchNorm2d(in_channels*expansion),
                               nn.ReLU6(inplace=True),
                               nn.Conv2d(in_channels*expansion,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
                               nn.BatchNorm2d(out_channels)
                                )
      def forward(self,x):
           if self.stride==1 and self.in_channels==self.out_channels:
              out = x+self.op(x)
           else:
              out = self.op(x)
           return out

class Dilation_Conv(nn.Module):
    def __init__(self,ratio,in_channels,out_channels,kernel_size,stride,padding):
        '''
        param ratio: dilation ratio
        '''
        super(Dilation_Conv,self).__init__()
        self.ratio = ratio
        self.op_standard = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=1))
        self.op_dilation = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernale_size=kernela_size,stride=stride,padding=padding,dilation=ratio))
    def forward(self,x):
        if self.ratio == 1:
            '''
            standard convolution
            '''
            return self.op_standard(x)
        else:
            '''
            dilation convolution
            '''
            return self.op_dilation(x)

class Group_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,groups):
        super(Group_Conv,self).__init__()
        '''
        param groups: groups == in_channels:depthwise convolution
                      groups == 1: standard convolution
                      1<groups<in_channels: group convolution 
        '''
        assert groups%in_channels == 0,'invalid groups,in_channel must be divisible by groups'
        assert groups%out_channels == 0,'invalid groups,out_channels must be divisible by groups'
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op_standard = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,groups=1)
                )
        self.op_group = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=kernel_Size,stride=stride,padding=padding,groups=groups)
                )
    def forward(self,x):
        if self.groups == 1:
            return self.op_standard(x)
        else:
            return self.op_group(x)
        

