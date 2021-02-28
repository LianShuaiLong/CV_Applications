import torch
import torch.nn as nn
import pdb
from backbones.Blocks import MB_Conv 

class Mobilenet_v2(nn.Module):
      def __init__(self,in_channels,num_class):
          super(Mobilenet_v2,self).__init__()
          self.num_channels=[32,16,24,32,64,96,160,320,1280]
          self.c = self.num_channels
          self.n=[1, 1, 2, 3, 4, 3, 3, 1, 1, 1]
          self.s=[2, 1, 2, 2, 2, 1, 2, 1, 1]
          self.t=[1, 6, 6, 6, 6, 6, 6]
          self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=self.c[0],kernel_size=3,stride=self.s[0],padding=1)
          self.bottleneck1 = self._make_stage(in_c=self.c[0],out_c=self.c[1],expansion=self.t[0],stride=self.s[1],rp=self.n[1])
          self.bottleneck2 = self._make_stage(in_c=self.c[1],out_c=self.c[2],expansion=self.t[1],stride=self.s[2],rp=self.n[2])
          self.bottleneck3 = self._make_stage(in_c=self.c[2],out_c=self.c[3],expansion=self.t[2],stride=self.s[3],rp=self.n[3])
          self.bottleneck4 = self._make_stage(in_c=self.c[3],out_c=self.c[4],expansion=self.t[3],stride=self.s[4],rp=self.n[4])
          self.bottleneck5 = self._make_stage(in_c=self.c[4],out_c=self.c[5],expansion=self.t[4],stride=self.s[5],rp=self.n[5])
          self.bottleneck6 = self._make_stage(in_c=self.c[5],out_c=self.c[6],expansion=self.t[5],stride=self.s[6],rp=self.n[6])
          self.bottleneck7 = self._make_stage(in_c=self.c[6],out_c=self.c[7],expansion=self.t[6],stride=self.s[7],rp=self.n[7])
          self.conv2=nn.Conv2d(in_channels=self.c[7],out_channels=self.c[8],kernel_size=1,stride=self.s[8])
          self.pool = nn.AdaptiveAvgPool2d(1)
          self.fc = nn.Linear(self.c[-1],num_class)
          
          for layer in self.modules():
              if isinstance(layer,nn.Conv2d):
                 self._init_conv(layer)
              elif isinstance(layer,nn.BatchNorm2d) or isinstance(layer,nn.InstanceNorm2d):
                 self._init_norm(layer)
              elif isinstance(layer,nn.Linear):
                 self._init_fc(layer)
          
          self._show_network()

      def _make_stage(self,in_c,out_c,expansion,stride,rp):
          stage = []
          stage.append(MB_Conv(in_channels=in_c,out_channels=out_c,expansion=expansion,stride=stride,padding=1))
          for i in range(rp-1):
              stage.append(MB_Conv(in_channels=out_c,out_channels=out_c,expansion=expansion,stride=1,padding=1))
          return nn.Sequential(*stage)

      def _show_network(self):
          print('Network:')
          for idx,m in enumerate(self.named_children()):
              print(idx,'->',m)
       
      def forward(self,x):
          b,_,_,_=x.size()
          #pdb.set_trace()
          x = self.conv1(x)
          x = self.bottleneck1(x)
          x = self.bottleneck2(x)
          x = self.bottleneck3(x)
          x = self.bottleneck4(x)
          x = self.bottleneck5(x)
          x = self.bottleneck6(x)
          x = self.bottleneck7(x)
          x = self.conv2(x)
          x = self.pool(x).view(b,self.c[-1])
          out = self.fc(x)
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

          

