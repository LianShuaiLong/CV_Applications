import torch
import torch.nn as nn
import pdb

class vgg19(nn.Module):
    '''
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    def __init__(self,in_channels,num_class,bn=False):
        super(vgg19,self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.bn = bn
        self.c = [3,64,128,256,512,512]
        self.stage1 = self._make_stage(in_channels=self.c[0],out_channels=self.c[1],rp=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.stage2 = self._make_stage(in_channels=self.c[1],out_channels=self.c[2],rp=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.stage3 = self._make_stage(in_channels=self.c[2],out_channels=self.c[3],rp=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.stage4 = self._make_stage(in_channels=self.c[3],out_channels=self.c[4],rp=4)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.stage5 = self._make_stage(in_channels=self.c[4],out_channels=self.c[5],rp=4)
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        '''
        classifier copy from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

        '''
        self.classifier = nn.Sequential(
                nn.Linear(512*7*7,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096,num_class)
                )

        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                self._init_conv(layer)
            elif isinstance(layer,nn.Linear):
                self._init_fc(layer)
            elif isinstance(layer,nn.BatchNorm2d):
                self._init_norm(layer)

        self._show_network()

    def _make_stage(self,in_channels,out_channels,rp):
        stage = []
        stage.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1))
        stage.append(nn.ReLU(inplace=True))
        for i in range(rp-1):
            stage.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,stride=1))
            stage.append(nn.ReLU(inplace=True))
            if self.bn:
                stage.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*stage)

    def _show_network(self):
        for idx,m in enumerate(self.children()):
            print(idx,'-->',m)

    def forward(self,x):
        b,c,h,w = x.size()
        #pdb.set_trace()
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.stage4(x)
        x = self.pool4(x)
        x = self.stage5(x)
        x = self.pool5(x)
        x = torch.flatten(x,1)#torch.flatten(input,start_dim=0,end_dim=-1)
        out = self.classifier(x)
        return out

    def _init_conv(self,conv):
        nn.init.kaiming_uniform_(conv.weight,a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias,0)
    def _init_norm(self,norm):
        if norm.weight is not None:
            '''
            norm.weight:Gamma
            norm.bias:beta
            https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html?highlight=batchnorm#torch.nn.BatchNorm2d
            '''
            nn.init.constant_(norm.weight,1)
            nn.init.constant_(norm.bias,0)
    def _init_fc(self,fc):
        nn.init.normal_(fc.weight,0,0.01)
        nn.init.constant_(fc.bias,0)

