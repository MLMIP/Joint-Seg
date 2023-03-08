# -*- coding: utf-8 -*-
"""
 @Time    : 2021/11/6 9:47
 @Author  : SvyJ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils import FeatureSelectionModule, FeatureAlignModule, MultiscaleSoftFusion


# #########--------- Components ---------#########
class DoubleConv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1), 
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace = True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace = True)  
            )

    def forward(self, x):
        return self.conv(x)


class SingleConv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1), 
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace = True) 
            )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x


# #########--------- Networks ---------#########
class base(nn.Module):

    def __init__(self, num_classes):
        super(base, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        self.conv1 = DoubleConv(3, filters[0])
        self.pool1 = nn.MaxPool2d(2) 

        self.conv2 = DoubleConv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(filters[3], filters[4])

        # FAZ path decoder
        self.faz_up6 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.faz_conv6 = DoubleConv(ch_in=filters[3]*2, ch_out=filters[3])

        self.faz_up7 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.faz_conv7 = DoubleConv(ch_in=filters[2]*2, ch_out=filters[2])

        self.faz_up8 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.faz_conv8 = DoubleConv(ch_in=filters[1]*2, ch_out=filters[1])

        self.faz_up9 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.faz_conv9 = DoubleConv(ch_in=filters[0]*2, ch_out=filters[0])

        # RV path decoder
        self.rv_up7 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.rv_conv7 = DoubleConv(ch_in=filters[2]*2, ch_out=filters[2])

        self.rv_up8 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.rv_conv8 = DoubleConv(ch_in=filters[1]*2, ch_out=filters[1])

        self.rv_up9 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.rv_conv9 = DoubleConv(ch_in=filters[0]*2, ch_out=filters[0])

        self.out = nn.Conv2d(filters[0], num_classes, 3, 1, 1)
        
    def forward(self, x):
        # encoding path
        conv_out_1 = self.conv1(x)
        pool_out_1 = self.pool1(conv_out_1)
        conv_out_2 = self.conv2(pool_out_1)
        pool_out_2 = self.pool2(conv_out_2)
        conv_out_3 = self.conv3(pool_out_2)
        pool_out_3 = self.pool3(conv_out_3)
        conv_out_4 = self.conv4(pool_out_3)
        pool_out_4 = self.pool4(conv_out_4)
        conv_out_5 = self.conv5(pool_out_4)

        # decoder path for FAZ branch
        faz_d6 = self.faz_up6(conv_out_5)
        faz_cat6 = torch.cat([faz_d6, conv_out_4], dim=1)
        faz_d6 = self.faz_conv6(faz_cat6)

        faz_d7 = self.faz_up7(faz_d6)
        faz_cat7 = torch.cat([faz_d7, conv_out_3], dim=1)
        faz_d7 = self.faz_conv7(faz_cat7)

        faz_d8 = self.faz_up8(faz_d7)
        faz_cat8 = torch.cat([faz_d8, conv_out_2], dim=1)
        faz_d8 = self.faz_conv8(faz_cat8)

        faz_d9 = self.faz_up9(faz_d8)
        faz_cat9 = torch.cat([faz_d9, conv_out_1], dim=1)
        faz_d9 = self.faz_conv9(faz_cat9)

        # decoder path for RV branch
        rv_d7 = self.rv_up7(conv_out_4)
        rv_cat7 = torch.cat([rv_d7, conv_out_3], dim=1)
        rv_d7 = self.rv_conv7(rv_cat7)

        rv_d8 = self.rv_up8(rv_d7)
        rv_cat8 = torch.cat([rv_d8, conv_out_2], dim=1)
        rv_d8 = self.rv_conv8(rv_cat8)

        rv_d9 = self.rv_up9(rv_d8)
        rv_cat9 = torch.cat([rv_d9, conv_out_1], dim=1)
        rv_d9 = self.rv_conv9(rv_cat9)
        
        return self.out(faz_d9), self.out(rv_d9)

class full(nn.Module):

    def __init__(self, num_classes=1):
        super(FRS_VGG, self).__init__()
        filters = [32, 64, 128, 256, 512]
        self.conv1 = DoubleConv(3, filters[0])
        self.pool1 = nn.MaxPool2d(2) 

        self.conv2 = DoubleConv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(filters[3], filters[4])

        # FAZ path decoder
        self.faz_fsm5 = FeatureSelectionModule(in_chan=filters[4], out_chan=filters[3])

        self.faz_up6 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.faz_fsm6 = FeatureSelectionModule(in_chan=filters[3], out_chan=filters[2])
        self.faz_fam6 = FeatureAlignModule(in_nc=filters[2], out_nc=filters[2])
        self.faz_conv6 = DoubleConv(ch_in=filters[3], ch_out=filters[3])

        self.faz_up7 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.faz_fsm7 = FeatureSelectionModule(in_chan=filters[2], out_chan=filters[1])
        self.faz_fam7 = FeatureAlignModule(in_nc=filters[1], out_nc=filters[1])
        self.faz_conv7 = DoubleConv(ch_in=filters[2], ch_out=filters[2])

        self.faz_up8 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.faz_fsm8 = FeatureSelectionModule(in_chan=filters[1], out_chan=filters[0])
        self.faz_fam8 = FeatureAlignModule(in_nc=filters[0], out_nc=filters[0])
        self.faz_conv8 = DoubleConv(ch_in=filters[1], ch_out=filters[1])

        self.faz_up9 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.faz_fsm9 = FeatureSelectionModule(in_chan=filters[0], out_chan=filters[0]//2)
        self.faz_fam9 = FeatureAlignModule(in_nc=filters[0]//2, out_nc=filters[0]//2)
        self.faz_conv9 = DoubleConv(ch_in=filters[0], ch_out=filters[0])

        # RV path decoder
        self.rv_fsm6 = FeatureSelectionModule(in_chan=filters[3], out_chan=filters[2])
        self.rv_attention = MultiscaleSoftFusion(filters[2])

        self.rv_up7 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.rv_fsm7 = FeatureSelectionModule(in_chan=filters[2], out_chan=filters[1])
        self.rv_conv7 = DoubleConv(ch_in=filters[1]*2, ch_out=filters[1])

        self.rv_up8 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.rv_fsm8 = FeatureSelectionModule(in_chan=filters[1], out_chan=filters[0])
        self.rv_conv8 = DoubleConv(ch_in=filters[0]*2, ch_out=filters[0])

        self.rv_up9 = up_conv(ch_in=filters[0], ch_out=filters[0]//2)
        self.rv_fsm9 = FeatureSelectionModule(in_chan=filters[0], out_chan=filters[0]//2)
        self.rv_conv9 = DoubleConv(ch_in=filters[0], ch_out=filters[0])

        self.out = nn.Conv2d(filters[0], num_classes, 3, 1, 1)
        
    def forward(self, x):
        # encoding path
        conv_out_1 = self.conv1(x)         
        pool_out_1 = self.pool1(conv_out_1)
        conv_out_2 = self.conv2(pool_out_1)
        pool_out_2 = self.pool2(conv_out_2)
        conv_out_3 = self.conv3(pool_out_2)
        pool_out_3 = self.pool3(conv_out_3)
        conv_out_4 = self.conv4(pool_out_3)
        pool_out_4 = self.pool4(conv_out_4)
        conv_out_5 = self.conv5(pool_out_4)

        # decoder path for FAZ branch
        faz_d6 = self.faz_up6(conv_out_5)
        faz_conv_out_4 = self.faz_fsm6(conv_out_4) 
        faz_cat6 = torch.cat([self.faz_fam6(faz_conv_out_4, faz_d6), faz_conv_out_4], dim=1)
        faz_d6 = self.faz_conv6(faz_cat6)

        faz_d7 = self.faz_up7(faz_d6) 
        faz_conv_out_3 = self.faz_fsm7(conv_out_3)
        faz_cat7 = torch.cat([self.faz_fam7(faz_conv_out_3, faz_d7), faz_conv_out_3], dim=1)
        faz_d7 = self.faz_conv7(faz_cat7)

        faz_d8 = self.faz_up8(faz_d7)    
        faz_conv_out_2 = self.faz_fsm8(conv_out_2) 
        faz_cat8 = torch.cat([self.faz_fam8(faz_conv_out_2, faz_d8), faz_conv_out_2], dim=1)
        faz_d8 = self.faz_conv8(faz_cat8)  

        faz_d9 = self.faz_up9(faz_d8)       
        faz_conv_out_1 = self.faz_fsm9(conv_out_1) 
        faz_cat9 = torch.cat([self.faz_fam9(faz_conv_out_1, faz_d9), faz_conv_out_1], dim=1) 
        faz_d9 = self.faz_conv9(faz_cat9) 

        # decoder path for RV branch
        msf = self.rv_fsm6(conv_out_4) + self.rv_attention(self.rv_fsm6(conv_out_4))

        rv_d7 = self.rv_up7(msf)    
        rv_cat7 = torch.cat([rv_d7, self.rv_fsm7(conv_out_3)], dim=1) 
        rv_d7 = self.rv_conv7(rv_cat7) 

        rv_d8 = self.rv_up8(rv_d7)     
        rv_cat8 = torch.cat([rv_d8, self.rv_fsm8(conv_out_2)], dim=1) 
        rv_d8 = self.rv_conv8(rv_cat8) 

        rv_d9 = self.rv_up9(rv_d8)   
        rv_cat9 = torch.cat([rv_d9, self.rv_fsm9(conv_out_1)], dim=1) 
        rv_d9 = self.rv_conv9(rv_cat9)
        
        return torch.sigmoid(self.out(faz_d9)), torch.sigmoid(self.out(rv_d9))
