import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from double_conv import DoubleConv
from attention_block import AttentionBlock
from depthwise_seperable_conv import DepthwiseSeperableConv
from ghost_conv import GhostConv
from up_conv import UpConv
import torch
import torch.nn as nn

class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(AtrousConv, self).__init__()
        self.atrous_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation_rate, dilation=dilation_rate, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UCrazyNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UCrazyNet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ghost conv for initial feature extraction 
        self.conv1 = GhostConv(in_channels, 64)
        self.conv2 = GhostConv(64, 128)

        # depthwise 
        self.conv3 = DepthwiseSeperableConv(128, 256)
        self.conv4 = DepthwiseSeperableConv(256, 512)
        self.conv5 = DepthwiseSeperableConv(512, 1024)

        # "skip" connections
        # AtrousConv per layer before downsampling
        self.atrous1 = AtrousConv(64, 64, dilation_rate=2)
        self.atrous2 = AtrousConv(128, 128, dilation_rate=4)
        self.atrous3 = AtrousConv(256, 256, dilation_rate=8)
        self.atrous4 = AtrousConv(512, 512, dilation_rate=16)

        # upsample
        self.up5 = UpConv(1024, 512)
        self.up4 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)

        # attn block for skip connections
        self.attn5 = AttentionBlock(512, 512, 256)
        self.attn4 = AttentionBlock(256, 256, 128)
        self.attn3 = AttentionBlock(128, 128, 64)
        self.attn2 = AttentionBlock(64, 64, 32)

        # decoder
        self.up_conv5 = DepthwiseSeperableConv(1024, 512)
        self.up_conv4 = DepthwiseSeperableConv(512, 256)
        self.up_conv3 = DepthwiseSeperableConv(256, 128)
        self.up_conv2 = GhostConv(128, 64)

        # segmentation map
        self.conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        # encoder
        x1 = self.conv1(x)
        x1_atrous = self.atrous1(x1) 
        x2 = self.maxpool(x1)

        x2 = self.conv2(x2)
        x2_atrous = self.atrous2(x2) 
        x3 = self.maxpool(x2)

        x3 = self.conv3(x3)
        x3_atrous = self.atrous3(x3) 
        x4 = self.maxpool(x3)

        x4 = self.conv4(x4)
        x4_atrous = self.atrous4(x4)  
        x5 = self.maxpool(x4)

        x5 = self.conv5(x5)

        # decoder (attn from decoder and AtrousConv)
        d5 = self.up5(x5)
        x4 = self.attn5(d5, x4_atrous)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.attn4(d4, x3_atrous)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.attn3(d3, x2_atrous)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.attn2(d2, x1_atrous)
        d2 = torch.cat((x1, d2), dim=1)
        
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        return d1
