import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# class UNetModel(nn.Module):
#     def __init__(self):
#         super(UNetModel, self).__init__()

#         self.first = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         self.down1 = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.down2 = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#         self.down3 = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(inplace=True)
#         )
#         self.down4 = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.up1 = nn.Sequential(
#             nn.Conv2d(1024, 512, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.up2 = nn.Sequential(
#             nn.Conv2d(512, 256, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.up3 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         )
#         self.up4 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(inplace=True)
#         )

#         self.outc = nn.Sequential(nn.Conv2d(64, 2, kernel_size=1))

#         # self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
#         # projection head
#         # self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.ReLU(inplace=True), nn.Linear(512, 256, bias=True))

#     def forward(self, x):
#         x1 = self.first(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         u1 = self.up1(torch.cat([x5, x4], dim=1))
#         u2 = self.up2(torch.cat([u1, x3], dim=1))
#         u3 = self.up3(torch.cat([u2, x2], dim=1))
#         u4 = self.up4(torch.cat([u3, x1], dim=1))
#         logits = self.outc(u4)
#         x = torch.sigmoid(logits)
#         out = torch.where(x>0.5, 1, 0)
#         return logits, out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128 // factor)
        
        # self.down4 = Down(128, 256 // factor)
        # self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.dropout = nn.Dropout(p=0.3)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.dropout(x)
        logits = self.outc(x)
        x = torch.sigmoid(logits)
        out = torch.where(x>0.5, 1, 0)

        return logits, out


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 16)
#         self.down1 = Down(16, 32)
#         self.down2 = Down(32, 64)
#         self.down3 = Down(64, 128)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(128, 256 // factor)
#         self.up1 = Up(256, 128 // factor, bilinear)
#         self.up2 = Up(128, 64 // factor, bilinear)
#         self.up3 = Up(64, 32 // factor, bilinear)
#         self.up4 = Up(32, 16, bilinear)
#         self.dropout = nn.Dropout(p=0.2)
#         self.outc = OutConv(16, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.dropout(x)
#         logits = self.outc(x)
#         x = torch.sigmoid(logits)
#         out = torch.where(x>0.5, 1, 0)

#         return logits, out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)