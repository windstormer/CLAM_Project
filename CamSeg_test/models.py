import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from collections import OrderedDict
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabModel(nn.Module):
    def __init__(self, pretrain_path=None):
        super(DeepLabModel, self).__init__()

        self.f = deeplabv3_resnet50(pretrained=False, num_classes=512)
        # print(dlab)
        # self.f = nn.Sequential(*list(dlab.children()))
        
        
        # print(list(dlab.children())[-1][0])
        c = list(self.f.children())[-1][0]
        list(c.children())[-1][-1] = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # projection head
        # self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.ReLU(inplace=True), nn.Linear(512, 256, bias=True))

        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            # print(state_dict_weights.keys())
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                # print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Model from scratch")
            
    def forward(self, x):
        x = self.f(x)
        mask = self.relu(x['out'])
        # x = self.p(x)
        x = self.gap(mask)
        feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        return mask, feature


class UNetModel(nn.Module):
    def __init__(self, pretrain_path=None):
        super(UNetModel, self).__init__()

        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.outc = nn.Sequential(nn.Conv2d(32, 512, kernel_size=1), nn.ReLU(inplace=True))
        # self.outc = nn.Sequential(nn.Conv2d(32, 512, kernel_size=1), nn.Sigmoid())

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # projection head
        # self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.ReLU(inplace=True), nn.Linear(512, 256, bias=True))
        
        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                # print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        u1 = self.up1(torch.cat([x5, x4], dim=1))
        u2 = self.up2(torch.cat([u1, x3], dim=1))
        u3 = self.up3(torch.cat([u2, x2], dim=1))
        
        u4 = self.up4(torch.cat([u3, x1], dim=1))
        
        x = self.outc(u4)
        mask = x
        x = self.gap(x)
        feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        return mask, feature

class Res50(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Res50, self).__init__()

        resnet = resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
        self.f = nn.Sequential(*list(resnet.children())[:-2])
        self.outc = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1), nn.ReLU(inplace=True))

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # projection head
        # self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.ReLU(inplace=True), nn.Linear(512, 256, bias=True))
        if pretrain_path != None:
            print("Model restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            # print(state_dict_weights.keys())
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Model from scratch")
            
    def forward(self, x):
        x = self.f(x)
        x = self.outc(x)
        mask = x
        x = self.gap(x)
        feature = torch.flatten(x, start_dim=1)
        # out = self.g(feature)
        return mask, feature

class Res18(nn.Module):
    def __init__(self, pretrain_path=None):
        super(Res18, self).__init__()
        self.handlers = []
        resnet = resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        self.f = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.feature = None

        if pretrain_path != None:
            print("Encoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                # print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Encoder from scratch")
        
    #     for name, m in self.f.named_modules():
    #         if isinstance(m, nn.Conv2d):
    #             layer_name = name

    #     for (name, module) in self.f.named_modules():
    #         if name == layer_name:
    #             self.handlers.append(module.register_forward_hook(self._get_features_hook))

    # def _get_features_hook(self, module, input, output):
    #     self.feature = output
    #     # print("feature shape:{}".format(output.size()))

    def forward(self, x):
        mask = self.f(x)
        x = self.gap(mask)
        feature = torch.flatten(x, start_dim=1)
        # mask = self.feature
        return mask, feature

class CModel(nn.Module):
    def __init__(self, input_dim, pretrain_path=None):
        super(CModel, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 1)
            )
        
        if pretrain_path != None:
            print("Decoder restore from", pretrain_path)
            state_dict_weights = torch.load(pretrain_path)
            state_dict_init = self.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
                # print(k, k_0)
            self.load_state_dict(new_state_dict, strict=False)
        else:
            print("Decoder from scratch")
            
    def forward(self, x):
        pred = self.decoder(x)
        return pred