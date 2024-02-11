import os
import torch
import torch.nn as nn
import torch.optim as optim
from base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
from SAattention import SAattention
from MSKA import MSKAattention
from TFB import CrossSwinTransformer, CrossSwinTransformer_1


class DMF_Net(nn.Module):
    def __init__(self, num_channels):
        super(DMF_Net, self).__init__()

        out_channels = 3
        n_resblocks = 4

        self.ms_feature_extract = ConvBlock(3, 32, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.pan_feature_extract = ConvBlock(1, 32, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.cross_attn_1 = CrossSwinTransformer_1(n_feats=32, n_heads=4, head_dim=8, win_size=4,
                                               cross_module=['ms', 'pan'], cat_feat=['ms', 'pan'])

        self.conv_SL = ConvBlock(64, 32, 5, 1, 2, activation='prelu')
        self.SA_1 = MSKAattention(channel=32)
        self.SA_2 = MSKAattention(channel=64)
        self.pan_features = ConvBlock(1, 3, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.four_conv_1 = four_conv(32, 20)
        self.convert_conv_1 = convert_conv(132, 32, 3)
        self.four_conv_2 = four_conv(32, 20)
        self.convert_conv_2 = convert_conv(232, 32, 3)
        self.four_conv_3 = four_conv(32, 20)
        self.convert_conv_3 = convert_conv(332, 32, 3)
        self.four_conv_4 = four_conv(32, 20)
        self.conv_GL = ConvBlock(560, 64, 1, stride=1, padding=0, activation='prelu', norm=None, bias=True)
        self.conv_IR_1 = ConvBlock(99, 32, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.conv_IR_2 = ConvBlock(32, 3, 3, 1, 1, activation='prelu', norm=None, bias=True)

        self.TDCN2_conv1 = ConvBlock(32, 64, 9, 1, 4, activation='prelu', norm=None, bias=True)
        self.TDCN2_conv2 = ConvBlock(64, 32, 5, 1, 2, activation='prelu', norm=None, bias=True)
        self.TDCN2_conv3 = ConvBlock(32, 32, 5, 1, 2, activation='prelu', norm=None, bias=True)
        self.TDCN2_conv4 = ConvBlock(32, 32, 5, 1, 2, activation='prelu', norm=None, bias=True)
        self.upsample = Upsampler(2, 3, activation='prelu')


        FUSION = [
            ConvBlock(3, 32, 3, 1, 1, activation='prelu', norm=None, bias=True),
        ]
        for i in range(n_resblocks):
            FUSION.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.FUSION = nn.Sequential(*FUSION)
        FUSION1 = [
            ConvBlock(1, 32, 3, 1, 1, activation='prelu', norm=None, bias=True),
        ]
        for i in range(n_resblocks):
            FUSION1.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.FUSION1 = nn.Sequential(*FUSION1)
        self.cross_attn = CrossSwinTransformer(n_feats=32, n_heads=4, head_dim=8, win_size=4,
                                               cross_module=['img', 'pan'], cat_feat=['img', 'pan'])
        self.conv_final1 = ConvBlock(128, 64, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.conv_final2 = ConvBlock(64, 32, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.conv_final3 = ConvBlock(32, 3, 3, 1, 1, activation='prelu', norm=None, bias=True)
        self.conv_final = ConvBlock(64, 3, 3, 1, 1, activation='prelu', norm=None, bias=True)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_pan, l_ms):

        hp_pan_4 = x_pan - F.interpolate(
            F.interpolate(x_pan, scale_factor=1 / 4, mode='bicubic', align_corners=False, recompute_scale_factor=True),
            scale_factor=4, mode='bicubic', align_corners=False, recompute_scale_factor=True)
        lr_pan = F.interpolate(x_pan, scale_factor=1 / 2, mode='bicubic', align_corners=False,
                               recompute_scale_factor=True)
        hp_pan_2 = lr_pan - F.interpolate(
            F.interpolate(lr_pan, scale_factor=1 / 2, mode='bicubic', align_corners=False, recompute_scale_factor=True),
            scale_factor=2, mode='bicubic', align_corners=False, recompute_scale_factor=True)
        a_ms = F.interpolate(l_ms, scale_factor=2, mode='bicubic', align_corners=False,
                             recompute_scale_factor=True)
        b_ms = F.interpolate(l_ms, scale_factor=4, mode='bicubic', align_corners=False,
                             recompute_scale_factor=True)
        ms_1 = self.ms_feature_extract(a_ms)
        pan_1 = self.pan_feature_extract(lr_pan)
        x1_f_1 = self.cross_attn_1(ms_1 , pan_1)
        x1_s11 = self.conv_SL(x1_f_1)
        x1_s12 = self.SA_1(x1_s11)
        x1_td_11 = self.TDCN2_conv1(x1_s12)
        x1_td_attention_11 = self.SA_2(x1_td_11)
        x1_td_12 = self.TDCN2_conv2(x1_td_attention_11)
        x1_td_attention_12 = self.SA_1(x1_td_12)
        x1_td_13 = self.TDCN2_conv3(x1_td_attention_12)
        x1_td_attention_13 = self.SA_1(x1_td_13)
        x1_td_14 = self.TDCN2_conv4(x1_td_attention_13)
        x1_f11 = self.four_conv_1(x1_s12)
        x1_f12 = torch.cat([x1_f11, x1_s12], 1)
        x1_f13 = self.convert_conv_1(x1_f12)
        x1_f21 = self.four_conv_2(x1_f13)
        x1_f22 = torch.cat([x1_f21, x1_s12, x1_f11], 1)
        x1_f23 = self.convert_conv_2(x1_f22)
        x1_f31 = self.four_conv_3(x1_f23)
        x1_f32 = torch.cat([x1_f31, x1_s12, x1_f11, x1_f21], 1)
        x1_f33 = self.convert_conv_3(x1_f32)
        x1_f41 = self.four_conv_4(x1_f33)
        x1_f1 = torch.cat([x1_s12, x1_f11, x1_f21, x1_f31, x1_f41, x1_td_11, x1_td_12,x1_td_13], dim=1)
        x1_f1 = self.conv_GL(x1_f1)
        x1_f1 = self.SA_2(x1_f1)
        x1_pan = self.pan_features(lr_pan)
        x1_f1 = torch.cat([x1_pan, x1_f1, x1_td_14], dim=1)
        x1_f1 = self.conv_IR_1(x1_f1)
        x1_f1 = self.conv_IR_2(x1_f1) + a_ms + hp_pan_2
        img1 = self.upsample(x1_f1) + b_ms
        ms_2 = self.ms_feature_extract(b_ms)
        pan_2 = self.pan_feature_extract(x_pan)
        x2_f_1 = self.cross_attn_1(ms_2,pan_2)
        x2_s11 = self.conv_SL(x2_f_1)
        x2_s12 = self.SA_1(x2_s11)
        x2_td_11 = self.TDCN2_conv1(x2_s12)
        x2_td_attention_11 = self.SA_2(x2_td_11)
        x2_td_12 = self.TDCN2_conv2(x2_td_attention_11)
        x2_td_attention_12 = self.SA_1(x2_td_12)
        x2_td_13 = self.TDCN2_conv3(x2_td_attention_12)
        x2_td_attention_13 = self.SA_1(x2_td_13)
        x2_td_14 = self.TDCN2_conv4(x2_td_attention_13)
        x2_f11 = self.four_conv_1(x2_s12)
        x2_f12 = torch.cat([x2_f11, x2_s12], 1)
        x2_f13 = self.convert_conv_1(x2_f12)
        x2_f21 = self.four_conv_2(x2_f13)
        x2_f22 = torch.cat([x2_f21, x2_s12, x2_f11], 1)
        x2_f23 = self.convert_conv_2(x2_f22)
        x2_f31 = self.four_conv_3(x2_f23)
        x2_f32 = torch.cat([x2_f31, x2_f21, x2_s12, x2_f11], 1)
        x2_f33 = self.convert_conv_3(x2_f32)
        x2_f41 = self.four_conv_4(x2_f33)
        x2_f1 = torch.cat([x2_s12, x2_f11, x2_f21, x2_f31, x2_f41,x2_td_11,x2_td_12,x2_td_13], dim=1)
        x2_f1 = self.conv_GL(x2_f1)
        x2_f1 = self.SA_2(x2_f1)
        x2_pan = self.pan_features(x_pan)
        x2_f1 = torch.cat([x2_pan, x2_f1, x2_td_14], dim=1)
        x2_f1 = self.conv_IR_1(x2_f1)
        img2 = self.conv_IR_2(x2_f1) + b_ms + hp_pan_4
        img1 = self.FUSION(img1)
        img2 = self.FUSION(img2)
        pan_detail = self.FUSION1(x_pan)
        fusion_1 = self.cross_attn(img=img1, pan=pan_detail)
        fusion_2 = self.cross_attn(img=img2, pan=pan_detail)
        img = torch.cat([fusion_1, fusion_2], dim=1)
        hrms = self.conv_final1(img)
        hrms = self.conv_final2(hrms)
        hrms = self.conv_final3(hrms) + hp_pan_4 + b_ms
        return hrms


if __name__ == '__main__':
    model = DMF_Net(num_channels=4)
    print("===> Parameter numbers : %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.train()
    input_ms, input_pan = torch.rand(1, 3, 64, 64), torch.rand(1, 1, 256, 256)
    sr = model(input_pan, input_ms)
    print('sr输出', sr.size())
