'''
Description: 
version: 

'''
from torch import nn
from unet_parts import OutConv
from unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from layer import CBAM
import os
import torch
import numpy as np
import netCDF4 as nc

class SmaAt_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(SmaAt_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)
        self.relu = nn.ReLU()

        '''加了波动太大， 值异常'''        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        logits = self.relu(logits)
        return logits
    
if __name__ == "__main__":
    Model = SmaAt_UNet(n_channels=6, n_classes=1, kernels_per_layer=2, bilinear=True, reduction_ratio=16)
    # 指定文件路径和排序顺序的基础目录
    base_dir = "/home/LVM_date/zhaolx/64to256/train/condition/20220601/0015"
    # 获取目录中的文件列表并按照排序顺序排序
    file_list = os.listdir(base_dir)
    file_list.sort()
    # 选择第1个文件
    selected_files = file_list[:1]
    # 创建一个用于存储数据的空数组
    data = np.zeros((1, 6, 64, 64))
    # 逐个加载文件数据
    for i, file_name in enumerate(selected_files):
        file_path = os.path.join(base_dir, file_name)
        # 加载数据到numpy数组
        dataset = nc.Dataset(file_path)
        channel10 = dataset.variables["Channel10"][:]
        channel11 = dataset.variables["Channel11"][:]
        channel12 = dataset.variables["Channel12"][:]
        channel13 = dataset.variables["Channel13"][:]
        channel14 = dataset.variables["Channel14"][:]
        channel15 = dataset.variables["Channel15"][:]
        # 组合数据成多维数组
        fy_data_array1 = np.stack([channel10, channel11, channel12, channel13, channel14, channel15])    
        dataset.close()
        # 将数据复制到对应的批量维度
        data[i] = fy_data_array1
    # 将numpy数组转换为PyTorch张量
    data_tensor = torch.from_numpy(data)
    # 将data_tensor的数据类型设置为torch.float32的数据类型
    data_tensor = data_tensor.float()
    fytorch = Model(data_tensor)
    print(fytorch)
    print(fytorch.shape)