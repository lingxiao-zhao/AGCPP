import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , log_loss ,accuracy_score,f1_score,recall_score,accuracy_score,precision_score
sys.path.append('/home/nvme/zhaolx/Code/SmArtUnetERA5/classification/')
from SmaAtUNet import SmaAt_UNet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import WeightedRandomSampler
from tensorboardX import SummaryWriter
import warnings
import shutil
import time
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler
from scipy import ndimage

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def worker_init_fn(worker_id):
    '''num of workers多进程读取数据,
        保持读取的数据相同'''
    setup_seed(2021 + worker_id)

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机数种子
    torch.cuda.manual_seed(seed)  # GPU随机数种子
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # True 的话会自动寻找最适合当前配置的高效算法
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的

def preprocess(img_x, img_y, imgpara):
    '''test'''
    for i in range(19):  # imgpara什么意思
        img_x[i, :, :] = (img_x[i, :, :] - imgpara[1, i]) / (imgpara[0, i] - imgpara[1, i])
    img_x = torch.from_numpy(np.array(img_x)).type(torch.FloatTensor)
    img_y = torch.from_numpy(np.array(img_y)).type(torch.long)
    return img_x, img_y

class Mydataset(Dataset):
    def __init__(self, path, imgpara):
        self.path = path
        self.imgpara = imgpara

    def __getitem__(self, item):
        imgpara = self.imgpara
        img_path = self.path[item]
        dataset = np.load(img_path, allow_pickle=True)
        img_x = dataset['gridsat'][:19]

        img_y0 = dataset['modis'][0, :, :]  # 0晴空 1水云 2冰云 3混合云 4无效值
        # 步骤1：将所有6值替换为4
        if np.any(img_y0 == 6):
            img_y0[img_y0 == 6] = 4

        # 处理后的矩阵结果
        img_y = img_y0
        img_x, img_y = preprocess(img_x, img_y, imgpara)
        dataset = None
        return img_x, img_y
    
    def __len__(self):
        return len(self.path)

def validate(model, valdataloaders):
    model.eval()
    acc_all = 0
    recall_all = 0
    precision_all = 0
    f1_score_all = 0
    with torch.no_grad():
        for idx, (vx, vy) in enumerate(valdataloaders):
            vx = vx.to(device, dtype=torch.float32)
            vy = vy.to(device, dtype=torch.long)
            out = model(vx)
            vy = vy.detach().cpu().flatten()
            out_class = torch.argmax(out, dim=1)
            out_class = torch.tensor(out_class, dtype=torch.long).detach().cpu().flatten()
            outputValid = out_class[vy != 4]
            yValid = vy[vy != 4]
            if len(outputValid) != 0:
                acc = accuracy_score(yValid, outputValid)
                recall = recall_score(yValid, outputValid, average='macro')
                precision = precision_score(yValid, outputValid, average='macro')
                f1 = f1_score(yValid, outputValid, average='macro')
                acc_all += acc
                recall_all += recall
                precision_all += precision
                f1_score_all += f1
    print("当前模型在验证集上的acc为: %.3f " % (acc_all / len(valdataloaders)))
    print("当前模型在验证集上的recall为: %.3f " % (recall_all / len(valdataloaders)))
    print("当前模型在验证集上的precision为: %.3f " % (precision_all / len(valdataloaders)))
    print("当前模型在验证集上的f1为: %.3f " % (f1_score_all / len(valdataloaders)))
    
    return acc_all / len(valdataloaders), recall_all / len(valdataloaders), precision_all / len(valdataloaders), f1_score_all / len(valdataloaders)

def train_model(model, criterion, optimizer, dataload, valdataloaders, num_epochs, scheduler):
    modelPath = "/home/LVM_date2/zhaolx/Gridsat/Unet_train/classification/modelsave/"
    modelFoldername = 'phase' + str(num_epochs)
    savePath = os.path.join(modelPath, modelFoldername)
    epochloss = []
    st = time.time()
    valrmse = 1e2

    logname = 'phase' + '_' + str(num_epochs) + '_singleGPU_log'
    logdir = "/home/LVM_date2/zhaolx/Gridsat/Unet_train/classification/runs/log/" + str(logname)
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.mkdir(logdir)
    writer = SummaryWriter(logdir=logdir)

    for epoch in range(num_epochs):
        model.train()  # validate结束后需要更新梯度
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_loss = 0
        for x, y in dataload:
            x = x.type(torch.FloatTensor).to(device)
            inputs = x.reshape(-1, 19, 64, 64)
            labels = y.reshape(-1, 64, 64).to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(inputs)  # forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # tensorboard
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        writer.add_scalar('Train/epoch loss', epoch_loss, epoch)
        
        acc_validate, recall_validate, precision_validate, f1_score_validate = validate(model, valdataloaders)
        writer.add_scalar('validate/acc_validate', acc_validate, epoch)
        writer.add_scalar('validate/recall_validate', recall_validate, epoch)
        writer.add_scalar('validate/precision_validate', precision_validate, epoch)
        writer.add_scalar('validate/f1_score_validate', f1_score_validate, epoch)

        if scheduler != None:
            scheduler.step()

        if epoch % 1 == 0 and epoch >= 20:
            modelname = os.path.join(savePath, 'weights_%d.pth' % (epoch))
            torch.save(model.state_dict(), modelname)
            print("checkpoint: %d, already saved!\n" % (epoch))

    writer.close()

def train(args, path_train, path_validate, imgpara):
    model = SmaAt_UNet(n_channels=19, n_classes=4, kernels_per_layer=2, bilinear=True, reduction_ratio=16).to(device)
    
    if args.ckp:
        model.load_state_dict(torch.load(args.ckp, map_location=device))
        print(f"启动模型{args.ckp}")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=4)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 250], 0.2)

    inputDatasets = Mydataset(path=path_train, imgpara=imgpara)
    valDatasets = Mydataset(path=path_validate, imgpara=imgpara)
    
    train_dataloader = DataLoader(inputDatasets, batch_size=int(args.batch_size), shuffle=True, num_workers=8, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(valDatasets, batch_size=int(args.batch_size), shuffle=False, num_workers=8)

    train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs=int(args.epoch), scheduler=scheduler)

def test(args, path_test, imgpara):
    model = SmaAt_UNet(n_channels=19, n_classes=4, kernels_per_layer=2, bilinear=True, reduction_ratio=16).to(device)

    if args.ckp:
        model.load_state_dict(torch.load(args.ckp, map_location=device))
        print(f"启动模型{args.ckp}")

    testDatasets = Mydataset(path=path_test, imgpara=imgpara)
    test_dataloader = DataLoader(testDatasets, batch_size=4, shuffle=False, num_workers=8)
    
    model.eval()
    for idx, (vx, vy) in enumerate(test_dataloader):
        vx = vx.to(device, dtype=torch.float32)
        vy = vy.to(device, dtype=torch.long)
        out = model(vx)
        # further testing actions can be added here

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SmaAt Unet ERA5 Weather Classification Training')
    parser.add_argument('--ckp', type=str, default=None, help="pretrained model")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--epoch', type=int, default=200, help="number of epochs")
    args = parser.parse_args()

    path_train = ["/home/nvme/zhaolx/Code/SmArtUnet/dataset_path/withERA5/test/test_dataset_2020_2023.csv"]
    path_validate = ["/home/nvme/zhaolx/Code/SmArtUnet/dataset_path/withERA5/test/test_dataset_2020_2023.csv"]
    imgpara = np.zeros((2, 19))
    imgpara[0, :] = [70.00,  180.00,  340.00, 300.00, 14, 14, 90, 90, 350, 90, 7, 325, 315, 280, 255, 135, 140, 150, 170]#19个通道的最大值
    imgpara[1, :] = [-70.00, -180.00, 145.00, 130.00, 8,  8,  0,  0,  220, 0,  0, 230, 220, 220, 200, -13, -11, -13, -16]#19个通道的最小值
    train(args, path_train, path_validate, imgpara)
