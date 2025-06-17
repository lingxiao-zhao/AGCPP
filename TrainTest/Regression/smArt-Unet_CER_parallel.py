'''
Description: multi-GPU to train CLP classification
version: 1.0

'''
import numpy as np
import pandas as pd
import torch
import os
import time
import numpy as np
import argparse
from torch.utils.data import DataLoader,Dataset
from torch import autograd, optim
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import *
import sys
from scipy import ndimage
import netCDF4 as nc
import torch.utils.data as data
sys.path.append('/home/nvme/zhaolx/Code/SmArtUnetERA5/regression/')
from SmaAtUNet import SmaAt_UNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import shutil
from torch.cuda.amp import autocast as autocast, GradScaler
import random
from scipy.stats import pearsonr
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def worker_init_fn(worker_id):
    '''num of workers多进程读取数据,
        保持读取的数据相同'''
    # 设置随机种子，保持读取的数据相同
    setup_seed(2021 + worker_id)

def reduce_tensor(tensor: torch.Tensor, proc):
    default_device = torch.device('cuda', proc)
    rt = tensor.clone().to(default_device)
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()  # 总进程数
    return rt

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # torch v1.9
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机数种子
    torch.cuda.manual_seed(seed)  # GPU随机数种子
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # True 的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题 
    torch.backends.cudnn.enabled = True  # 同 benchmark
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的

def get_backend_file():
    '''本地通信'''
    file_dir = os.path.dirname(__file__)
    file_name = os.path.basename(__file__) + '.bin'
    if os.path.exists(file_name):
        os.remove(file_name)
    backend_file = os.path.realpath(os.path.join(file_dir, file_name))
    init_method = 'file://%s' % backend_file
    return init_method

def preprocess(img_x, img_y, imgpara, labelpara):
    '''test'''
    for i in range(19):
        img_x[i][:, :] =  (img_x[i][:, :]-imgpara[1, i])/(imgpara[0, i]-imgpara[1, i])
    # 计算phase时候不需要下面对于y的归一化
    img_y[img_y<=0] = 0 # 0 代表晴空, 排除   
    key = img_y == 0
    img_y = (img_y-labelpara[1])/(labelpara[0]-labelpara[1])
    img_y[key] = 0
    return img_x, img_y

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, target, ignored_index):
        mask = target == ignored_index
        out = (input[~mask] - target[~mask]) ** 2
        return out

class PartMSE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, output, label):
        loss = calPartMSE(output, label)
        return loss

def calPartMSE(output, label):
    '''torch'''
    output = output.type(torch.FloatTensor)
    label = label.type(torch.FloatTensor)
    key = label != 0
    a = output[key]
    b = label[key]
    loss = torch.mean(torch.pow((a - b), 2))
    return loss

class loadDataset(data.Dataset):
    
    def __init__(self, path, imgpara, labelpara, productname, augment=True):
        self.path = path
        self.imgpara = imgpara
        self.labelpara = labelpara
        self.productname = productname
        self.augmented = []  

        # **如果 augment=True（训练集），扩充 6 倍；如果 augment=False（验证集），只使用原始数据**
        if augment:
            for p in self.path:
                self.augmented.append((p, "original"))   # 原始数据
                self.augmented.append((p, "horizontal")) # 水平翻转
                self.augmented.append((p, "vertical"))   # 垂直翻转
        else:
            for p in self.path:
                self.augmented.append((p, "original"))  # 只加入原始数据

    def __getitem__(self, index):
        img_path, aug_type = self.augmented[index]
        dataset = np.load(img_path)
        img_x = dataset['gridsat'][:19]  
        img_y = dataset['modis'][2, :, :]    # ["phase", "cth", "cer", "cot", "ctt", "lat", "lon"]

        # 归一化
        img_x, img_y = preprocess(img_x, img_y, self.imgpara, self.labelpara)

        # 转换为Tensor
        img_x = torch.from_numpy(np.array(img_x)).float()
        img_y = torch.from_numpy(np.array(img_y)).float()

        # **只有训练集（augment=True）才执行数据增强**
        if aug_type == "horizontal":
            img_x = torch.flip(img_x, dims=[2])  
            img_y = torch.flip(img_y, dims=[1])
        elif aug_type == "vertical":
            img_x = torch.flip(img_x, dims=[1])  
            img_y = torch.flip(img_y, dims=[0])
        return img_x, img_y

    def __len__(self):
        return len(self.augmented)  



def train_model(model, criterion, optimizer, dataloaders, valdataloaders, num_epochs, scheduler, proc, train_sampler,
                labelpara, productName):
    modelPath = "/home/nvme/zhaolx/Unet_train/regression/modelsave/"
    modelFoldername = str(productName) + str(num_epochs)
    savePath = os.path.join(modelPath, modelFoldername)
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    default_device = torch.device('cuda', proc)
    epochloss = []
    allStep = 0
    valrmse = 1e2
    if proc == 0:
        '''记录时间, 写入日志, 有点错误'''
        st = time.time()
        dt_size = len(dataloaders)
        print("数据划分为%d个batch" % len(dataloaders))

        logname = str(productName) + '_' + str(num_epochs) + '_parallelGPU_log'
        logdir = "/home/nvme/zhaolx/Unet_train/runs/log/" + str(logname)
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        os.mkdir(logdir)
        writer = SummaryWriter(logdir=logdir)
    for epoch in range(num_epochs):
        model.train()
        '''通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果'''
        train_sampler.set_epoch(epoch)
        epoch_loss = 0
        steps = 0
        rmseEpochMean = []
        for x, y in dataloaders:
            allStep += 1  # tensorboard
            steps += 1
            inputs = x.reshape(-1, 19, 64, 64).to(default_device, dtype=torch.float32,non_blocking=True)
            labels = y.reshape(-1, 1, 64, 64).to(default_device, dtype=torch.float32,non_blocking=True)


            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(inputs)  # forward

            loss = criterion(outputs, labels)  # GHM

            loss.sum().backward()
            optimizer.step()
            reduced_loss = reduce_tensor(loss.sum().data, proc)
            epoch_loss += reduced_loss.item()
            if proc == 0:
                '''tensorboard'''
                print(f"epoch: {epoch}")
                outputStretch = (outputs.detach().cpu()[:, 0, :, :]) * (labelpara[0] - labelpara[1]) + labelpara[1]
                print("预测最值:")
                print(outputStretch.data.max())
                print(outputStretch.data.min())
                y = y.detach().cpu() * (labelpara[0] - labelpara[1]) + labelpara[1]
                print("标签最值:")
                print(y.max())
                print(y.min())
                print("\n")

                writer.add_scalar('Train/Loss', loss.sum().item(), allStep)
                writer.add_scalar('Train/Max', outputStretch.max().item(), allStep)
                writer.add_scalar('Train/Min', outputStretch.min().item(), allStep)
                writer.add_scalar('Label/Max', y.max(), allStep)
                writer.add_scalar('Label/Min', y.min(), allStep)
                outputStretch = outputStretch.detach().cpu().flatten()

                y = y.detach().cpu().flatten()
                if productName == 'cer':
                    outputStretch = outputStretch[(y <= 100) & (y > 0)]
                    y = y[(y <= 100) & (y > 0)]
                elif productName == 'cot':
                    outputStretch = outputStretch[(y <= 150) & (y > 0)]
                    y = y[(y <= 150) & (y > 0)]                
                elif productName == 'cth':
                    outputStretch = outputStretch[(y <= 18) & (y > 0)]
                    y = y[(y <= 18) & (y > 0)]
                elif productName == 'ctt':
                    outputStretch = outputStretch[(y <= 340) & (y >= 140)] 
                    y = y[(y <= 340) & (y >= 140)]   
                writer.add_scalar('Rmse/step mean', np.sqrt(mean_squared_error(y.flatten(), outputStretch.flatten())),
                                  allStep)
                rmseEpochMean.append(np.sqrt(mean_squared_error(y.flatten(), outputStretch.flatten())))
                writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], allStep)

                print("%d/%d,train_loss:%0.3f" % (steps, dt_size, reduced_loss.item()))
                print("消耗时间:%.2f" % (time.time() - st))
                st = time.time()
            ##  释放显存, 不建议释放, 以防其他脚本干扰
            # torch.cuda.empty_cache()
        if proc == 0:
            print("epoch %d loss:%0.3f" % (epoch, epoch_loss))  # epoch_loss是多卡loss的sum
            writer.add_scalar('Train/epoch loss', epoch_loss, epoch)
            writer.add_scalar('Rmse/epoch mean', np.mean(rmseEpochMean), epoch)
            epochloss.append("%.2f" % (epoch_loss))
        '''validate'''
        valrmse = validate(model, valdataloaders, labelpara, proc,productname=productName)
        if proc == 0:
            writer.add_scalar('validate/rmse', valrmse, epoch)
        '''update lr'''

        if scheduler != None:  # lrMulti
            scheduler.step()  # CosineAnnealingLR需要传入epoch
        print("proc: %d , 当前学习率为: %.2f" % (proc, optimizer.param_groups[0]['lr']))
        # cheackpoint
        dist.barrier()
        # if proc == 0 and epoch % 100 == 0:
        if (proc == 0) and (epoch % 1 == 0) and (epoch >= 0):
            modelname = os.path.join(savePath, 'weights_%d_%s_ddp.pth' % (epoch, str(productName)))
            torch.save(model.module.state_dict(), modelname, _use_new_zipfile_serialization=False)
            print("checkpoint: %d, already saved!\n" % (epoch))
        dist.barrier()
    if proc == 0:
        print("训练完毕, 全部epoch如下:\n")
        print(epochloss)
        writer.close()
    dist.barrier()
    modelname = os.path.join(savePath, 'weights_%d_%s_ddp.pth' % (epoch, str(productName)))
    if proc == 0:
        torch.save(model.module.state_dict(), modelname, _use_new_zipfile_serialization=False)  # 并行保存方式
    dist.barrier()
    dist.destroy_process_group()
    return model


def train(proc, nprocs, args, path_train, path_validate, imgpara, labelpara):
    default_device = torch.device('cuda', proc)
    print(f"当前进程: {proc}")
    print(f"启动显卡: {default_device}")
    batch_size = int(args.batch_size)
    #print(f"单卡batchsize： {batch_size * 64}")
    '''ddp'''
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:15435', world_size=nprocs, rank=proc) 

    torch.cuda.set_device(proc)
    torch.manual_seed(2021)
    '''model'''

    model=SmaAt_UNet(n_channels=19, n_classes=1, kernels_per_layer=2, bilinear=True, reduction_ratio=16).to(default_device,
                                                                                                      dtype=torch.float32)
    if args.ckp:
        model.load_state_dict(torch.load(args.ckp, map_location=default_device))
        print(f"启动模型{args.ckp}")
    model = model.to(default_device, dtype=torch.float32)    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[proc], output_device=proc,
                                                      find_unused_parameters=True)
    keys = ['cer', 'cot', 'cth', 'ctt'] 
    productIndex = int(args.pi)
    productName = keys[productIndex]

    '''loss func'''

    criterion = PartMSE()

    criterion = criterion.to(default_device, dtype=torch.float32)

    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr))
    '''scheduler'''
    if productIndex == 0:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 200], 0.2)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, min_lr=0.0001, patience=25)
    elif productIndex == 1:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 200], 0.2)  # 低学习率
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], 0.2)     # 高学习率
    elif productIndex == 2:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 200], 0.2)
    elif productIndex == 3:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 200], 0.2)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    '''dataloader'''
    torch.manual_seed(2021)
    inputDatasets = loadDataset(path=path_train, imgpara=imgpara, labelpara=labelpara, productname=productName, augment=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(inputDatasets, shuffle=True)
    dataloaders = DataLoader(inputDatasets, batch_size=batch_size, shuffle=False, drop_last=True, sampler=train_sampler,
                             num_workers=8, prefetch_factor=2, pin_memory=True, worker_init_fn=worker_init_fn)
    torch.manual_seed(2021)
    validateDatasets = loadDataset(path=path_validate, imgpara=imgpara, labelpara=labelpara, productname=productName, augment=False)
    valdataloaders = DataLoader(validateDatasets, shuffle=False, batch_size=4, num_workers=4, prefetch_factor=2,
                                pin_memory=True, worker_init_fn=worker_init_fn)
    if proc == 0:
        print("dataloader训练集数量: %d" % len(inputDatasets))
        print("dataloader验证集数量: %d" % len(validateDatasets))
    train_model(model, criterion, optimizer, dataloaders, valdataloaders, num_epochs=args.epoch, scheduler=scheduler,
                proc=proc, train_sampler=train_sampler, labelpara=labelpara, productName=productName)


def validate(model, valdataloaders, labelpara, proc, productname):
    # default_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    default_device = "cpu"
    model.eval()
    # valOutput = []
    # valY = []
    valRmse = []
    with torch.no_grad():
        for idx, (vx, vy) in enumerate(valdataloaders):
            if proc == 0:
                print("进度: %d | %d" % (idx, len(valdataloaders) - 1))
            vx = vx.to(default_device, dtype=torch.float32)  # 一定要转为cuda, 这样才能加速
            vy = vy.to(default_device, dtype=torch.float32)

            out = model(vx)
            outputstretch = (out[:, 0, :, :]) * (labelpara[0] - labelpara[1]) + labelpara[1]

            '''剔除晴空，计算RMSE'''

            vy = (vy * (labelpara[0] - labelpara[1]) + labelpara[1])

            outputstretch = outputstretch.detach().cpu().flatten()
            vy = vy.detach().cpu().flatten()
            if productname == 'cer':
                outputstretch = outputstretch[(vy <= 60) & (vy > 0)]
                vy = vy[(vy <= 60) & (vy > 0)]
            elif productname == 'cot':
                outputstretch = outputstretch[(vy <= 100) & (vy > 0)]
                vy = vy[(vy <= 100) & (vy > 0)]                
            elif productname == 'cth':
                outputstretch = outputstretch[(vy <= 18) & (vy > 0)]
                vy = vy[(vy <= 18) & (vy > 0)]
            elif productname == 'ctt':
                outputstretch = outputstretch[(vy <= 340) & (vy >= 140)] 
                vy = vy[(vy <= 340) & (vy >= 140)]   
            if len(outputstretch) != 0:
                if proc == 0:
                    print("RMSE为%.2f" % np.sqrt(mean_squared_error(outputstretch.flatten(), vy.flatten())))
                valRmse.append(np.sqrt(mean_squared_error(outputstretch.flatten(), vy.flatten())))
                # valOutput.extend(outputstretch)
                # valY.extend(vy)
            else:
                pass


    if proc == 0:
        # print("当前模型在验证集上的平均RMSE为: %.3f \n"%(meanRmse))
        print("当前模型在验证集上的平均RMSE为: %.3f \n" % np.mean(valRmse))
    return np.mean(valRmse)


def test(path):
    default_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(default_device)
    keys = ['cer', 'cot', 'cth', 'ctt'] 
    productIndex = int(args.pi)
    productName = keys[productIndex]

    model = SmaAt_UNet(n_channels=19, n_classes=1, kernels_per_layer=2, bilinear=True, reduction_ratio=16)
    model.load_state_dict(torch.load(args.ckp, map_location=default_device))
    model = model.to(default_device, dtype=torch.float32)  # model一定要转为cuda, 这样才能加速
    np.random.seed(2021)
    inputDatasets = loadDataset(path, imgpara=imgpara, labelpara=labelpara,productname=productName)
    dataloaders = DataLoader(inputDatasets, shuffle=False, batch_size=batch_size, num_workers=2, prefetch_factor=2,
                             pin_memory=True, worker_init_fn=worker_init_fn)
    print(len(inputDatasets))
    preImg = np.zeros((len(inputDatasets), 2,64, 64))

    model.eval()
    outputValid = []
    yValid = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloaders):
            print("进度: %d | %d" % (idx, len(dataloaders) - 1))
            x = x.to(default_device, dtype=torch.float32)  # 一定要转为cuda, 这样才能加速
            y = y.to(default_device, dtype=torch.float32)

            out = model(x)
            outputstretch = (out[:, 0, :, :]) * (labelpara[0] - labelpara[1]) + labelpara[1]  # batchsize X 1 X H X W
            outputstretch = outputstretch.detach().cpu()
            if productname != 'cot':
                outputstretch[outputstretch>100] = 100

            if idx != len(dataloaders) - 1:
                preImg[idx * batch_size:(idx + 1) * batch_size, 0, :, :] = outputstretch
                preImg[idx * batch_size:(idx + 1) * batch_size, 1, :, :] = y.detach().cpu() * (
                            labelpara[0] - labelpara[1]) + labelpara[1]
            else:
                preImg[idx * batch_size:, 0, :, :] = outputstretch
                preImg[idx * batch_size:, 1, :, :] = y.detach().cpu() * (labelpara[0] - labelpara[1]) + labelpara[1]

            '''剔除晴空，计算RMSE'''


            y = (y * (labelpara[0] - labelpara[1]) + labelpara[1])
            y = y.detach().cpu()



            if len(outputstretch) != 0:
                print("预测最大值%.2f" % outputstretch.max())
                print("预测最小值%.2f" % outputstretch.min())
                print("标签最大值%.2f" % y.max())
                print("标签最小值%.2f" % y.min())
                print("RMSE为%.2f" % np.sqrt(mean_squared_error(outputstretch.flatten(), y.flatten())))
                print("MAE为%.2f" % (mean_absolute_error(outputstretch.flatten(), y.flatten())))

                print("\n")

                outputValid.extend(outputstretch)
                yValid.extend(y)
            else:
                pass
    print("平均RMSE为%.3f" % np.sqrt(mean_squared_error(outputValid.flatten(), yValid.flatten())))
    print("平均MAE为%.3f" % (mean_absolute_error(outputValid.flatten(), yValid.flatten())))
    print("平均R2为%.3f" % (r2_score(outputValid.flatten(), yValid.flatten())))
    r, p = pearsonr(outputValid, yValid)
    print("平均pearsonr为%.3f" % (r))
    np.save('/home/nvme/zhaolx/Unet_train/regression/savedata/%s_%s.npy' % (epochs, keys[productIndex]), preImg)  # linux上使用


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("-action", type=str, dest='action', help="train or test", default="train")
    parse.add_argument("-batch_size", type=int, dest='batch_size', default=512)
    parse.add_argument("-epoch", type=int, dest='epoch', default=300)
    parse.add_argument("-ckp", type=str, dest='ckp', help="the path of model weight file",
                        default="")
    parse.add_argument("-pi", type=int, dest='pi', help="choose the product index", default=0)   ###['cer', 'cot', 'cth', 'ctt']
    parse.add_argument("-lr", type=float, dest='lr', help="choose the learning rate", default=0.001)

    args = parse.parse_args()
    batch_size = int(args.batch_size)
    epochs = int(args.epoch)
    productIndex = int(args.pi)
    learningRate = float(args.lr)
    args.nprocs = torch.cuda.device_count()

    keys = ['cer', 'cot', 'cth', 'ctt']
    productname=keys[productIndex]
    setup_seed(2021)  # 加载随机数种子, 确保每次运算的网络参数、卷积模式都一样

    if keys[productIndex] == 'cer':
        labelpara = np.array([100, 0])
    elif keys[productIndex] == 'cot':
        labelpara = np.array([100, 0])
    elif keys[productIndex] == 'cth':
        labelpara = np.array([18, 0])
    elif keys[productIndex] == 'ctt':
        labelpara = np.array([340, 130])
    imgpara = np.zeros((2, 19))
    imgpara[0, :] = [70.00,  180.00,  340.00, 300.00, 14, 14, 90, 90, 350, 90, 7, 325, 315, 280, 255, 135, 140, 150, 170]#19个通道的最大值
    imgpara[1, :] = [-70.00, -180.00, 145.00, 130.00, 8,  8,  0,  0,  220, 0,  0, 230, 220, 220, 200, -13, -11, -13, -16]#19个通道的最小值
    '''
    @linux
    '''
    modelPath = "/home/nvme/zhaolx/Unet_train/regression/modelsave/"
    modelFoldername = str(keys[productIndex]) + str(epochs)
    savePath = os.path.join(modelPath, modelFoldername)
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    '''csv读取，方便数据筛选（生成新csv即可）'''
    if args.action == "train":
        # Read the CSV files for path_train and path_validate
        validate_csv_file = "................"
        train_csv_file = "................"
        path_train = pd.read_csv(train_csv_file)["path"].values.tolist()
        path_validate = pd.read_csv(validate_csv_file)["path"].values.tolist()

        mp.spawn(train, nprocs=args.nprocs, args=(args.nprocs, args, path_train, path_validate, imgpara, labelpara),
                 join=True) 
        print("Finished!")

    elif "test" in args.action:
        test_csv_file = "................"
        path_test = pd.read_csv(test_csv_file)["path"].values.tolist()
        test(path_test)