'''
Description: multi-GPU to train CLP classification
version: 1.0 -> 2.0
Revised: Added ERA5 weather field
Author: ZhaoLX
Date: 2024-11-06 00:01:39

'''
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
import torch.distributed as dist
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , log_loss ,accuracy_score,f1_score,recall_score,accuracy_score,precision_score
sys.path.append('/home/nvme/zhaolx/Code/SmArtUnetERA5/classification/')
from SmaAtUNet import SmaAt_UNet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import WeightedRandomSampler
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import warnings
import shutil
import code
import time
import random
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torch.cuda.amp import autocast as autocast, GradScaler
from scipy.stats import pearsonr
from scipy import ndimage
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def worker_init_fn(worker_id):
    '''num of workers多进程读取数据,
        保持读取的数据相同'''
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


def preprocess(img_x, img_y, imgpara):
    '''test'''
    for i in range(19):  # imgpara什么意思
        img_x[i, :, :] = (img_x[i, :, :] - imgpara[1, i]) / (imgpara[0, i] - imgpara[1, i])
    img_x = torch.from_numpy(np.array(img_x)).type(torch.FloatTensor)
    img_y = torch.from_numpy(np.array(img_y)).type(torch.long)
    return img_x, img_y



class Mydataset(Dataset):
    def __init__(self, path, imgpara, augment=True):
        self.path = path
        self.imgpara = imgpara
        self.augment = augment  # 是否进行数据增强

        # **如果 augment=True，扩充数据集为6倍**
        if self.augment:
            self.augmented = []  # 存储扩充后的数据路径
            for p in self.path:
                self.augmented.append((p, "original"))   # 原始数据
                self.augmented.append((p, "horizontal")) # 水平翻转
                self.augmented.append((p, "vertical"))   # 垂直翻转
                self.augmented.append((p, "rot90"))      # 旋转 90°
                self.augmented.append((p, "rot180"))     # 旋转 180°
                self.augmented.append((p, "rot270"))     # 旋转 270°
        else:
            self.augmented = [(p, "original") for p in self.path]  # 只用原始数据

    def __getitem__(self, item):
        img_path, aug_type = self.augmented[item]
        dataset = np.load(img_path)
        img_x = dataset['gridsat'][:19]  # 获取19个通道的输入数据

        img_y0 = dataset['modis'][0, :, :]  # 标签数据，0代表晴空，1代表水云，2代表冰云，3代表混合云，4代表无效值

        # 处理标签，将大于3的值和-999的标签替换为3
        if np.any(img_y0 > 3):
            img_y0[img_y0 > 3] = 3  # 将大于3的值替换为3
        if np.any(img_y0 == -999):
            img_y0[img_y0 == -999] = 3  # 将-999的值替换为3

        img_y = img_y0

        # 预处理：归一化和标准化
        img_x, img_y = preprocess(img_x, img_y, self.imgpara)

        # **根据增强类型进行翻转或旋转**
        if aug_type == "horizontal":
            img_x = torch.flip(img_x, dims=[2])  # 水平翻转
            img_y = torch.flip(img_y, dims=[1])
        elif aug_type == "vertical":
            img_x = torch.flip(img_x, dims=[1])  # 垂直翻转
            img_y = torch.flip(img_y, dims=[0])
        elif aug_type == "rot90":
            img_x = torch.rot90(img_x, 1, dims=[1, 2])  # 旋转90°
            img_y = torch.rot90(img_y, 1, dims=[0, 1])
        elif aug_type == "rot180":
            img_x = torch.rot90(img_x, 2, dims=[1, 2])  # 旋转180°
            img_y = torch.rot90(img_y, 2, dims=[0, 1])
        elif aug_type == "rot270":
            img_x = torch.rot90(img_x, 3, dims=[1, 2])  # 旋转270°
            img_y = torch.rot90(img_y, 3, dims=[0, 1])
        return img_x, img_y

    def __len__(self):
        return len(self.augmented)  # 数据集长度扩充3倍


def validate(model, valdataloaders,proc):
    # device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model.eval()
    acc_all=0
    recall_all=0
    precision_all=0
    f1_score_all=0
    with torch.no_grad():
        for idx, (vx, vy) in enumerate(valdataloaders):
            if proc == 0:
                print("进度: %d | %d" % (idx, len(valdataloaders) - 1))
            vx = vx.to(device, dtype=torch.float32)
            vy = vy.to(device, dtype=torch.long)
            out = model(vx)
            vy = vy.detach().cpu().flatten()
            out_class = torch.argmax(out,dim=1)
            out_class = torch.tensor(out_class,dtype=torch.long).detach().cpu().flatten()
            outputValid = out_class[vy!=3]
            yValid = vy[vy!=3]
            if len(outputValid) != 0:
                acc = (accuracy_score(yValid, outputValid))
                recall = (recall_score(yValid, outputValid,average='macro'))
                precision = (precision_score(yValid, outputValid,average='macro'))
                f1 = (f1_score(yValid, outputValid,average='macro'))
                acc_all+=acc
                recall_all+=recall
                precision_all+=precision
                f1_score_all+=f1
            else:
                pass
    if proc == 0:
        print("当前模型在验证集上的acc为: %.3f " % (acc_all/len(valdataloaders)))
        print("当前模型在验证集上的recall为: %.3f " % (recall_all/len(valdataloaders)))
        print("当前模型在验证集上的precision为: %.3f " % (precision_all/len(valdataloaders)))
        print("当前模型在验证集上的f1为: %.3f " % (f1_score_all/len(valdataloaders)))
    return acc_all/len(valdataloaders), recall_all/len(valdataloaders), precision_all/len(valdataloaders), f1_score_all/len(valdataloaders)

def train_model(model, criterion, optimizer, dataload, valdataloaders, num_epochs, scheduler, proc, train_sampler):
    modelPath = "/home/nvme/zhaolx/Unet_train/classification/modelsave/"
    modelFoldername = 'phase' + str(num_epochs)
    savePath = os.path.join(modelPath, modelFoldername)
    default_device = torch.device('cuda', proc)
    epochloss = []
    allStep = 0
    st = time.time()
    valrmse = 1e2
    if proc == 0:
        st = time.time()
        logname = 'phase' + '_' + str(num_epochs) + '_parallelGPU_log'
        logdir = "/home/nvme/zhaolx/Unet_train/runs/log/" + str(logname)
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
        os.mkdir(logdir)
        writer = SummaryWriter(logdir=logdir)
    for epoch in range(num_epochs):
        model.train()  # validate结束后需要更新梯度
        train_sampler.set_epoch(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload)
        epoch_loss = 0
        step = 0
        accEpochMean = []
        for x, y in dataload:
            allStep += 1  # tensorboard
            step += 1
            x = x.type(torch.FloatTensor)  # double => float
            inputs = x.reshape(-1, 19, 64, 64).to(default_device, dtype=torch.float32, non_blocking=True)
            labels = y.reshape(-1, 64, 64).to(default_device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(inputs)  # forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            reduced_loss = reduce_tensor(loss.data, proc)
            epoch_loss += reduced_loss.item()
            if proc == 0:
                '''tensorboard'''
                print(f"epoch: {epoch}")
                key = (labels!=3).detach().cpu()
                predict = torch.argmax(outputs, axis=1)
                predict = predict[key]
                y = labels[key]
                
                print("预测最值:")
                print(predict.max())
                print(predict.min())    
                print("标签最值:")
                print(y.max())
                print(y.min())
                print("\n")

                writer.add_scalar('Train/Loss', loss.item(), allStep)
                writer.add_scalar('Train/Max', predict.max().item(), allStep)
                writer.add_scalar('Train/Min', predict.min().item(), allStep)
                writer.add_scalar('Label/Max', y.max(), allStep)
                writer.add_scalar('Label/Min', y.min(), allStep)

                accuracy = accuracy_score(y.cpu().numpy().flatten(), predict.cpu().numpy().flatten())
                writer.add_scalar('Acc_score/step mean', accuracy, allStep)
                accEpochMean.append(accuracy)
                writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], allStep)
                # print("%d/%d,train_loss:%0.3f" % (steps, dt_size, reduced_loss.item()))
                print("消耗时间:%.2f"%(time.time()-st))
                st = time.time()
        if proc == 0:
            print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
            writer.add_scalar('Train/epoch loss', epoch_loss, epoch)
            epochloss.append("%.2f" % (epoch_loss))
        acc_validate, recall_validate, precision_validate, f1_score_validate = validate(model, valdataloaders,proc)
        if proc == 0:
            writer.add_scalar('validate/acc_validate', acc_validate, epoch)
            writer.add_scalar('validate/recall_validate', recall_validate, epoch)
            writer.add_scalar('validate/precision_validate', precision_validate, epoch)
            writer.add_scalar('validate/f1_score_validate', f1_score_validate, epoch)
        if scheduler != None:  # lrMulti
            scheduler.step()
        print("proc: %d , 当前学习率为: %.2f" % (proc, optimizer.param_groups[0]['lr']))
        dist.barrier()
        if (proc == 0) and (epoch % 1 == 0) and (epoch >= 0):
            modelname = os.path.join(savePath, 'weights_%d_phase_ddp.pth' % (epoch))
            torch.save(model.module.state_dict(), modelname, _use_new_zipfile_serialization=False)
            print("checkpoint: %d, already saved!\n" % (epoch))

    if proc == 0:
        print("训练完毕, 全部epoch如下:\n")
        print(epochloss)
        writer.close()

    dist.barrier()
    modelname = os.path.join(savePath, 'weights_%d_cp_ddp.pth' % (epoch))
    if proc == 0:
        torch.save(model.module.state_dict(), modelname, _use_new_zipfile_serialization=False)  # 并行保存方式
    dist.barrier()
    dist.destroy_process_group()
    return model

def train(proc, nprocs, args, path_train, path_validate, imgpara):
    default_device = torch.device('cuda', proc)
    print(f"当前进程: {proc}")
    print(f"启动显卡: {default_device}")
    '''ddp'''
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:11984', world_size=nprocs, rank=proc)  
    torch.cuda.set_device(proc)
    torch.manual_seed(2021)
    model = SmaAt_UNet(n_channels=19, n_classes=3, kernels_per_layer=2, bilinear=True, reduction_ratio=16).to(
        default_device,dtype=torch.float32)
    if args.ckp:
        model.load_state_dict(torch.load(args.ckp, map_location=default_device))
        print(f"启动模型{args.ckp}")
    model = model.to(default_device, dtype=torch.float32)        
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[proc], output_device=proc,
                                                      find_unused_parameters=True)
    batch_size = args.batch_size

    criterion = torch.nn.CrossEntropyLoss(ignore_index=3)  # 没有ignore_index. 除了pahse都用这个  自己写 ignore——index=3 无效值
    criterion = criterion.to(default_device, dtype=torch.float32)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 250], 0.2)
    torch.manual_seed(2021)
    inputDatasets = Mydataset(path=path_train, imgpara=imgpara, augment=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(inputDatasets, shuffle=True)
    dataloaders = DataLoader(inputDatasets, batch_size=batch_size, shuffle=False, drop_last=True, sampler=train_sampler,
                             num_workers=8, prefetch_factor=2, pin_memory=True, worker_init_fn=worker_init_fn)

    torch.manual_seed(2021)
    validateDatasets = Mydataset(path=path_validate, imgpara=imgpara, augment=False)
    valdataloaders = DataLoader(validateDatasets, shuffle=False, batch_size=16, num_workers=4, prefetch_factor=2,
                                pin_memory=True, worker_init_fn=worker_init_fn)
    if proc == 0:
        print("dataloader训练集数量: %d" % len(inputDatasets))
        print("dataloader验证集数量: %d" % len(validateDatasets))
    train_model(model, criterion, optimizer, dataloaders, valdataloaders, num_epochs=args.epoch, scheduler=scheduler,proc=proc,train_sampler=train_sampler)

def test(path):
    default_device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(default_device)
    
    model = SmaAt_UNet(n_channels=19, n_classes=3, kernels_per_layer=2, bilinear=True, reduction_ratio=16)
    model.load_state_dict(torch.load(args.ckp, map_location=default_device))
    model = model.to(default_device, dtype=torch.float32)  # 确保模型在正确的设备上
    np.random.seed(2021)

    inputDatasets = Mydataset(path, imgpara=imgpara)
    dataloaders = DataLoader(inputDatasets, shuffle=False, batch_size=batch_size, num_workers=4, prefetch_factor=2,
                             pin_memory=True, worker_init_fn=worker_init_fn)

    print(len(inputDatasets))
    
    preImg = np.zeros((len(inputDatasets), 2, 64, 64))
    model.eval()

    outputValid = []
    yValid = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloaders):
            print("进度: %d | %d" % (idx, len(dataloaders) - 1))

            x = x.to(default_device, dtype=torch.float32)
            y = y.to(default_device, dtype=torch.long)

            out = model(x)

            # 正确获取类别预测
            out_class = torch.argmax(out, dim=1).detach().cpu()  # 形状 (batch_size, 64, 64)
            y = y.detach().cpu()

            # 保存预测结果
            if idx != len(dataloaders) - 1:
                preImg[idx * batch_size:(idx + 1) * batch_size, 0, :, :] = out_class.numpy()
                preImg[idx * batch_size:(idx + 1) * batch_size, 1, :, :] = y.numpy()
            else:
                preImg[idx * batch_size:, 0, :, :] = out_class.numpy()
                preImg[idx * batch_size:, 1, :, :] = y.numpy()

            # 逐 batch 收集预测值
            outputValid.append(out_class.numpy())  
            yValid.append(y.numpy())  

    # 将 list 转换为 numpy 数组
    outputValid = np.concatenate(outputValid, axis=0)  # (9000, 64, 64)
    yValid = np.concatenate(yValid, axis=0)  # (9000, 64, 64)

    # 过滤掉 yValid == 4 的数据
    mask = yValid != 3
    outputValid = outputValid[mask]
    yValid = yValid[mask]

    # 计算分类指标
    acc_test = accuracy_score(yValid, outputValid)
    recall_test = recall_score(yValid, outputValid, average='macro')
    precision_test = precision_score(yValid, outputValid, average='macro')
    f1_test = f1_score(yValid, outputValid, average='macro')

    print("test平均acc为%.3f" % acc_test)
    print("test平均recall为%.3f" % recall_test)
    print("test平均precision为%.3f" % precision_test)
    print("test平均f1_score为%.3f" % f1_test)

    # 保存预测结果
    np.save('/home/nvme/zhaolx/Unet_train/classification/savedata/%s_phase.npy' % (epochs), preImg)  



if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("-action", type=str, dest='action', help="train or test", default='train')
    parse.add_argument("-batch_size", type=int, dest='batch_size', default=512)
    parse.add_argument("-epoch", type=int, dest='epoch', default=300)
    parse.add_argument("-ckp", type=str, dest='ckp', help="the path of model weight file",
                        default="")# 继续计算的时候填入
    parse.add_argument("-lr", type=float, dest='lr', help="choose the learning rate", default=0.001)
    parse.add_argument("-loadm", type=str, dest='loadm', help="load the model", default=None)

    args = parse.parse_args()
    batch_size = int(args.batch_size)
    epochs = int(args.epoch)
    learningRate = float(args.lr)
    loadModelPath = str(args.loadm)
    args.nprocs = torch.cuda.device_count()
    imgpara = np.zeros((2, 19))
    imgpara[0, :] = [70.00,  180.00,  340.00, 300.00, 14, 14, 90, 90, 350, 90, 7, 325, 315, 280, 255, 135, 140, 150, 170]#19个通道的最大值
    imgpara[1, :] = [-70.00, -180.00, 145.00, 130.00, 8,  8,  0,  0,  220, 0,  0, 230, 220, 220, 200, -13, -11, -13, -16]#19个通道的最小值
    '''
    @linux
    '''
    modelPath = "/home/nvme/zhaolx/Unet_train/classification/modelsave/"
    if not os.path.exists(modelPath):
        os.mkdir(modelPath)
    modelFoldername = 'phase' + str(epochs)
    savePath = os.path.join(modelPath, modelFoldername)
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    '''csv读取，方便数据筛选（生成新csv即可）'''
    if args.action == "train":
        validate_csv_file = '................'
        train_csv_file = '................'

        path_train = pd.read_csv(train_csv_file)["path"].values.tolist()
        path_validate = pd.read_csv(validate_csv_file)["path"].values.tolist()
        
        mp.spawn(train, nprocs=args.nprocs, args=(args.nprocs, args, path_train, path_validate, imgpara),
                 join=True)
        print("Finished!")

    elif "test" in args.action:
        test_csv_file = '................'
        path_test = pd.read_csv(test_csv_file)["path"].values.tolist()
        test(path_test)

