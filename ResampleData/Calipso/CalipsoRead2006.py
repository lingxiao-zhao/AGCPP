
import os
from pyhdf.SD import SD, SDC
from pyhdf.VS import *
from pyhdf.HDF import *
import numpy as np
import code
import datetime
import argparse
import time
import pandas as pd


def read_clay(ifile):
    '''
    @desc: read hdf:pyhdf读取hdf数据集
          sd + vdata
    '''
    SD_file = SD(ifile)
    ds = SD_file.datasets()  # 所有数据集名称
    cot_uncertain = SD_file.select('Column_Optical_Depth_Cloud_Uncertainty_532')[:].flatten()
    cot_certain = SD_file.select('Column_Optical_Depth_Cloud_532')[:].flatten()
    cot_certain[cot_certain < 0] = -9999
    ch_layer = SD_file.select('Layer_Top_Altitude')[:]
    cth = [ch_layer[i, :][ch_layer[i, :] != -9999][0] if len(ch_layer[i, :][ch_layer[i, :] != -9999]) > 0 else -9999 for i in range(ch_layer.shape[0])]
    cth = np.array(cth) 
    lat = SD_file.select('Latitude')[:, 0]
    lon = SD_file.select('Longitude')[:, 0]
    daynight = SD_file.select('Day_Night_Flag')[:].flatten()  # 0 day ; 1 night

    ts = SD_file.select('Profile_UTC_Time')[:, 0]
    ts_read = [        datetime.datetime(int('200' + str(t)[:1]), int(str(t)[1:3]), int(str(t)[3:5]), 0, 0, 0) + datetime.timedelta(
            seconds=float('0.' + str(t)[6:]) * 24 * 3600) for t in ts]
    ts_str = ['%04d-%02d-%02d %02d-%02d-%02d-%06d' % (t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond)
              for t in ts_read]

    layerNum = [np.sum(ch_layer[i,:]>0) for i in range(ch_layer.shape[0])]
    singleLayerFlag = [1 if ln==1 else 0 for ln in layerNum]  # 1 单层云 / 0 其他

    flag = SD_file.select('Feature_Classification_Flags')[:]
    flag1d = flag.astype(int).flatten()
    featureType = np.array([str('{:0>16}'.format(bin(q)[2:])) for q in flag1d]).reshape(flag1d.shape)
    # featureType = np.array([str('{:016b}'.format(q)) for q in flag1d]).reshape(flag1d.shape)
    # ——————————————————————————————————————————————————————————————————
    # Calipso数据的矩阵是倒置的，所以[-3:]是前3位，[-7:-5]是第6-7位
    # 4是晴空，1是冰，2是水
    # ——————————————————————————————————————————————————————————————————
    clearsky = np.array([int(str(f[-3:]), base=2) for f in featureType]).reshape(flag.shape)    # 1 means clear air
    iwphase = np.array([int(str(f[-7:-5]), base=2) for f in featureType]).reshape(flag.shape)
    # iwqa = np.array([int(str(f[-9:-7]), base=2) for f in featureType]).reshape(flag.shape)   # 1 low, 2 med, 3 high
    iwphase[clearsky==1] = 4
    iwphase = [iwphase[i, :][iwphase[i, :]!=4][0] if len(iwphase[i,:][iwphase[i,:]!=4])>0 else 4 for i in range(iwphase.shape[0])]
    iwphase = np.array(iwphase)
    # '''高置信度，可用于与云顶高度对应'''
    # confmask = np.array([int(str(f[-13]), base=2) for f in featureType]).reshape(flag.shape)   
    # confmaskcth = np.array([confmask[i, :][ch_layer[i, :] != -9999][0] if len(ch_layer[i, :][ch_layer[i, :] != -9999]) > 0 else 0 for i in range(ch_layer.shape[0])])

    signal = [1 if 7 not in clearsky[i, :] else 0 for i in range(clearsky.shape[0])]  # 1 repreesent signal ; 0 represent no signal
    samePhaseFlag = [1 if (len(np.unique(iwphase[idx-1:idx+2]))==1) and (idx-1>=0) and (idx+1<=len(iwphase)-1) else 0 for idx in range(len(iwphase))]  # 判断该点前后2个像元，是否属于同一类型的相态（共计连续5个点），严格筛选，1为符合要求，0为不符合

    return cth, cot_certain, cot_uncertain, daynight, lon, lat, ts_str, iwphase, singleLayerFlag, signal, samePhaseFlag
