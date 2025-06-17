import numpy as np
import math
from scipy.interpolate import griddata
from scipy.spatial import KDTree
def resampleScipy2d(arr, xx_org, yy_org, x_tag_1d, y_tag_1d):
    '''
    @desc: scipy griddata插值
    @xx_org: 2-d 
    @yy_org: 2-d
    @x_tag: 1-d
    @y_tag: 1-d
    '''
    xx_org_1d = xx_org.reshape(-1)  # xx_org_1d / yy_org_1d 即 对应站点数据, 输入应是所有格点依次的 x/y 坐标, xx_org / yy_org 是类似 meshgrid 的结果
    yy_org_1d = yy_org.reshape(-1)

    xx_tag_1d, yy_tag_1d = np.meshgrid(x_tag_1d, y_tag_1d)
    data = griddata(
        np.vstack([[xx_org_1d], [yy_org_1d]]).T,  # 这样写, 速度快
        arr.reshape(-1).T,
        (xx_tag_1d, yy_tag_1d),
        method='nearest',
        fill_value=np.nan
    )
    # Use KDTree to answer the question: "which point of set (x,y) is the nearest neighbors of those in (xp, yp)"
    tree = KDTree(np.c_[xx_org_1d, yy_org_1d])  
    dist, _ = tree.query(np.c_[xx_tag_1d.ravel(), yy_tag_1d.ravel()], k=1)  
    dist = dist.reshape(xx_tag_1d.shape) 
    data = data.astype(np.float16)
    data[dist > 0.2] = -999 # 距离 > sqrt(2)*0.01 degree 的插值点被判断为无效插值（该距离一般设置为单元格对角线距离，这里根据情况调整为两个对角线距离，实际酌情而定）
    data = data.astype(np.float32)
    key = (dist > 0.2) 
    data[(~key)*(data==-999)] = 0  # set clear sky and retrieval failed as 0
    return data, key
