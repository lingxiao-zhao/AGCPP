import numpy as np
import numpy.ma as ma

def shift_zenith_angle(based, satlon_i, satlat_i):
    """
    对goes16数据进行经纬度平移
    :param based: 天顶角数据 (2D数组或MaskedArray)
    :param satlon_i: 单个卫星星下点经度 (scalar或Masked)
    :param satlat_i: 单个卫星星下点纬度 (scalar或Masked)
    :return: 平移后的天顶角数据
    """
    # 如果星下点缺失，则返回全 -1 标记值
    if ma.is_masked(satlon_i) or ma.is_masked(satlat_i):
        return np.full_like(based, -1, dtype=based.dtype)

    # 确保输入为浮点数
    satlon = float(satlon_i)
    satlat = float(satlat_i)

    # 计算经纬度的偏移量 (单位：度)
    lon_diff = satlon  # 因为 GOES16 星下点基准为 (0,0)
    lat_diff = satlat

    # 将偏移量转换为网格单位 (0.07度 = 1 网格)
    lon_shift = int(np.round(lon_diff / 0.07))
    lat_shift = int(np.round(lat_diff / 0.07))

    # 对天顶角数据进行平移 (先经度后纬度)
    # 如果 based 是 MaskedArray，先用填充值替换 mask 区域
    if ma.isMaskedArray(based):
        filled = ma.filled(based, -1)
    else:
        filled = based

    shifted = np.roll(filled, lon_shift, axis=1)
    shifted = np.roll(shifted, lat_shift, axis=0)
    return shifted
