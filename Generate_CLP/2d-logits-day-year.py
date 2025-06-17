import os
import numpy as np
import numpy.ma as ma
from datetime import datetime
from netCDF4 import Dataset
import torch
import torch.nn.functional as F
from netCDF4 import Dataset as NC_Dataset
from ShiftZenith import shift_zenith_angle
from SmaAtUNet import SmaAt_UNet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import netCDF4 as nc
# ── 配置 ───────────────────────────────────────────────────────────
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = '/home/nvme/zhaolx/Unet_train/classification/modelsave/phase300-Class3Best/weights_102_phase_ddp.pth'
IMGPARA = np.zeros((2, 19))
IMGPARA[0, :] = [70.00,180.00,340.00,300.00,14,14,90,90,350,90,7,325,315,280,255,135,140,150,170]
IMGPARA[1, :] = [-70.00,-180.00,145.00,130.00,8,8,0,0,220,0,0,230,220,220,200,-13,-11,-13,-16]
PATCH = 64
STRIDE = 10

# ── 加载模型 ───────────────────────────────────────────────────────
model = SmaAt_UNet(n_channels=19, n_classes=3, kernels_per_layer=2, bilinear=True, reduction_ratio=16)
state = torch.load(CKPT_PATH, map_location=DEVICE)
# 如果是 module.state_dict()
if any(key.startswith('module.') for key in state.keys()):
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k.replace('module.', '')] = v
    state = new_state
model.load_state_dict(state)
model.to(DEVICE).eval()

# ── 路径与网格 ─────────────────────────────────────────────────────
year_path = 2009
start_day = f'{year_path}0101'
data = np.load('/home/nvme/zhaolx/Dataset/grid_data.npz')
lon_grid = data['lon_grid']      # shape (2000, 5143)
lat_grid = data['lat_grid']      # shape (2000, 5143)
based = np.load('/home/nvme/zhaolx/Dataset/meteosat11_filtered_interpolated_values.npy')
masked_values = np.ma.masked_equal(based, -1)
output_root = f"/home/Data_Pool/zhaolx/CLP-Dataset/{year_path}"
os.makedirs(output_root, exist_ok=True)

# ── 打开并预加载 ERA5 数据 ─────────────────────────────────────────────────────────
ERA5_PATHS = {
    'temp1': f"/home/pcie_ssd_107/zhaolx/Resample/Retemp/Retemp{year_path}_850-1000.nc",
    'temp2': f"/home/pcie_ssd_107/zhaolx/Resample/Retemp/Retemp{year_path}_300-500.nc",
    'humi1': f"/home/pcie_ssd_107/zhaolx/Resample/Rehumi/Rehumi{year_path}_850-1000.nc",
    'humi2': f"/home/pcie_ssd_107/zhaolx/Resample/Rehumi/Rehumi{year_path}_300-500.nc",
    'skt':   f"/home/pcie_ssd_107/zhaolx/Resample/Reskt/Reskt{year_path}.nc",
    'tcwv':  f"/home/pcie_ssd_107/zhaolx/Resample/Retcwv/Retcwv{year_path}.nc",
    'soil':  f"/home/pcie_ssd_107/zhaolx/Resample/Resoil/Resoil{year_path}.nc",
}

# 一次性打开所有文件
nc_objs = {k: NC_Dataset(path) for k, path in ERA5_PATHS.items()}
# 读取时间轴并构建 datetime 索引映射
time = nc_objs['skt'].variables['valid_time'][:]  # 单位: 秒 since 1970-01-01

# 预载所有变量
temp1_all = nc_objs['temp1'].variables['t'] # 形状 (T, H, W)
temp2_all = nc_objs['temp2'].variables['t']
humi1_all = nc_objs['humi1'].variables['r']
humi2_all = nc_objs['humi2'].variables['r']
skt_all   = nc_objs['skt'].variables['skt']
tcwv_all  = nc_objs['tcwv'].variables['tcwv']
soil_all  = nc_objs['soil'].variables['slt']

# Gridsat 文件夹
gridsat_file_folder = "/home/pcie_ssd_107/zhaolx/Gridsat/access"
gridsat_year_folder = os.path.join(gridsat_file_folder, str(year_path))
all_files = sorted([f for f in os.listdir(gridsat_year_folder) if f.endswith('.nc')])

# 按天分组
from collections import defaultdict
files_by_day = defaultdict(list)
for fname in all_files:
    parts = fname.split('.')
    yyyy, mm, dd, hh = parts[1:5]
    ymd = yyyy + mm + dd
    files_by_day[ymd].append((hh, fname))

# ── 工具函数：blend 窗口 ────────────────────────────────────────────
def make_blend_window(PATCH, STRIDE, i, j, H2, W2):
    P, S = PATCH, STRIDE
    wx = np.ones(P, dtype=np.float32)
    if j > 0:
        wx[:S] = np.linspace(0, 1, S, dtype=np.float32)
    if j + P < W2:
        wx[-S:] = np.linspace(1, 0, S, dtype=np.float32)
    wy = np.ones(P, dtype=np.float32)
    if i > 0:
        wy[:S] = np.linspace(0, 1, S, dtype=np.float32)
    if i + P < H2:
        wy[-S:] = np.linspace(1, 0, S, dtype=np.float32)
    return np.outer(wy, wx)

# ── 开始按天处理 ───────────────────────────────────────────────────
for ymd, hh_fname_list in sorted(files_by_day.items()):
    if ymd < start_day:
        continue
    hh_fname_list.sort(key=lambda x: x[0])
    pred_list = []
    times = []

    for hh, fname in hh_fname_list:
        parts = fname.split('.')
        year_info, month_info, day_info, time_info = parts[1:5]
        # 拼文件路径并打开
        path = os.path.join(gridsat_year_folder, fname)
        nc_file = NC_Dataset(path)

        satlon = nc_file.variables['satlon'][0, :]  # 星下点经度
        satlat = nc_file.variables['satlat'][0, :]  # 星下点纬度
        # 读取经纬度
        lat = nc_file.variables['lat'][:]
        lon = nc_file.variables['lon'][:]
        # 获取经度的排序索引
        sorted_indices = np.argsort(lon)
        # 重新排序经度
        lon_sorted = lon[sorted_indices]
        # 生成二维经纬度网格
        lon2d, lat2d = np.meshgrid(lon_sorted, lat)
        # 读取栅格数据
        irwin_cdr0 = nc_file.variables['irwin_cdr'][:]
        irwin_cdr1 = irwin_cdr0[0, :, :][:, sorted_indices]
        irwin_cdr1.mask=ma.nomask
        irwin_cdr = (irwin_cdr1 * 0.01) + 200.0
        irwvp0 = nc_file.variables['irwvp'][:]
        irwvp1 = irwvp0[0, :, :][:, sorted_indices]
        irwvp1.mask=ma.nomask
        irwvp = (irwvp1 * 0.01) + 200.0

        # 直接在原来位置：
        nan_mask_ir = (irwin_cdr1 == -31999.0) 
        nan_mask_vp = (irwvp1     == -31999.0) 
        nan_mask    = nan_mask_ir | nan_mask_vp

        irwin_vza_adj0 = nc_file.variables['irwin_vza_adj'][:]
        irwin_vza_adj = irwin_vza_adj0[0, :, :][:, sorted_indices]
        irwin_vza_adj = np.where(irwin_vza_adj < -1, 0, irwin_vza_adj)
        # 创建一个新的掩码图层，将所有掩盖位置标记为 -999
        mask_layer = np.zeros_like(irwin_cdr, dtype=np.int16)
        mask_layer[np.where((irwin_cdr < 0) | (irwvp < 0) | (irwin_vza_adj == -1))] = -999
        # Original_temperature_observed = variable(IRWIN) - variable(IRWIN_vza_adj)
        irwin1 = irwin_cdr - irwin_vza_adj
        # 读取satid_ir变量
        satid_ir0 = nc_file.variables['satid_ir'][:]
        satid_ir0.mask=ma.nomask
        # 去除第一维度
        satid_ir = satid_ir0[0, :, :][:, sorted_indices]
        # 将缺失值标记为掩码
        satid_ir = np.ma.masked_equal(satid_ir, -1)
        # 转换为无符号字节
        satid_ir_unsigned = satid_ir.astype(np.int16) + 128
        # 解包
        SATID = np.floor(satid_ir_unsigned / 15.0)

        # 读取satid_wv变量
        satid_wv0 = nc_file.variables['satid_wv'][:]
        satid_wv0.mask=ma.nomask
        # 去除第一维度
        satid_wv = satid_wv0[0, :, :][:, sorted_indices]
        # 将缺失值标记为掩码
        satid_wv = np.ma.masked_equal(satid_wv, -1)
        # 转换为无符号字节
        satid_wv_unsigned = satid_wv.astype(np.int16) + 128
        # 解包
        SATID_wv = np.floor(satid_wv_unsigned / 15.0)
        # ----------------------读取ERA5数据----------------------

        # 假设时间是以秒为单位，从1970年1月1日开始
        time_dates = [datetime.utcfromtimestamp(t) for t in time]
        # 将date_info_fy（例如："2023.01.01"）和time_info（例如："0900"）结合成一个完整的时间字符串
        date_info_fy = year_info +"."+ month_info +"." + day_info
        time_info_mod23 = time_info[0:2]
        target_time_str = f"{date_info_fy} {time_info_mod23[:2]}:00:00"
        # 将字符串转换为datetime对象
        target_time = datetime.strptime(target_time_str, "%Y.%m.%d %H:%M:%S")
        # 在time_dates中找到与target_time匹配的索引
        for i, time_point in enumerate(time_dates):
            if time_point == target_time:
                idx = i
                print(f"匹配的时间: {time_point}，索引: {idx}")
                # 直接索引对应时刻数据
                temp_data1 = temp1_all[idx]
                temp_data2 = temp2_all[idx]
                humi_data1 = humi1_all[idx]
                humi_data2 = humi2_all[idx]
                skt_data   = skt_all[idx]
                tcwv_data  = tcwv_all[idx]
                soil_data  = soil_all[idx]
        # 计算并平移天顶角数据
        shifted_zenith_angles = []
        for i in range(len(satlat)):  # 遍历每个卫星
            satlon_i = satlon[i]
            satlat_i = satlat[i]    
            shifted_values = shift_zenith_angle(based, satlon_i, satlat_i)
            shifted_zenith_angles.append(shifted_values)
        # 将所有卫星的天顶角数据保存为一个新的数组
        shifted_zenith_angles = np.array(shifted_zenith_angles)
        ##---------------------##
        # 获取 SATID 中的唯一值
        unique_sat_ids = np.unique(SATID)
        # 初始化结果数组，初始值设为无效值 -1
        final_SATID_angles = np.full_like(SATID, -1, dtype=np.float32)
        # 初步替换：按 best_coverage 替换 SATID 匹配的区域
        for sat_id in unique_sat_ids:
            sat_id_mask = (SATID == sat_id)
            best_shifted_values = None
            best_coverage_count = 0
            # 寻找最匹配的 shifted_values
            for shifted_values in shifted_zenith_angles:
                coverage_count = np.sum((shifted_values != -1) & sat_id_mask)
                if coverage_count > best_coverage_count:
                    best_shifted_values = shifted_values
                    best_coverage_count = coverage_count
            # 如果找到最佳的 shifted_values，则进行初步替换
            if best_shifted_values is not None:
                final_SATID_angles[sat_id_mask] = best_shifted_values[sat_id_mask]
        # 进一步填充未被替换的 -1 值
        for i in range(len(shifted_zenith_angles)):
            shifted_values = shifted_zenith_angles[i]
            # 找到 final_SATID_angles 中为 -1 的位置，且 shifted_values 中有有效值的位置
            unfilled_mask = (final_SATID_angles == -1) & (shifted_values != -1)
            final_SATID_angles[unfilled_mask] = shifted_values[unfilled_mask]
        # 最后补值为 0
        final_SATID_angles[final_SATID_angles == -1] = 0
        ##---------------------##
        # 获取 SATID_wv 中的唯一值
        unique_sat_ids_wv = np.unique(SATID_wv)
        # 初始化结果数组，初始值设为无效值 -1
        final_SATID_wv_angles = np.full_like(SATID_wv, -1, dtype=np.float32)
        # 初步替换：按 best_coverage 替换 SATID_wv 匹配的区域
        for sat_id in unique_sat_ids_wv:
            sat_id_mask_wv = (SATID_wv == sat_id)
            best_shifted_values_wv = None
            best_coverage_count_wv = 0
            # 寻找最匹配的 shifted_values
            for shifted_values in shifted_zenith_angles:
                coverage_count_wv = np.sum((shifted_values != -1) & sat_id_mask_wv)
                if coverage_count_wv > best_coverage_count_wv:
                    best_shifted_values_wv = shifted_values
                    best_coverage_count_wv = coverage_count_wv
            # 如果找到最佳的 shifted_values，则进行初步替换
            if best_shifted_values_wv is not None:
                final_SATID_wv_angles[sat_id_mask_wv] = best_shifted_values_wv[sat_id_mask_wv]
        # 进一步填充未被替换的 -1 值
        for i in range(len(shifted_zenith_angles)):
            shifted_values = shifted_zenith_angles[i]
            # 找到 final_SATID_wv_angles 中为 -1 的位置，且 shifted_values 中有有效值的位置
            unfilled_mask_wv = (final_SATID_wv_angles == -1) & (shifted_values != -1)
            final_SATID_wv_angles[unfilled_mask_wv] = shifted_values[unfilled_mask_wv]
        # 最后补值为 0
        final_SATID_wv_angles[final_SATID_wv_angles == -1] = 0
        # 组合数据成多维数组~
        mask_layer = np.expand_dims(mask_layer, axis=0)  # 变成 (1, 2000, 5143)
        # 将除 temp_data_at_time 和 humi_data_at_time 之外的数据堆叠
        other_data = np.stack([lat2d, lon2d, irwin1, irwvp, SATID, SATID_wv, final_SATID_angles, final_SATID_wv_angles,
                            skt_data, tcwv_data, soil_data], axis=0)
        gridsat_data_array1 = np.concatenate([other_data, temp_data1, temp_data2,humi_data1,humi_data2], axis=0)
        grid = gridsat_data_array1


        C,H,W  = grid.shape
        pad_h  = (PATCH - H%PATCH)%PATCH
        pad_w  = (PATCH - W%PATCH)%PATCH
        pad    = np.pad(grid,((0,0),(0,pad_h),(0,pad_w)),mode='constant')
        H2,W2  = pad.shape[1:]

        logits_sum  = np.zeros((3,H2,W2),dtype=np.float32)
        weight_sum  = np.zeros((H2,W2),dtype=np.float32)
        img_max = IMGPARA[0].reshape(1,-1,1,1).astype(np.float32)
        img_min = IMGPARA[1].reshape(1,-1,1,1).astype(np.float32)
        i_list = list(range(0,H2-PATCH+1,STRIDE))
        j_list = list(range(0,W2-PATCH+1,STRIDE))
        batch_size = 8192

        model.eval()
        with torch.no_grad():
            for i in i_list:
                row = []
                for j in j_list:
                    row.append(pad[:,i:i+PATCH,j:j+PATCH])
                row = np.stack(row,axis=0).astype(np.float32)
                row = (row-img_min)/(img_max-img_min)
                N_j = row.shape[0]
                logits_row = np.zeros((N_j,3,PATCH,PATCH),dtype=np.float32)
                for b0 in range(0,N_j,batch_size):
                    b1 = min(N_j,b0+batch_size)
                    out = model(torch.from_numpy(row[b0:b1]).to(DEVICE))
                    logits_row[b0:b1] = out.cpu().numpy()
                for idx,j in enumerate(j_list):
                    W2d = make_blend_window(PATCH,STRIDE,i,j,H2,W2)
                    logits_sum[:,i:i+PATCH,j:j+PATCH] += logits_row[idx]*W2d[None]
                    weight_sum[i:i+PATCH,j:j+PATCH]   += W2d

        avg_logits = logits_sum / weight_sum[None]
        pred_full  = np.argmax(avg_logits,axis=0).astype(np.int8)[:H,:W]
        # —— 在这里，对 any NaN 的像元直接赋值 -1 ——  # *** ADD
        pred_full[nan_mask] = -1

        # 存到列表 & 时间戳
        pred_list.append(pred_full)
        dt = datetime.strptime(ymd+hh, "%Y%m%d%H")
        times.append(int(dt.timestamp()))

        nc_file.close()

    # ── 合并 & 写 NetCDF ─────────────────────────────────────────────
    pred_full_day = np.stack(pred_list,axis=0)  # (N, H, W)
    times = np.array(times, dtype=np.int64)

    out_nc = os.path.join(output_root, f"{ymd}_CLP.nc")
    with Dataset(out_nc, 'w', format='NETCDF4') as dst:
        dst.createDimension('time', pred_full_day.shape[0])
        dst.createDimension('lat',  lat_grid.shape[0])
        dst.createDimension('lon',  lon_grid.shape[1])

        var_t = dst.createVariable('time', 'i8', ('time',))
        var_t.units     = 'seconds since 1970-01-01 00:00:00'
        var_t.long_name = 'time'
        var_t[:]        = times

        var_lat = dst.createVariable('latitude',  'f4', ('lat',))
        var_lat.units   = 'degrees_north'
        var_lat[:]      = lat_grid[:,0]

        var_lon = dst.createVariable('longitude', 'f4', ('lon',))
        var_lon.units   = 'degrees_east'
        var_lon[:]      = lon_grid[0,:]

        var_p = dst.createVariable(
            'pred_full','i1',('time','lat','lon',),
            zlib=True, complevel=5, fill_value=-1
        )
        var_p.long_name = 'cloud_phase_inversion (CLP)'
        var_p.comment   = '0: Clear, 1: Liquid, 2: Ice; missing: -1'
        var_p[:]        = pred_full_day

    print(f"Saved daily CLP file: {out_nc}")
