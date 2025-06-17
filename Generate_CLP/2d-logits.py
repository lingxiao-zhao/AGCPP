import numpy as np
import numpy.ma as ma
import os
import netCDF4 as nc
from netCDF4 import Dataset
from datetime import datetime, timedelta
from ShiftZenith import shift_zenith_angle
import torch
import torch.nn.functional as F
from SmaAtUNet import SmaAt_UNet
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
# 配置
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = '/home/nvme/zhaolx/Unet_train/classification/modelsave/phase300-Class3Best/weights_102_phase_ddp.pth'  # 替换为你的权重路径
IMGPARA = np.zeros((2, 19))
IMGPARA[0, :] = [70.00,180.00,340.00,300.00,14,14,90,90,350,90,7,325,315,280,255,135,140,150,170]
IMGPARA[1, :] = [-70.00,-180.00,145.00,130.00,8,8,0,0,220,0,0,230,220,220,200,-13,-11,-13,-16]
PATCH = 64
STRIDE = 10
# 1. 加载模型
model = SmaAt_UNet(n_channels=19, n_classes=3, kernels_per_layer=2, bilinear=True, reduction_ratio=16)
state = torch.load(CKPT_PATH, map_location=DEVICE)
# 若保存的是module.state_dict()
if any(key.startswith('module.') for key in state.keys()):
    # 去掉 module.
    from collections import OrderedDict
    new_state = OrderedDict()
    for k,v in state.items(): new_state[k.replace('module.','')] = v
    state = new_state
model.load_state_dict(state)
model.to(DEVICE).eval()

year_path = 2011  # 设置年份
# 读取网格的经纬度（lon_grid 和 lat_grid）
data = np.load('/home/nvme/zhaolx/Dataset/grid_data.npz')
lon_grid = data['lon_grid']
lat_grid = data['lat_grid']
# 读取天顶角数据 (meteosat11_filtered_interpolated_values.npy)
based = np.load('/home/nvme/zhaolx/Dataset/meteosat11_filtered_interpolated_values.npy')
output_root   = f"/home/Data_Pool/zhaolx/CLP-Dataset/{year_path}"
masked_values = np.ma.masked_equal(based, -1)  # 掩蔽无效值
ERA5_TEMP_PATH1 = f"/home/Data_Pool/zhaolx/Resample/Retemp/Retemp{year_path}_300-500.nc"
ERA5_TEMP_PATH2 = f"/home/Data_Pool/zhaolx/Resample/Retemp/Retemp{year_path}_850-1000.nc"
ERA5_HUMI_PATH1 = f"/home/Data_Pool/zhaolx/Resample/Rehumi/Rehumi{year_path}_300-500.nc"
ERA5_HUMI_PATH2 = f"/home/Data_Pool/zhaolx/Resample/Rehumi/Rehumi{year_path}_850-1000.nc"
ERA5_SKT_PATH = f"/home/Data_Pool/zhaolx/Resample/Reskt/Reskt{year_path}.nc"
ERA5_TCWV_PATH = f"/home/Data_Pool/zhaolx/Resample/Retcwv/Retcwv{year_path}.nc"
ERA5_SOIL_PATH = f"/home/Data_Pool/zhaolx/Resample/Resoil/Resoil{year_path}.nc"

# 构建Gridsat对应的文件路径
gridsat_file_folder = "/home/Data_Pool/zhaolx/Gridsat/access"
gridsat_year_folder = os.path.join(gridsat_file_folder, str(year_path))
files = sorted(os.listdir(gridsat_year_folder))
for fname in files:
    if not fname.endswith('.nc'): continue
    parts = fname.split('.')
    year_info, month_info, day_info, time_info = parts[1:5]
    gridsat_file_path = os.path.join(gridsat_year_folder, fname)

    # 打开netCDF文件
    nc_file = nc.Dataset(gridsat_file_path)
    # 读取卫星星下点经纬度 (satlon 和 satlat)
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
    # 打开ERA5文件
    TEMP_file1 = nc.Dataset(ERA5_TEMP_PATH1)
    TEMP_file2 = nc.Dataset(ERA5_TEMP_PATH2)
    HUMI_file1 = nc.Dataset(ERA5_HUMI_PATH1)
    HUMI_file2 = nc.Dataset(ERA5_HUMI_PATH2)

    SKT_file = nc.Dataset(ERA5_SKT_PATH)
    TCWV_file = nc.Dataset(ERA5_TCWV_PATH)
    SOIL_file = nc.Dataset(ERA5_SOIL_PATH)
    # 获取时间戳
    time = SKT_file.variables['valid_time'][:]
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
            print(f"匹配的时间: {time_point}，索引: {i}")
            # 直接读取第i个时间点的数据
            temp_data_at_time1 = TEMP_file1.variables['t'][i]  # 获取对应时间点的温度数据
            temp_data_at_time2 = TEMP_file2.variables['t'][i]  # 获取对应时间点的温度数据
            humi_data_at_time1 = HUMI_file1.variables['r'][i]  # 获取对应时间点的湿度数据
            humi_data_at_time2 = HUMI_file2.variables['r'][i]  # 获取对应时间点的湿度数据
            skt_data_at_time = SKT_file.variables['skt'][i]    # 获取对应时间点的地表温度数据
            tcwv_data_at_time = TCWV_file.variables['tcwv'][i]  # 获取对应时间点的TCWV数据
            soil_data_at_time = SOIL_file.variables['slt'][i]  # 获取对应时间点的土壤数据
            # 例如，打印或存储结果
            print(f"Temperature: {temp_data_at_time1.shape}")
            print(f"Humidity: {humi_data_at_time1.shape}")
            print(f"Surface Temperature: {skt_data_at_time.shape}")
            print(f"TCWV: {tcwv_data_at_time.shape}")
            print(f"Soil Temperature: {soil_data_at_time.shape}")
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
    ###---------------------------------------###
    # 解析输入文件的日期和时间信息
    file_name = os.path.basename(gridsat_file_path)
    file_name_parts = file_name.split(".")
    year_info = file_name_parts[1] # 年份，如"2023"
    date_info = file_name_parts[2] + file_name_parts[3] # 提取日期信息的前8个字符，如 "0101"
    time_info = file_name_parts[4]  # 提取日期信息后4个字符，如 "09"

    # 组合数据成多维数组~
    mask_layer = np.expand_dims(mask_layer, axis=0)  # 变成 (1, 2000, 5143)
    # 将除 temp_data_at_time 和 humi_data_at_time 之外的数据堆叠
    other_data = np.stack([lat2d, lon2d, irwin1, irwvp, SATID, SATID_wv, final_SATID_angles, final_SATID_wv_angles,
                        skt_data_at_time, tcwv_data_at_time, soil_data_at_time], axis=0)
    gridsat_data_array1 = np.concatenate([other_data, temp_data_at_time1, temp_data_at_time2,humi_data_at_time1,humi_data_at_time2], axis=0)
    gridsat = gridsat_data_array1
    C, H, W = gridsat.shape

    # ——— Padding，仅需定义一次 H2, W2 ———
    pad_h = (PATCH - H % PATCH) % PATCH
    pad_w = (PATCH - W % PATCH) % PATCH
    padded = np.pad(gridsat, ((0,0),(0,pad_h),(0,pad_w)), mode='constant')
    H2, W2 = padded.shape[1], padded.shape[2]

    def make_blend_window(PATCH, STRIDE, i, j, H2, W2):
        P, S = PATCH, STRIDE

        # —— 水平权重 wx ——  
        wx = np.ones(P, dtype=np.float32)
        # 如果有左邻（j>0），就在前 S 列做 0→1 的线性
        if j > 0:
            wx[:S] = np.linspace(0, 1, S, dtype=np.float32)
        # 如果有右邻（j+P < W2），就在后 S 列做 1→0 的线性
        if j + P < W2:
            wx[-S:] = np.linspace(1, 0, S, dtype=np.float32)

        # —— 垂直权重 wy ——  
        wy = np.ones(P, dtype=np.float32)
        if i > 0:
            wy[:S] = np.linspace(0, 1, S, dtype=np.float32)
        if i + P < H2:
            wy[-S:] = np.linspace(1, 0, S, dtype=np.float32)

        # —— 外积得到 2D 窗口 ——  
        return np.outer(wy, wx)  # shape (P, P)

    # —— 推理与加权累加（按 i 行分批处理） ——  
    n_classes = 3
    logits_sum = np.zeros((n_classes, H2, W2), dtype=np.float32)
    weight_sum = np.zeros((H2, W2),      dtype=np.float32)

    # 先把 IMGPARA 转成 numpy，方便广播
    img_max = IMGPARA[0].reshape(1, -1, 1, 1).astype(np.float32)  # (1, C, 1, 1)
    img_min = IMGPARA[1].reshape(1, -1, 1, 1).astype(np.float32)

    # 预先计算所有 i_list, j_list
    i_list = list(range(0, H2 - PATCH + 1, STRIDE))
    j_list = list(range(0, W2 - PATCH + 1, STRIDE))

    batch_size = 8192  # 根据你的显存调整

    model.eval()
    with torch.no_grad():
        for i in i_list:
            # 1) 收集这一行上所有 j 对应的 patch，形成 (N_j, C, P, P)
            row_patches = []
            for j in j_list:
                row_patches.append(padded[:, i:i+PATCH, j:j+PATCH])
            row_patches = np.stack(row_patches, axis=0).astype(np.float32)  # (N_j, C, P, P)

            # 2) 在 CPU 上做归一化
            row_patches = (row_patches - img_min) / (img_max - img_min)

            # 3) 分批移到 GPU 并推理
            N_j = row_patches.shape[0]
            logits_row = np.zeros((N_j, n_classes, PATCH, PATCH), dtype=np.float32)
            for b0 in range(0, N_j, batch_size):
                b1 = min(N_j, b0 + batch_size)
                batch = torch.from_numpy(row_patches[b0:b1]).to(DEVICE)      # (b, C, P, P)
                out   = model(batch)                                          # (b, n_classes, P, P)
                logits_row[b0:b1] = out.cpu().numpy()

            # 4) 累加到全图
            for idx, j in enumerate(j_list):
                W2d = make_blend_window(PATCH, STRIDE, i, j, H2, W2)
                logits_sum[:, i:i+PATCH, j:j+PATCH] += logits_row[idx] * W2d[None]
                weight_sum[i:i+PATCH, j:j+PATCH]    += W2d

    # —— 最终合并 & 裁剪 ——  
    avg_logits = logits_sum / weight_sum[None, :, :]
    pred_full  = np.argmax(avg_logits, axis=0).astype(np.uint8)
    pred_full  = pred_full[:H, :W]

    # 定义明确的颜色映射（与 tab10 前三种颜色一致）
    colors = [plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)]
    cmap = ListedColormap(colors)
    # 可视化图像（带图例，颜色一致）
    plt.figure(figsize=(12,6))
    im = plt.imshow(pred_full, cmap=cmap, vmin=0, vmax=2)
    plt.axis('off')
    plt.title('Cloud Phase Prediction')

    # 添加图例
    labels = ['0: Clear', '1: Liquid', '2: Ice']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)

    plt.tight_layout()
    plt.savefig(f'/home/nvme/zhaolx/Dataset/CLP/{year_info}{month_info}{day_info}{time_info}_cloud_phase_full_prediction.png',
                bbox_inches='tight', dpi=200)
    plt.close()

    print('Saved numpy array and PNG to ./output/')