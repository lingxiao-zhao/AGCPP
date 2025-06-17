import numpy as np
import netCDF4
from pyhdf.SD import SD
from MODread import readMod06
from resampleScipy2d import resampleScipy2d
import os
import netCDF4 as nc
# 指定年份和起始天数
year = 2015
start_day = "001"
end_day = "365"
# 文件路径
file_path = "/home/nvme/zhaolx/satellite_zenith_angle/lon_lat.npz"
# 读取 .npz 文件
data = np.load(file_path)
girdsat_lon = data['lon']
girdsat_lat = data['lat']
# 循环处理每一天的文件
for day in range(int(start_day), int(end_day) + 1):
    # 构建MOD03文件夹路径
    mod03_folder_path = f"/home/LVM_date2/data/GridSatall/MODIS/MYD03/{year}/{day:03d}/"
    # 检查MOD03文件夹是否存在
    if os.path.exists(mod03_folder_path):
        # 获取MOD03文件夹下的所有文件
        mod03_files = os.listdir(mod03_folder_path)
        # 遍历MOD03文件夹下的文件，按时间顺序找到文件
        for mod03_file_name in sorted(mod03_files, key=lambda x: int(x.split(".")[2])):
            if mod03_file_name.startswith("MYD03.A"):
                mod03_file_path = os.path.join(mod03_folder_path, mod03_file_name)

                # 检查MOD03文件是否存在
                if os.path.exists(mod03_file_path):
                    # 打开MOD03文件
                    mod03_file = netCDF4.Dataset(mod03_file_path, "r")
                    # 读取经纬度数据
                    latitude = mod03_file.variables["Latitude"][:]
                    # 判断latitude是否在指定范围内
                    if np.all((latitude >= -70) & (latitude <= 70)):
                        # 继续运行代码
                        longitude = mod03_file.variables["Longitude"][:]
                        mod03_file.close()
                    else:
                        # 如果latitude不在指定范围内，跳过当前文件，继续下一个文件的读取和判定
                        mod03_file.close()
                        continue
                else:
                    # 如果MOD03文件不存在，跳过当前文件
                    continue

                # 获取MOD03文件名中的时间信息
                mod03_file_name_parts = mod03_file_name.split(".")
                mod03_time_info = mod03_file_name_parts[-5][:8] + "." + mod03_file_name_parts[-4]
                # 构建MOD06_L2文件夹路径
                mod06_folder_path = f"/home/LVM_date2/data/GridSatall/MODIS/MYD06_L2/{year}/{day:03d}/"
                # 检查MOD06_L2文件夹是否存在
                if os.path.exists(mod06_folder_path):
                    # 获取MOD06_L2文件夹下的所有文件
                    mod06_files = os.listdir(mod06_folder_path)
                    # 遍历MOD06_L2文件夹下的文件，找到匹配的文件
                    for mod06_file_name in mod06_files:
                        mod06_file_name_parts = mod06_file_name.split(".")
                        mod06_time_info = mod06_file_name_parts[-5][:8] + "." + mod06_file_name_parts[-4]
                        if mod06_time_info == mod03_time_info:
                            mod06_file_path = os.path.join(mod06_folder_path, mod06_file_name)
                            break
                    # 检查MOD06_L2文件是否存在
                    if os.path.exists(mod06_file_path):
                        # 读取 cth、cer、cot、phase 和 ctt 数据
                        cth, cer, cot, phase, ctt = readMod06(mod06_file_path)
                        # 主程序
                        # 组合数据成多维数组
                        data_array = np.stack([cth, cer, cot, phase, ctt, latitude, longitude])
                        # 从 data_array 中提取经纬度数据
                        xx_org = data_array[-1, :, :]
                        yy_org = data_array[-2, :, :]
                        # 剔除 NaN 和 -999 值
                        valid_indices = (
                            np.isfinite(xx_org) & np.isfinite(yy_org) &  # 剔除 NaN
                            (xx_org != -999) & (yy_org != -999)          # 剔除值为 -999
                        )
                        xx_valid = xx_org[valid_indices]
                        yy_valid = yy_org[valid_indices]
                        # 目标网格的 x 和 y 坐标范围
                        x_min, x_max = np.min(xx_valid), np.max(xx_valid)
                        y_min, y_max = np.min(yy_valid), np.max(yy_valid)
                        # 剔除超出有效范围的网格
                        x_mask = (girdsat_lon >= x_min) & (girdsat_lon <= x_max)
                        y_mask = (girdsat_lat >= y_min) & (girdsat_lat <= y_max)
                        x_tag_1d = girdsat_lon[x_mask]
                        y_tag_1d = girdsat_lat[y_mask]
                        # y_min = max(y_min, -70)
                        # y_max = min(y_max, 70)
                        # x_tag_1d = np.arange(x_min, x_max + 0.07, 0.07)  #lon经度
                        # y_tag_1d = np.arange(y_min, y_max + 0.035, 0.07)  #lat纬度

                        # 创建一个数组来保存所有通道的数据
                        all_channel_data = np.zeros((5, y_tag_1d.shape[0], x_tag_1d.shape[0]))
                        print("当前处理的MYD06_L2文件:", mod06_file_name)
                        print("数据维度:", all_channel_data.shape)
                        # 循环处理每个通道数据
                        for i in range (5):
                            channel_data = data_array[i]
                            # 对当前通道进行插值和重采样
                            resampled_data, _ = resampleScipy2d(channel_data, xx_org, yy_org, x_tag_1d, y_tag_1d)
                            # 保存插值和重采样后的数据到数组中
                            all_channel_data[i] = resampled_data

                        # 解析输入文件的日期和时间信息
                        file_name = os.path.basename(mod06_file_name)
                        file_name_parts = file_name.split(".")
                        date_info = file_name_parts[1][1:]  # 提取日期信息，如 "2022152"
                        time_info = file_name_parts[2]  # 提取时间信息，如 "0030"
                        # 输出目录和文件名
                        output_dir = "/home/LVM_date2/zhaolx/Gridsat/Resample_data/MODIS/MYD0.07/" + str(year) + "/" + date_info
                        output_file_name = "ReMYD06_L2_" + date_info + "_" + time_info + ".nc"
                        output_file_path = os.path.join(output_dir, output_file_name)
                        # 创建输出目录
                        os.makedirs(output_dir, exist_ok=True)
                        # 创建.nc文件
                        nc_file = nc.Dataset(output_file_path, "w", format="NETCDF4")
                        # 创建维度
                        nc_file.createDimension("y", y_tag_1d.shape[0])
                        nc_file.createDimension("x", x_tag_1d.shape[0])
                        # 创建变量
                        y_var = nc_file.createVariable("lat", "f4", ("y", "x"))
                        x_var = nc_file.createVariable("lon", "f4", ("y", "x"))
                        cth_var = nc_file.createVariable("cth", "f4", ("y", "x"))
                        cer_var = nc_file.createVariable("cer", "f4", ("y", "x"))
                        cot_var = nc_file.createVariable("cot", "f4", ("y", "x"))
                        phase_var = nc_file.createVariable("phase", "f4", ("y", "x"))
                        ctt_var = nc_file.createVariable("ctt", "f4", ("y", "x"))
                        # 写入数据
                        y_var[:] = np.reshape(y_tag_1d, (y_tag_1d.shape[0], 1))
                        x_var[:] = np.reshape(x_tag_1d, (1, x_tag_1d.shape[0]))
                        cth_var[:] = all_channel_data[0]
                        cer_var[:] = all_channel_data[1]
                        cot_var[:] = all_channel_data[2]
                        phase_var[:] = all_channel_data[3]
                        ctt_var[:] = all_channel_data[4]
                        # 关闭.nc文件
                        nc_file.close()

                        print("保存成功！")                       
                else:
                    # 如果MOD06_L2文件夹不存在，跳过当前文件
                    continue
    else:
        # 如果MOD03文件夹不存在，跳过当前天数
        continue


