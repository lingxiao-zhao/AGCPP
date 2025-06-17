import os
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
from CalipsoRead import read_clay

# 读取插值网格数据
file_path = "/home/nvme/zhaolx/satellite_zenith_angle/lon_lat.npz"
data = np.load(file_path)
girdsat_lon = data['lon']  # shape:(5143,)
girdsat_lat = data['lat']  # shape:(2000,)

# 生成网格点坐标 (meshgrid)
girdsat_lon_grid, girdsat_lat_grid = np.meshgrid(girdsat_lon, girdsat_lat)
grid_points = np.column_stack([girdsat_lon_grid.ravel(), girdsat_lat_grid.ravel()])

# 读取 Calipso 数据文件夹
folder_path = "/home/Data_Pool/zhaolx/Calipso_day/2025104114642_83676/"
all_files = os.listdir(folder_path)
hdf_files = [f for f in all_files if f.endswith('.hdf')]

# 生成 KDTree 用于最近邻搜索
tree = cKDTree(grid_points)

for file_name in hdf_files:
    file_path = os.path.join(folder_path, file_name)
    file_name_parts = file_name.split("-")
    year = file_name_parts[3][3:7]
    month = file_name_parts[4]
    day = file_name_parts[5][:2]
    time = file_name_parts[5][3:] + ":" + file_name_parts[6]
    print(f"Processing file: {file_path}")
    # 构建输出路径并检查是否已存在，若已处理则跳过
    output_dir = f"/home/Data_Pool/zhaolx/ResampleCalipso-day/{year}"
    output_file_name = f"ReCalipso_{year}.{month}.{day}.{time}.nc"
    output_file_path = os.path.join(output_dir, output_file_name)
    if os.path.exists(output_file_path):
        print(f"文件已存在，跳过: {output_file_path}")
        continue
    # 读取 Calipso 数据
    cth, cot_certain, cot_uncertain, daynight, lon, lat, ts_str, iwphase, singleLayerFlag, signal, samePhaseFlag = read_clay(file_path)

    # 只保留纬度在 -70 到 70 之间的点
    mask = (lat >= -70) & (lat <= 70)
    lon_filtered = lon[mask]
    lat_filtered = lat[mask]
    cth_filtered = cth[mask]
    cot_certain_filtered = cot_certain[mask]
    iwphase_filtered = iwphase[mask]

    # 最近邻插值
    query_points = np.column_stack([lon_filtered, lat_filtered])
    _, nearest_idx = tree.query(query_points)

    # 创建网格大小的数组，初始化为 NaN
    cth_grid = np.full(girdsat_lon_grid.shape, np.nan)
    cot_certain_grid = np.full(girdsat_lon_grid.shape, np.nan)
    iwphase_grid = np.full(girdsat_lon_grid.shape, np.nan)

    # 将值填充到最近的网格点
    for i, idx in enumerate(nearest_idx):
        grid_y, grid_x = np.unravel_index(idx, (girdsat_lat.shape[0], girdsat_lon.shape[0]))  # 修正索引转换
        cth_grid[grid_y, grid_x] = cth_filtered[i]
        cot_certain_grid[grid_y, grid_x] = cot_certain_filtered[i]
        iwphase_grid[grid_y, grid_x] = iwphase_filtered[i]

    # 创建输出文件夹
    output_dir = f"/home/Data_Pool/zhaolx/ResampleCalipso-day/{year}"
    os.makedirs(output_dir, exist_ok=True)

    # 输出 NetCDF 文件名
    output_file_name = f"ReCalipso_{year}.{month}.{day}.{time}.nc"
    output_file_path = os.path.join(output_dir, output_file_name)

    # 创建 NetCDF 文件
    with nc.Dataset(output_file_path, "w", format="NETCDF4") as nc_file:
        # 创建维度
        nc_file.createDimension("y", girdsat_lat.shape[0])
        nc_file.createDimension("x", girdsat_lon.shape[0])

        # 创建变量
        y_var = nc_file.createVariable("lat", "f4", ("y",))
        x_var = nc_file.createVariable("lon", "f4", ("x",))
        cth_var = nc_file.createVariable("cth", "f4", ("y", "x"))
        cot_var = nc_file.createVariable("cot", "f4", ("y", "x"))
        iwphase_var = nc_file.createVariable("phase", "f4", ("y", "x"))

        # 写入数据
        y_var[:] = girdsat_lat
        x_var[:] = girdsat_lon
        cth_var[:] = cth_grid
        cot_var[:] = cot_certain_grid
        iwphase_var[:] = iwphase_grid

    print(f"保存成功: {output_file_path}")
