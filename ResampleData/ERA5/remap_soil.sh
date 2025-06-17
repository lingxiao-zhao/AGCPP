#!/bin/bash

# 定义输入和输出路径
input_dir="/home/Data_Pool/zhaolx/ERA5/Soil"
output_dir="/home/Data_Pool/zhaolx/Resample/Resoil"
regrid_file="/home/Data_Pool/zhaolx/ERA5/Soil/regrid.txt"

# 循环处理每年的数据
for year in {2000..2009}
do
    # 输入和输出文件路径
    input_file="${input_dir}/SST${year}.nc"
    output_file="${output_dir}/Resoil${year}.nc"
    
    # 执行 cdo remapcon 命令
    cdo remapcon,$regrid_file $input_file $output_file
    
    # 输出日志
    echo "Processed SST${year}.nc -> Resoil${year}.nc"
done
