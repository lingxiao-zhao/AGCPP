#!/bin/bash

# 输入文件和输出目录
input_dir="/home/Data_Pool/zhaolx/ERA5/Temp"
output_dir="/home/Data_Pool/zhaolx/Resample/Retemp"
regrid_file="regrid.txt"

# 处理每一年的数据（2000-2023）
for year in {2000..2023}
do
  # 输入和输出文件路径
  input_file="${input_dir}/Temp_${year}_850-1000.nc"
  output_file="${output_dir}/Retemp${year}_850-1000.nc"
  
  # 使用 cdo 命令进行重映射和压缩
  cdo -O -z zip_9 remapcon,${regrid_file} "${input_file}" "${output_file}"
  
  # 输出处理结果
  echo "Processed: ${input_file} -> ${output_file}"
done
