#!/bin/bash

# �����ļ������Ŀ¼
input_dir="/home/Data_Pool/zhaolx/ERA5/Temp"
output_dir="/home/Data_Pool/zhaolx/Resample/Retemp"
regrid_file="regrid.txt"

# ����ÿһ������ݣ�2000-2023��
for year in {2000..2023}
do
  # ���������ļ�·��
  input_file="${input_dir}/Temp_${year}_850-1000.nc"
  output_file="${output_dir}/Retemp${year}_850-1000.nc"
  
  # ʹ�� cdo ���������ӳ���ѹ��
  cdo -O -z zip_9 remapcon,${regrid_file} "${input_file}" "${output_file}"
  
  # ���������
  echo "Processed: ${input_file} -> ${output_file}"
done
