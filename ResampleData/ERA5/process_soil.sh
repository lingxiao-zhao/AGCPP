#!/bin/bash

# �����ļ������Ŀ¼
input_dir="/home/Data_Pool/zhaolx/ERA5/Soil"
output_dir="/home/Data_Pool/zhaolx/Resample/Resoil"
regrid_file="regrid.txt"

# ����ÿһ������ݣ�2000-2019��
for year in {2000..2019}
do
  # ���������ļ�·��
  input_file="${input_dir}/Soil${year}.nc"
  output_file="${output_dir}/Resoil${year}.nc"
  
  # ʹ�� cdo ���������ӳ���ѹ��
  cdo -O -z zip_9 remapcon,${regrid_file} "${input_file}" "${output_file}"
  
  # ���������
  echo "Processed: ${input_file} -> ${output_file}"
done
