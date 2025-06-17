#!/bin/bash

# �����ļ������Ŀ¼
input_dir="/home/Data_Pool/zhaolx/ERA5/SKT"
output_dir="/home/Data_Pool/zhaolx/Resample/Reskt"
regrid_file="regrid.txt"

# ����ÿһ������ݣ�2000-2023��
for year in {2017..2023}
do
  # ���������ļ�·��
  input_file="${input_dir}/SKT${year}.nc"
  output_file="${output_dir}/Reskt${year}.nc"
  
  # ʹ�� cdo ���������ӳ���ѹ��
  cdo -O -z zip_9 remapcon,${regrid_file} "${input_file}" "${output_file}"
  
  # ���������
  echo "Processed: ${input_file} -> ${output_file}"
done
