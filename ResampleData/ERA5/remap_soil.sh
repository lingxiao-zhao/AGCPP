#!/bin/bash

# ������������·��
input_dir="/home/Data_Pool/zhaolx/ERA5/Soil"
output_dir="/home/Data_Pool/zhaolx/Resample/Resoil"
regrid_file="/home/Data_Pool/zhaolx/ERA5/Soil/regrid.txt"

# ѭ������ÿ�������
for year in {2000..2009}
do
    # ���������ļ�·��
    input_file="${input_dir}/SST${year}.nc"
    output_file="${output_dir}/Resoil${year}.nc"
    
    # ִ�� cdo remapcon ����
    cdo remapcon,$regrid_file $input_file $output_file
    
    # �����־
    echo "Processed SST${year}.nc -> Resoil${year}.nc"
done
