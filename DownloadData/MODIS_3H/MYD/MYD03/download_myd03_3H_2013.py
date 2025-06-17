import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
import argparse

def main(file):

    if not os.path.exists(os.path.join(savedir, os.path.basename(file))):
        print("command: ")
        commd = f'wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 "{file}" --header "Authorization: Bearer {token}" -P {savedir}'
        print(commd)
        print("start download!")
        os.system(commd)
        print("Finished!\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='selectable')
    parser.add_argument('-st', action="store", dest='st', default="2019")
    parser.add_argument('-pro', action="store", dest='pro', default="MOD06")
    parser.add_argument('-token', action="store", dest='tok', default=".....")
    args = parser.parse_args()
    year = args.st
    pro = args.pro
    token = args.tok

    downloadfile = f'/home/nvme/zhaolx/Download_MODIS/MODIS_3H/MYD03/MYD03_2013.txt'      # /{pro}_{year}.txt'
    savedir = '/home/LVM_date2/data/GridSatall/MODIS'
    f1 = open(downloadfile, 'r')
    lines = f1.readlines()
    "/home/LVM_date/data/MODIS/MYD06/2016/MYD06_L2/2016/001/MYD06_L2.A2016001.0000.061.2018059011445.hdf"
    if pro =='MYD06':
        files = [line[:-1] for line in lines if int(os.path.basename(line[:-1])[18:22]) in [0, 300, 600, 900, 1200, 1500, 1800, 2100]]  # download file at specific times
        files = sorted(files, key=lambda x: os.path.basename(x)[10:22])
    elif pro =='MOD06':
        files = [line[:-1] for line in lines if int(os.path.basename(line[:-1])[15:19]) in [0, 300, 600, 900, 1200, 1500, 1800, 2100]]  # download file at specific times
        files = sorted(files, key=lambda x: os.path.basename(x)[10:22])
    elif pro =='MYD03':
        files = [line[:-1] for line in lines if int(os.path.basename(line[:-1])[15:19]) in [0, 300, 600, 900, 1200, 1500, 1800, 2100]]  # download file at specific times
        files = sorted(files, key=lambda x: os.path.basename(x)[7:19])
    elif pro =='MOD03':
        files = [line[:-1] for line in lines if int(os.path.basename(line[:-1])[15:19]) in [0, 300, 600, 900, 1200, 1500, 1800, 2100]]  # download file at specific times
        files = sorted(files, key=lambda x: os.path.basename(x)[7:19])
    f1.close()

    with ProcessPoolExecutor(max_workers=4) as pool:
        all_task = [pool.submit(main, file) for file in files]
        wait(all_task, return_when=ALL_COMPLETED)
    print("Finished!\n")