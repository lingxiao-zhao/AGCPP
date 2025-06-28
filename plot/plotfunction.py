import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import FixedLocator

def plot_classification(data, title, Lon2d, Lat2d):
    # 掩盖 nan
    data_masked = np.ma.masked_invalid(data)

    # 定义三类颜色
    colors = [plt.cm.tab10(0), plt.cm.tab10(6), plt.cm.tab10(9)]
    cmap = ListedColormap(colors)
    cmap.set_bad(color='white')

    # 离散化
    norm = BoundaryNorm(boundaries=[-0.5,0.5,1.5,2.5], ncolors=cmap.N)

    # 绘图
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -70.1, 70.1],
                  crs=ccrs.PlateCarree())
    im = ax.pcolormesh(Lon2d, Lat2d, data_masked,
                       cmap=cmap, norm=norm, shading='auto')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    # 网格线和刻度
    gl = ax.gridlines(draw_labels=False, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 10}
    lon_ticks = list(range(-180, 181, 30))   # [-180, -150, …, 150, 180]
    lat_ticks = [70,50,30,10,-10,-30,-50,-70]
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_title(title, fontsize=14)

    # 把分类也用 colorbar 来表示：位置、宽度与 plot_single_variable 保持一致
    cbar = plt.colorbar(im, ax=ax,
                        orientation='vertical',
                        pad=0.02,      # 距离图幅的距离
                        shrink=0.5,    # 缩放比例
                        ticks=[0,1,2])
    cbar.set_ticklabels(['Clear','Liquid','Ice'])
    cbar.set_label('Cloud Phase (CLP)', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_single_variable(data, title, Lon2d, Lat2d, vmin, vmax, cmap, cbar_label, save_name=None):
    # ----------------------------
    # 1. 先把传入的 cmap 字符串变成 colormap 对象
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # 2. 掩掉所有等于 0 的点
    data = np.ma.masked_where(data == 0, data)
    # 3. 让坏值（mask）显示为白色
    cmap.set_bad('white')
    # ----------------------------
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -70.1, 70.1],
                  crs=ccrs.PlateCarree())
    pcm = ax.pcolormesh(Lon2d, Lat2d, data,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='auto')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    gl = ax.gridlines(draw_labels=False, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 10}
    lon_ticks = list(range(-180, 181, 30))   # [-180, -150, …, 150, 180]
    lat_ticks = [70,50,30,10,-10,-30,-50,-70]
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(pcm, ax=ax,
                        orientation='vertical',
                        pad=0.02,
                        shrink=0.5)
    cbar.set_label(cbar_label, fontsize=12)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
