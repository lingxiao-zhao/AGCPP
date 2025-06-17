
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def plot_classification(data, title, Lon2d, Lat2d):
    # 定义颜色映射（与 tab10 前三种颜色一致）
    colors = ['lightgray',plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)]
    cmap = ListedColormap(colors)

    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(Lon2d, Lat2d, data, cmap=cmap, vmin=-1, vmax=2, shading='auto')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    ax.set_title(title, fontsize=14)

    # 添加图例
    labels = ['-1: Missing','0: Clear', '1: Liquid', '2: Ice']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(4)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_single_variable(data, title, Lon2d, Lat2d, vmin, vmax, cmap, cbar_label, save_name=None):
    fig = plt.figure(figsize=(12, 6), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())

    pcm = ax.pcolormesh(Lon2d, Lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='white')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    ax.set_title(title, fontsize=14)
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, shrink=0.5)
    cbar.set_label(cbar_label, fontsize=12)

    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()