"""
Quick script for plotting the slopetype field from a GDAS parameter file
"""
import netCDF4 as nc4
import numpy as np
from pathlib import Path

from testbed import plotting

if __name__=="__main__":
    #root_dir = Path("/Users/mtdoddson/Desktop/slopetype-tiling")
    root_dir = Path("/rhome/mdodson/testbed")
    gdas_file = root_dir.joinpath("data/static").joinpath(
            "lis71_input_GDAStbot_viirsgvf_GDASforc.d01_conus3km.nc")
    ncd = nc4.Dataset(root_dir.joinpath(gdas_file))
    int_data = ncd["SLOPETYPE"][::-1]
    print(np.unique(int_data))
    print(int_data.shape)
    int_data[0,0] = 8
    print(np.unique(int_data))
    plotting.plot_geo_ints(
            #int_data=np.where(
            #    ncd["LANDMASK"][::-1], ncd["SLOPETYPE"][::-1], np.nan),
            int_data=int_data,
            #int_ticks=np.array(list(range(10)))+.5,
            int_ticks=np.array(list(range(10)))*(9/10)+.5,
            int_labels=list(map(str,range(10))),
            lat=ncd["lat"][::-1],
            lon=ncd["lon"][::-1],
            geo_bounds=None,
            fig_path=root_dir.joinpath(
                f"figures/static/static_slopetype_3km.png"),
            color_list=["white", "blue", "orange", "green", "red",
                    "purple", "olive", "cyan", "pink", "gray"],
            plot_spec={
                #"cmap":"gist_ncar",
                "cbar_pad":0.02,
                "cbar_orient":"horizontal",
                "cbar_tick_rotation":90,
                "cbar_fontsize":14,
                "title":"GDAS Slopetype Parameter",
                "title_fontsize":18,
                "interpolation":"none",
                },
            )
