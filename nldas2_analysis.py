""" Module for NLDAS-2 data visualization and analysis """

import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
#'''
from aes670hw2 import enhance as enh
from aes670hw2 import guitools as gt
from aes670hw2 import geo_plot as gp
#'''

from get_nldas2 import get_grib1_data, wgrib, nldas2_to_time

if __name__=="__main__":
    # Full hourly directory of FORA0125 files
    data_dir = Path("data")
    nldas2_data_dir = data_dir.joinpath("nldas2_2019")
    static_data_dir = data_dir.joinpath("lis_static")
    fig_dir = Path("figures/")
    init_time = dt(year=2019, month=1, day=1)
    final_time = dt(year=2020, month=1, day=1)

    #'''
    """ Generate sample graphics for each product """
    lis_files = list(nldas2_data_dir.iterdir())
    sample_file = lis_files[-1]
    data, info, geo = get_grib1_data(sample_file)
    for i in range(len(data)):
        #amin = np.amin(data[i])
        #data[i][np.where(type(data[i])==np.ma.core.MaskedConstant)] = amin
        d = data[i][::-1]
        m = np.copy(data[i].mask[::-1])
        d[m] = np.amin(d)
        tmp_rgb = gt.scal_to_rgb(enh.linear_gamma_stretch(d))
        tmp_rgb[m] = np.array([0,0,0])
        gp.generate_raw_image(
                enh.norm_to_uint(tmp_rgb,256,np.uint8),
                fig_dir.joinpath(
                    f"sample_{info[i]['record']}_{info[i]['name']}.png"))
        #gt.quick_render(tmp_rgb)
    #'''

    """
    Get a time series of LIS data for a pixel chosen with pickle_pixel_picker
    """
    # Pixel in SEUS with abundance of soil types nearby
    # (coordinates of SPoRT-LIS pixel)
    px_center = [(622, 1209)]

    lis_soiltype = pkl.load(static_data_dir.joinpath(
        "soiltype_lis_3KM_conus.pkl").open("rb"))
    print(enh.array_stat(lis_soiltype))
    print(lis_soiltype.dtype)
