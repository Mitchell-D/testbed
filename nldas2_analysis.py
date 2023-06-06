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
from pickle_pixel_picker import pick_pixels

def grib_parse_pixels(pixels:list, datafiles:list):
    """
    Opens a grib1 file and extracts the values of pixels at a list of
    indeces like [ (j1,i1), (j2,i2), (j3,i3), ], returning a list of
    tuples like [ (fname, [val1,val2,val3]), (fname, [val1,val2,val3]), ]
    """
    pass

if __name__=="__main__":
    # Full hourly directory of FORA0125 files
    data_dir = Path("data")
    nldas2_dir = data_dir.joinpath("nldas2_2019")
    noahlsm_dir = data_dir.joinpath("noahlsm_2019")
    static_dir = data_dir.joinpath("lis_static")
    fig_dir = Path("figures/")
    init_time = dt(year=2019, month=1, day=1)
    final_time = dt(year=2020, month=1, day=1)

    #'''
    """ Open a sample file """
    nldas_files = list(nldas2_dir.iterdir())
    lsm_files = list(noahlsm_dir.iterdir())
    sample_file = nldas_files[-1]
    #sample_file = lsm_files[-1]
    data, info, geo = get_grib1_data(sample_file)
    #'''

    '''
    """ Generate sample graphics for each product """
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
                    f"sample_nldas2_{info[i]['record']}_{info[i]['name']}.png"))
        gt.quick_render(tmp_rgb)
    '''

    '''
    """
    Pick a time series pixel with pickle_pixel_picker using the LIS soil type
    """
    # Find a pixel in SEUS or plains with consistent soil types
    # (coordinates of SPoRT-LIS pixel)
    #lis_soiltype = pkl.load(static_dir.joinpath(
    #    "soiltype_lis_3KM_conus.pkl").open("rb"))[::-1]
    #pick_pixels(lis_soiltype, replace_val=-3)
    #print(enh.array_stat(lis_soiltype))
    #print(lis_soiltype.dtype)
    '''

    #'''
    """
    Pick a pixel using the  NLDAS soil type
    """
    px = pick_pixels(pkl.load(static_dir.joinpath(
        "soiltype_nldas_14km_conus.pkl").open("rb"))[::-1])[0]
    #'''
