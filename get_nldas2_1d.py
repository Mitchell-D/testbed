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

from grib_tools import get_grib1_data, wgrib, grib_parse_pixels
from gesdisc import nldas2_to_time, noahlsm_to_time
from pickle_pixel_picker import pick_pixels

def load_textures(texture_path:Path):
    rows = list(map(lambda L: L.replace("\n","").split(" "), texture_path.open("r").readlines()))
    rows = [r for r in rows if "#" not in r[0]]
    for i in range(len(rows)):
        rows[i][0] = int(rows[i][0])
    return rows


if __name__=="__main__":
    # Full hourly directory of FORA0125 files
    data_dir = Path("data")
    nldas2_dir = data_dir.joinpath("nldas2_2019")
    noahlsm_dir = data_dir.joinpath("noahlsm_2019")
    static_dir = data_dir.joinpath("lis_static")
    texture_path = data_dir.joinpath("statsgo_texture_classes.tbl")
    fig_dir = Path("figures/")
    init_time = dt(year=2019, month=1, day=1)
    final_time = dt(year=2020, month=1, day=1)
    nldas_files = sorted(list(nldas2_dir.iterdir()))
    lsm_files = sorted(list(noahlsm_dir.iterdir()))

    '''
    """ Open a sample file for basic operations, if needed """

    lsm_files, lsm_times = tuple(zip(
        *[(f,noahlsm_to_time(f)) for f in noahlsm_dir.iterdir()]))
    nldas_files, nldas_times = tuple(zip(
        *[(f,nldas2_to_time(f)) for f in nldas2_dir.iterdir()]))

    sample_file = nldas_files[-1]
    #sample_file = lsm_files[-1]
    data, info, geo = get_grib1_data(sample_file)
    '''
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

    #'''
    """
    Pick a time series pixel with pickle_pixel_picker using the LIS soil type
    """
    # Find a pixel in SEUS or plains with consistent soil types
    # (coordinates of SPoRT-LIS pixel)
    #lis_soiltype = pkl.load(static_dir.joinpath(
    #    "soiltype_lis_3KM_conus.pkl").open("rb"))[::-1]
    #pick_pixels(lis_soiltype, replace_val=-3)
    #'''

    # These pkls will store the pixels chosen by the user. Be careful
    # not to overwrite previous sets.
    nldas_pkl = data_dir.joinpath("silty-loam_nldas2_forcings_2019.pkl")
    noahlsm_pkl = data_dir.joinpath("silty-loam_noahlsm_soilm_2019.pkl")
    chunk_size = 24
    workers = 12

    #'''
    """
    Pick a pixel using the  NLDAS soil type, and extract
    """
    # Prompt the user to choose one or more pixels to analyze
    stype = pkl.load(static_dir.joinpath(
        "soiltype_nldas_14km_conus.pkl").open("rb"))[::-1]
    pixels = pick_pixels(stype)

    # Print selected soil type IDs for convenience
    print([ stype[T] for T in pixels])
    print(load_textures(texture_path))

    rgb = gt.scal_to_rgb(stype)
    rgb[np.where(stype==12)] = np.array([0,0,0])
    gt.quick_render(rgb)

    # Retrieve the pixels selected by the user from every grib grid.
    # points are (t,p,b) shaped arrays for t times, p pixels, and b grib grids.
    lsm_points = grib_parse_pixels(pixels=pixels, grib1_files=lsm_files,
                                   chunk_size=chunk_size, workers=workers)
    pkl.dump(np.stack(lsm_points), noahlsm_pkl.open("wb"))

    nldas_points = grib_parse_pixels(pixels=pixels, grib1_files=nldas_files,
                                     chunk_size=chunk_size, workers=workers)
    pkl.dump(np.stack(nldas_points), nldas_pkl.open("wb"))
    print(f"Data successfully acquired with pixels at indeces\n{pixels}")
    #'''
