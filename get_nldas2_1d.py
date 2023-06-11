"""
Procedural script enabling the user to choose a set pixels based on soil type
and extract a time series from a directory of GES DISC style grib1 files of
14km NLDAS-2 and Noah-LSM.

The chosen pixel indeces and extracted 1D datasets are stored as a pkl
with a tuple like (pixels, array) where is a (t,p,b) shaped array for t times,
p pixels, and b grib datasets.
"""

import numpy as np
import pickle as pkl
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
from pprint import pprint as ppt
#'''
from aes670hw2 import enhance as enh
from aes670hw2 import guitools as gt
from aes670hw2 import geo_plot as gp
from aes670hw2 import TextFormat
#'''

from grib_tools import get_grib1_data, wgrib, grib_parse_pixels
from gesdisc import nldas2_to_time, noahlsm_to_time
from pickle_pixel_picker import pick_pixels

if __name__=="__main__":
    # Full hourly directory of FORA0125 files
    data_dir = Path("data")
    nldas2_dir = data_dir.joinpath("nldas2_2019")
    noahlsm_dir = data_dir.joinpath("noahlsm_2019")
    static_dir = data_dir.joinpath("lis_static")
    fig_dir = Path("figures/pixel_curves")
    init_time = dt(year=2019, month=1, day=1)
    final_time = dt(year=2020, month=1, day=1)
    # These pkls will store the pixels chosen by the user. Be careful
    # not to overwrite previous sets.
    set_label = "silty-loam"
    nldas_pkl = data_dir.joinpath(f"1D/{set_label}_nldas2_all-forcings_2019.pkl")
    noahlsm_pkl = data_dir.joinpath(f"1D/{set_label}_noahlsm_all-fields_2019.pkl")

    nldas_files = sorted(list(nldas2_dir.iterdir()))
    lsm_files = sorted(list(noahlsm_dir.iterdir()))
    """ --------------------------------------------------------------- """

    #'''
    """
    Open a sample file to get the lists of info dicts provided by
    grib_tools.get_grib1_data, which will be loaded into the time
    series pkls.
    """
    # Get a full-domain NLDAS-2 grid for the last timestep in the series.
    nldas_files, nldas_times = tuple(zip(
        *[(f,nldas2_to_time(f)) for f in nldas2_dir.iterdir()]))
    nldas_grid, nldas_info, geo = get_grib1_data(nldas_files[-1])
    nldas_grid = np.dstack(nldas_grid)
    # Get a full-domain Noah-LSM grid for the last timestep in the series.
    lsm_files, lsm_times = tuple(zip(
        *[(f,noahlsm_to_time(f)) for f in noahlsm_dir.iterdir()]))
    noahlsm_grid, noahlsm_info, _ = get_grib1_data(lsm_files[-1])
    noahlsm_grid = np.dstack(noahlsm_grid)

    """ --------------------------------------------------------------- """

    '''
    """ Pick a series of pixels using the NLDAS soil type scalar dataset """
    chunk_size = 24
    workers = 12

    # Load the soil type static data to choose the pixels to parse
    # as a 1-dimensional time series
    stype = pkl.load(static_dir.joinpath(
        "soiltype_nldas_14km_conus.pkl").open("rb"))[::-1]
    #stype = pkl.load(static_dir.joinpath(
    #    "soiltype_lis_3KM_conus.pkl").open("rb"))[::-1]

    # Prompt the user to choose one or more pixels to analyze
    pixels = pick_pixels(stype, replace_val=0)
    '''

    '''
    """
    Call a multiprocessed method to open all NLDAS-2 and/or Noah-LSM files
    and extract the data at each pixel for each time step. The returned points
    arrays are (t,p,b) shaped arrays for t times, p pixels, and b grib grids.

    This script is intended to parse all the fields by default, even the
    other currently-irrelevant land surface features. build_1d_dataset
    extracts specific fields from the pkl created here while assembling the
    total 1D dataset.
    """
    if noahlsm_pkl.exists or nldas_pkl.exists:
        response = input(TextFormat.RED(f"Dataset {set_label} already " + \
            "exists! Overwrite? (y/n)", bold=True))
        if response.lower != "y":
            print("exiting.")
            exit(0)

    # Load the Noah-LSM time series from all files into a pkl
    lsm_points = grib_parse_pixels(pixels=pixels, grib1_files=lsm_files,
                                   chunk_size=chunk_size, workers=workers)
    pkl.dump((np.stack(lsm_points),pixels,noahlsm_info),
             noahlsm_pkl.open("wb"))
    # Load the NLDAS-2 time series from all files into a pkl
    nldas_points = grib_parse_pixels(pixels=pixels, grib1_files=nldas_files,
                                     chunk_size=chunk_size, workers=workers)
    pkl.dump((np.stack(nldas_points),pixels,noahlsm_info),
             nldas_pkl.open("wb"))
    print(f"Data successfully acquired with pixels at indeces\n{pixels}")
    '''

    """ --------------------------------------------------------------- """

    # Re-load the pkl of 1D datasets
    nldas, pixels, nldas_info = pkl.load(nldas_pkl.open("rb"))
    noahlsm, _, noahlsm_info = pkl.load(noahlsm_pkl.open("rb"))

    #'''
    """
    Plot the Noah-LSM soil moisture for selected pixels across the time series.
    """
    #d0, df = 0, 365
    # time range in days, starting from the first day in the dataset
    #d0, df = 0, 365
    d0, df = 160, 200

    """ Noah-LSM settings """
    # record numbers (per wgrib) of the Noah-LSM fields to plot
    noahlsm_records = (33,)
    # labels only used for plot title and file name
    #noahlsm_field_label = "LSOIL_0-10cm_160-200day"
    #noahlsm_field_label = "LSOIL_10-40cm_160-200day"
    #noahlsm_field_label = "LSOIL_40-100cm_160-200day"
    noahlsm_field_label = "LSOIL_100-200cm_160-200day"
    noahlsm_image_path = fig_dir.joinpath(
            f"noahlsm_2019_{set_label.lower()}_{noahlsm_field_label.lower()}.png")
    noahlsm_title = f"Noah-LSM 12-pixel {set_label}, {noahlsm_field_label.replace('_',' ')}"
    noahlsm_ylabel = "Soil moisture ($\\frac{kg}{m^2}$)"
    noahlsm_yrange = (250,400)

    """ NLDAS-2 settings """
    # record numbers (per wgrib) of the NLDAS-2 fields to plot
    nldas_records = (8,)
    # label only used for plot title and file name
    nldas_field_label = "cape_160-200day"
    nldas_image_path = fig_dir.joinpath(
            f"nldas_2019_{set_label.lower()}_{nldas_field_label.lower()}.png")
    nldas_title = f"NLDAS-2 12-pixel {set_label}, {nldas_field_label.replace('_',' ')}"
    nldas_ylabel = "CAPE ($\\frac{J}{kg}$)"
    nldas_yrange = (0,7000)

    """ --------------------------------------------------------------- """

    """ Noah-LSM time series plotting methods """
    noahlsm_fields = np.dstack(
            [noahlsm[:,:,i] for i in range(noahlsm.shape[2])
             if noahlsm_info[i]["record"] in noahlsm_records])
    ylines = [noahlsm_fields[d0*24:df*24,i,:].data
              for i in range(noahlsm_fields.shape[1])]
    print(TextFormat.GREEN(
        f"Noah-LSM {noahlsm_field_label} time series for each pixel", bright=True))
    for yl in ylines:
        print(enh.array_stat(yl))
    gp.plot_lines(
            domain=np.linspace(d0*24,df*24,(df-d0)*24)/24,
            ylines=ylines,
            labels = [f"px{i+1:02d}" for i in range(len(ylines))],
            image_path=noahlsm_image_path,
            show=True,
            plot_spec={
                "yrange":noahlsm_yrange,
                "title":noahlsm_title,
                "xlabel":"Day of the year",
                "ylabel":noahlsm_ylabel,
                "line_width":1.2,
                "legend_ncols":2,
                "dpi":200,
                }
            )
    #'''

    #'''
    """ NLDAS-2 time series plotting methods """
    nldas_fields = np.dstack([nldas[:,:,i] for i in range(nldas.shape[2])
                        if nldas_info[i]["record"] in nldas_records])
    ylines = [nldas_fields[d0*24:df*24,px,:].data
              for px in range(nldas_fields.shape[1])]
    print(TextFormat.GREEN(
        f"NLDAS-2 {nldas_field_label} time series for each pixel",bright=True))
    for yl in ylines:
        print(enh.array_stat(yl))
    gp.plot_lines(
            domain=np.linspace(d0*24,df*24,(df-d0)*24)/24,
            ylines=ylines,
            image_path=nldas_image_path,
            labels = [f"px{i+1:02d}" for i in range(len(ylines))],
            show=True,
            plot_spec={
                "yrange":nldas_yrange,
                "title":nldas_title,
                "xlabel":"Day of the year",
                "ylabel":nldas_ylabel,
                "line_width":1.2,
                "dpi":200,
                "legend_ncols":2,
                }
            )
    #'''
