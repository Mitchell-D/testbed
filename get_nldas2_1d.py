""" Module for NLDAS-2 data visualization and analysis """

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
#'''

from grib_tools import get_grib1_data, wgrib, grib_parse_pixels
from gesdisc import nldas2_to_time, noahlsm_to_time
from pickle_pixel_picker import pick_pixels

# hardcoded version of the nldas2 default statsgo lookup table
statsgo_texture_default = {
        1: ('sand', 'S',                  np.array([0.92, 0.05, 0.03])),
        2: ('loamy_sand', 'LS',           np.array([0.82, 0.12, 0.06])),
        3: ('sandy_loam', 'SL',           np.array([0.58, 0.32, 0.1 ])),
        4: ('silty_loam', 'SiL',          np.array([0.17, 0.7 , 0.13])),
        5: ('silt', 'Si',                 np.array([0.1 , 0.85, 0.05])),
        6: ('loam', 'L',                  np.array([0.43, 0.39, 0.18])),
        7: ('sandy_clay_loam', 'SCL',     np.array([0.58, 0.15, 0.27])),
        8: ('silty_clay_loam', 'SiCL',    np.array([0.1 , 0.56, 0.34])),
        9: ('clay_loam', 'CL',            np.array([0.32, 0.34, 0.34])),
        10: ('sandy_clay', 'SC',          np.array([0.52, 0.06, 0.42])),
        11: ('silty_clay', 'SiC',         np.array([0.06, 0.47, 0.47])),
        12: ('clay', 'C',                 np.array([0.22, 0.2 , 0.58])),
        13: ('organic_materials', 'OM',   np.array([0., 0., 0.])),
        14: ('water', 'W',                np.array([0., 0., 0.])),
        15: ('bedrock', 'BR',             np.array([0., 0., 0.])),
        16: ('other', 'O',                np.array([0., 0., 0.])),
        0: ('other', 'O',                 np.array([0., 0., 0.])),
        }

def load_textures(texture_path:Path=None):
    """
    Loads the statsgo_texture_classes.tbl file adapted from (Miller, 1998)
    See: https://doi.org/10.1175/1087-3562(1998)002<0001:ACUSMS>2.3.CO;2
    I should've hard-coded these... develop a config system in the future.

    :@param texture_path: optional path to a file formatted identically to
        table 7 in (Miller, 1998). Otherwise, uses hardcoded string.
    :@return: dictionary lookup table mapping
    """
    if not texture_path:
        return statsgo_texture_default
    lines = texture_path.open("r").readlines()
    rows = list(map(lambda L: L.replace("\n","").strip().split(" "), lines))
    rows = [r for r in rows if "#" not in r[0]]
    newrows = {} # Formatted
    for i in range(len(rows)):
        rows[i][0] = int(rows[i][0])
        # (id, desc, abbrev, sand_pct, silt_pct, clay_pct)
        # converts soil triangle percentiles into a decimal vector
        newrows.update({int(rows[i][0]):(rows[i][1], rows[i][2],
                        np.array(list(map(int,rows[i][3:])))/100)})
    return newrows

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

    #'''
    """ Open a sample file for basic operations, if needed """

    lsm_files, lsm_times = tuple(zip(
        *[(f,noahlsm_to_time(f)) for f in noahlsm_dir.iterdir()]))
    nldas_files, nldas_times = tuple(zip(
        *[(f,nldas2_to_time(f)) for f in nldas2_dir.iterdir()]))

    sample_file = nldas_files[-1]
    #sample_file = lsm_files[-1]
    data, info, geo = get_grib1_data(sample_file)
    for i in info:
        print(i)
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

    #'''
    # These pkls will store the pixels chosen by the user. Be careful
    # not to overwrite previous sets.
    set_label = "silty-loam"
    nldas_pkl = data_dir.joinpath(f"1D/{set_label}_nldas2_forcings_2019.pkl")
    noahlsm_pkl = data_dir.joinpath(f"1D/{set_label}_noahlsm_soilm_2019.pkl")
    #nldas_pkl = data_dir.joinpath("buffer/tmp_nldas.pkl")
    #noahlsm_pkl = data_dir.joinpath("buffer/tmp_noahlsm.pkl")
    chunk_size = 24
    workers = 12

    #'''
    """
    Pick a series of pixels using the NLDAS soil type scalar static cast to RGB
    """
    # Prompt the user to choose one or more pixels to analyze
    #stype = pkl.load(static_dir.joinpath(
    #    "soiltype_nldas_14km_conus.pkl").open("rb"))[::-1]
    # Find a pixel in SEUS or plains with consistent soil types
    # (coordinates of SPoRT-LIS pixel)
    stype = pkl.load(static_dir.joinpath(
        "soiltype_lis_3KM_conus.pkl").open("rb"))[::-1]
    pixels = pick_pixels(stype)

    """
    Generate an RGB for soil type
    """
    # Print selected soil type IDs for convenience
    ppt([ stype[T] for T in pixels])
    #print(load_textures(texture_path))
    texture_dict = load_textures()

    print(np.unique(stype))
    rgb = np.zeros((*stype.shape,3))
    for i in range(stype.shape[0]):
        for j in range(stype.shape[1]):
            if not stype[i,j]:
                rgb[i,j] = np.array([0,0,0])
            else:
                label, _, rgb[i,j] = texture_dict[int(stype[i,j])]
    #rgb = gt.scal_to_rgb(stype)
    #rgb[np.where(stype==12)] = np.array([0,0,0])
    #gt.quick_render(rgb)
    gp.generate_raw_image(rgb, fig_dir.joinpath("rgb_stype_sand-silt-clay_3KM.png"))
    exit(0)

    '''
    """
    Retrieve the pixels selected by the user from every grib grid.
    points are (t,p,b) shaped arrays for t times, p pixels, and b grib grids.
    """
    lsm_points = grib_parse_pixels(pixels=pixels, grib1_files=lsm_files,
                                   chunk_size=chunk_size, workers=workers)
    pkl.dump((pixels,np.stack(lsm_points)), noahlsm_pkl.open("wb"))

    nldas_points = grib_parse_pixels(pixels=pixels, grib1_files=nldas_files,
                                     chunk_size=chunk_size, workers=workers)
    pkl.dump((pixels, np.stack(nldas_points)), nldas_pkl.open("wb"))
    print(f"Data successfully acquired with pixels at indeces\n{pixels}")
    '''

    pixels, nldasTPB = pkl.load(nldas_pkl.open("rb"))
    _, lsmTPB = pkl.load(noahlsm_pkl.open("rb"))

    '''
    """
    Generate an RGB showing the domain of the selected pixels on the
    soil texture map from before, which is in (sand, silt, clay) space.
    """
    for j,i in pixels:
        rgb[j,i] = np.array([1.,1.,1.])
    gp.generate_raw_image(rgb, fig_dir.joinpath(f"selection_{set_label}.png"))
    '''

    '''
    """
    Plot the LSM selected pixels across the time series.
    """
    soilm_total = lsmTPB[:,:,24]
    soilm_layers = lsmTPB[:,:,25:29]
    print(enh.array_stat(soilm_total))
    print(enh.array_stat(soilm_layers))
    print(soilm_layers.shape)
    #d0, df = 0, 365
    d0, df = 160, 200
    ylines = [soilm_layers[d0*24:df*24,i,0].data
              for i in range(soilm_layers.shape[1])]

    #depth_label = "40-100cm_160-200day"
    depth_label = "0-10cm_160-200day"
    gp.plot_lines(
            domain=np.linspace(d0*24,df*24,(df-d0)*24)/24,
            ylines=ylines,
            image_path=fig_dir.joinpath(
                f"soilm_2019_{set_label}_{depth_label}.png"),
            show=True,
            plot_spec={
                "yrange":(15,50),
                "title":f"NLDAS-2 12-pixel soil moisture response; {set_label}, {depth_label}",
                "xlabel":"Day of the year",
                "ylabel":"Soil moisture ($\\frac{kg}{m^2}$)",
                "line_width":.8,
                "dpi":500,
                }
            )
    '''

    """
    """
    print(enh.array_stat(nldasTPB))
    print(enh.array_stat(nldasTPB))
    print(nldasTPB.shape)
    d0, df = 0, 365
    #d0, df = 160, 200
    ylines = [nldasTPB[d0*24:df*24,i,9].data
              for i in range(nldasTPB.shape[1])]

    force_label = "precip"
    gp.plot_lines(
            domain=np.linspace(d0*24,df*24,(df-d0)*24)/24,
            ylines=ylines,
            image_path=fig_dir.joinpath(
                f"nldas_2019_{set_label}_{force_label}.png"),
            show=True,
            plot_spec={
                #"yrange":(0,1000),
                "title":f"NLDAS-2 12-pixel time series, {set_label}, {force_label}",
                "xlabel":"Day of the year",
                "ylabel":"Hourly precip total ($\\frac{kg}{m^2}$)",
                "line_width":.8,
                "dpi":500,
                }
            )
