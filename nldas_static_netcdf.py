"""
Basic helper script for interfacing with the NLDAS static soil texture classes
and parametric data available Here: https://ldas.gsfc.nasa.gov/nldas/soils

You might have to switch your python path to import xarray; I don't recommend
installing it on the default conda environment for this codebase.
"""

import xarray as xr
import numpy as np
import pickle as pkl

from pathlib import Path
from aes670hw2 import enhance as enh
from aes670hw2 import guitools as gt
from aes670hw2 import geo_plot as gp

# UMD Land cover vegetation classes ordered according to
# https://ldas.gsfc.nasa.gov/nldas/vegetation-class
umd_veg_classes = [
        "water", "evergreen_needleleaf", "evergreen_broadleaf",
        "deciduous_needleleaf", "deciduous_broadleaf", "mixed_cover",
        "woodland", "wooded_grassland", "closed_shrubland", "open_shrubland",
        "grassland", "cropland", "bare", "urban"
        ]

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

def soil_class_lookup(soil_classes:np.ndarray, fill=[0.,0.,0.]):
    """
    Apply the (Miller, 1998) STATSGO soil class lookup table to a 2D
    integer array, returning a new scalar 2D array in [0,1] corresponding
    to the approximate percentiles of soil material on (sand, silt, clay)
    axes.
    """
    SC = soil_classes.astype(int)
    newfill = np.array(fill)
    assert len(newfill.shape)==1
    assert len(SC.shape)==2
    rgb = np.zeros((*SC.shape,3))
    textures = load_texture_table()
    for i in range(SC.shape[0]):
        for j in range(SC.shape[1]):
            # replace null values will fill value in data coords
            # zero is a safe bet here for soil classes.
            if not SC[i,j]:
                rgb[i,j] = newfill
            else:
                rgb[i,j] = textures[SC[i,j]][2]
    return rgb

def load_texture_table(texture_path:Path=None):
    """
    Loads the statsgo_texture_classes.tbl file adapted from (Miller, 1998)
    See: https://doi.org/10.1175/1087-3562(1998)002<0001:ACUSMS>2.3.CO;2
    I should've hard-coded these... develop a config system in the future.

    :@param texture_path: optional path to a file formatted identically to
        table 7 in (Miller, 1998). Otherwise, uses hardcoded string.
    :@return: dictionary lookup table mapping the texture integer label
        to a tuple of information like: (label_str, abbrev_str, soil_vec)
        where soil_vec is a percentage in [0,1] of sand, silt, and clay
        according to the (Miller, 1998) table. Effectively an RGB color.
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

def get_soil_parameters(nldas_static_nc:Path, fig_dir:Path=None, show=False):
    """
    Get the static soil parameters dataset and information about it.

    Source: https://ldas.gsfc.nasa.gov/nldas/soils

    :@param nldas_static_nc: Path to the GSFC soil parameter netCDF file
    :@param fig_dir: if provided, generates RGB renders of each
        static dataset.
    :@return: (bands, info) with (224,464) shaped (NLDAS-2 domain) ndarrays
        for each scalar static dataset (bands), and a list of dictionaries
        corresponding to meta-information about each band (info)
    """
    nldas_static = xr.open_dataset(nldas_static_nc)
    bands = []
    info = []
    for k in nldas_static.keys():
        _, name = k.split("_")
        bands.append(np.squeeze(nldas_static[k].data)[::-1])
        info.append(nldas_static[k].attrs)
        info[-1].update({"key":name})

    bands = bands[1:]
    info = info[1:]

    for i in range(len(bands)):
        nan_mask = np.isnan(bands[i])
        rgb = gt.scal_to_rgb(enh.linear_gamma_stretch(
            np.nan_to_num(bands[i],np.amin(bands[i]))))
        rgb[np.where(nan_mask)] = np.array([0,0,0])
        if show:
            gt.quick_render(rgb)
        if fig_dir:
            image_path = fig_dir.joinpath(
                    "static_nldas2_"+info[i]["key"]+".png")
            gp.generate_raw_image(rgb, image_path)
    return bands, info

def get_veg_and_soil_classes(nldas_types_nc:Path):
    """
    Get the vegetation and soil types datasets as 2d integer arrays, which
    abide by the UMD and STATSGO class numbers, respectively.

    This ignores the "conus mask" from the netCDF

    :@return: (veg, soil, land, (lat, lon)), all (224,464) integer ndarrays
    """
    nldas_static = xr.open_dataset(nldas_types_nc)
    bands = []
    info = []
    for k in nldas_static.keys():
        _, name = k.split("_")
        bands.append(np.squeeze(nldas_static[k].data)[::-1].astype(np.uint8))
        info.append(nldas_static[k].attrs)
        info[-1].update({"key":k})
        print(info[-1])
    lon, lat = np.meshgrid(nldas_static["lon"], nldas_static["lat"][::-1])
    return (bands[3], bands[4], bands[1], (lon, lat))

if __name__=="__main__":
    fig_dir = Path("figures/static")

    nldas_static_nc = Path("data/NLDAS_soil_Noah.nc4")
    nldas_types_nc = Path("data/NLDAS_masks-veg-soil.nc4")

    # New pickle containing all relevant static datasets on the NLDAS2 grid
    nldas_static_pkl = Path("data/nldas2_static_all.pkl")

    # Load numerical parameter datasets
    params, params_info = get_soil_parameters(nldas_static_nc)
    # Get integer class arrays for vegetation and soil type
    veg, soil, _, latlon = get_veg_and_soil_classes(nldas_types_nc)
    # Look up the (sand,silt,clay) percentages for each pixel
    soil_pct = soil_class_lookup(soil)
    gt.quick_render(soil_pct)

    """
    Build a comprehensive dictionary of NLDAS-2 static datasets, and
    load it into a pikle
    """
    static_data = {
            # List of (M,N) float arrays corresponding to static
            # numerical soil parameters
            "params":params,
            # List of dictionaries describing the soil parameters
            "params_info":params_info,
            # (M,N) array of UMD vegetation type integers
            "veg_type_ints":veg,
            # (M,N) array of STATSGO soil type integers
            "soil_type_ints":soil,
            # (M,N,3) shaped array in [0,1] for soil material percentages
            "soil_comp":soil_pct,
            # (lat,lon) tuple of 2d float arrays
            "geo":latlon,
            }

    with nldas_static_pkl.open("wb") as pklfp:
        pkl.dump(static_data, pklfp)
