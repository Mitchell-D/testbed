"""
Basic helper script for reading the NLDAS static soil texture classes
and parametric data from the netCDFs at: https://ldas.gsfc.nasa.gov/nldas/
"""

import xarray as xr
import numpy as np
import pickle as pkl

from pathlib import Path

# UMD Land cover vegetation classes ordered according to
# https://ldas.gsfc.nasa.gov/nldas/vegetation-class
umd_veg_classes = [
        "water", "evergreen_needleleaf", "evergreen_broadleaf",
        "deciduous_needleleaf", "deciduous_broadleaf", "mixed_cover",
        "woodland", "wooded_grassland", "closed_shrubland", "open_shrubland",
        "grassland", "cropland", "bare", "urban"
        ]

## hardcoded version of the statsgo composition lookup table (sand, silt, clay)
## http://www.soilinfo.psu.edu/index.cgi?soil_data&conus&data_cov&fract&methods
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

def get_static_params(
        nc_masks:Path, nc_soil:Path, nc_gfrac:Path, nc_elev:Path):
    """
    :@parameter nc_masks: integer class arrays for UMD vegetation and STATSGO
        soil types, as well as a boolean land mask and latitude/longitude.
        Doesn't consider fractional coverage of vegetation components.
        https://ldas.gsfc.nasa.gov/nldas/vegetation-class

    :@parameter nc_soil: soil parameter arrays and descriptions
        https://ldas.gsfc.nasa.gov/nldas/soils

    :@parameter nc_gfrac: Monthly greenness fraction
        https://ldas.gsfc.nasa.gov/nldas/lai-greenness

    :@param nc_elev: GTOP elevation
        https://ldas.gsfc.nasa.gov/nldas/elevation
    """
    data = []
    labels  = []

    assert nc_masks.exists()
    assert nc_soil.exists()
    assert nc_gfrac.exists()
    assert nc_elev.exists()

    ## vegetation, soil, and CONUS integer masks (and lat/lon)
    int_masks = xr.open_dataset(nc_masks)
    lon, lat = np.meshgrid(int_masks["lon"], int_masks["lat"][::-1])
    mask = np.squeeze(int_masks["CONUS_mask"].data)[::-1].astype(np.uint8)
    mask = (mask == 1)
    veg = np.squeeze(int_masks["NLDAS_veg"].data)[::-1].astype(np.uint8)
    soil = np.squeeze(int_masks["NLDAS_soil"].data)[::-1].astype(np.uint8)
    soil_pct = soil_class_lookup(soil)
    soil_pct = [soil_pct[...,i] for i in range(soil_pct.shape[-1])]
    labels += ["lat", "lon", "m_conus", "int_veg", "int_soil",
            "pct_sand", "pct_silt", "pct_clay"]
    data += [lat, lon, mask, veg, soil, *soil_pct]

    ## Soil properties (contains NaN where not CONUS mask)
    nldas_static = xr.open_dataset(nc_soil)
    for k in nldas_static.keys():
        if k == "time_bnds":
            continue
        _,name = k.split("_")
        data.append(np.squeeze(nldas_static[k].data)[::-1])
        labels.append(name)

    '''
    ## Monthly greeness fraction (skipped due to odd decode problems)
    nldas_gfrac = xr.open_dataset(nc_gfrac)
    print(nldas_gfrac.keys())
    '''

    ## Elevation (contains NaN where not CONUS mask)
    nldas_elev = xr.open_dataset(nc_elev)
    for k in ["NLDAS_elev", "NLDAS_elev_std", "NLDAS_slope", "NLDAS_aspect"]:
        data.append(np.squeeze(nldas_elev[k].data)[::-1])
    labels += ["elev", "elev_std", "slope", "aspect"]
    return labels, data


if __name__=="__main__":
    fig_dir = Path("figures/static")
    labels, data = get_static_params(
            nc_masks=Path("data/static/NLDAS_masks-veg-soil.nc4"),
            nc_gfrac=Path("data/static/NLDAS_gfrac.nc4"),
            nc_soil=Path("data/static/NLDAS_soil_Noah.nc4"),
            nc_elev=Path("data/static/NLDAS_elevation.nc4"),
            )

    # New pickle containing all relevant static datasets on the NLDAS2 grid
    nldas_static_pkl = Path("data/static/nldas_static.pkl")
    assert not nldas_static_pkl.exists()
    pkl.dump((labels,data), nldas_static_pkl.open("wb"))
