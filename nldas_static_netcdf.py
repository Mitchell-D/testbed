"""
Helper script for interfacing with the NLDAS static soil texture data
available Here: https://ldas.gsfc.nasa.gov/nldas/soils

You might have to switch your python path to import xarray; I don't recommend
installing it on the default conda environment for this codebase.
"""

import xarray as xr
from pathlib import Path
from aes670hw2 import enhance as enh
from aes670hw2 import guitools as gt

nldas_static_path = Path("data/NLDAS_masks-veg-soil.nc4")
nldas_static = xr.open_dataset(nldas_static_path)
print(nldas_static.attrs)
for k in nldas_static.keys():
    print(k, nldas_static[k].attrs)

""" Format a pkl of info like (arrays, info, geo) """
