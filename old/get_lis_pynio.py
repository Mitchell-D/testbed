import requests
import json
from pprint import pprint as ppt
from datetime import datetime as dt
from pathlib import Path
import xarray as xr
import numpy as np

awips_url = "https://geo.nsstc.nasa.gov/SPoRT/modeling/lis/conus3km/awips/"
awips_template = "sportlis_conus3km_awips_%Y%m%d_%H%M.grb2"
model_url = "https://geo.nsstc.nasa.gov/SPoRT/modeling/lis/conus3km/"
model_template = "sportlis_conus3km_model_%Y%m%d_%H%M.grb2"
json_append = "?format=json"
awips_dir = Path("data/lis_awips")

def list_awips():
    listing = requests.get(awips_url+json_append).json()["directory_listing"]
    return sorted([(dt.strptime(f["filename"],awips_template),
                    Path(f["filename"]))
                   for f in listing if ".grb2" in f["filename"]],
                  key=lambda T: T[0])
def list_model():
    listing = requests.get(model_url+json_append).json()["directory_listing"]
    return sorted([(dt.strptime(f["filename"],model_template),
                    Path(f["filename"]))
                   for f in listing if ".grb2" in f["filename"]],
                  key=lambda T: T[0])

def download_awips(webfile:Path, dest_dir:Path, replace=False):
    url = awips_url + webfile.name
    dest_file = dest_dir.joinpath(webfile)
    if dest_file.exists():
        print(f"{dest_file.as_posix()} already exists.")
        return dest_file
    print(f"Getting {url}")
    awips_file = requests.get(url)
    with open(dest_file.as_posix(), "wb") as dest_fp:
        dest_fp.write(awips_file.content)
    return dest_file

if __name__=="__main__":
    awips_files = list_awips()
    dl_file = download_awips(awips_files[-1][1], awips_dir, replace=True)
    data = xr.open_dataset(dl_file.as_posix(), engine="pynio")
    for k in data.variables.keys():
        v = data.variables[k]
        print(k,v.attrs["long_name"])
    VSM = data.variables["SOILW_P0_2L106_GLL0"]
    TS = data.variables["TSOIL_P0_2L106_GLL0"]
    print(VSM.data)
    print(VSM.data.shape)
    print(VSM.attrs)
    #lon,lat = np.meshgrid(data.coords["lon_0"],data.coords["lat_0"])
    #print(lon,lat)
