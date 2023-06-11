import requests
import json
from pprint import pprint as ppt
from datetime import datetime as dt
from pathlib import Path
import numpy as np

awips_url = "https://geo.nsstc.nasa.gov/SPoRT/modeling/lis/conus3km/awips/"
awips_template = "sportlis_conus3km_awips_%Y%m%d_%H%M.grb2"
model_url = "https://geo.nsstc.nasa.gov/SPoRT/modeling/lis/conus3km/"
model_template = "sportlis_conus3km_model_%Y%m%d_%H%M.grb2"
json_append = "?format=json"
lis_dir = Path("data/lis_model")

def _is_grb2(f:Path):
    return Path(f).suffix=="grb2"

def list_lis(awips:bool=False):
    listing = requests.get(model_url+json_append).json()["directory_listing"]
    files = [(dt.strptime(f["filename"],model_template), Path(f["filename"]))
             for f in listing
             if "grb" in f["filename"] and awips==_is_grb2(f["filename"])]
    return sorted(files, key=lambda T: T[0])

def download_lis(webfile:Path, dest_dir:Path, replace=False):
    if _is_grb2(webfile):
        url = awips_url + webfile.name
    else:
        url = model_url + webfile.name
    dest_file = dest_dir.joinpath(webfile)
    if dest_file.exists():
        print(f"{dest_file.as_posix()} already exists.")
        return dest_file
    print(f"Getting {url}")
    grib_file = requests.get(url)
    with open(dest_file.as_posix(), "wb") as dest_fp:
        dest_fp.write(grib_file.content)
    return dest_file

if __name__=="__main__":
    lis_options = list_lis()
    times = []
    lis_files = []
    for t,l in lis_options:
        times.append(t)
        lis_files.append(download_lis(l, lis_dir, replace=True))
