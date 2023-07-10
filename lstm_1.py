
from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime

from SparseTimeGrid import SparseTimeGrid
from GeoTimeSeries import GeoTimeSeries

def init_stg(static_pkl:Path):
    """
    Load static datasets from pkl created by nldas_static_netcdf.py into a new
    SparseTimeGrid object
    """
    static = pkl.load(static_pkl.open("rb"))
    stg = SparseTimeGrid(*static["geo"][::-1])
    stg.add_data_dir("data/GTS")
    stg.add_static("sand_pct", static["soil_comp"][:,:,0])
    stg.add_static("silt_pct", static["soil_comp"][:,:,1])
    stg.add_static("clay_pct", static["soil_comp"][:,:,2])
    stg.add_static("veg_type_ints", static["veg_type_ints"])
    stg.add_static("soil_type_ints", static["soil_type_ints"])
    for i in range(len(static["params_info"])):
        stg.add_static(
                static["params_info"][i]["standard_name"].replace(" ","_"),
                static["params"][i])
    return stg

if __name__=="__main__":
    data_dir = Path("data")
    static_pkl = data_dir.joinpath("static/nldas2_static_all.pkl")
    stg = init_stg(static_pkl)

    gtss_train = stg.search(
            time_range=(datetime(year=2018, month=4, day=1),
                        datetime(year=2018, month=9, day=1)),
            #yrange=(30,37),
            #xrange=(-93,-80),
            static={"soil_type_ints":4}, # sandy loam
            group_pixels=True
            )
    exit(0)
    feature_labels = ("APCP", "CAPE", "DLWRF", "DSWRF",
                      "PEVAP", "PRES", "SPFH", "TMP", "SOILM-0-10")
    soilm_train = [g.data for g in gtss_train if g.flabel=="SOILM-0-10"]
    feat_train_names, feat_train = zip(
            *[(g.flabel, g.data) for g in gtss_train
              if g.flabel in feature_labels])

    gtss_val = stg.search(
            time_range=(datetime(year=2021, month=4, day=1),
                        datetime(year=2021, month=9, day=1)),
            #yrange=(30,37),
            #xrange=(-93,-80),
            static={"soil_type_ints":4} # sandy loam
            )

    print(gtss_val)
