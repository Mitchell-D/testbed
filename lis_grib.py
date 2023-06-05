
from pathlib import Path
from datetime import datetime as dt
import pygrib
import numpy as np
from pprint import pprint as ppt
import pickle as pkl

#import pygrib
#import matplotlib.pyplot as plt
#import xarray as xr
#lut = lis_dir.joinpath(Path("202007/LIS_RST_NOAH33_202007010000.d01.nc"))
#print(xr.open_dataset(lut.as_posix()))

#gefs_dir = Path("/raid1/sport/people/mdodson/test/")
lis_dir = Path("/raid1/sport/data/LIS7CLIMO/conus3km/SURFACEMODEL")


def get_lis_awips_vsm(run_time:dt):
    """ """
    # Vars: soilw (vsm), avsft (avg skin temp),  st (soil temp)
    awips_lis = lis_dir.joinpath(Path(run_time.strftime(
            "%Y%m/sportlis_conus3km_awips_%Y%m%d_%H00.grb2")))
    print(f"Retrieving AWIPS LIS file {awips_lis.as_posix()}")

    lis = xr.open_dataset(awips_lis.as_posix(), engine='cfgrib',
            backend_kwargs={
                'indexpath':'',
                'errors':'ignore',
                #'filter_by_keys':{'typeOfLevel': 'depthBelowLandLayer'},
                })
    lon, lat = np.meshgrid(lis["longitude"], lis["latitude"].data)
    print(lis["soilw"])
    print(lis["soilw"]["depthBelowLandLayer"])
    print(lis["soilw"]["surface"])
    vsm = lis["soilw"].data
    return vsm, lat, lon

def get_lis_paths_in_range(t0:dt, tf:dt):
    lis_hist = []
    assert tf > t0
    tmp_m = dt(year=t0.year, month=t0.month, day=1)
    monthf = dt(year=tf.year, month=tf.month, day=1)
    while tmp_m <= monthf:
        month_dir = lis_dir.joinpath(dt.strftime(tmp_m,"%Y%m"))
        lis_hist += [(dt.strptime(f.name,"LIS_HIST_%Y%m%d%H%M.d01.grb"),f)
                     for f in month_dir.iterdir() if "LIS_HIST" in f.name]
        print(tmp_m.year, tmp_m.month)
        tmp_m = dt(year=tmp_m.year+int(tmp_m.month==12),
                month=(tmp_m.month)%12+1, day=1)
    return sorted([r for r in lis_hist if t0<=r[0]<=tf], key=lambda a: a[0])

def get_lis_hist_latlon(lis_grb:Path):
    """
    Returns a (lat,lon) tuple numpy float arrays for the provided grb file.
    """
    gf = pygrib.open(lis_grb.as_posix())
    gf.seek(0)
    return tuple(gf[1].latlons())

def get_lis_hist_data(lis_grb:Path, grib_message_ids:list):
    """
    Returns a tuple of data arrays and corresponding boolean quality masks for
    the provided grib message IDs contained in the lis_grb file.
    """
    print(f"Parsing {lis_grb.as_posix()}")
    gf = pygrib.open(lis_grb.as_posix())
    gf.seek(0)
    return [gf.message(d).data()[0] for d in grib_message_ids]
    '''
    for g in gf:
        print(g["globalDomain"], g["parameterName"],
                g["parameterUnits"], g["level"])
        print(g, g["typeOfLevel"])
    '''

if __name__=="__main__":
    '''
    in_range = get_lis_paths_in_range(dt(year=2011,month=11,day=12),
            dt(year=2012,month=3,day=10))
    grb_time, grb_path = in_range.pop(0)
    print(grb_time, grb_path)
    '''
    static_data_dir = Path("data/lis_static")
    fig_dir = Path("figures")

    grb_time = dt(year=2011,month=11,day=12,hour=0)
    grb_path = Path("data/LIS_HIST_202007220000.d01.grb")

    # SM bins (x4), land mask, veg type, soil type
    soilm = get_lis_hist_data(grb_path, [10,11,12,13,27,28,29])
    latlon = np.dstack(get_lis_hist_latlon(grb_path))
    #print(soilm, soilm.shape)
    #with open("./data/soiltexture_statsgo_1KM_conus.pkl", "wb") as pklfp:
    #    pkl.dump(soilm[-1].astype(np.uint8), pklfp)

    pkl.dump(soilm[4][::-1], static_data_dir.joinpath(
        "lis_3KM_conus_landmask.pkl").open("wb"))
    enh.generate_raw_image(
            soilm[4][::-1], Path("./figures/static_lis_3KM_conus_landmask.png"))

    pkl.dump(soilm[5][::-1], static_data_dir.joinpath(
        "lis_3KM_conus_vegtype.pkl").open("wb"))
    enh.generate_raw_image(
            soilm[5][::-1], Path("./figures/static_lis_3KM_conus_vegtype.png"))

    pkl.dump(soilm[6][::-1], static_data_dir.joinpath(
        "lis_3KM_conus_soiltype.pkl").open("wb"))
    enh.generate_raw_image(
            soilm[6][::-1], Path("./figures/static_lis_3KM_conus_soiltype.png"))

    pkl.dump(latlon[::-1], static_data_dir.joinpath(
        "lis_3KM_conus_latlon.pkl").open("wb"))
    enh.generate_raw_image(
            soilm[6][::-1], Path("./figures/static_lis_3KM_conus_latlon.png"))
