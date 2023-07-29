from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime
import multiprocessing as mp

from TimeGrid import TimeGrid
from grib_tools import get_grib1_data
import gesdisc

from aes670hw2 import guitools as gt
from aes670hw2 import enhance as enh

if __name__=="__main__":
    data_dir = Path("data")
    static_pkl = data_dir.joinpath("static/nldas2_static_all.pkl")
    subgrid_dir = data_dir.joinpath("subgrids")
    # Dimensions of the .npy file subgrids wrt original nldas grid
    yrange, xrange = slice(64, 192), slice(200,328)
    # Initial and final time of the extracted time series
    init_time = datetime(2021, 4, 1)
    final_time = datetime(2021, 8, 1)
    # Feature labels of the datasets to extract. The order of these lists
    # corresponds to the order of the features dimension in returned arrays.
    nldas_feats = ["TMP", "PRES", "NCRAIN", "SPFH", "DLWRF", "DSWRF", "PEVAP"]
    noahlsm_feats = ['SOILM-0-10', 'SOILM-10-40', 'SOILM-40-100',
                     'SOILM-100-200']

    # Use a GUI to select a set of pixels to extract.
    soilpct = pkl.load(static_pkl.open("rb"))["soil_comp"][yrange,xrange]
    pixels = gt.get_category(soilpct)

    # Extract a timeseries for each selected pixels as a (T,F) shaped array
    # for T timesteps in range (init_time, final_time) and F features.
    nldas_paths = [p for p in subgrid_dir.iterdir()
                   if p.stem.split("_")[0]=="FORA0125"]
    tg_nldas = TimeGrid(
            time_file_tuples = [
                (datetime.strptime(p.stem.split("_")[1], "%Y%m%d-%H"),p)
                for p in nldas_paths],
            labels = ['TMP', 'SPFH', 'PRES', 'UGRD', 'VGRD', 'DLWRF',
                      'NCRAIN', 'CAPE', 'PEVAP', 'APCP', 'DSWRF']
            ).subset(init_time,final_time)
    nldas_arrays = tg_nldas.extract_timeseries(
            pixels, nldas_feats, nworkers=4)

    # Same thing for Noah-LSM data
    noahlsm_paths = [p for p in subgrid_dir.iterdir()
                   if p.stem.split("_")[0]=="NOAH0125"]
    tg_noahlsm = TimeGrid(
            time_file_tuples = [
                (datetime.strptime(p.stem.split("_")[1], "%Y%m%d-%H"),p)
                for p in noahlsm_paths],
            labels=['SOILM-0-100', 'SOILM-0-10', 'SOILM-10-40', 'SOILM-40-100',
                    'SOILM-100-200', 'LSOIL-0-10', 'LSOIL-10-40',
                    'LSOIL-40-100', 'LSOIL-100-200']
            ).subset(init_time,final_time)
    noahlsm_arrays = tg_noahlsm.extract_timeseries(
            pixels, noahlsm_feats, nworkers=4)

    feats = nldas_feats + noahlsm_feats
    px_arrays = [np.hstack([A,B]) for A,B in zip(nldas_arrays, noahlsm_arrays)]
    print(px_arrays[0].shape)
