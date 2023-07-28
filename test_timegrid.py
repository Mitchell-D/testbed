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
    static = pkl.load(static_pkl.open("rb"))
    gt.quick_render(static["soil_comp"][64:192,200:328])

    nldas_paths = [p for p in subgrid_dir.iterdir()
                   if p.stem.split("_")[0]=="FORA0125"]
    noahlsm_paths = [p for p in subgrid_dir.iterdir()
                   if p.stem.split("_")[0]=="NOAH0125"]
    TG = TimeGrid()
    TG.register_files(
            dataset_label="nldas",
            files=[(datetime.strptime(p.stem.split("_")[1], "%Y%m%d-%H"),p)
                   for p in nldas_paths ],
            feature_labels=[
                'TMP', 'SPFH', 'PRES', 'UGRD', 'VGRD', 'DLWRF',
                'NCRAIN', 'CAPE', 'PEVAP', 'APCP', 'DSWRF']
            )
    TG.register_files(
            dataset_label="noahlsm",
            files=[(datetime.strptime(p.stem.split("_")[1], "%Y%m%d-%H"),p)
                   for p in noahlsm_paths ],
            feature_labels=[
                'SOILM-0-100', 'SOILM-0-10', 'SOILM-10-40', 'SOILM-40-100',
                'SOILM-100-200', 'LSOIL-0-10', 'LSOIL-10-40', 'LSOIL-40-100',
                'LSOIL-100-200']
            )
    TG.validate_dataset("nldas")
    TG.validate_dataset("noahlsm")
