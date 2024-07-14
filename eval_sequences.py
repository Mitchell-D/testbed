""" """
import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from pprint import pprint as ppt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from list_feats import nldas_record_mapping,noahlsm_record_mapping
from generators import gen_sequence_samples

def sequence_info(sequence_h5:Path):
    """ """
    print(f"Opening {sequence_h5.name}")
    with h5py.File(sequence_h5, mode="r") as F:

        W = F["/data/window"]
        H = F["/data/horizon"]
        P = F["/data/pred"]
        S = F["/data/static"]
        SI = F["/data/static_int"]
        T = F["/data/time"]
        #T = [datetime.fromtimestamp(int(t))
        #        for t in tuple(F["/data/time"][...])]

        print(T.shape, H.shape, P.shape, S.shape, SI.shape, T.shape)
        '''
        print(W[0])
        print(H[0])
        print(P[0])
        print(S[0])
        print(SI[0])
        print([datetime.fromtimestamp(int(t)) for t in T[0]])
        '''

        #ppt(json.loads(F["data"].attrs["gen_params"]))
    return

if __name__=="__main__":
    data_dir = Path("data")
    sequence_dir = data_dir.joinpath("sequences")

    seq_h5s = mm.get_seq_paths(
            sequence_h5_dir=sequence_dir,
            region_strs=("se", "sc", "sw", "ne", "nc"),
            season_strs=("warm", "cold"),
            time_strs=("2013-2018", "2018-2023"),
            )

    gen = gen_sequence_samples(
            sequence_hdf5s=seq_h5s,

            num_procs=5,
            frequency=1,
            sample_on_frequency=True,
            deterministic=False,
            buf_size_mb=1024,
            block_size=8,

            #dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            #static_norm_coeffs=dict(static_coeffs),

            seed=1,
            window_feats=[
                    "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                    "dlwrf", "dswrf", "apcp",
                    "soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd"
                    ],
            horizon_feats=[
                    "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                    "dlwrf", "dswrf", "apcp"
                    ],
            pred_feats=[
                    "soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd"
                    ],
            static_feats=[
                    "pct_sand", "pct_silt", "pct_clay", "elev", "elev_std"
                    ],
            static_int_feats=["int_veg"],
            total_static_int_input_size=14,
            )

    '''
    """ Sampling sanity check """
    for (tw,th,ts,tsi),tp in data_t.batch(64):
        break
