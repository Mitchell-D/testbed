"""
basic setup for sampling from sequence hdf5s using generators.sequence_dataset
"""
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

import model_methods as mm
from list_feats import nldas_record_mapping,noahlsm_record_mapping
from generators import sequence_dataset

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
            region_strs=("se", "sc", "sw", "ne", "nc", "nw"),
            season_strs=("warm", "cold"),
            time_strs=("2012-2015", "2015-2018", "2018-2021", "2021-2024"),
            )
    from list_feats import dynamic_coeffs,static_coeffs,derived_feats

    #ppt(seq_h5s)
    gen = sequence_dataset(
            sequence_hdf5s=seq_h5s,

            num_procs=6,
            frequency=1,
            sample_on_frequency=True,
            deterministic=False,
            buf_size_mb=1024,
            block_size=4,
            yield_times=True,
            derived_feats=derived_feats,
            #seed=1,
            static_conditions=[
                #(("pct_sand",), "lambda s:s[0]>.55"),
                (("pct_clay",), "lambda s:s[0]>.4"),
                #(("pct_silt",), "lambda s:s[0]>.5"),
                ],
            #dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            #static_norm_coeffs=dict(static_coeffs),

            window_feats=[
                    "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                    "dlwrf", "dswrf", "apcp",
                    #"soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd"
                    "rsm-10", "rsm-40", "rsm-100", "weasd"
                    ],
            horizon_feats=[
                    "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
                    "dlwrf", "dswrf", "apcp",
                    "weasd",
                    ],
            pred_feats=[
                    #"soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd"
                    "rsm-10", "rsm-40", "rsm-100", "rsm-200", "rsm-fc",
                    ],
            static_feats=[
                    "pct_sand", "pct_silt", "pct_clay", "elev", "elev_std"
                    ],
            static_int_feats=["int_veg"],
            total_static_int_input_size=14,
            debug=True,
            )

    sample_batches = 4096
    all_y = []
    all_h = []
    all_s = []
    for (w,h,s,si,t),ys in gen.batch(16):
        if sample_batches == 0:
            break
        sample_batches -= 1
        all_y.append(ys)
        all_h.append(h)
        all_s.append(s)

    all_y = np.concatenate(all_y, axis=0)
    num_samples,num_sequence,num_feats = all_y.shape
    print(f"{num_samples=} {num_sequence=}, {num_feats=}")

    print()
    print(f"pred state: ")
    print(np.average(all_y, axis=(0,1)))
    print(np.std(all_y, axis=(0,1)))
    res_y = np.diff(all_y, axis=1)
    print(f"pred residual")
    print(np.average(res_y, axis=(0,1)))
    print(np.std(res_y, axis=(0,1)))

    all_h = np.concatenate(all_h, axis=0)
    print()
    print(f"horizon state: ")
    print(np.average(all_h, axis=(0,1)))
    print(np.std(all_h, axis=(0,1)))
    res_h = np.diff(all_h, axis=1)
    print(f"horizon residual")
    print(np.average(res_h, axis=(0,1)))
    print(np.std(res_h, axis=(0,1)))

    all_s = np.concatenate(all_s, axis=0)
    print()
    print(f"static state: ")
    print(np.average(all_s, axis=(0)))
    print(np.std(all_s, axis=(0)))

