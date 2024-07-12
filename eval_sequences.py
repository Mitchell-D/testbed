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
    seq_dir = data_dir.joinpath("sequences")
    for p in filter(lambda p:"sequence" in p.stem, seq_dir.iterdir()):
        sequence_info(p)
