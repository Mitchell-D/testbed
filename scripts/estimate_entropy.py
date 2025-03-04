""" """
import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from time import perf_counter
from datetime import datetime
import pickle as pkl
from pathlib import Path
from pprint import pprint as ppt

from testbed.list_feats import dynamic_coeffs,static_coeffs,derived_feats
from tracktrain import ModelDir,ModelSet
from testbed import evaluators

def calc_entropy(counts:np.array, log_base=None):
    """
    Discretely estimate entropy of an Nd array of counts using the method of:
    https://doi.org/10.1175/JHM-D-15-0063.1
    """
    nbins = counts.size
    idxs = np.reshape(np.indices(counts.shape), (len(counts.shape), nbins)).T
    if not log_base is None:
        entropy = counts/nbins * np.emath.logn(log_base, counts/nbins)
    else:
        entropy = counts/nbins * np.log(counts/nbins)
    return np.nansum(entropy)

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/testbed")
    performance_dir = proj_root.joinpath("data/eval_sequence_pkls")
    model_root_dir = proj_root.joinpath("data/models/new")

    model_dirs = [ModelDir(d) for d in model_root_dir.iterdir()]

    plot_data_sources = ["test"]
    plot_models_named = [
            md.name for md in model_dirs
            if md.config["model_type"]=="lstm-s2s"
            ]
    plot_eval_feats = ["rsm-10", "rsm-40", "rsm-100"]

    eval_pkls = [
            (p,pt) for p,pt in map(
                lambda f:(f,f.stem.split("_")),
                sorted(performance_dir.iterdir()))
            if pt[0] == "eval"
            and pt[1] in plot_data_sources
            and any(s==pt[2] for s in plot_models_named)
            and pt[3] in plot_eval_feats
            and pt[4] == "hist-true-pred"
            and pt[5] == "na"
            ]

    for p,pt in eval_pkls:
        ev = evaluators.EvalJointHist().from_pkl(p)
        counts = ev.get_results()["counts"]
        y_ent = calc_entropy(np.sum(counts, axis=0))
        p_ent = calc_entropy(np.sum(counts, axis=1))
        all_ent = calc_entropy(counts)
        mutual_info = y_ent + p_ent - all_ent
        print(pt[2], int(mutual_info), int(p_ent), mutual_info/y_ent)
