"""
use eval_grids.eval_model_on_grids to extract a series of subgrids of timegrid
files based on domains described by GridDomain objects from eval_grids.

This script extracts the entire prediction array, so be careful setting
extract_only_first_sequence to False.
"""
import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from time import perf_counter
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from pprint import pprint as ppt
from dataclasses import dataclass

from testbed import eval_grids
from testbed.list_feats import dynamic_coeffs,static_coeffs,derived_feats

proj_root = Path("/rhome/mdodson/testbed")
timegrid_dir = proj_root.joinpath("data/timegrids/")
model_parent_dir = proj_root.joinpath("data/models/new")
grid_pred_dir = proj_root.joinpath("data/pred_grids")
pkl_dir = proj_root.joinpath("data/eval_grid_ensembles")

"""
Generate subgrid sample h5s from the dictionary in subgrid_samples.py
"""
## keyword arguments to generators.gen_timegrid_subgrid
base_generator_args = {
        "buf_size_mb":4096,
        "load_full_grid":False,
        "include_init_state_in_predictors":True,
        "derived_feats":derived_feats,
        "seed":200007221750,
        "max_delta_hours":2,
        }
## keyword arguments to eval_models.grid_preds_to_hdf5
rsm_models = [
        "lstm-rsm-9_final.weights.h5",
        "accfnn-rsm-8_final.weights.h5",
        "acclstm-rsm-4_final.weights.h5",
        ]
soilm_models = [ "lstm-20_final.weights.h5", ]

domains_to_eval = [
        #"dakotas-flash-drought",
        #"gtlb-drought-fire",
        "sandhills",
        #"high-sierra",
        #"hurricane-laura",
        "hurricane-florence",
        #"eerie-mix",
        "kentucky-flood",
        ]
pred_feat_unit = "rsm"
eval_feat_unit = "rsm"

## If True, extracts only the first valid init time, which is useful since
## the data size will get enormous after very many. This is mainly
## important since this script shares config with the standard evaluation
extract_only_first_sequence = True

rsm_grid_eval_getter_args = [
        {
        "eval_types":["keep-all"],
        "eval_feat":"rsm-10",
        "pred_feat":f"{pred_feat_unit}-10",
        "coarse_reduce_func":"mean",
        "use_absolute_error":True,
        },
        {
        "eval_types":["keep-all"],
        "eval_feat":"rsm-10",
        "pred_feat":f"{pred_feat_unit}-10",
        "coarse_reduce_func":"mean",
        "use_absolute_error":False,
        },
        ]
soilm_grid_eval_getter_args = [{}]

""" ---------------------- end of configuration ---------------------- """

#'''
""" Extract the domains to a series of pkls """
## For each of the model/domain combos,
for model in {"soilm":soilm_models,"rsm":rsm_models}[pred_feat_unit]:
    ## Parse information about the model from the weights file naming
    mname,epoch = Path(Path(model).stem
            ).stem.split("_")[:2]
    model_dir_path = model_parent_dir.joinpath(mname)
    for dkey in domains_to_eval:
        cur_domain = next(gd for gd in eval_grids.domains if gd.name==dkey)
        ## silly way of making sure only one init time is evaluated
        if extract_only_first_sequence:
            cur_domain.frequency = 1e21
        out_pkls = eval_grids.eval_model_on_grids(
                pkl_dir=pkl_dir,
                grid_domain=cur_domain,
                model_dir_path=model_dir_path,
                timegrid_h5_dir=timegrid_dir,
                weights_file=model,
                m_valid=None,
                extract_valid_mask=True,
                eval_getter_args={
                    "soilm":soilm_grid_eval_getter_args,
                    "rsm":rsm_grid_eval_getter_args,
                    }[eval_feat_unit],
                grid_gen_args=base_generator_args,
                output_conversion={
                    "soilm":"rsm_to_soilm",
                    "rsm":"soilm_to_rsm",
                    }[eval_feat_unit],
                dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
                static_norm_coeffs=dict(static_coeffs),
                debug=True,
                )
#'''
