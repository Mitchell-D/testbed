"""
use eval_grids.grid_preds_to_hdf5 to extract a series of subgrids of timegrid
files, execute a model over them, and store the results as a new hdf5.
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

import model_methods as mm
import tracktrain as tt
from subgrid_samples import subgrid_samples_bad,subgrid_samples_good
import generators
import eval_grids

## Mapping between timegrid pixel range substrings and the abbreviated labels
timegrid_region_mapping = (
        ("y000-098_x000-154", "nw"),
        ("y000-098_x154-308", "nc"),
        ("y000-098_x308-462", "ne"),
        ("y098-195_x000-154", "sw"),
        ("y098-195_x154-308", "sc"),
        ("y098-195_x308-462", "se"),
        )

@dataclass
class subgrid_sample:
    """
    simple dataclass for storing subgrid spatial/temporal coordinate properties
    """
    init_time_str:str
    v_range:tuple
    h_range:tuple
    desc:str=""
    final_time_str:str=None
    @property
    def v_slice(self):
        return slice(*self.v_range)
    @property
    def h_slice(self):
        return slice(*self.h_range)
    @property
    def init_time(self):
        return datetime.strptime(self.init_time_str,"%Y%m%d-%H%M")
    @property
    def init_epoch(self):
        return int(self.init_time.strftime("%s"))
    @property
    def final_time(self):
        if self.final_time_str is None:
            return None
        return datetime.strptime(self.final_time_str,"%Y%m%d-%H%M")
    @property
    def final_epoch(self):
        return int(self.final_time.strftime("%s"))

if __name__=="__main__":
    timegrid_dir = Path("data/timegrids/")
    model_parent_dir = Path("data/models/new")
    grid_pred_dir = Path("data/pred_grids")
    bulk_grid_dir = Path("data/pred_grids/")
    #subgrid_dir = Path(f"data/subgrid_samples_bad")
    subgrid_dir = Path(f"data/subgrid_samples_good")

    from list_feats import dynamic_coeffs,static_coeffs,derived_feats
    #'''
    """
    Generate subgrid sample h5s from the dictionary in subgrid_samples.py
    """
    ## keyword arguments to generators.gen_timegrid_subgrid
    base_generator_args = {
            "static_int_feats":[("int_veg",14)],
            "buf_size_mb":4096,
            "load_full_grid":False,
            "include_init_state_in_predictors":True,
            "derived_feats":derived_feats,
            "seed":200007221750,
            }
    ## keyword arguments to eval_grids.grid_preds_to_hdf5
    base_hdf5_args = {
            "pixel_chunk_size":16,
            "sample_chunk_size":1,
            "dynamic_norm_coeffs":{k:v[2:] for k,v in dynamic_coeffs},
            "static_norm_coeffs":dict(static_coeffs),
            "yield_normed_inputs":False,
            "yield_normed_outputs":False,
            "save_window":True,
            "save_horizon":True,
            "save_static":True,
            "save_static_int":True,
            "extract_valid_mask":True,
            "debug":True,

            ## (!!!) Model configuration (!!!)
            "weights_file_name":"lstm-rsm-1_458_0.001.weights.h5",
            #"weights_file_name":"lstm-20_353_0.053.weights.h5",
            #"weights_file_name":"lstm-21_445_0.327.weights.h5",
            #"weights_file_name":"lstm-23_217_0.569.weights.h5"
            }

    ## Format model label (ie lstm-23-217) using the weights file name
    #regions_to_eval = ("nw", "nc", "ne")
    regions_to_eval = ("nc",)
    num_procs = 11

    ## Select the subgrid boundary dict to evaluate
    #cur_samples = subgrid_samples_bad
    cur_samples = subgrid_samples_good

    ## Get lists of timegrids per region, relying on the expected naming
    ## scheme timegrid_{YYYY}q{Q}_y{vmin}-{vmax}_x{hmin}-{hmax}.h5
    eval_time_substrings = tuple(map(str,range(2013,2022)))

    """ ---------------------- end of configuration ---------------------- """

    ## Parse information about the model from the weights file naming scheme
    mname,epoch = Path(base_hdf5_args["weights_file_name"]).stem.split("_")[:2]
    model_label = "-".join((mname,epoch))
    model_dir_path = model_parent_dir.joinpath(mname)

    ## Make a dict mapping region shorthand labels to corresponding timegrids
    timegrid_paths = {
            region_short:sorted([
                (tg,tuple(tg_tup)) for tg,tg_tup in map(
                    lambda p:(p,p.stem.split("_")),
                    timegrid_dir.iterdir())
                if tg_tup[0] == "timegrid"
                and any(ss in tg_tup[1] for ss in eval_time_substrings)
                and tg_tup[2] in region_str
                and tg_tup[3] in region_str
                ])
            for region_str,region_short in timegrid_region_mapping
            }

    ## arguments for gen_timegrid_subgrids that specify model size/features
    md = tt.ModelDir(model_dir_path)
    model_feature_args = {
            "window_size":md.config["model"]["window_size"],
            "horizon_size":md.config["model"]["horizon_size"],
            "window_feats":md.config["feats"]["window_feats"],
            "horizon_feats":md.config["feats"]["horizon_feats"],
            "pred_feats":md.config["feats"]["pred_feats"],
            ## append a valid mask feature to the static feats so that
            ## the grid can be unraveled and re-raveled
            "static_feats":md.config["feats"]["static_feats"],
            }

    ## make a list of arguments to gen_gridded_predictions specifying the
    ## dimensions of each unique sample, and carrying the arguments from
    ## base_generator_args and model_feature_args
    subgrid_gen_args = []
    for region in regions_to_eval:
        for ix,sample in enumerate(sorted(cur_samples[region])):
            ## create a subgrid sample object to store the attributes
            sg = subgrid_sample(*sample)
            ## determine the path to the newly created file
            pred_h5_path = subgrid_dir.joinpath(
                    f"subgrid-sample_{region}-{ix:03}_"
                    f"{model_label}_{sg.init_time_str}.h5"
                    )
            ## update the base arguments with the values specific to the sample
            tmp_gen_args = {
                **base_generator_args,
                **model_feature_args,
                "timegrid_paths":[
                    p.as_posix() for p,_ in timegrid_paths[region]],
                "init_pivot_epoch":sg.init_epoch,
                ## only want single pivot time so dt = 1 hour
                "final_pivot_epoch":sg.init_epoch + 60 * 60,
                ## 24 hour frequency, so only 1 grid sequence will be extracted
                "frequency":24,
                "vidx_min":sg.v_range[0],
                "vidx_max":sg.v_range[1],
                "hidx_min":sg.h_range[0],
                "hidx_max":sg.h_range[1],
                }
            subgrid_gen_args.append((pred_h5_path, tmp_gen_args))

    ## assemble a final list of keyword arguments to grid_preds_to_hdf5,
    ## each one corresponding to a model run over a test area.
    grid_preds_to_hdf5_args = [{
            "model_dir":model_dir_path,
            "grid_generator_args":gen_args,
            "pred_h5_path":h5_path,
            **base_hdf5_args,
            } for (h5_path,gen_args) in subgrid_gen_args]

    ## multiprocess predicting over and saving the sample subgrids to hdf5s
    with Pool(num_procs) as pool:
        results = pool.imap_unordered(
                eval_grids.mp_grid_preds_to_hdf5,
                grid_preds_to_hdf5_args
                )
        for p in results:
            print(f"Finished: {p.name}")
    #'''
