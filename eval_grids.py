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

import model_methods as mm
import tracktrain as tt
from list_feats import dynamic_coeffs,static_coeffs
import generators

def grid_preds_to_hdf5(model_dir:tt.ModelDir, grid_generator_args:dict,
        pred_h5_path:Path, weights_file_name:str=None, pixel_chunk_size=64,
        sample_chunk_size=32, pred_norm_coeffs={}, debug=False):
    """
    Evaluate a trained model on spatial grids on data from a
    generators.gen_timegrid_subgrid, and save the predictions, true values,
    timestamps, and spatial grid indeces separately over multiple init times.

    :@param model_dir: ModelDir object associated with the model to run
    :@param grid_generator_args: JSON-serializable dict of arguments sufficient
        to initialize a generators.gen_timegrid_subgrids as a dataset.
    :@param pred_h5_path: Path to a non-existent hdf5 file that will be written
        with the results from each sample timestep of this model.
    :@param weights_file_name: String name of the weights file within the
        provided ModelDir to use for inference. Defualts to _final.weights.h5
    :@param pixel_chunk_size: Number of elements per chunk along the pixel axis
    :@param sample_chunk_size: Number of samples per chunk along the first axis
    :@param pred_norm_coeffs: Dict containing prediction coefficient names
        mapped to (mean,stdev) normalization coefficients
    """
    model = model_dir.load_weights(weights_path=weights_file)
    ## declare a grid generator for this region/model combination
    gen_tg = generators.gen_timegrid_subgrids(**grid_generator_args)
    m_valid = None
    p_norm = np.array([
        tuple(pred_norm_coeffs[k])
        if k in pred_norm_coeffs.keys() else (0,1)
        for k in md.config["feats"]["pred_feats"]
        ])[np.newaxis,:]

    F = None
    h5_idx = 0
    for (w,h,s,si,t),y in gen_tg:
        #if debug:
        #    pidx = w.shape[0]
        #    print("Loading timestep", datetime.fromtimestamp(int(t[pidx])))
        ## Extract the boolean mask for valid pixels
        if m_valid is None:
            m_valid = s[...,-1].astype(bool)
        s = s[...,:-1]
        ## Apply the mask and organize the batch axis across pixels
        w = w[:,m_valid].transpose((1,0,2))
        h = h[:,m_valid].transpose((1,0,2))
        y = y[:,m_valid].transpose((1,0,2))
        s = s[m_valid]
        si = si[m_valid]
        ## evaluate the model on the inputs and rescale the results
        if debug:
            clock_0 = perf_counter()
        p = model((w,h,s,si)) * p_norm[...,1]
        if debug:
            clock_f = perf_counter()
            tmp_time = datetime.fromtimestamp(int(t[w.shape[1]]))
            tmp_time = tmp_time.strftime("%Y%m%d %H%M")
            tmp_dt = f"{clock_f-clock_0:.3f}"
            print(f"{tmp_time} evaluated {p.shape[0]} px in {tmp_dt} sec")
        ## rescale the truth values before storing them
        y = y * p_norm[...,1] + p_norm[...,0]

        ## initialize the file if it hasn't already been created
        if F is None:
            ## Create a new h5 file with datasets for the model (residual)
            ## predictions, true (state) values, and timesteps
            valid_idxs = np.stack(np.where(m_valid), axis=-1)
            F = h5py.File(
                    name=pred_h5_path,
                    mode="w-",
                    ## use a 256MB cache (shouldn't matter)
                    rdcc_nbytes=256*1024**2,
                    )
            ## (N, P, S_p, F) Predicted values
            chunks = (sample_chunk_size,pixel_chunk_size)
            P = F.create_dataset(
                    name="/data/preds",
                    shape=(0, *p.shape),
                    maxshape=(None, *p.shape),
                    chunks=(*chunks,*p.shape[1:]),
                    compression="gzip",
                    )
            ## (N, P, S_y, F) True values
            Y = F.create_dataset(
                    name="/data/truth",
                    shape=(0, *y.shape),
                    maxshape=(None, *y.shape),
                    chunks=(*chunks, *y.shape[1:]),
                    compression="gzip",
                    )
            ## (N, S_p) Epoch times
            T = F.create_dataset(
                    name="/data/time",
                    shape=(0, y.shape[1]),
                    maxshape=(None, y.shape[1]),
                    compression="gzip",
                    )
            ## (P, 2) Valid pixel indeces
            IDX = F.create_dataset(
                    name="/data/idxs",
                    shape=valid_idxs.shape,
                    maxshape=valid_idxs.shape,
                    compression="gzip",
                    )
            ## Go ahead and load the indeces
            IDX[...] = valid_idxs
            ## Store the generator arguments so the same can be re-initialized
            grid_generator_args["timegrid_paths"] = [
                    p.as_posix() for p in grid_generator_args["timegrid_paths"]
                    ]
            F["data"].attrs["gen_args"] = json.dumps(grid_generator_args)
            F["data"].attrs["grid_shape"] = str(m_valid.shape)

        ## Incrementally expand the file and load data per timestep sample
        P.resize((h5_idx+1, *p.shape))
        Y.resize((h5_idx+1, *y.shape))
        T.resize((h5_idx+1, y.shape[1]))
        P[h5_idx,...] = p
        Y[h5_idx,...] = y
        T[h5_idx,...] = t[-y.shape[1]:]
        h5_idx += 1
    F.close()
    return pred_h5_path

if __name__=="__main__":

    timegrid_dir = Path("data/timegrids/")
    model_parent_dir = Path("data/models/new")
    grid_pred_dir = Path("data/pred_grids")

    eval_regions = (
            ("y000-098_x000-154", "nw"),
            ("y000-098_x154-308", "nc"),
            ("y000-098_x308-462", "ne"),
            ("y098-195_x000-154", "sw"),
            ("y098-195_x154-308", "sc"),
            ("y098-195_x308-462", "se"),
            )
    eval_time_substrings = tuple(map(str,range(2013,2022)))

    #start_datetime = datetime(2018,5,1)
    #end_datetime = datetime(2018,11,1)
    start_datetime = datetime(2018,1,1)
    end_datetime = datetime(2021,12,16)

    model_name = "lstm-16"
    weights_file = "lstm-16_505_0.047.weights.h5"
    model_label = f"{model_name}-505"

    """
    Get lists of timegrids per region, relying on the expected naming
    scheme timegrid_{YYYY}q{Q}_y{vmin}-{vmax}_x{hmin}-{hmax}.h5
    """
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
            for region_str,region_short in eval_regions
            }

    """ Load a specific trained model's ModelDir for evaluation """
    md = tt.ModelDir(
            model_parent_dir.joinpath(model_name),
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                }
            )
    grid_generator_args = {
            "timegrid_paths":None,
            "window_size":md.config["model"]["window_size"],
            "horizon_size":md.config["model"]["horizon_size"],
            "window_feats":md.config["feats"]["window_feats"],
            "horizon_feats":md.config["feats"]["horizon_feats"],
            "pred_feats":md.config["feats"]["pred_feats"],
            ## append a valid mask feature to the static feats so that
            ## the grid can be unraveled and re-raveled
            "static_feats":md.config["feats"]["static_feats"] + ["m_valid"],
            "static_int_feats":[("int_veg",14)],
            "init_pivot_epoch":float(start_datetime.strftime("%s")),
            "final_pivot_epoch":float(end_datetime.strftime("%s")),
            "frequency":7*24,
            #"vidx_min":10,
            #"vidx_max":58,
            #"hidx_min":10,
            #"hidx_max":58,
            "buf_size_mb":4096,
            "load_full_grid":False,
            "include_init_state_in_predictors":True,
            "seed":200007221750,
            }

    for tmp_region,v in timegrid_paths.items():
        rpaths,rtups = zip(*v)
        grid_generator_args["timegrid_paths"] = rpaths
        t0 = start_datetime.strftime("%Y%m%d")
        tf = end_datetime.strftime("%Y%m%d")
        tmp_path = f"pred-grid_{tmp_region}_{t0}_{tf}_{model_label}.h5"
        grid_preds_to_hdf5(
            model_dir=md,
            grid_generator_args=grid_generator_args,
            pred_h5_path=grid_pred_dir.joinpath(tmp_path),
            weights_file_name=weights_file,
            pixel_chunk_size=64,
            sample_chunk_size=16,
            pred_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            debug=True,
            )
