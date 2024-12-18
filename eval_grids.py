"""
:@module eval_grids: This module contains methods and generators for evaluating
    models using data from generators.gen_timegrid_subgrid, and calculating
    bulk statistics over the results with bulk_grid_error_stats_to_hdf5.
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

import model_methods as mm
import tracktrain as tt
import generators

def pearson_coeff(y, p, axis=1, keepdims=True):
    """ Calculate the pearson coefficient of sequences along an axis """
    y_mdiff = y - np.average(y, axis=axis, keepdims=keepdims)
    p_mdiff = p - np.average(p, axis=axis, keepdims=keepdims)
    num = np.sum( (y-y_mdiff)*(p-p_mdiff), axis=axis, keepdims=keepdims)
    y_denom = np.sum(y_mdiff**2, axis=axis, keepdims=keepdims)**(1/2)
    p_denom = np.sum(p_mdiff**2, axis=axis, keepdims=keepdims)**(1/2)
    return num / (y_denom * p_denom)

def gen_gridded_predictions(model_dir:tt.ModelDir, grid_generator_args:dict,
        weights_file_name=None, m_valid=None, dynamic_norm_coeffs:dict={},
        static_norm_coeffs:dict={}, yield_normed_inputs:bool=False,
        yield_normed_outputs:bool=False, debug=False):
    """
    Generates model inputs, true outputs, and model predictions as a 3-tuple.

    Executes a trained model (given its ModelDir object) over the data
    returned by a generators.gen_timegrid_subgrids initialized with
    grid_generator_args.

    :@param model_dir: ModelDir object associated with the model to run
    :@param grid_generator_args: JSON-serializable dict of arguments sufficient
        to initialize a generators.gen_timegrid_subgrids as a dataset.
    :@param weights_file_name: String name of the weights file within the
        provided ModelDir to use for inference. Defualts to _final.weights.h5
    :@param dynamic_norm_coeffs: Dict containing feature names mapped to
        (mean,stdev) dynamic normalization coefficients.
    :@param static_norm_coeffs: Dict containing feature names mapped to
        (mean,stdev) static normalization coefficients.
    """
    model = model_dir.load_weights(weights_path=weights_file_name)
    ## extract the coarseness in order to sub-sample the labels
    coarseness = model_dir.config.get("feats").get("pred_coarseness", 1)

    ## declare a grid generator for this region/model combination
    gen_tg = generators.gen_timegrid_subgrids(**grid_generator_args)

    ## collect normalization coefficients
    w_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in model_dir.config["feats"]["window_feats"]
        ])[np.newaxis,:]
    h_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in model_dir.config["feats"]["horizon_feats"]
        ])[np.newaxis,:]
    s_norm = np.array([
        tuple(static_norm_coeffs[k])
        if k in static_norm_coeffs.keys() else (0,1)
        for k in model_dir.config["feats"]["static_feats"]
        ])[np.newaxis,:]
    p_norm = np.array([
        tuple(dynamic_norm_coeffs[k])
        if k in dynamic_norm_coeffs.keys() else (0,1)
        for k in model_dir.config["feats"]["pred_feats"]
        ])[np.newaxis,:]

    F = None
    ix = None
    h5_idx = 0
    for (w,h,s,si,t),y in gen_tg:
        ## Extract the boolean mask for valid pixels
        if m_valid is None:
            m_valid = np.full(s.shape[:-1], True)
        if ix is None:
            hslice = slice(
                    grid_generator_args.get("hidx_min"),
                    grid_generator_args.get("hidx_max"))
            vslice = slice(
                    grid_generator_args.get("vidx_min"),
                    grid_generator_args.get("vidx_max"))
            m_valid = m_valid[vslice,hslice]
            ix = np.stack(np.where(m_valid), axis=-1)

        ## Apply the mask and organize the batch axis across pixels
        y = y[:,m_valid].transpose((1,0,2))
        w = (w[:,m_valid].transpose((1,0,2))-w_norm[...,0])/w_norm[...,1]
        h = (h[:,m_valid].transpose((1,0,2))-h_norm[...,0])/h_norm[...,1]
        s = (s[m_valid]-s_norm[...,0])/s_norm[...,1]
        si = si[m_valid]

        ## evaluate the model on the inputs and rescale the results
        if debug:
            clock_0 = perf_counter()
            p = model((w,h,s,si))
            clock_f = perf_counter()
            tmp_time = datetime.fromtimestamp(int(t[w.shape[1]]))
            tmp_time = tmp_time.strftime("%Y%m%d %H%M")
            tmp_dt = f"{clock_f-clock_0:.3f}"
            print(f"{tmp_time} evaluated {p.shape[0]} px in {tmp_dt} sec")
        else:
            p = model((w,h,s,si))

        ## invert the residual's normalization scale if requested
        if yield_normed_outputs:
            y = (y-p_norm[...,0])/p_norm[...,1]
        else:
            p = p * p_norm[...,1]
        if not yield_normed_inputs:
            w = w * w_norm[...,1] + w_norm[...,0]
            h = h * h_norm[...,1] + h_norm[...,0]
            s = s * s_norm[...,1] + s_norm[...,0]

        ## subsample y and t to the output coarseness of this model,
        ## and restrict t to only span the output sequence range
        y = y[:,::coarseness,:]
        t = t[-y.shape[1]:][::coarseness]

        ## yield 3-tuple of  inputs, true values, model predictions, indeces
        yield (w,h,s,si,t),y,p,ix


def grid_preds_to_hdf5(model_dir:tt.ModelDir, grid_generator_args:dict,
        pred_h5_path:Path, weights_file_name:str=None, pixel_chunk_size=64,
        sample_chunk_size=32, dynamic_norm_coeffs={}, static_norm_coeffs={},
        yield_normed_inputs=False, yield_normed_outputs=False,
        save_window=False, save_horizon=False, save_static=False,
        save_static_int=False, extract_valid_mask:bool=False, debug=False):
    """
    Evaluate a trained model on spatial grids on data from a
    generators.gen_timegrid_subgrid, and save the predictions, true values,
    timestamps, and spatial grid indeces separately over multiple init times.

    Predictions and true values are stored as (T,P,S,F) shaped arrays such that
    T : Timestep , P : Pixel (valid only) , S : Sequence step , F : pred feat

    /data/preds     : (N, P, S_p, F_p)  Model residual outputs
    /data/truth     : (N, P, S_y, F_p)  True state outputs
    /data/time      : (N, S_p)          Integer epoch times
    /data/idxs      : (P, 2)            Integer indeces
    /data/window    : (N, P, S_w, F_w)  Window inputs
    /data/horizon   : (N, P, S_p, F_h)  Horizon inputs
    /data/static    : (P, F_s)          Static inputs
    /data/static_int: (P, F_si)         Static integer inputs

    :@param model_dir: ModelDir object associated with the model to run
    :@param grid_generator_args: JSON-serializable dict of arguments sufficient
        to initialize a generators.gen_timegrid_subgrids as a dataset.
    :@param pred_h5_path: Path to a non-existent hdf5 file that will be written
        with the results from each sample timestep of this model.
    :@param weights_file_name: String name of the weights file within the
        provided ModelDir to use for inference. Defualts to _final.weights.h5
    :@param pixel_chunk_size: Number of elements per chunk along the pixel axis
    :@param sample_chunk_size: Number of samples per chunk along the first axis
    :@param dynamic_norm_coeffs: Dict containing prediction coefficient names
        mapped to (mean,stdev) normalization coefficients
    :@param save_*: If True, save the corresponding dataset in the hdf5
    :@param extract_valid_mask: If True, establishes the valid mask for the
        whole timegrid series by extracting the static parameter labeled
        "m_valid" from the first provided file.
    """
    ## apply valid mask labeled m_valid if key found in timegrid attributes
    tmpf = h5py.File(name=grid_generator_args["timegrid_paths"][0], mode="r")
    if extract_valid_mask:
        _,_,tg_static_args = generators.parse_timegrid_attrs(
                grid_generator_args["timegrid_paths"][0])
        mask_idx = tg_static_args["flabels"].index("m_valid")
        m_valid = tmpf["/data/static"][...,mask_idx].astype(bool)
    else:
        m_valid = np.full(tmpf["/data/static"][...,0].shape, True)


    gen = gen_gridded_predictions(
            model_dir=model_dir,
            grid_generator_args=grid_generator_args,
            weights_file_name=weights_file_name,
            dynamic_norm_coeffs=dynamic_norm_coeffs,
            static_norm_coeffs=static_norm_coeffs,
            yield_normed_inputs=yield_normed_inputs,
            yield_normed_outputs=yield_normed_outputs,
            m_valid=m_valid,
            debug=debug,
            )

    F = None
    h5_idx = 0
    for (w,h,s,si,t),y,p,ix in gen:
        ## initialize the file if it hasn't already been created
        if F is None:
            ## Create a new h5 file with datasets for the model (residual)
            ## predictions, true (state) values, and timesteps
            #valid_idxs = np.stack(np.where(m_valid), axis=-1)
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
            if save_window:
                W = F.create_dataset(
                        name="/data/window",
                        shape=(0, *w.shape),
                        maxshape=(None, *w.shape),
                        chunks=(*chunks, *w.shape[1:]),
                        compression="gzip",
                        )
            if save_horizon:
                H = F.create_dataset(
                        name="/data/horizon",
                        shape=(0, *h.shape),
                        maxshape=(None, *h.shape),
                        chunks=(*chunks, *h.shape[1:]),
                        compression="gzip",
                        )
            if save_static:
                S = F.create_dataset(
                        name="/data/static",
                        shape=s.shape,
                        maxshape=s.shape,
                        )
                S[...] = s
            if save_static_int:
                SI = F.create_dataset(
                        name="/data/static_int",
                        shape=si.shape,
                        maxshape=si.shape,
                        )
                SI[...] = si

            ## (P, 2) Valid pixel indeces
            IDX = F.create_dataset(
                    name="/data/idxs",
                    shape=ix.shape,
                    maxshape=ix.shape,
                    compression="gzip",
                    )
            ## Go ahead and load the indeces
            IDX[...] = ix
            ## Store the generator arguments so the same can be re-initialized
            grid_generator_args["timegrid_paths"] = [
                    Path(p).as_posix()
                    for p in grid_generator_args["timegrid_paths"]
                    ]
            F["data"].attrs["gen_args"] = json.dumps(grid_generator_args)
            F["data"].attrs["grid_shape"] = np.array(m_valid.shape)
            F["data"].attrs["model_config"] = json.dumps(model_dir.config)

        ## Incrementally expand the file and load data per timestep sample
        P.resize((h5_idx+1, *p.shape))
        Y.resize((h5_idx+1, *y.shape))
        T.resize((h5_idx+1, *t.shape))
        P[h5_idx,...] = p
        Y[h5_idx,...] = y
        T[h5_idx,...] = t

        ## Rescale and save additional datasets that were requested for the h5
        if save_window:
            W.resize((h5_idx+1, *w.shape))
            W[h5_idx,...] = w
        if save_horizon:
            H.resize((h5_idx+1, *h.shape))
            H[h5_idx,...] = h
        h5_idx += 1
    F.close()
    return pred_h5_path

def mp_grid_preds_to_hdf5(kwargs):
    """
    Helper method for multiprocessing over grid_preds_to_hdf5.

    ModelDir objects are initialized here because the custom_model_builders
    methods cannot be serialized for multiprocessing. Thus, kwargs['model_dir']
    is expected to be the string path of the model directory rather than the
    corresponding ModelDir object, unlike grid_preds_to_hdf5.

    custom_model_builders here must be kept up to date.

    :@param kwargs: dictionary of keyword arguments to
        eval_grids.grid_preds_to_hdf5. These are identical to the arguments
        passed to the method, EXCEPT model_dir must be the Path representing
        the ModelDir object.
    """
    kwargs["model_dir"] = tt.ModelDir(
            kwargs["model_dir"],
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                "acclstm":lambda args:mm.get_acclstm(**args),
                },
            )
    return grid_preds_to_hdf5(**kwargs)


def gen_grid_prediction_combos(grid_h5:Path):
    """
    Simple generator returning gridded sequence predictions one-by-one from a
    hdf5 file populated by grid_preds_to_hdf5.

    Sequences are yielded per timestep as 4-tuples (true, pred, idxs, time) st:
    true: (P, S, F_p) P valid pixels having S sequence members and F_p feats
    pred: (P, S, F_p) Predictions associted directly with the true sequences
    idxs: (P, 2)      Indeces of each of the P valid pixels on a larger grid
    time: float       Epoch time of the currently yielded timestep's pivot time
                      (the first predicted timestep)
    """
    with h5py.File(grid_h5, mode="r") as grid_file:
        P = grid_file["/data/preds"]
        Y = grid_file["/data/truth"]
        T = grid_file["/data/time"][:]
        IDX = grid_file["/data/idxs"][:]
        assert T.shape[0] == Y.shape[0]
        assert T.shape[0] == P.shape[0]
        assert IDX.shape[0] == Y.shape[1]
        assert IDX.shape[0] == P.shape[1]
        for i in range(P.shape[0]):
            yield (Y[i,...], P[i,...], IDX, T[i,1])

def parse_grid_params(grid_h5:Path):
    """
    Simple method to extract the parameter dict from a grid h5.

    Returns the original grid shape and the generators.gen_timegrid_subgrids
    arguments used to initialize the generator for the file as a 2-tuple:
    (grid_shape, gen_args)
    """
    with h5py.File(grid_h5, "r") as tmpf:
        gen_args = json.loads(tmpf["data"].attrs["gen_args"])
        grid_shape = tmpf["data"].attrs["grid_shape"]
        model_config = json.loads(tmpf["data"].attrs["model_config"])
        #grid_shape = tuple(int(v) for v in grid_shape if v.isnumeric())
    return (grid_shape, model_config, gen_args)

def bulk_grid_error_stats_to_hdf5(grid_h5:Path, stats_h5:Path,
        timesteps_chunk:int=32, debug=False):
    """
    Make a new hdf5 file given a gridded predictions hdf5 file, which contains
    error magnitude and bias statistics, collected across each pixel sequence

    Predictions and true values are stored as (T,P,Q,F) shaped arrays such that
    T : Timestep , P : Pixel (valid only) , Q : Stat quantity ,  F : pred feat

    Where the 'Q' axis has size 7 representing residual & state absolute error:
    (state_max, state_mean, state_stdev, state_final,
     res_max, res_mean, res_stdev)
    """
    gen = gen_grid_prediction_combos(grid_h5)
    grid_shape,model_config,gen_args = parse_grid_params(grid_h5)
    coarseness = gen_args.get("pred_coarseness", 1)
    F = None
    h5idx = 0
    for (ys,pr,ix,t) in gen:
        if debug:
            t = datetime.fromtimestamp(int(t))
            print(f"Loading timestep {t.strftime('%Y%m%d %H%M')}")
        if F is None:
            err_shape = (pr.shape[0], 7, pr.shape[-1])
            F = h5py.File(
                    name=stats_h5,
                    mode="w-",
                    ## use a 256MB cache (shouldn't matter)
                    rdcc_nbytes=256*1024**2,
                    )
            ## (N, P, Q, F) Predicted values
            S = F.create_dataset(
                    name="/data/stats",
                    shape=(0, *err_shape),
                    maxshape=(None, *err_shape),
                    chunks=(timesteps_chunk, *err_shape),
                    compression="gzip",
                    )
            ## (N,) Epoch times
            T = F.create_dataset(
                    name="/data/time",
                    shape=(0,),
                    maxshape=(None,),
                    compression="gzip",
                    )
            ## (P, 2) Valid pixel indeces
            IDX = F.create_dataset(
                    name="/data/idxs",
                    shape=ix.shape,
                    maxshape=ix.shape,
                    compression="gzip",
                    )
            IDX[...] = ix

            ## Load grid generator, original grid shape, and statistic labels
            ## as hdf5 attributes
            F["data"].attrs["gen_args"] = json.dumps(gen_args)
            F["data"].attrs["grid_shape"] = np.array(grid_shape)
            F["data"].attrs["model_config"] = json.dumps(model_config)
            F["data"].attrs["stat_labels"] = [
                    "state_error_max",
                    "state_error_mean",
                    "state_error_stdev",
                    "state_kge",
                    "state_pearson",
                    "state_bias_final",
                    "res_error_max",
                    "res_error_mean",
                    "res_error_stdev",
                    "res_kge",
                    "res_pearson",
                    ]

        ## subsample labels to the model's coarseness
        ys = ys[:,::coarseness,:]
        ## Predicted state
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## True residual
        yr = ys[:,1:] - ys[:,:-1]
        ## Bias in state
        bs = ps - ys[:,1:,:]
        ## Error in state
        es = np.abs(bs)
        ## Error in residual
        er = np.abs(pr - yr)

        ## Calculate state and residual kling-gupta efficiency
        pc_s = mm.pearson_coeff(ys[:,1:], ps)
        alpha_s = np.std(ps, axis=1, keepdims=True) \
                / np.std(ys[:,1:], axis=1, keepdims=True)
        beta_s = np.average(ps, axis=1, keepdims=True) \
                / np.average(ys[:,1:], axis=1, keepdims=True)
        kge_s = 1 - ((pc_s-1)**2 + (alpha_s-1)**2 + (beta_s-1)**2)**(1/2)
        pc_r = mm.pearson_coeff(yr, pr)
        alpha_r = np.std(pr, axis=1, keepdims=True) \
                / np.std(yr, axis=1, keepdims=True)
        beta_r = np.average(pr, axis=1, keepdims=True) \
                / np.average(yr, axis=1, keepdims=True)
        kge_r = 1 - ((pc_r-1)**2 + (alpha_r-1)**2 + (beta_r-1)**2)**(1/2)

        ## Stack to (P, Q, F) array
        stats = np.stack([
            np.amax(es, axis=1),
            np.average(es, axis=1),
            np.std(es, axis=1),
            np.squeeze(kge_s, axis=1),
            np.squeeze(pc_s, axis=1),
            bs[:,-1,:],
            np.amax(er, axis=1),
            np.average(er, axis=1),
            np.std(er, axis=1),
            np.squeeze(kge_r, axis=1),
            np.squeeze(pc_r, axis=1),
            ], axis=1)

        S.resize((h5idx+1, *err_shape))
        T.resize((h5idx+1,))
        S[h5idx,...] = stats
        T[h5idx] = t.strftime("%s")
        h5idx += 1
    return

def parse_bulk_grid_params(bulk_grid_path:Path):
    """
    Simple method to extract the parameter dict from a grid h5.

    Returns the original grid shape and the generators.gen_timegrid_subgrids
    arguments used to initialize the generator for the file as a 3-tuple:
    (grid_shape, gen_args, stat_labels)
    """
    with h5py.File(bulk_grid_path, "r") as tmpf:
        gen_args = json.loads(tmpf["data"].attrs["gen_args"])
        grid_shape = tmpf["data"].attrs["grid_shape"]
        #grid_shape = tuple(int(v) for v in grid_shape if v.isnumeric())
        stat_labels = tuple(tmpf["data"].attrs["stat_labels"])
        model_config = json.loads(tmpf["data"].attrs["model_config"])
    return (grid_shape, model_config, gen_args, stat_labels)


def gen_bulk_grid_stats(bulk_grid_path:Path, init_time=None, final_time=None,
        buf_size_mb=128):
    """
    Yields a bulk statistic grid's data by timestep as a 3-tuple like
    (stats, idxs, time) such that:

    stats := (P,7,F) 7 stats for the P valid pixels' F features
    idxs := (P,2) 2d integer indeces for each of the valid pixels
    time := float epoch time for the current sample's pivot

    :@param bulk_grid_path: Path to a hdf5 from bulk_grid_error_stats_to_hdf5
    :@param init_time: datetime of initial pivot to include in yielded results
    :@param final_time: datetime of last pivot to include in yielded results
    :@param buf_size_mb: Size of hdf5 chunk read buffer in MB
    """
    with h5py.File(
            bulk_grid_path,
            mode="r",
            rdcc_nbytes=buf_size_mb*1024**2,
            rdcc_nslots=buf_size_mb*16,
            ) as grid_file:
        S = grid_file["/data/stats"]
        T = grid_file["/data/time"][...]
        IDX = grid_file["/data/idxs"][...]

        ## restrict timestep indeces by applying optional bounds
        if not init_time is None:
            m_init = (T >= int(init_time.strftime("%s")))
        else:
            m_init = np.full(T.shape, True)
        if not final_time is None:
            m_final = (T < int(final_time.strftime("%s")))
        else:
            m_final = np.full(T.shape, True)

        ## yield timesteps in order, one at a time
        time_idxs = np.where(np.logical_and(m_init,m_final))
        for tidx in time_idxs[0]:
            yield (S[tidx,...], IDX, T[tidx])

if __name__=="__main__":
    from list_feats import dynamic_coeffs,static_coeffs,derived_feats
    timegrid_dir = Path("data/timegrids/")
    model_parent_dir = Path("data/models/new")
    grid_pred_dir = Path("data/pred_grids")
    bulk_grid_dir = Path("data/pred_grids/")

    ## Create a grid hdf5 file using generators.gen_timegrid_subgrids
    #'''
    eval_regions = (
            ("y000-098_x000-154", "nw"),
            ("y000-098_x154-308", "nc"),
            ("y000-098_x308-462", "ne"),
            ("y098-195_x000-154", "sw"),
            ("y098-195_x154-308", "sc"),
            ("y098-195_x308-462", "se"),
            )
    eval_time_substrings = tuple(map(str,range(2017,2020)))

    #start_datetime = datetime(2018,5,1)
    #end_datetime = datetime(2018,11,1)
    start_datetime = datetime(2018,1,1)
    end_datetime = datetime(2019,1,1)

    #weights_file = "lstm-23_217_0.569.weights.h5"
    #weights_file = "lstm-20_353_0.053.weights.h5"
    #weights_file = "lstm-rsm-9_231_0.003.weights.h5"
    weights_file = "acclstm-rsm-4_final.weights.h5"

    #model_name = "lstm-23"
    #model_label = f"{model_name}-217"

    model_name,model_epoch = weights_file.split("_")[:2]
    model_label = f"{model_name}-{model_epoch}"

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
                "acclstm":lambda args:mm.get_acclstm(**args),
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
            "static_feats":md.config["feats"]["static_feats"],# + ["m_valid"],
            "static_int_feats":[("int_veg",14)],
            "init_pivot_epoch":float(start_datetime.strftime("%s")),
            "final_pivot_epoch":float(end_datetime.strftime("%s")),
            "derived_feats":derived_feats,
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

    ## Not multiprocessing right now due to gpu memory limitations
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
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            extract_valid_mask=True,
            debug=True,
            )
    #'''

    ## Populate a new hdf5 with weekly error statistics on a valid pixel grid
    '''
    pred_h5s = [
            #Path("pred-grid_nw_20180101_20211216_lstm-20-353.h5"),
            #Path("pred-grid_nc_20180101_20211216_lstm-20-353.h5"),
            #Path("pred-grid_ne_20180101_20211216_lstm-20-353.h5"),
            #Path("pred-grid_sw_20180101_20211216_lstm-20-353.h5"),
            #Path("pred-grid_sc_20180101_20211216_lstm-20-353.h5"),
            #Path("pred-grid_se_20180101_20211216_lstm-20-353.h5"),
            #Path("pred-grid_nc_20180101_20211216_lstm-23-217.h5"),
            #Path("pred-grid_ne_20180101_20211216_lstm-23-217.h5"),
            #Path("pred-grid_nw_20180101_20211216_lstm-23-217.h5"),
            #Path("pred-grid_sc_20180101_20211216_lstm-23-217.h5"),
            #Path("pred-grid_se_20180101_20211216_lstm-23-217.h5"),
            #Path("pred-grid_sw_20180101_20211216_lstm-23-217.h5"),
            Path("pred-grid_nc_20180101_20181231_lstm-rsm-9-231.h5"),
            Path("pred-grid_ne_20180101_20181231_lstm-rsm-9-231.h5"),
            Path("pred-grid_nw_20180101_20181231_lstm-rsm-9-231.h5"),
            Path("pred-grid_sc_20180101_20181231_lstm-rsm-9-231.h5"),
            Path("pred-grid_se_20180101_20181231_lstm-rsm-9-231.h5"),
            Path("pred-grid_sw_20180101_20181231_lstm-rsm-9-231.h5"),
            ]
    for p in pred_h5s:
        ftype,region,t0,tf,model = p.stem.split("_")
        bulk_file = f"bulk-grid_{region}_{t0}_{tf}_{model}.h5"
        bulk_grid_error_stats_to_hdf5(
                grid_h5=bulk_grid_dir.joinpath(p),
                stats_h5=bulk_grid_dir.joinpath(bulk_file),
                debug=True,
                )
    '''
