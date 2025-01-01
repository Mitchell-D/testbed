""" """
import gc
import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
from time import perf_counter
from datetime import datetime
from pathlib import Path
from pprint import pprint

import tracktrain as tt

import generators
import model_methods as mm
from eval_models import gen_gridded_predictions
from list_feats import dynamic_coeffs,static_coeffs
from list_feats import derived_feats,hist_bounds

## define a couple namedtuples to expediate configuration of domains that
## can span multiple timegrid regions.
GridDomain = namedtuple(
    "GridDomain",
    ["name", "tiles", "mosaic_shape", "start_time", "end_time", "frequency"],
    )
GridTile = namedtuple("GridTile", ["region", "px_bounds"])

## Configration for domain subsets
domains = [
        GridDomain(
            name="full",
            tiles=[
                GridTile(region="nw", px_bounds=(None, None, None, None)),
                GridTile(region="nc", px_bounds=(None, None, None, None)),
                GridTile(region="ne", px_bounds=(None, None, None, None)),
                GridTile(region="sw", px_bounds=(None, None, None, None)),
                GridTile(region="sc", px_bounds=(None, None, None, None)),
                GridTile(region="se", px_bounds=(None, None, None, None)),
                ],
            mosaic_shape=(2,3),
            start_time=datetime(2018, 1, 1, 0),
            end_time=datetime(2023, 12, 16, 23),
            frequency=24*7,
            ),
        GridDomain(
            name="kentucky-flood",
            tiles=[
                GridTile(region="nc", px_bounds=(80,98,144,154)),
                GridTile(region="ne", px_bounds=(80,98,0,50)),
                GridTile(region="sc", px_bounds=(0,15,144,154)),
                GridTile(region="se", px_bounds=(0,15,0,50)),
                ],
            mosaic_shape=(2,2),
            start_time=datetime(2022, 7, 22, 0),
            end_time=datetime(2022, 7, 29, 0),
            frequency=24,
            ),
        GridDomain(
            name="sandhills",
            tiles=[ GridTile(region="nc", px_bounds=(40,75,10,50)) ],
            mosaic_shape=(1,1),
            start_time=datetime(2018,1,1,0),
            end_time=datetime(2023,12,16,23),
            frequency=24*7,
            ),
        ]


def get_grid_evaluator_objects(eval_types:list, model_dir:tt.ModelDir,
        data_source:str, eval_feat:str, pred_fat:str, use_absolute_error:bool,
        hist_resolution=128, coarse_reduce_func="mean", debug=False):
    """
    Returns a list of pre-configured gridded Evaluator subclass objects
    identified by unique strings in the eval_types list.
    """
    pass

def eval_model_on_grids(pkl_dir:Path, grid_domain:GridDomain,
        model_dir_path:Path, weights_file:str, eval_getter_args:list,
        grid_gen_args:dict, output_conversion="soilm_to_rsm", m_valid=None,
        extract_valid_mask=False, dynamic_norm_coeffs={},
        static_norm_coeffs={}, debug=False):
    """
    High-level method that executes a model over a subset of a timegrid dataset
    using eval_models.gen_gridded_predictions, and runs a series of Evaluator
    subclass objects on the results batch-wise.

    :@param pkl_dir: Directory where Evaluator pkl files are generated
    :@param GridDomain: GridDomain namedtuple describing a subgrid and time
        range that may span multiple timegrid regions.
    :@param model_dir_path: Path to the ModelDir-created directory of the model
        to be evaluated
    :@param weights_file: File name (only) of the ".weights.hdf5 " model file
        to execute, which is anticipated to be stored in the above model dir.
    :@param eval_getter_args: a list of dictionary keyword arguments to
        get_grid_evaluator_objects excluding only the model_dir argument. Each
        entry may list multiple Evaluator objects to evaluate for a
        particular feature, absolute error/bias, reduction function, or
        histogram resolution
    :@param grid_gen_args: dict of arguments to gen_sequence_predictions
        specifying how to declare the data generator. Exclude the "*_feats"
        and "sequence_hdf5s" arguments which are provided based on the ModelDir
        configuration and argument to this method, respectively.
    :@param output_conversion: Specify which conversion function to run within
        the generator if the provided model produces the opposite unit type.
        Must be either "soilm_to_rsm" or "rsm_to_soilm".
    :@param extract_valid_mask: If True, a mask of valid pixels will be
        extracted from the timegrid per-region using the static feat "m_valid"
    """
    ## initialize the ModelDir instance associated with the requested weights
    md = tt.ModelDir(
            model_dir_path,
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                "acclstm":lambda args:mm.get_acclstm(**args),
                "accrnn":lambda args:mm.get_accrnn(**args),
                "accfnn":lambda args:mm.get_accfnn(**args),
                },
            )

    ## timegrid region file substrings, mapped to by their shorthand labels.
    eval_regions = {
            "nw":"y000-098_x000-154", "nc":"y000-098_x154-308",
            "ne":"y000-098_x308-462", "sw":"y098-195_x000-154",
            "sc":"y098-195_x154-308", "se":"y098-195_x308-462",
            }

    print(grid_domain)
    ## Evaluate over each tile in this domain, making each a generator that's
    ## specific to its timegrid source files and spatial range in the domain.
    for tile in grid_domain.tiles:
        ## Get a list of years forming a superset around requested times.
        eval_time_substrings = tuple(map(str, range(
            grid_domain.start_time.year-1, grid_domain.end_time.year+1
            )))
        print(tile)

        ## Develop a sorted list of timegrid files for this tile's region
        tmp_timegrid_h5s = list(sorted([
            tg for tg,tg_tup in map(
                lambda p:(p,p.stem.split("_")),
                timegrid_h5_dir.iterdir())
            if tg_tup[0] == "timegrid"
            and any(ss in tg_tup[1] for ss in eval_time_substrings)
            and tg_tup[2] in eval_regions[tile.region]
            and tg_tup[3] in eval_regions[tile.region]
            ]))

        ## Collect the generator params from ModelDir, GridDomain, gen_args
        tmp_gen_args = {
                **grid_gen_args,
                **md.config["feats"],
                "window_size":md.config["model"]["window_size"],
                "horizon_size":md.config["model"]["horizon_size"],
                "timegrid_paths":list(tmp_timegrid_h5s),
                "init_pivot_epoch":float(
                    grid_domain.start_time.strftime("%s")),
                "final_pivot_epoch":float(
                    grid_domain.end_time.strftime("%s")),
                "vidx_min":tile.px_bounds[0],
                "vidx_max":tile.px_bounds[1],
                "hidx_min":tile.px_bounds[2],
                "hidx_max":tile.px_bounds[3],
                "frequency":grid_domain.frequency,
                }

        ## Optionally extract the valid mask from the timegrid static data
        ext_mask = None
        if extract_valid_mask and ext_mask is None:
            if not m_valid is None:
                raise ValueError(
                        "Don't provide a valid mask if you want it to "
                        "be extracted from the timegrid")
            ## apply valid mask labeled m_valid if key found in timegrid attrs
            with h5py.File(name=tmp_timegrid_h5s[0], mode="r") as tmpf:
                _,_,tg_static_args = generators.parse_timegrid_attrs(
                        tmp_timegrid_h5s[0])
                mask_idx = tg_static_args["flabels"].index("m_valid")
                ext_mask = tmpf["/data/static"][...,mask_idx].astype(bool)

        ## initialize a grid prediction generator instance with the ModelDir
        ## and domain configuration
        gen = gen_gridded_predictions(
                model_dir=md,
                grid_generator_args=tmp_gen_args,
                weights_file_name=weights_file,
                m_valid=ext_mask,
                dynamic_norm_coeffs=dynamic_norm_coeffs,
                static_norm_coeffs=static_norm_coeffs,
                yield_normed_inputs=True,
                yield_normed_outputs=True,
                debug=debug,
                output_conversion=output_conversion,
                )

        ## Iterate over this tile's generator, running evaluators on each step
        for inputs,true_states,predicted_residuals,idxs in gen:
            print(f"{md.name} new batch; {true_states.shape = }")
            continue
        ## reset the extracted mask since the next tile's mask can be different
        ext_mask = None

        '''
        ## initialize some evaluator objects to run batch-wise on the generator
        evals = []
        for eargs in eval_getter_args:
            evals += get_grid_evaluator_objects(
                model_dir=md, **eargs, debug=debug)
        ## run each of the evaluators on every batch from the generator
        for inputs,true_states,predicted_residuals in gen:
            print(f"{md.name} new batch; {true_states.shape = }")
            for _,ev in evals
        pass
        '''
    return None

if __name__=="__main__":
    timegrid_h5_dir = Path("data/timegrids/")
    model_parent_dir = Path("data/models/new")
    pkl_dir = Path("data/performance/grid-eval")

    ## only models that predict rsm at 3 depth levels (tf 2.14)
    rsm_models = [
        ## Fully-connected models (feedforward only)
        "accfnn-rsm-0_final.weights.h5", "accfnn-rsm-1_final.weights.h5",
        "accfnn-rsm-2_final.weights.h5", "accfnn-rsm-3_final.weights.h5",
        "accfnn-rsm-4_final.weights.h5", "accfnn-rsm-5_final.weights.h5",
        "accfnn-rsm-6_final.weights.h5", "accfnn-rsm-7_final.weights.h5",
        "accfnn-rsm-8_final.weights.h5", "accfnn-rsm-9_final.weights.h5",

        ## State-accumulating LSTMs
        "acclstm-rsm-0_final.weights.h5",
        "acclstm-rsm-1_056_0.003.weights.h5",
        "acclstm-rsm-2_final.weights.h5", "acclstm-rsm-3_final.weights.h5",
        "acclstm-rsm-4_final.weights.h5", "acclstm-rsm-5_final.weights.h5",
        "acclstm-rsm-6_final.weights.h5", "acclstm-rsm-7_final.weights.h5",
        "acclstm-rsm-8_final.weights.h5", "acclstm-rsm-9_final.weights.h5",
        "acclstm-rsm-10_final.weights.h5", "acclstm-rsm-11_final.weights.h5",
        "acclstm-rsm-12_final.weights.h5",

        ## Accumulating RNNs
        "accrnn-rsm-0_final.weights.h5", "accrnn-rsm-1_final.weights.h5",
        "accrnn-rsm-2_final.weights.h5", "accrnn-rsm-3_final.weights.h5",
        "accrnn-rsm-4_final.weights.h5", "accrnn-rsm-5_final.weights.h5",
        "accrnn-rsm-6_final.weights.h5",

        ## Basic LSTMs
        "lstm-rsm-0_final.weights.h5", "lstm-rsm-2_final.weights.h5",
        "lstm-rsm-3_final.weights.h5", "lstm-rsm-5_final.weights.h5",
        "lstm-rsm-6_final.weights.h5", "lstm-rsm-7_021_0.015.weights.h5",
        "lstm-rsm-8_final.weights.h5", "lstm-rsm-9_final.weights.h5",
        "lstm-rsm-10_final.weights.h5", "lstm-rsm-11_final.weights.h5",
        "lstm-rsm-12_final.weights.h5", "lstm-rsm-19_final.weights.h5",
        "lstm-rsm-20_final.weights.h5",
        ]

    ## Basic LSTMs predicting 4-layer soilm + snow (tf 2.15)
    soilm_models = [
        "lstm-1_final.weights.h5", "lstm-2_final.weights.h5",
        "lstm-3_final.weights.h5", "lstm-4_final.weights.h5",
        "lstm-5_final.weights.h5", "lstm-6_final.weights.h5",
        "lstm-7_final.weights.h5", "lstm-8_final.weights.h5",
        "lstm-9_final.weights.h5", "lstm-10_final.weights.h5",
        "lstm-11_final.weights.h5", "lstm-12_final.weights.h5",
        "lstm-13_final.weights.h5", "lstm-14_final.weights.h5",
        "lstm-15_final.weights.h5", "lstm-16_final.weights.h5",
        "lstm-17_final.weights.h5", "lstm-18_final.weights.h5",
        "lstm-19_final.weights.h5", "lstm-20_final.weights.h5",
        "lstm-21_final.weights.h5", "lstm-22_final.weights.h5",
        "lstm-23_final.weights.h5", "lstm-24_final.weights.h5",
        "lstm-25_final.weights.h5", "lstm-26_final.weights.h5",
        "lstm-27_final.weights.h5",
        ]

    ## Model predicted unit. Used to identify feature indeces in truth/pred
    pred_feat_unit = "rsm"
    ## Output unit. Determines which set of evaluators are executed
    eval_feat_unit = "soilm"

    ## Subset of model weights to evaluate
    #weights_to_eval = soilm_models
    #weights_to_eval = [m for m in rsm_models if m[:10]=="lstm-rsm-9"]
    weights_to_eval = [m for m in rsm_models if m[:12]=="accfnn-rsm-8"]
    #weights_to_eval = [m for m in rsm_models if m[:12]=="accrnn-rsm-2"]
    #weights_to_eval = [m for m in rsm_models if m[:12]=="accfnn-rsm-5"]
    #weights_to_eval = [m for m in rsm_models if m[:13]=="acclstm-rsm-4"]
    #weights_to_eval = [m for m in soilm_models if m[:7]=="lstm-20"]

    ## Keywords for subgrid domains to evaluate per configuration dict above
    domains_to_eval = [
            "full",
            "kentucky-flood",
            "sandhills",
            ]

    ## generators.gen_timegrid_subgrids arguments for domains to evaluate.
    ## feat_*, pred_coarseness, timegrid_paths, spatial bounds, temporal
    ## range, and frequency arguments are provided by the eval_model_on_grids
    ## method based on model and domain configuration.
    grid_gen_args = {
            "derived_feats":derived_feats,
            "buf_size_mb":128,
            "load_full_grid":False,
            "max_delta_hours":2,
            "include_init_state_in_predictors":True,
            "seed":200007221752,
            }

    rsm_grid_eval_getter_args = [{}]
    soilm_grid_eval_getter_args = [{}]

    for w in weights_to_eval:
        ## parse the unique model name from its weights path
        mname,epoch = Path(Path(w).stem).stem.split("_")[:2]
        model_dir_path = model_parent_dir.joinpath(mname)
        for dstr in domains_to_eval:
            out_pkls = eval_model_on_grids(
                    pkl_dir=pkl_dir,
                    grid_domain=next(gd for gd in domains if gd.name==dstr),
                    model_dir_path=model_dir_path,
                    weights_file=w,
                    m_valid=None,
                    extract_valid_mask=True,
                    eval_getter_args={
                        "soilm":soilm_grid_eval_getter_args,
                        "rsm":rsm_grid_eval_getter_args,
                        }[eval_feat_unit],
                    grid_gen_args=grid_gen_args,
                    output_conversion={
                        "soilm":"rsm_to_soilm",
                        "rsm":"soilm_to_rsm",
                        }[eval_feat_unit],
                    dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
                    static_norm_coeffs=dict(static_coeffs),
                    )
            print(f"Generated evaluator pkls:")
            pprint(out_pkls)
            gc.collect()
