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

from testbed import generators
from testbed import model_methods as mm
from testbed.evaluators import EvalGridAxes,EvalTemporal,EvalHorizon
from testbed.evaluators import EvalStatic,EvalJointHist
from testbed.eval_models import gen_gridded_predictions
from testbed.list_feats import dynamic_coeffs,static_coeffs
from testbed.list_feats import derived_feats,hist_bounds

## define a couple namedtuples to expediate configuration of domains that
## can span multiple timegrid regions.
GridDomain = namedtuple(
    "GridDomain",
    ["name", "tiles", "mosaic_shape", "start_time", "end_time", "frequency"],
    )
GridTile = namedtuple("GridTile", ["region", "px_bounds"])

## Configration for domain subsets. The tiles should be ordered such that they
## index across rows first from east to west, then columns from north to south!
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
            frequency=6,
            #frequency=24*7,
            ),
        GridDomain(
            name="sandhills",
            tiles=[ GridTile(region="nc", px_bounds=(40,75,10,50)) ],
            mosaic_shape=(1,1),
            start_time=datetime(2018,1,1,0),
            end_time=datetime(2023,12,16,23),
            frequency=24*7,
            ),
        GridDomain(
            name="high-sierra",
            tiles=[ GridTile(region="nw", px_bounds=(70,90,20,40)) ],
            mosaic_shape=(1,1),
            start_time=datetime(2019,4,1,0),
            end_time=datetime(2019,6,1,0),
            frequency=24*7,
            ),
        GridDomain(
            name="hurricane-laura",
            tiles=[ GridTile(region="sc", px_bounds=(0,70,75,115)) ],
            mosaic_shape=(1,1),
            start_time=datetime(2020,8,20,0),
            end_time=datetime(2020,8,29,0),
            frequency=6,
            ),
        ]

def get_grid_evaluator_objects(eval_types:list, model_dir:tt.ModelDir,
        data_source:str, eval_feat:str, pred_feat:str, use_absolute_error:bool,
        hist_resolution=128, coarse_reduce_func="mean", grid_gen_args={},
        store_static=None, store_time=None, attrs={}, debug=False):
    """
    Returns a list of pre-configured gridded Evaluator subclass objects
    identified by unique strings in the eval_types list.
    """
    md = model_dir
    pred_feat_idx = md.config["feats"]["pred_feats"].index(pred_feat)
    apcp_idx = md.config["feats"]["horizon_feats"].index("apcp")
    temp_idx = md.config["feats"]["horizon_feats"].index("tmp")
    spfh_idx = md.config["feats"]["horizon_feats"].index("spfh")
    output_idxs = tuple(range(len(md.config["feats"]["pred_feats"])))

    ## list the evaluator labels for which it matters whether error bias vs
    ## absolute error value is distinguished in the output file name
    absolute_error_relevant = [
            "temporal", "static-combos", "hist-humidity-temp",
            "hist-state-increment", "spatial-stats",  "init-time-stats",
            ]
    ## Evaluator instances that consider all feats simultaneously, so the
    ## eval_feat field in the file name should be general (ie rsm not rsm-10)
    contains_all_feats = ["horizon", "temporal", "static-combos",
            "spatial-stats", "init-time-stats"]
    ## initialize some evaluator objects to run batch-wise on the generator
    grid_evals = {
            ## EvalGridAxes assumes (T, P, S, F) array structure
            ## Get mean and variance of critical forcings, outputs, and error
            ## stats, marginalizing over the sequence and init time axes.
            "spatial-stats":EvalGridAxes(
                feat_args=[
                    ("horizon", apcp_idx), ("horizon", temp_idx),
                    ("horizon", spfh_idx),
                    *[("true_res", ix) for ix in output_idxs],
                    *[("pred_res", ix) for ix in output_idxs],
                    *[("err_res", ix) for ix in output_idxs],
                    *[("err_state", ix) for ix in output_idxs],
                    ],
                axes=1,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                store_static=True if store_static is None else store_static,
                store_time=False if store_time is None else store_time,
                use_absolute_error=use_absolute_error,
                coarse_reduce_func="mean",
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    ## Store the feature labels identifying the datasets
                    "flabels":[
                        ("horizon","apcp"),
                        ("horizon","tmp"),
                        ("horizon","spfh"),
                        *[("true_res", md.config["feats"]["pred_feats"][ix])
                            for ix in output_idxs],
                        *[("pred_res", md.config["feats"]["pred_feats"][ix])
                            for ix in output_idxs],
                        *[("err_res", md.config["feats"]["pred_feats"][ix])
                            for ix in output_idxs],
                        *[("err_state", md.config["feats"]["pred_feats"][ix])
                            for ix in output_idxs],
                        ],
                    **attrs,
                    }
                ),
            "init-time-stats":EvalGridAxes(
                feat_args=[
                    ("horizon", apcp_idx), ("horizon", temp_idx),
                    ("horizon", spfh_idx),
                    *[("true_res", ix) for ix in output_idxs],
                    *[("pred_res", ix) for ix in output_idxs],
                    *[("err_res", ix) for ix in output_idxs],
                    *[("err_state", ix) for ix in output_idxs],
                    ],
                axes=0,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                store_static=True if store_static is None else store_static,
                store_time=True if store_time is None else store_time,
                use_absolute_error=use_absolute_error,
                coarse_reduce_func="mean",
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    "flabels":[
                        ("horizon", "apcp"),
                        ("horizon", "tmp"),
                        ("horizon", "spfh"),
                        *[("true_res", md.config["feats"]["pred_feats"][ix])
                            for ix in output_idxs],
                        *[("pred_res", md.config["feats"]["pred_feats"][ix])
                            for ix in output_idxs],
                        *[("err_res", md.config["feats"]["pred_feats"][ix])
                            for ix in output_idxs],
                        *[("err_state", md.config["feats"]["pred_feats"][ix])
                            for ix in output_idxs],
                        ],
                    **attrs,
                    }
                ),
            ## carry-overs from sequence evaluation. Treats pixels as batch
            ## so marginalizes over the initialization time axis
            f"horizon":EvalHorizon(
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    "plot_spec":{
                        "xlabel":"Forecast distance (hours)",
                        },
                    **attrs,
                    },
                ),
            f"temporal":EvalTemporal(
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    "plot_spec":{
                        },
                    **attrs,
                    },
                use_absolute_error=use_absolute_error,
                ),
            f"static-combos":EvalStatic(
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    **attrs,
                    },
                soil_idxs=[md.config["feats"]["static_feats"].index(l)
                    for l in ("pct_sand", "pct_silt", "pct_clay")],
                use_absolute_error=use_absolute_error,
                ),
            ## validation histogram
            f"hist-true-pred":EvalJointHist(
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    "plot_spec":{
                        "title":"Validation Histogram " + \
                                f"{eval_feat} ({md.name})",
                        "ylabel":"True Increment Change",
                        "xlabel":"Predicted Increment Change",
                        },
                    **attrs,
                    },
                ax1_args=(
                    ("true_res", pred_feat_idx),
                    (*hist_bounds[f"res-{eval_feat}"], hist_resolution),
                    ),
                ax2_args=(
                    ("pred_res", pred_feat_idx),
                    (*hist_bounds[f"res-{eval_feat}"], hist_resolution),
                    ),
                ),
            ## residual error wrt saturation level
            f"hist-saturation-error":EvalJointHist(
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    "plot_spec":{
                        "title":"Joint distribution of increment error in" + \
                                f" {eval_feat} wrt state",
                        "xlabel":"Hourly increment error in ({eval_feat})",
                        "ylabel":"True state magnitude for ({eval_feat})",
                        },
                    **attrs,
                    },
                ax1_args=(
                    ("true_state", pred_feat_idx),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                ax2_args=(
                    ("err_res", pred_feat_idx),
                    (*hist_bounds[f"err-res-{eval_feat}"], hist_resolution),
                    ),
                use_absolute_error=use_absolute_error,
                ),
            ## error rates wrt true state / true residual configuration
            "hist-state-increment":EvalJointHist(
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    "plot_spec":{
                        "title":"Joint distribution of true state and true" + \
                                "increment with MAE contours",
                        "ylabel":"True state ({eval_feat})",
                        "xlabel":"True increment change ({eval_feat}) ",
                        },
                    **attrs,
                    },
                ax1_args=(
                    ("true_state", pred_feat_idx),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                ax2_args=(
                    ("true_res", pred_feat_idx),
                    (*hist_bounds["res-"+eval_feat], hist_resolution),
                    ),
                ## Calculate the mean residual error per bin
                covariate_feature=("err_res", pred_feat_idx),
                use_absolute_error=use_absolute_error,
                ignore_nan=True,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                ),
            ## error rates wrt humidity/temperature residual configuration
            "hist-humidity-temp":EvalJointHist(
                attrs={
                    "model_config":md.config,
                    "gen_args":grid_gen_args,
                    "plot_spec":{
                        "title":"Joint distribution of humidity and temp" + \
                                "with MAE contours",
                        "ylabel":"Specific humidity (kg/kg)",
                        "xlabel":"Temperature (K)",
                        },
                    **attrs,
                    },
                ax1_args=(
                    ("horizon",
                        md.config["feats"]["horizon_feats"].index("spfh")),
                    (*hist_bounds["spfh"], hist_resolution),
                    ),
                ax2_args=(
                    ("horizon",
                        md.config["feats"]["horizon_feats"].index("tmp")),
                    (*hist_bounds["tmp"], hist_resolution),
                    ),
                ## Calculate the mean residual error per bin
                coarse_reduce_func="mean",
                covariate_feature=("err_res", pred_feat_idx),
                use_absolute_error=use_absolute_error,
                ignore_nan=True,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                ),
            }
    selected_evals = []
    for et in eval_types:
        assert et in grid_evals.keys(), f"{et} must be one of" + \
                f"\n{list(grid_evals.keys())}"
        tmp_name = list(map(str,
            ("eval-grid", data_source, md.name, eval_feat, et)))
        if et in absolute_error_relevant:
            if use_absolute_error is None:
                raise ValueError(
                        f"You must define use_absolute_error for {et}")
            tmp_name.append(["bias", "abs-err"][use_absolute_error])
        else:
            tmp_name.append("na")
        if et in contains_all_feats:
            tmp_name[3] = eval_feat.split("-")[0]
        tmp_eval = grid_evals[et]
        selected_evals.append(("_".join(tmp_name),tmp_eval))
    return selected_evals


def eval_model_on_grids(pkl_dir:Path, grid_domain:GridDomain,
        model_dir_path:Path, weights_file:str, eval_getter_args:list,
        grid_gen_args:dict, output_conversion="soilm_to_rsm", m_valid=None,
        extract_valid_mask=False, extract_latlon=True, dynamic_norm_coeffs={},
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
    agg_by_add = [EvalJointHist, EvalHorizon, EvalStatic, EvalTemporal]
    agg_by_concat = [EvalGridAxes]

    if debug:
        print(grid_domain)
    ## Evaluate over each tile in this domain, making each a generator that's
    ## specific to its timegrid source files and spatial range in the domain.
    out_evals = []
    tile_latlons = []
    for tile in grid_domain.tiles:
        ## Get a list of years forming a superset around requested times.
        eval_time_substrings = tuple(map(str, range(
            grid_domain.start_time.year-1, grid_domain.end_time.year+1
            )))
        if debug:
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

        ## If requested, extract the latlon array around this tile.
        lat,lon = None,None
        if extract_latlon:
            ## apply valid mask labeled m_valid if key found in timegrid attrs
            with h5py.File(name=tmp_timegrid_h5s[0], mode="r") as tmpf:
                _,_,tg_static_args = generators.parse_timegrid_attrs(
                        tmp_timegrid_h5s[0])
                lat_idx = tg_static_args["flabels"].index("lat")
                lon_idx = tg_static_args["flabels"].index("lon")
                lat = tmpf["/data/static"][...,lat_idx]
                lon = tmpf["/data/static"][...,lon_idx]
                tile_slice = (
                        slice(*tile.px_bounds[:2]),
                        slice(*tile.px_bounds[2:]))
                lat = lat[*tile_slice]
                lon = lon[*tile_slice]
        tile_latlons.append(np.stack([lat, lon], axis=-1))

        ## initialize a grid prediction generator instance with the ModelDir
        ## and domain configuration
        gen = gen_gridded_predictions(
                model_dir=md,
                grid_generator_args=tmp_gen_args,
                weights_file_name=weights_file,
                m_valid=ext_mask,
                dynamic_norm_coeffs=dynamic_norm_coeffs,
                static_norm_coeffs=static_norm_coeffs,
                yield_normed_inputs=False,
                yield_normed_outputs=False,
                debug=debug,
                output_conversion=output_conversion,
                )

        ## initialize some evaluator objects to run batch-wise on the generator
        evals = []
        for eargs in eval_getter_args:
            evals += get_grid_evaluator_objects(
                    model_dir=md,
                    grid_gen_args=grid_gen_args,
                    **eargs,
                    data_source="-".join([grid_domain.name, tile.region]),
                    attrs={"domain":grid_domain, "tiles":[tile]},
                    debug=debug,
                    )

        ## Iterate over this tile's generator, running evaluators on each step
        for inputs,true_states,predicted_residuals,idxs in gen:
            if debug:
                print(f"{md.name} new batch; {true_states.shape = }")
            for _,ev in evals:
                ev.add_batch(inputs, true_states, predicted_residuals, idxs)

        tmp_out_evals = []
        for name,ev in evals:
            ## If there are multiple tiles to concatenate, is a partial pkl
            if len(grid_domain.tiles) != 1:
                name += "_PARTIAL"
            tmp_out_evals.append(
                    (type(ev), pkl_dir.joinpath(f"{name}.pkl"))
                    )
            ev.to_pkl(tmp_out_evals[-1][1])

        out_evals.append(tmp_out_evals)
        ## reset the extracted mask since the next tile's mask can be different
        ext_mask = None

    '''
    ## Concatenate the Tiles' latlon objects in order of appearence,
    ## wrapping row-wise.
    ytiles,xtiles = grid_domain.mosaic_shape
    rows = [tile_latlons[i:i + xtiles]
            for i in range(0, len(tile_latlons), xtiles)]
    assert len(rows) == ytiles
    assert len(rows[0]) == xtiles
    latlons = np.concatenate([np.concatenate(x, axis=1) for x in rows], axis=0)
    '''

    ## aggregate tiles into a single pkl per grid domain
    if len(grid_domain.tiles) != 1:
        agg_eval_paths = []
        ## iterate over evaluator type
        for tile_eval_series in zip(*out_evals):
            tmp_agg_eval = None
            ## iterate over tiles of this evaluator type
            tmp_tiles = []
            for ev_type,ev_path in tile_eval_series:
                tmp_ev = ev_type().from_pkl(ev_path)
                ## keep track of this tile's configuration
                tmp_tiles += tmp_ev.attrs["tiles"]
                if tmp_agg_eval is None:
                    tmp_agg_eval = tmp_ev
                else:
                    tmp_ev = ev_type().from_pkl(ev_path)
                    ## Pathetically assume that the first saved axis will be
                    ## the concatenation axis. Need better way in the future.
                    if ev_type in agg_by_concat:
                        tmp_agg_eval = tmp_agg_eval.concatenate(
                                tmp_ev, axis=tmp_ev._axes[0])
                    ## otherwise add Evaluators that support it
                    elif ev_type in agg_by_add:
                        tmp_agg_eval = tmp_agg_eval.add(tmp_ev)
                    else:
                        raise ValueError(f"Evaluator type {ev_type}" + \
                                " doesn't support aggregation")
            ## Generalize the data source to not specify the tile
            new_path = list(ev_path.stem.replace("_PARTIAL","").split("_"))
            new_path[1] = grid_domain.name
            new_path = ev_path.parent.joinpath("_".join(new_path)+".pkl")
            ## Add latlon arrays and save the aggregated pkl as a new path
            tmp_agg_eval._attrs.update({
                "latlon":tile_latlons,
                "tiles":tmp_tiles,
                })
            tmp_agg_eval.to_pkl(new_path)
            agg_eval_paths.append(new_path)
            if debug:
                print(f"aggregated from: {list(zip(*tile_eval_series))[1]}")
        return agg_eval_paths
    else:
        ## If only 1 tile, just return the generated pkls after adding latlon
        for ev_type,ev_path in out_evals[0]:
            ev = ev_type().from_pkl(ev_path)
            ev._attrs.update({ "latlon":tile_latlons })
            ## remove the partial label from the pkl and re-save it
            ev.to_pkl(Path(Path(ev_path).as_posix().replace("_PARTIAL","")))
        return list(out_evals[0])

if __name__=="__main__":
    root_proj = Path("/rhome/mdodson/testbed/")
    timegrid_h5_dir = root_proj.joinpath("data/timegrids/")
    model_parent_dir = root_proj.joinpath("data/models/new")
    pkl_dir = root_proj.joinpath("data/eval_grid_pkls")

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
    #pred_feat_unit = "rsm"
    pred_feat_unit = "rsm"
    ## Output unit. Determines which set of evaluators are executed
    eval_feat_unit = "rsm"

    ## Subset of model weights to evaluate
    #weights_to_eval = soilm_models
    #weights_to_eval = [m for m in rsm_models if m[:10]=="lstm-rsm-9"]
    #weights_to_eval = [m for m in rsm_models if m[:12]=="accfnn-rsm-8"]
    #weights_to_eval = [m for m in rsm_models if m[:12]=="accrnn-rsm-2"]
    #weights_to_eval = [m for m in rsm_models if m[:12]=="accfnn-rsm-5"]
    #weights_to_eval = [m for m in rsm_models if m[:13]=="acclstm-rsm-4"]
    #weights_to_eval = [m for m in soilm_models if m[:7]=="lstm-20"]

    #weights_to_eval = [m for m in soilm_models
    #        if m.split("_")[0] in ["lstm-20"]]
    weights_to_eval = [m for m in rsm_models if m.split("_")[0] in [
            "lstm-rsm-9", ]]#"accfnn-rsm-8", "acclstm-rsm-4", ]]

    ## Keywords for subgrid domains to evaluate per configuration dict above
    domains_to_eval = [
            #"full",
            #"kentucky-flood",
            #"sandhills",
            #"high-sierra",
            "hurricane-laura",
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

    rsm_grid_eval_getter_args = [
            {
            "eval_types":[
                "spatial-stats", "init-time-stats", "hist-humidity-temp",
                "hist-true-pred", "static-combos", "horizon",
                "hist-state-increment"
                ],
            "eval_feat":"rsm-10",
            "pred_feat":f"{pred_feat_unit}-10",
            "coarse_reduce_func":"mean",
            "use_absolute_error":True,
            },
            {
            "eval_types":[
                "spatial-stats", "init-time-stats", "hist-humidity-temp",
                "hist-true-pred", "hist-saturation-error", "static-combos",
                "hist-state-increment"
                ],
            "eval_feat":"rsm-10",
            "pred_feat":f"{pred_feat_unit}-10",
            "coarse_reduce_func":"mean",
            "use_absolute_error":False,
            },
            {
            "eval_types":[
                "hist-true-pred", "hist-saturation-error",
                "hist-state-increment",
                ],
            "eval_feat":"rsm-40",
            "pred_feat":f"{pred_feat_unit}-40",
            "use_absolute_error":False,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            {
            "eval_types":[
                "hist-true-pred", "hist-saturation-error",
                "hist-state-increment",
                ],
            "eval_feat":"rsm-100",
            "pred_feat":f"{pred_feat_unit}-100",
            "use_absolute_error":False,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            {
            "eval_types":[ "hist-state-increment", ],
            "eval_feat":"rsm-40",
            "pred_feat":f"{pred_feat_unit}-40",
            "use_absolute_error":True,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            {
            "eval_types":[ "hist-state-increment", ],
            "eval_feat":"rsm-100",
            "pred_feat":f"{pred_feat_unit}-100",
            "use_absolute_error":True,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            ]
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
                    debug=True,
                    )
            print(f"Generated evaluator pkls:")
            pprint(out_pkls)
            gc.collect()