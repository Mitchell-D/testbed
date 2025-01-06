"""
This script contains main context code for evaluating models and generating
prediction grid hdf5 susing eval_models.grid_preds_to_hdf5, and calculating
bulk statistics over the results with eval_models.bulk_grid_error_stats_to_hdf5
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path

import tracktrain as tt
from testbed import eval_models
from testbed import model_methods as mm

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
        eval_models.grid_preds_to_hdf5(
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
        eval_models.bulk_grid_error_stats_to_hdf5(
                grid_h5=bulk_grid_dir.joinpath(p),
                stats_h5=bulk_grid_dir.joinpath(bulk_file),
                debug=True,
                )
    '''
