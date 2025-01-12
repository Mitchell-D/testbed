"""
Script for running plotting methods on pkls associated with gridded Evaluator
objects, which were probably produced by testbed.eval_grids.eval_model_on_grids
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
from pprint import pprint

from testbed import evaluators
from testbed.eval_grids import GridDomain,GridTile
from testbed import plotting

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/testbed")
    fig_dir = proj_root.joinpath("figures/eval_grid_figs")
    eval_pkl_dir = proj_root.joinpath("data/eval_grid_ensembles")

    ## Specify a subset of grid Evaluator pkls to plot based on name fields:
    ## eval-grid_{domain}_{md.name}_{eval_feat}_{et}_{na|bias|abs-err}.pkl

    ## Spatiotemporal domains to plot (2nd field of file name)
    plot_domains = [
            "kentucky-flood",
            "high-sierra",
            "sandhills",
            "hurricane-laura",
            ]
    ## substrings of model names to plot (3rd field of file name)
    plot_models_contain = [
            "lstm-rsm-9",
            "accfnn-rsm-8",
            "acclstm-rsm-4",
            "lstm-20",
            ]
    ## evlauated features to plot (4th field of file name)
    plot_eval_feats = [
            "rsm",
            "rsm-10",
            "rsm-40",
            "rsm-100",
            "soilm-10"
            ]
    ## Evaluator instance types to include (5th field of file name)
    plot_eval_type = [ "keep-all", ]

    ## error types of evaluators to plot (6th field of file name)
    plot_error_type = [
            "na",
            "bias",
            "abs-err"
            ]
    plot_quad_sequence_type = [
            "state-err-bias-rgb",
            "res-err-bias-rgb",
            "state-seq-textures",
            "res-seq-textures",
            ]

    common_sequence_plot_spec = {
                        "main_title":"",
                        "quad_titles":[],
                        "xlabel":"",
                        "ylabel":"",
                        "yscale":"linear",
                        "lines_rgb":"",
                        "line_opacity":.4,
                        "true_linestyle":"-",
                        "pred_linestyle":"-",
                        "main_title_size":18,
                        "legend_location":"lower left",
                        "pred_legend_label":"",
                        "true_legend_label":"",
                        "figsize":(11,7),
                        }
    quad_sequence_plot_info = {
            "state-err-bias-rgb":{
                "single_feature_per_axis":False,
                "soil_texture_rgb":True,
                "average_soil_textures":False,
                "feats":[
                    ("err_state", "rsm-10", "mean"),
                    ("err_state", "rsm-40", "mean"),
                    ("err_state", "rsm-100", "mean"),
                    ],
                "error_type":"bias",
                "plot_spec":{
                    **common_sequence_plot_spec,
                    },
                },
            "res-err-bias-rgb":{
                "single_feature_per_axis":False,
                "soil_texture_rgb":True,
                "average_soil_textures":False,
                "feats":[
                    ("err_res", "rsm-10", "mean"),
                    ("err_res", "rsm-40", "mean"),
                    ("err_res", "rsm-100", "mean"),
                    ],
                "error_type":"bias",
                "plot_spec":{
                    **common_sequence_plot_spec,
                    },
                },
            "state-seq-textures":{
                "single_feature_per_axis":False,
                "soil_texture_rgb":False,
                "average_soil_textures":True,
                "feats":[
                    [("true_state", "rsm-10", "mean"),
                        ("pred_state", "rsm-10", "mean"),],
                    [("true_state", "rsm-40", "mean"),
                        ("pred_state", "rsm-40", "mean"),],
                    [("true_state", "rsm-100", "mean"),
                        ("pred_state", "rsm-100", "mean"),],
                    ],
                "error_type":"bias",
                "plot_spec":{
                    **common_sequence_plot_spec,
                    },
                },
            "res-seq-textures":{
                "single_feature_per_axis":False,
                "soil_texture_rgb":False,
                "average_soil_textures":True,
                "feats":[
                    [("true_res", "rsm-10", "mean"),
                        ("pred_res", "rsm-10", "mean"),],
                    [("true_res", "rsm-40", "mean"),
                        ("pred_res", "rsm-40", "mean"),],
                    [("true_res", "rsm-100", "mean"),
                        ("pred_res", "rsm-100", "mean"),],
                    ],
                "error_type":"bias",
                "plot_spec":{
                    **common_sequence_plot_spec,
                    },
                },
            }

    ## --------( END BASIC CONFIGURATION )--------

    ## subset available pkls according to selection string configuration
    eval_pkls = [
            (p,pt) for p,pt in map(
                lambda f:(f,f.stem.split("_")),
                sorted(eval_pkl_dir.iterdir()))
            if pt[0] == "eval-grid"
            and pt[1] in plot_domains
            and any(s in pt[2] for s in plot_models_contain)
            and pt[3] in plot_eval_feats
            and pt[4] in plot_eval_type
            and (len(pt)==5 or pt[5] in plot_error_type)
            and "PARTIAL" not in pt
            ]
    ## Ignore spatial stats with error types not needed
    eval_pkls = list(filter(
            lambda p:p[1][4] != "spatial-stats" or any([
                spatial_plot_info[k]["error_type"] == p[1][5]
                for k in plot_spatial_stats
                ]),
            eval_pkls
            ))

    print(f"Found {len(eval_pkls)} matching eval pkls:")
    print("\n".join([p[0].name for p in eval_pkls]))

    ## plot
    for p,pt in filter(lambda p:p[1][4]=="keep-all", eval_pkls):
        ev = evaluators.EvalGridAxes().from_pkl(p)
        _,data_source,model,eval_feat,eval_type,error_type = pt
        model_config = ev.attrs["model_config"]
        feat_labels = [
                "-".join((eval_feat, f.split("-")[-1]))
                for f in model_config["feats"]["pred_feats"]
                if f != "weasd"
                ]
        ev_dict = ev.get_results()
        print(f'{ev_dict["avg"].shape = }')
        print(f'{ev_dict["var"].shape = }')
        print(f'{ev_dict["static"].shape = }')
        print(f'{ev_dict["time"].shape = }')
        print(f'{ev_dict["indeces"].shape = }')
        print(f'{ev_dict["static"].shape = }')
        print(f'{ev_dict["counts"].shape = }')

        for qst in plot_quad_sequence_type:
            tmp_cfg = quad_sequence_plot_info[qst]

            if tmp_cfg["average_soil_textures"] or tmp_cfg["soil_texture_rgb"]:
                sfeats = model_config["feats"]["static_feats"]
                soil_feats = ("pct_sand", "pct_silt", "pct_clay")
                soil_idxs = tuple(sfeats.index(s) for s in soil_feats)
                soil_rgb = np.clip(ev_dict["static"][...,soil_idxs], 0, 1)
                unique_textures = np.unique(soil_rgb, axis=0)
                print(unique_textures)
            continue

            tmp_path = fig_dir.joinpath("_".join([*pt, qst])+".png")
            print(tmp_path)
            ## Plot residual values
            plot_quad_sequence(
                    true_array=yr[...,feat_idxs],
                    pred_array=pr[...,feat_idxs],
                    fig_path=tmp_path,
                    pred_coarseness=model_config["feats"]["pred_coarseness"],
                    plot_spec=tmp_cfg["plot_spec"],
                    show=False,
                    fill_sigma=1,
                    )
