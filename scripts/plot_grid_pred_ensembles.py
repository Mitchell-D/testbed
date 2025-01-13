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
from testbed import list_feats

## Collect soil texture arrays and their corresponding text labels
soil_mapping = [
        (np.array(texture), label)
        for label,abbrv_label,texture in [
            list_feats.statsgo_texture_default[ix]
            for ix in list(range(1,12))+[16]
            ]
        ]

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/testbed")
    fig_dir = proj_root.joinpath("figures/eval_grid_figs_ensembles")
    eval_pkl_dir = proj_root.joinpath("data/eval_grid_ensembles")

    ## Specify a subset of grid Evaluator pkls to plot based on name fields:
    ## eval-grid_{domain}_{md.name}_{eval_feat}_{et}_{na|bias|abs-err}.pkl

    ## Specify which initialization time to use. Probably gonna B zero
    init_time_idx = 0

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
            "state-err-bias-textures",
            "res-err-bias-textures",
            "state-seq-textures",
            "res-seq-textures",
            ]

    common_sequence_plot_spec = {
                        "main_title":"",
                        "quad_titles":["","","",""],
                        "xlabel":"",
                        "ylabel":"",
                        "yscale":"linear",
                        "lines_rgb":None,
                        "line_opacity":.4,
                        "true_linestyle":"-",
                        "pred_linestyle":"-",
                        "true_linewidth":2,
                        "pred_linewidth":2,
                        "xlabel":"Forecast hour from {init_time}",
                        "main_title_size":18,
                        "legend_location":"lower left",
                        "pred_legend_label":"",
                        "true_legend_label":"",
                        "figsize":(11,7),

                        "legend_location":"lower right",
                        "legend_size":24,
                        "legend_ncols":2,
                        }
    quad_sequence_plot_info = {
            ## Error biases wrt soil texture types
            "state-err-bias-textures":{
                "single_feature_per_axis":True,
                "soil_texture_rgb":True,
                "averaging":"soil_texture",
                "plot_feats":[
                    ("err_state", "rsm-10", "mean"),
                    ("err_state", "rsm-40", "mean"),
                    ("err_state", "rsm-100", "mean"),
                    ],
                "error_type":"bias",
                "plot_spec":{
                    },
                },
            "res-err-bias-textures":{
                "single_feature_per_axis":True,
                "soil_texture_rgb":True,
                "averaging":"soil_texture",
                "plot_feats":[
                    ("err_res", "rsm-10", "mean"),
                    ("err_res", "rsm-40", "mean"),
                    ("err_res", "rsm-100", "mean"),
                    ],
                "error_type":"bias",
                "plot_spec":{
                    },
                },
            ## State and increment time series wrt soil texture types
            "state-seq-textures":{
                "single_feature_per_axis":False,
                "soil_texture_rgb":True,
                "averaging":"soil_texture",
                "plot_feats":[
                    [("true_state", "rsm-10", "mean"),
                        ("pred_state", "rsm-10", "mean"),],
                    [("true_state", "rsm-40", "mean"),
                        ("pred_state", "rsm-40", "mean"),],
                    [("true_state", "rsm-100", "mean"),
                        ("pred_state", "rsm-100", "mean"),],
                    ],
                "error_type":"bias",
                "plot_spec":{
                    "true_linestyle":"-",
                    "pred_linestyle":":",
                    "quad_titles":[
                        "0-10cm RSM State",
                        "10-40cm RSM State",
                        "40-100cm RSM State",
                        ""
                        ],
                    "main_title":"True and Predicted RSM State wrt Time, " + \
                            "Colored by Soil Texture",
                    },
                },
            "res-seq-textures":{
                "single_feature_per_axis":False,
                "soil_texture_rgb":True,
                "averaging":"soil_texture",
                "plot_feats":[
                    [("true_res", "rsm-10", "mean"),
                        ("pred_res", "rsm-10", "mean"),],
                    [("true_res", "rsm-40", "mean"),
                        ("pred_res", "rsm-40", "mean"),],
                    [("true_res", "rsm-100", "mean"),
                        ("pred_res", "rsm-100", "mean"),],
                    ],
                "error_type":"bias",
                "plot_spec":{
                    "true_linestyle":"-",
                    "pred_linestyle":":",
                    "quad_titles":[
                        "0-10cm RSM Increment Change",
                        "10-40cm RSM Increment Change",
                        "40-100cm RSM Increment Change",
                        ""
                        ],
                    "main_title":"True and Predicted RSM Increment wrt " + \
                            "Time, Colored by Soil Texture",
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

    for p,pt in filter(lambda p:p[1][4]=="keep-all", eval_pkls):
        ev = evaluators.EvalGridAxes().from_pkl(p)
        _,data_source,model,eval_feat,eval_type,error_type = pt
        model_cfg = ev.attrs["model_config"]
        feat_labels = [
                "-".join((eval_feat, f.split("-")[-1]))
                for f in model_cfg["feats"]["pred_feats"]
                if f != "weasd"
                ]
        ev_dict = ev.get_results()
        all_flabels = ev.attrs["flabels"]
        all_feats = ev.average[init_time_idx]

        ## Plots should have options:
        ##  - Average by soil texture, overall, or not at all
        ##  - Color by soil texture, or from configuration
        for qst in plot_quad_sequence_type:
            tmp_cfg = quad_sequence_plot_info[qst]
            tmp_path = fig_dir.joinpath("_".join([*pt, qst])+".png")

            ## ignore Evalator objects with the wrong error type
            if tmp_cfg["error_type"] != error_type:
                continue

            ## Extract a (P,3) RGB of soil textures if RGB requested
            if tmp_cfg["soil_texture_rgb"]:
                sfeats = model_cfg["feats"]["static_feats"]
                soil_feats = ("pct_sand", "pct_silt", "pct_clay")
                soil_idxs = tuple(sfeats.index(s) for s in soil_feats)
                soil_rgb = np.clip(ev_dict["static"][...,soil_idxs], 0, 1)
            ## Otherwise default to plot_spec specified true_color,pred_color
            else:
                soil_rgb = None

            ## mean-reduce pixels along the 2nd axis according to the requested
            ## method, and collect standard deviations within each for unc bars
            if tmp_cfg["averaging"] == "soil_texture":
                unique_textures = np.unique(soil_rgb, axis=0)
                ## Make a boolean mask for each unique soil texture
                texture_masks = [
                        np.all(soil_rgb == ut, axis=1)
                        for ut in unique_textures
                        ]
                ## Average the features within a soil texture if requested
                tmp_stdevs = np.stack([
                    np.std(all_feats[tm], axis=0)
                    for tm in texture_masks
                    ], axis=0)
                tmp_feats = np.stack([
                    np.average(all_feats[tm], axis=0)
                    for tm in texture_masks
                    ], axis=0)
                if not soil_rgb is None:
                    soil_rgb = unique_textures
                legend_labels = []
                ## If averaging soil textures, assume the legend labels them
                for i in range(unique_textures.shape[0]):
                    for txtr,label in soil_mapping:
                        if np.all(np.isclose(unique_textures[i],txtr)):
                            legend_labels.append(label)
                    tmp_cfg["plot_spec"]["soil_texture_legend"] = \
                            legend_labels
            ## Average all pixels together
            elif tmp_cfg["averaging"] == "all_px":
                tmp_stdevs = np.std(all_feats, axis=0, keepdims=True)
                tmp_feats = np.average(all_feats, axis=0, keepdims=True)
                if not soil_rgb is None:
                    soil_rgb = np.average(soil_rgb, axis=0, keepdims=True)
            ## Make no change
            else:
                tmp_feats = all_feats
                tmp_stdevs = np.full(all_feats.shape, 0)

            ## Split between single features per axis and two per
            tmp_feat_dict = {"mean":ev.average, "stdev":ev.variance**(1/2)}
            plot_flabels = tmp_cfg["plot_feats"]
            if tmp_cfg["single_feature_per_axis"]:
                pred_fidxs = [all_flabels.index(c[:2]) for c in plot_flabels]
                true_feats = None
                pred_feats = np.stack([
                    tmp_feats[...,ix]
                    for ix,m in zip(pred_fidxs,(c[2] for c in plot_flabels))
                    ], axis=-1)
            else:
                true_fidxs = [all_flabels.index(c[0][:2])
                        for c in plot_flabels]
                true_feats = np.stack([
                    tmp_feats[...,ix]
                    for ix,m in zip(
                        true_fidxs, (c[0][2] for c in plot_flabels))
                    ], axis=-1)
                pred_fidxs = [all_flabels.index(c[1][:2])
                        for c in plot_flabels]
                pred_feats = np.stack([
                    tmp_feats[...,ix]
                    for ix,m in zip(
                        pred_fidxs, (c[1][2] for c in plot_flabels))
                    ], axis=-1)

            plot_spec = {
                    **common_sequence_plot_spec,
                    **tmp_cfg["plot_spec"],
                    "lines_rgb":soil_rgb,
                    }
            time_fmt = "%m/%d/%Y %H:00"
            substitute_strings = {
                    "init_time":datetime.fromtimestamp(
                        int(ev.time[init_time_idx][0])).strftime(time_fmt),
                    "model_name":model,
                    "eval_feat":eval_feat,
                    "eval_type":eval_type,
                    "error_type":error_type,
                    }

            for k,v in plot_spec.items():
                if type(v) == str:
                    v = v.format(**substitute_strings)

            ## Generate the plot
            plotting.plot_quad_sequence(
                    pred_array=pred_feats,
                    true_array=true_feats,
                    fig_path=tmp_path,
                    pred_coarseness=model_cfg["feats"]["pred_coarseness"],
                    plot_spec=plot_spec,
                    show=False,
                    )
