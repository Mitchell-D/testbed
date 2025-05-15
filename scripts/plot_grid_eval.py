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
    #fig_dir = proj_root.joinpath("figures/eval_grid_slope-tiles")
    eval_pkl_dir = proj_root.joinpath("data/eval_grid_pkls")

    ## Specify a subset of grid Evaluator pkls to plot based on name fields:
    ## eval-grid_{domain}_{md.name}_{eval_feat}_{et}_{na|bias|abs-err}.pkl

    ## Spatiotemporal domains to plot (2nd field of file name)
    plot_domains = [
            #"kentucky-flood",
            #"high-sierra",
            #"sandhills",
            #"hurricane-laura",
            #"gtlb-drought-fire",
            #"dakotas-flash-drought",
            #"hurricane-florence",
            #"eerie-mix",
            #"full"
            #"lt-cascades",
            #"lt-fourcorners",

            #"lt-miss-alluvial",
            #"lt-high-sierra",
            #"lt-high-plains",
            #"lt-north-michigan",
            "lt-atlanta",
            ]
    ## substrings of model names to plot (3rd field of file name)
    plot_models_contain = [
            #"accfnn",
            #"accrnn",
            #"lstm-rsm",
            #"acclstm-rsm-1",
            "lstm-rsm-9",
            #"accfnn-rsm-8",
            #"accrnn-rsm-2",
            #"accfnn-rsm-5",
            #"lstm-20",
            #"lstm-18",
            #"acclstm-rsm-4",

            #"lstm-rsm-46" ## sand-dominant model
            ]
    ## evlauated features to plot (4th field of file name)
    plot_eval_feats = [
            "rsm",
            "rsm-10",
            "rsm-40",
            "rsm-100",
            #"soilm"
            #"soilm-10"
            #"soilm-40"
            #"soilm-100"
            #"soilm-200"
            ]
    ## Evaluator instance types to include (5th field of file name)
    plot_eval_type = [
            "horizon",
            "static-combos",
            "hist-true-pred",
            #"hist-saturation-error",
            "hist-state-increment",
            "hist-humidity-temp",
            "hist-weasd-increment",
            "hist-weasd-temp",
            #"hist-infiltration",
            "spatial-stats",
            ]
    ## error types of evaluators to plot (6th field of file name)
    plot_error_type = [
            "na",
            "bias",
            "abs-err",
            ]

    ## Select which 4-panel configurations to plot (from plot_spatial_stats)
    plot_spatial_stats = [
            "res-mean",
            "state-mean",
            "res-err-bias-mean",
            "res-err-bias-stdev",
            "res-err-abs-mean",
            "res-err-abs-stdev",
            "state-err-bias-mean",
            "state-err-bias-stdev",
            "state-err-abs-mean",
            "state-err-abs-stdev",
            "temp-spfh-apcp-mean",
            "temp-spfh-apcp-stdev",
            ]

    ## --------( END BASIC CONFIGURATION )--------

    domain_plot_specs = {
            "lt-high-sierra":{
                "figsize":(24,24),
                "cbar_orient":"vertical",
                },
            "lt-miss-alluvial":{
                "figsize":(14,22),
                "cbar_orient":"vertical",
                },
            "lt-north-michigan":{
                "figsize":(22,22),
                "cbar_orient":"horizontal",
                },
            "lt-atlanta":{
                "figsize":(22,18),
                "cbar_orient":"horizontal",
                },
            "lt-high-plains":{
                "figsize":(20,18),
                "cbar_orient":"horizontal",
                },
            }

    ## Specify 4-panel figure configurations of spatial statistics data
    common_spatial_plot_spec = {
            "text_size":24,
            "show_ticks":False,
            "cmap":"gnuplot2",
            "figsize":(32,16), ## best for full domain
            #"figsize":(18,12),
            "tight_layout":False,
            "title_fontsize":32,
            "use_pcolormesh":True,
            "cbar_orient":"horizontal",
            #"cbar_orient":"vertical",
            "cbar_shrink":.9,
            #"cbar_shrink":.6,
            "cbar_pad":.02,
            #"geo_bounds":[-95,-80,32,42],
            }
    spatial_plot_info = {
            "res-mean":{
                "feats":[
                    ("true_res", "rsm-10", "mean"),
                    ("true_res", "rsm-40", "mean"),
                    ("true_res", "rsm-100", "mean"),
                    ],
                "error_type":"bias",
                "plot_spec":{},
                },
            "state-mean":{
                "feats":[
                    ("true_state", "rsm-10", "mean"),
                    ("true_state", "rsm-40", "mean"),
                    ("true_state", "rsm-100", "mean"),
                    ],
                "error_type":"bias",
                "plot_spec":{
                    #"vmin":[.4,.44,.42],
                    #"vmax":[.7,.56,.57],
                    },
                },
            "res-err-bias-mean":{
                "feats":[
                    ("err_res", "rsm-10", "mean"),
                    ("err_res", "rsm-40", "mean"),
                    ("err_res", "rsm-100", "mean"),
                    ],
                "subplot_titles":[
                    "Mean Per-Pixel Bias in Increment RSM (0-10cm)",
                    "Mean Per-Pixel Bias in Increment RSM (10-40cm)",
                    "Mean Per-Pixel Bias in Increment RSM (40-100cm)",
                    ],
                "error_type":"bias",
                "plot_spec":{
                    "vmin":[-3e-4,-1.5e-4, -1.5e-4],
                    "vmax":[3e-4, 1.5e-4, 1.5e-4],
                    "cmap":"seismic_r",
                    },
                },
            "res-err-bias-stdev":{
                "feats":[
                    ("err_res", "rsm-10", "stdev"),
                    ("err_res", "rsm-40", "stdev"),
                    ("err_res", "rsm-100", "stdev"),
                    ],
                "subplot_titles":[
                    "Standard Deviation of Per-Pixel Bias in Increment " + \
                            "RSM (0-10cm)",
                    "Standard Deviation of Per-Pixel Bias in Increment " + \
                            "RSM (10-40cm)",
                    "Standard Deviation of Per-Pixel Bias in Increment " + \
                            "RSM (40-100cm)",
                    ],
                "error_type":"bias",
                "plot_spec":{
                    "cmap":"gnuplot2",
                    },
                },
            "res-err-abs-mean":{
                "feats":[
                    ("err_res", "rsm-10", "mean"),
                    ("err_res", "rsm-40", "mean"),
                    ("err_res", "rsm-100", "mean"),
                    ],
                "subplot_titles":[
                    "Mean Per-Pixel MAE in Increment RSM (0-10cm)",
                    "Mean Per-Pixel MAE in Increment RSM (10-40cm)",
                    "Mean Per-Pixel MAE in Increment RSM (40-100cm)",
                    ],
                "error_type":"abs_err",
                "plot_spec":{
                    "cmap":"gnuplot2",
                    },
                },
            "res-err-abs-stdev":{
                "feats":[
                    ("err_res", "rsm-10", "stdev"),
                    ("err_res", "rsm-40", "stdev"),
                    ("err_res", "rsm-100", "stdev"),
                    ],
                "subplot_titles":[
                    "Standard Deviation of Per-Pixel MAE in Increment " + \
                            "RSM (0-10cm)",
                    "Standard Deviation of Per-Pixel MAE in Increment " + \
                            "RSM (10-40cm)",
                    "Standard Deviation of Per-Pixel MAE in Increment " + \
                            "RSM (40-100cm)",
                    ],
                "error_type":"abs_err",
                "plot_spec":{
                    "cmap":"gnuplot2",
                    },
                },
            "state-err-bias-mean":{
                "feats":[
                    ("err_state", "rsm-10", "mean"),
                    ("err_state", "rsm-40", "mean"),
                    ("err_state", "rsm-100", "mean"),
                    ],
                "subplot_titles":[
                    "Mean Per-Pixel Bias in RSM State (0-10cm)",
                    "Mean Per-Pixel Bias in RSM State (10-40cm)",
                    "Mean Per-Pixel Bias in RSM State (40-100cm)",
                    ],
                "error_type":"bias",
                "plot_spec":{
                    "vmin":[-6e-2, -3e-2, -3e-2],
                    "vmax":[6e-2, 3e-2, 3e-2],
                    "cmap":"seismic_r",
                    },
                },
            "state-err-bias-stdev":{
                "feats":[
                    ("err_state", "rsm-10", "stdev"),
                    ("err_state", "rsm-40", "stdev"),
                    ("err_state", "rsm-100", "stdev"),
                    ],
                "subplot_titles":[
                    "Standard Deviation of Per-Pixel Bias in RSM " + \
                            "State (0-10cm)",
                    "Standard Deviation of Per-Pixel Bias in RSM " + \
                            "State (10-40cm)",
                    "Standard Deviation of Per-Pixel Bias in RSM " + \
                            "State (40-100cm)",
                    ],
                "error_type":"bias",
                "plot_spec":{
                    "cmap":"gnuplot2",
                    "vmin":[0,0,0],
                    "vmax":[.12,.8,.8],
                    },
                },
            "state-err-abs-mean":{
                "feats":[
                    ("err_state", "rsm-10", "mean"),
                    ("err_state", "rsm-40", "mean"),
                    ("err_state", "rsm-100", "mean"),
                    ],
                "subplot_titles":[
                    "Per-Pixel MAE in RSM State (0-10cm)",
                    "Per-Pixel MAE in RSM State (10-40cm)",
                    "Per-Pixel MAE in RSM State (40-100cm)",
                    ],
                "error_type":"abs-err",
                "plot_spec":{
                    "cmap":"gnuplot2",
                    "vmin":[0,0,0],
                    "vmax":[.07,.05,.05],
                    },
                },
            "state-err-abs-stdev":{
                "feats":[
                    ("err_state", "rsm-10", "stdev"),
                    ("err_state", "rsm-40", "stdev"),
                    ("err_state", "rsm-100", "stdev"),
                    ],
                "subplot_titles":[
                    "Standard Deviation of Per-Pixel MAE in " + \
                            "RSM State (0-10cm)",
                    "Standard Deviation of Per-Pixel MAE in " + \
                            "RSM State (10-40cm)",
                    "Standard Deviation of Per-Pixel MAE in " + \
                            "RSM State (40-100cm)",
                    ],
                "error_type":"abs-err",
                "plot_spec":{
                    "cmap":"gnuplot2",
                    "vmin":[0,0,0],
                    "vmax":[.1, .07, .05],
                    },
                },
            "temp-spfh-apcp-mean":{
                "feats":[
                    ("horizon", "tmp", "mean"),
                    ("horizon", "spfh", "mean"),
                    ("horizon", "apcp", "mean"),
                    ],
                "error_type":"abs-err", ## doesn't matter which type here.
                "plot_spec":{
                    },
                },
            "temp-spfh-apcp-stdev":{
                "feats":[
                    ("horizon", "tmp", "stdev"),
                    ("horizon", "spfh", "stdev"),
                    ("horizon", "apcp", "stdev"),
                    ],
                "error_type":"abs-err", ## doesn't matter which type here.
                "plot_spec":{
                    },
                },
            }

    hist_plot_info = {
            "hist-true-pred":{
                "na":{
                    "title":"Increment RSM True/Pred Joint Histogram " + \
                            "\n{model_name} {eval_feat} {domain}",
                    "xlabel":"Predicted change in RSM",
                    "ylabel":"Actual change in RSM",
                    "aspect":1,
                    "norm":"log",
                    "text_size":14,
                    }
                },
            "hist-saturation-error":{
                "na":{
                    "title":"Increment RSM Error Bias wrt Saturation " + \
                            "Percentage\n{model_name} {eval_feat} {domain}",
                    "xlabel":"Error in hourly change in RSM",
                    "ylabel":"Relative soil moisture ({eval_feat})",
                    "aspect":1,
                    "norm":"log",
                    "text_size":14,
                    },
                },
            "hist-state-increment":{
                "abs-err":{
                    "title":"Hourly RSM MAE wrt Saturation and Increment " + \
                            "Change in RSM\n{model_name} {eval_feat} {domain}",
                    "xlabel":"Hourly increment change in {eval_feat}",
                    "ylabel":"Soil saturation in RSM",
                    "norm":"log",
                    "cov_norm":"log",
                    #"cov_vmin":0,
                    "cov_vmax":{
                        "rsm-10":.05,
                        "rsm-40":.01,
                        "rsm-100":.005,
                        "rsm-200":.005,
                        },
                    "cb_size":.8,
                    "cov_cmap":"jet",
                    "aspect":1,
                    "fig_size":(18,8),
                    },
                "bias":{
                    "title":"Hourly RSM Bias wrt Saturation and Increment " + \
                            "Change in RSM\n{model_name} {eval_feat} {domain}",
                    "xlabel":"Hourly increment change in {eval_feat}",
                    "ylabel":"Soil saturation in RSM",
                    "norm":"log",
                    "cov_vmin":{
                        "rsm-10":-.05,
                        "rsm-40":-.01,
                        "rsm-100":-.005,
                        "rsm-200":-.005,
                        },
                    "cov_vmax":{
                        "rsm-10":.05,
                        "rsm-40":.01,
                        "rsm-100":.005,
                        "rsm-200":.005,
                        },
                    "cb_size":.8,
                    "cov_cmap":"seismic_r",
                    "cov_norm":"symlog",
                    "aspect":1,
                    "fig_size":(18,8),
                    },
                },
            "hist-weasd-increment":{
                "abs-err":{
                    "title":"Hourly MAE wrt SWE/Increment RSM Distribution" + \
                            " {model_name} {eval_feat} {domain} ",
                    "xlabel":"Increment Change in RSM (RSM/hour)",
                    "ylabel":"Snow Water Equivalent ($kg\/m^2$)",
                    "norm":"log",
                    #"cov_vmin":0,
                    "cb_label":"Sample Counts",
                    "cb_pad":.01,
                    #"cov_vmax":1.2e-3,
                    "cov_vmin":1e-5,
                    "cov_vmax":1e-2,
                    "cov_norm":"log",
                    "cov_cmap":"jet",
                    "xlim":(-.025,0.05),
                    "cov_cb_label":"Increment RSM MAE",
                    "aspect":1,
                    "fig_size":(14,8),
                    "cb_size":.9,
                    "text_size":16,
                    "hist_title":"SWE/Increment RSM Joint Histogram",
                    "cov_title":"Increment Absolute Error in Histogram Bins",
                    },
                "bias":{
                    "title":"Hourly Bias wrt SWE/Increment RSM Distribution"+\
                            " {model_name} {eval_feat} {domain}",
                    "norm":"log",
                    "xlabel":"Temperature (K)",
                    "ylabel":"Snow Water Equivalent (kg/kg)",
                    #"cov_vmin":-1.2e-3,
                    #"cov_vmax":1.2e-3,
                    "cov_norm":"symlog",
                    "cov_cmap":"seismic_r",
                    "cb_label":"Sample Counts",
                    "cov_cb_label":"Increment RSM Error Bias",
                    "cov_vmin":-1e-3,
                    "cov_vmax":1e-3,
                    "xlim":(-.025,0.05),
                    "aspect":1,
                    "fig_size":(18,8),
                    "cb_size":.9,
                    "text_size":16,
                    "hist_title":"SWE/Increment RSM Joint Histogram",
                    "cov_title":"Increment Error Bias in Histogram Bins",
                    }
                },
            "hist-weasd-temp":{
                "abs-err":{
                    "title":"Hourly MAE wrt SWE/Temp Distribution" + \
                            " {model_name} {eval_feat} {domain} ",
                    "norm":"log",
                    "xlabel":"Temperature (K)",
                    "ylabel":"Snow Water Equivalent ($kg\/m^2$)",
                    "norm":"log",
                    #"cov_vmin":0,
                    "cb_label":"Sample Counts",
                    "cb_pad":.01,
                    #"cov_vmax":1.2e-3,
                    "cov_vmin":1e-5,
                    "cov_vmax":1e-2,
                    "cov_norm":"log",
                    "cov_cmap":"jet",
                    "xlim":(245,290),
                    "cov_cb_label":"Increment RSM MAE",
                    "aspect":1,
                    "fig_size":(18,6),
                    "cb_size":.9,
                    "text_size":16,
                    "hist_title":"SWE/Temperature Joint Histogram",
                    "cov_title":"Increment Absolute Error in Histogram Bins",
                    },
                "bias":{
                    "title":"Hourly Bias wrt SWE/Temp Distribution" + \
                            " {model_name} {eval_feat} {domain}",
                    "norm":"log",
                    "xlabel":"Temperature (K)",
                    "ylabel":"Snow Water Equivalent (kg/kg)",
                    #"cov_vmin":-1.2e-3,
                    #"cov_vmax":1.2e-3,
                    "cov_norm":"symlog",
                    "cov_cmap":"seismic_r",
                    "cb_label":"Sample Counts",
                    "cov_cb_label":"Increment RSM Error Bias",
                    "cov_vmin":-1e-3,
                    "cov_vmax":1e-3,
                    "aspect":1,
                    "xlim":(245,290),
                    "fig_size":(18,8),
                    "cb_size":.9,
                    "text_size":16,
                    "hist_title":"SWE/Temperature Joint Histogram",
                    "cov_title":"Increment Error Bias in Histogram Bins",
                    }
                },
            "hist-humidity-temp":{
                "abs-err":{
                    "title":"Hourly MAE wrt Humidity/Temp Distribution" + \
                            " {model_name} {eval_feat} {domain} ",
                    "norm":"log",
                    "xlabel":"Temperature (K)",
                    "ylabel":"Absolute humidity (kg/kg)",
                    "norm":"log",
                    #"cov_vmin":0,
                    "cb_label":"Sample Counts",
                    "cb_pad":.01,
                    #"cov_vmax":1.2e-3,
                    "cov_vmin":1e-5,
                    "cov_vmax":1e-2,
                    "cov_norm":"log",
                    "cov_cmap":"jet",
                    "cov_cb_label":"Increment RSM MAE",
                    "aspect":1,
                    "fig_size":(18,8),
                    "cb_size":.9,
                    "text_size":16,
                    "hist_title":"Temperature/Humidity Joint Histogram",
                    "cov_title":"Increment Absolute Error in Histogram Bins",
                    },
                "bias":{
                    "title":"Hourly Bias wrt Humidity/Temp Distribution" + \
                            " {model_name} {eval_feat} {domain}",
                    "norm":"log",
                    "xlabel":"Temperature (K)",
                    "ylabel":"Absolute humidity (kg/kg)",
                    #"cov_vmin":-1.2e-3,
                    #"cov_vmax":1.2e-3,
                    "cov_norm":"symlog",
                    "cov_cmap":"seismic_r",
                    "cb_label":"Sample Counts",
                    "cov_cb_label":"Increment RSM Error Bias",
                    "cov_vmin":-1e-3,
                    "cov_vmax":1e-3,
                    "aspect":1,
                    "fig_size":(18,8),
                    "cb_size":.9,
                    "text_size":16,
                    "hist_title":"Temperature/Humidity Joint Histogram",
                    "cov_title":"Increment Error Bias in Histogram Bins",
                    }
                },
            "hist-infiltration":{
                    "na":{
                        "title":"Infiltration Validation and Mean Layer " + \
                                "Water Content ($kg\/m^2$) " + \
                                "{model_name} {eval_feat} {domain} ",
                        "norm":"log",
                        "vmax":100,
                        "vmin":0,
                        }
                    },
            }

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

    ## plot error wrt forecast horizon
    for p,pt in filter(lambda p:p[1][4]=="horizon", eval_pkls):
        ev = evaluators.EvalHorizon().from_pkl(p)
        _,data_source,model,eval_feat,eval_type,error_type = pt
        feat_labels = [
                "-".join((eval_feat, f.split("-")[-1]))
                for f in ev.attrs["model_config"]["feats"]["pred_feats"]
                if f != "weasd"
                ]
        ev.plot(
                fig_path=fig_dir.joinpath(p.stem+"_state.png"),
                feat_labels=["State Error in "+l for l in feat_labels],
                state_or_res="state",
                fill_sigma=1,
                bar_sigma=1,
                plot_spec={
                    "title":f"{model} {eval_feat} {data_source} state MAE " + \
                            "wrt Forecast Hour",
                    "xlabel":"Forecast hour",
                    "ylabel":"Mean absolute state error " + \
                            f"({eval_feat.upper()})",
                    "alpha":.6,
                    "line_width":2,
                    "error_line_width":.5,
                    "error_every":4,
                    "fill_alpha":.25,
                    "yrange":(0,.04)
                    },
                use_stdev=False,
                )
        print(f"Generated {fig_dir.joinpath(p.stem+'_state.png')}")
        ev.plot(
                fig_path=fig_dir.joinpath(p.stem+"_res.png"),
                feat_labels=["Increment Error in "+l for l in feat_labels],
                state_or_res="res",
                fill_sigma=1,
                bar_sigma=1,
                plot_spec={
                    "title":f"{model} {eval_feat} {data_source} increment " + \
                            "MAE wrt Forecast Hour",
                    "xlabel":"Forecast hour",
                    "ylabel":"Mean absolute increment error " + \
                            f"({eval_feat.upper()})",
                    "alpha":.6,
                    "line_width":2,
                    "error_line_width":.5,
                    "error_every":4,
                    "fill_alpha":.25,
                    "yrange":(0,.002)
                    },
                use_stdev=False,
                )
        print(f"Generated {fig_dir.joinpath(p.stem+'_res.png')}")

    ## plot joint histograms
    for p,pt in filter(lambda p:"hist" in p[1][4], eval_pkls):
        ev = evaluators.EvalJointHist().from_pkl(p)
        tmp_ps = {
                **ev.attrs.get("plot_spec", {}),
                **hist_plot_info.get(pt[4], {}).get(pt[-1], {}),
                }
        for s in ["title", "xlabel", "ylabel", "cov_xlabel", "cov_ylabel"]:
            if s in tmp_ps.keys():
                tmp_ps[s] = tmp_ps[s].format(
                        eval_feat=pt[3],
                        model_name=pt[2],
                        domain=pt[1],
                        )
        if type(tmp_ps.get("cov_vmin"))==dict:
            tmp_ps["cov_vmin"] = tmp_ps["cov_vmin"][pt[3]]
        if type(tmp_ps.get("cov_vmax"))==dict:
            tmp_ps["cov_vmax"] = tmp_ps["cov_vmax"][pt[3]]
        ev.plot(
                show_ticks=True,
                plot_covariate=True,
                separate_covariate_axes=True,
                plot_diagonal=False,
                normalize_counts=False,
                fig_path=fig_dir.joinpath(p.stem+".png"),
                plot_spec=tmp_ps,
                )

    ## plot static combination matrices
    ## error label mapping
    _elm = {"bias":"Error Bias", "abs-err":"Mean Absolute Error"}
    for p,pt in filter(lambda p:p[1][4]=="static-combos", eval_pkls):
        ev = evaluators.EvalStatic().from_pkl(p)
        pred_feats = ev.attrs["model_config"]["feats"]["pred_feats"]
        _,data_source,model,eval_feat,eval_type,error_type = pt
        for ix,pf in enumerate(pred_feats):
            try:
                new_feat = eval_feat.split("-")[0] + "-" + pf.split("-")[1]
            except:
                continue
            new_path_base = [
                    "eval-grid",data_source,model,new_feat,eval_type,error_type
                    ]
            res_fig_path = fig_dir.joinpath(
                    "_".join(new_path_base+["res"]) + ".png")
            state_fig_path = fig_dir.joinpath(
                    "_".join(new_path_base+["state"]) + ".png")
            res_nonorm_fig_path = fig_dir.joinpath(
                    "_".join(new_path_base+["res-total"]) + ".png")
            state_nonorm_fig_path = fig_dir.joinpath(
                    "_".join(new_path_base+["state-total"]) + ".png")
            ev.plot(
                    state_or_res="res",
                    fig_path=res_nonorm_fig_path,
                    norm_by_counts=False,
                    plot_index=ix,
                    plot_spec={
                        "title":f"Total Increment {_elm[error_type]} per " + \
                            f"Static Combo\n{model} {new_feat} {data_source}",
                        "cmap":{
                            "bias":"seismic_r",
                            "abs-err":"gnuplot2"
                            }[error_type],
                        "vmin":{
                            "bias":[None,-40]["rsm" in new_feat],
                            "abs-err":0
                            }[error_type],
                        "vmax":{
                            "bias":[None,40]["rsm" in new_feat],
                            "abs-err":{
                                "rsm-10":800,
                                "rsm-40":400,
                                "rsm-100":400,
                                "rsm-200":400,
                                }[new_feat],
                            }[error_type],
                        }
                    )
            '''
            ev.plot(
                    state_or_res="state",
                    fig_path=state_nonorm_fig_path,
                    plot_index=ix,
                    norm_by_counts=False,
                    plot_spec={
                        "title":f"Total State {_elm[error_type]} per " + \
                            f"Static Combo\n{model} {new_feat} {data_source}",
                        #"vmax":.1,
                        "cmap":{
                            "bias":"seismic_r",
                            "abs-err":"gnuplot2",
                            }[error_type],
                        "vmin":{
                            "bias":[-30.,-.015]["rsm" in new_feat],
                            "abs-err":0
                            }[error_type],
                        "vmax":{
                            "bias":[30,.015]["rsm" in new_feat],
                            "abs-err":{
                                "rsm-10":.06,
                                "rsm-40":.03,
                                "rsm-100":.03,
                                "rsm-200":.03,
                                }[new_feat],
                            }[error_type],
                        }
                    )
            '''
            ev.plot(
                    state_or_res="res",
                    fig_path=res_fig_path,
                    plot_index=ix,
                    plot_spec={
                        "title":f"Increment {_elm[error_type]} per Static " + \
                                f"Combo\n{model} {new_feat} {data_source}",
                        "cmap":{
                            "bias":"seismic_r",
                            "abs-err":"gnuplot2"
                            }[error_type],
                        "vmin":{
                            "bias":[-5.,-.00015]["rsm" in new_feat],
                            "abs-err":0
                            }[error_type],
                        "vmax":{
                            "bias":[5.,.00015]["rsm" in new_feat],
                            "abs-err":{
                                "rsm-10":.0015,
                                "rsm-40":.0006,
                                "rsm-100":.0004,
                                "rsm-200":.0003,
                                "soilm-10":5.,
                                "soilm-40":5.,
                                "soilm-100":5.,
                                "soilm-200":5.,
                                }[new_feat],
                            }[error_type],
                        }
                    )
            ev.plot(
                    state_or_res="state",
                    fig_path=state_fig_path,
                    plot_index=ix,
                    plot_spec={
                        "title":f"State {_elm[error_type]} per Static " + \
                                f"Combo\n{model} {new_feat} {data_source}",
                        #"vmax":.1,
                        "cmap":{
                            "bias":"seismic_r",
                            "abs-err":"gnuplot2",
                            }[error_type],
                        "vmin":{
                            "bias":[-30.,-.015]["rsm" in new_feat],
                            "abs-err":0
                            }[error_type],
                        "vmax":{
                            "bias":[30,.015]["rsm" in new_feat],
                            "abs-err":{
                                "rsm-10":.06,
                                "rsm-40":.03,
                                "rsm-100":.03,
                                "rsm-200":.03,
                                "soilm-10":30.,
                                "soilm-40":30.,
                                "soilm-100":30.,
                                "soilm-200":30.,
                                }[new_feat],
                            }[error_type],
                        }
                    )

    ## plot 4-panel spatial statistics
    for p,pt in filter(lambda p:p[1][4]=="spatial-stats", eval_pkls):
        print(f"plotting from {pt}")
        ev = evaluators.EvalGridAxes().from_pkl(p)
        _,data_source,model,eval_feat,_,error_type = pt

        csps = {**common_spatial_plot_spec}
        if data_source in domain_plot_specs.keys():
            csps.update(domain_plot_specs[data_source])

        ## Gotta do this since indeces are concatenated along along axis 1
        ## with EvalGridAxis concatenation. Probably need to just keep a list.
        idx_zero_splits = list(np.where(
            ev.indeces[1:,0]-ev.indeces[:-1,0] < 0
            )[0] + 1)
        idx_zero_splits = [0] + idx_zero_splits + [ev.indeces.shape[0]]
        tile_slices = [slice(start_tile,end_tile) for start_tile,end_tile
                in zip(idx_zero_splits[:-1], idx_zero_splits[1:])]
        tiles_info = list(zip(
            ev.attrs["latlon"], ev.attrs["tiles"], tile_slices))

        ## iterate over requested spatial feature quad plot configurations
        for spt in plot_spatial_stats:
            ## Extract features needed for this plot type
            tmp_cfg = spatial_plot_info[spt]
            if tmp_cfg["error_type"] != error_type:
                continue
            fidxs = [ev.attrs["flabels"].index(c[:2])
                    for c in tmp_cfg["feats"]]
            feats = [
                    {"mean":ev.average[...,ix],
                        "stdev":ev.variance[...,ix]**(1/2)}[m]
                    for ix,m in zip(fidxs, (c[2] for c in tmp_cfg["feats"]))
                    ]
            feats = np.stack(feats, axis=-1)

            ## independently grid each of the tiles
            gridded_feats = []
            for ll,tl,slc in tiles_info:
                tmp_tile_shape = (*ll.shape[:2], len(tmp_cfg["feats"]))
                tmp_tile_feats = np.full(tmp_tile_shape, np.nan)
                ix = ev.indeces[slc]
                ## Batch and sequence axes should be size 1 (marginalized)
                tmp_tile_feats[ix[:,0], ix[:,1],:] = feats[0,slc,0,:]
                gridded_feats.append(tmp_tile_feats)

            ## plot each of the requested spatial plots
            xt = ev.attrs["domain"].mosaic_shape[-1]
            tile_arrays = [ev.attrs["latlon"], gridded_feats]
            for j,ta in enumerate(tile_arrays):
                rows = [ta[i:i + xt] for i in range(0,len(ta),xt)]
                tile_arrays[j] = np.concatenate(
                        [np.concatenate(x, axis=1) for x in rows], axis=0)
            latlon,feats = tile_arrays

            _spt = tmp_cfg.get("subplot_titles")
            has_subplot_titles = not _spt==None and len(_spt)==feats.shape[-1]
            plotting.geo_quad_plot(
                    data=[feats[...,i] for i in range(feats.shape[-1])],
                    flabels=[
                        " ".join(fl) for fl in tmp_cfg["feats"]]
                        if not has_subplot_titles
                        else [
                            s + "\n"+" ".join([model, data_source])
                            for s in tmp_cfg.get("subplot_titles")
                            ],
                    latitude=latlon[...,0],
                    longitude=latlon[...,1],
                    plot_spec={
                        **csps,
                        "title":f"{model} {eval_feat} {data_source} " + \
                                "Bulk Gridded Statistics",
                        **tmp_cfg.get("plot_spec", {}),
                        },
                    fig_path=fig_dir.joinpath("_".join(pt)+f"_{spt}.png"),
                    )
