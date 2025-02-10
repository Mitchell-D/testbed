"""
Script for running plotting methods on pkls associated with Evaluator objects,
which were probably produced by testbed.eval_sequences.eval_model_on_sequences
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
from pprint import pprint

from testbed import evaluators

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/testbed")
    fig_dir = proj_root_dir.joinpath("figures/performance-partial")
    performance_dir = proj_root_dir.joinpath("data/eval_sequence_pkls")

    ## Specify a subset of Evaluator pkls to plot based on their name fields:
    ## eval_{data_source}_{md.name}_{eval_feat}_{et}_{na|bias|abs-err}.pkl

    ##  datasets to evaluate (2nd name field)
    plot_data_sources = ["test"]
    ## models to evaluate (3rd name field)
    plot_models_contain = [
            #"accfnn",
            #"accrnn",
            #"lstm-rsm",
            #"acclstm-rsm-1",
            #"lstm-rsm-9","accfnn-rsm-8",#"accrnn-rsm-2",
            #"accfnn-rsm-5", "lstm-20",
            #"acclstm-rsm-4",
            "lstm-rsm-0", "lstm-rsm-2", "lstm-rsm-3", "lstm-rsm-5",
            "lstm-rsm-6", "lstm-rsm-7", "lstm-rsm-8", "lstm-rsm-9",
            "lstm-rsm-10", "lstm-rsm-11", "lstm-rsm-12", "lstm-rsm-19",
            "lstm-rsm-20",
            ]
    ## evlauated features to include (4th name field)
    plot_eval_feats = [
            "rsm",
            "rsm-10",
            "rsm-40",
            "rsm-100",
            "soilm-10"
            ]
    ## Evaluator instance types to include (5th name field)
    plot_eval_type = [
            "horizon",
            "temporal",
            "static-combos",
            "hist-true-pred",
            "hist-saturation-error",
            "hist-state-increment",
            "hist-humidity-temp",
            #"hist-infiltration",
            ]
    ## Types of error to include (6th name field)
    plot_error_type = [
            "na",
            "bias",
            "abs-err"
            ]

    ## ---- ( end evaluator pkl selection config ) ----

    hist_plot_specs = {
            "hist-true-pred":{
                "na":{
                    "title":"{model_name} {eval_feat} validation " + \
                            "joint histogram",
                    "xlabel":"Predicted RSM (%)",
                    "ylabel":"Actual RSM (%)",
                    "aspect":1,
                    "norm":"log",
                    }
                },
            "hist-saturation-error":{
                "na":{
                    "title":"{model_name} {eval_feat} error bias wrt " + \
                            "saturation percentage",
                    "xlabel":"Hourly absolute error in RSM",
                    "ylabel":"Relative soil moisture ({eval_feat})",
                    "aspect":1,
                    "norm":"log",
                    },
                },
            "hist-state-increment":{
                "abs-err":{
                    "title":"Mean hourly absolute error wrt saturation " + \
                            "and increment percent change in RSM",
                    "xlabel":"Hourly increment change in {eval_feat} (%)",
                    "ylabel":"Soil saturation in RSM (%)",
                    "norm":"log",
                    "cov_vmin":0.,
                    "cov_vmax":.005,
                    "cov_cmap":"jet",
                    "aspect":1,
                    "fig_size":(18,8),
                    },
                "bias":{
                    "title":"Mean hourly error bias wrt saturation and " + \
                            "increment percent change in {eval_feat}",
                    "xlabel":"Hourly increment change in RSM (%)",
                    "ylabel":"Soil saturation in RSM (%)",
                    "norm":"log",
                    "cov_vmin":-.05,
                    "cov_vmax":.05,
                    "cov_cmap":"seismic",
                    "cov_norm":"linear",
                    "aspect":1,
                    "fig_size":(18,8),
                    },
                },
            "hist-humidity-temp":{
                "abs-err":{
                    "title":"{eval_feat} absolute error wrt humidity and " + \
                            "temp distribution",
                    "norm":"log",
                    "xlabel":"Temperature (K)",
                    "ylabel":"Absolute humidity (kg/kg)",
                    "norm":"log",
                    "cov_vmin":0.,
                    "cov_vmax":1.2e-3,
                    "cov_norm":"linear",
                    "cov_cmap":"jet",
                    "aspect":1,
                    "fig_size":(18,8),
                    },
                "bias":{
                    "title":"{eval_feat} error bias wrt humidity and " + \
                            "temp distribution",
                    "norm":"log",
                    "xlabel":"Temperature (K)",
                    "ylabel":"Absolute humidity (kg/kg)",
                    "cov_vmin":-1.2e-3,
                    "cov_vmax":1.2e-3,
                    "cov_norm":"linear",
                    "cov_cmap":"seismic",
                    "aspect":1,
                    "fig_size":(18,8),
                    }
                },
            "hist-infiltration":{
                    "na":{
                        "title":"{model_name} infiltration validation and " + \
                                "mean layer water content (kg/m^2)",
                        "norm":"log",
                        "vmax":100,
                        "vmin":0,
                        }
                    },
            }

    eval_pkls = [
            (p,pt) for p,pt in map(
                lambda f:(f,f.stem.split("_")),
                sorted(performance_dir.iterdir()))
            if pt[0] == "eval"
            and pt[1] in plot_data_sources
            and any(s in pt[2] for s in plot_models_contain)
            and pt[3] in plot_eval_feats
            and pt[4] in plot_eval_type
            and (len(pt)==5 or pt[5] in plot_error_type)
            ]

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
                    "title":"Mean Absolute State Error wrt Forecast Hour " + \
                            f"({model})",
                    "xlabel":"Forecast hour",
                    "ylabel":"Mean absolute state error ({eval_feat.upper()})",
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
                    "title":"Mean Increment Error wrt Forecast Hour " + \
                            f"({model})",
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

    for p,pt in filter(lambda p:"hist" in p[1][4], eval_pkls):
        ev = evaluators.EvalJointHist().from_pkl(p)
        tmp_ps = {
                **ev.attrs.get("plot_spec", {}),
                **hist_plot_specs.get(pt[4], {}).get(pt[-1], {}),
                }
        for s in ["title", "xlabel", "ylabel", "cov_xlabel", "cov_ylabel"]:
            if s in tmp_ps.keys():
                tmp_ps[s] = tmp_ps[s].format(
                        eval_feat=pt[3],
                        model_name=pt[2],
                        )
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
                    "eval", data_source, model, new_feat,
                    eval_type, error_type]
            res_fig_path = fig_dir.joinpath(
                    "_".join(new_path_base + ["res"]) + ".png")
            state_fig_path = fig_dir.joinpath(
                    "_".join(new_path_base + ["state"]) + ".png")
            ev.plot(
                    state_or_res="res",
                    fig_path=res_fig_path,
                    plot_index=ix,
                    plot_spec={
                        "title":f"{model} increment {new_feat} " + \
                                f"{error_type} {data_source}",
                        "vmax":.005,
                        }
                    )
        print(f"Generated {res_fig_path.name}")
            ev.plot(
                    state_or_res="state",
                    fig_path=state_fig_path,
                    plot_index=ix,
                    plot_spec={
                        "title":f"{model} state {new_feat} " + \
                                f"{error_type} {data_source}",
                        "vmax":.1,
                        }
                    )
        print(f"Generated {state_fig_path.name}")
