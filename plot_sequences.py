"""
Script for running plotting methods on pkls associated with Evaluator objects,
which were probably produced by eval_sequences.eval_model
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
from pprint import pprint

from evaluators import EvalHorizon,EvalTemporal,EvalStatic,EvalJointHist

if __name__=="__main__":
    eval_dir = Path(f"data/performance")
    fig_dir = Path("figures/performance-partial")
    sequence_h5_dir = Path("data/sequences/")
    performance_dir = Path("data/performance/partial-new-2")

    ## Specify a subset of Evaluator pkls to plot based on their name fields:
    ## eval_{data_source}_{md.name}_{eval_feat}_{et}_{bias|abs-err}.pkl
    plot_data_sources = ["test"]
    plot_models_contain = [
            #"accfnn",
            #"accrnn",
            #"acclstm-rsm-1",
            "lstm-20",
            #"lstm-rsm",
            ]
    ## evlauated features to include.
    plot_eval_feats = [
            #"rsm",
            "rsm-10",
            #"rsm-40",
            #"rsm-100",
            ]
    ## Evaluator instance types to include
    plot_eval_type = [
            #"horizon",
            #"temporal",
            #"static-combos",
            "hist-true-pred",
            "hist-saturation-error",
            "hist-state-increment",
            "hist-humidity-temp",
            #"hist-infiltration",
            ]
    plot_error_type = [
            "na",
            "bias",
            "abs-err"
            ]

    ## ---- ( end evaluator pkl selection config ) ----

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
        ev = EvalHorizon().from_pkl(p)
        _,data_source,model,eval_feat,eval_type,error_type = pt
        feat_labels = [
                "-".join((eval_feat, f.split("-")[-1]))
                for f in ev.attrs["model_config"]["feats"]["pred_feats"]
                ]
        ev.plot(
                fig_path=fig_dir.joinpath(p.stem+".png"),
                feat_labels=["State Error in "+l for l in feat_labels],
                state_or_res="state",
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
                    "yrange":(0,.1)
                    },
                use_stdev=False,
                )
        ev.plot(
                fig_path=fig_dir.joinpath(p.stem+".png"),
                feat_labels=["Increment Error in "+l for l in feat_labels],
                state_or_res="res",
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
                    "yrange":(0,.1)
                    },
                use_stdev=False,
                )

    for p,pt in filter(lambda p:"hist" in p[1][4], eval_pkls):
        ev = EvalJointHist().from_pkl(p)
        ev.plot(
                show_ticks=True,
                plot_covariate_contours=True,
                plot_diagonal=False,
                normalize_counts=False,
                fig_path=fig_dir.joinpath(p.stem+".png"),
                plot_spec={
                    **ev.attrs.get("plot_spec", {}),
                    "norm":"log",
                    },
                )

    ## plot static combination matrices
    for p,pt in filter(lambda p:p[1][4]=="static-combos", eval_pkls):
        ev = EvalStatic().from_pkl(p)
        pred_feats = ev.attrs["model_config"]["feats"]["pred_feats"]
        _,data_source,model,eval_feat,_,error_type = pt
        for ix,pf in enumerate(pred_feats):
            new_feat = eval_feat.split("-")[0] + "-" + pf.split("-")[1]
            res_fig_path = fig_dir.joinpath(
                    p.stem.replace(eval_feat, new_feat) + "_res.png")
            state_fig_path = fig_dir.joinpath(
                    p.stem.replace(eval_feat, new_feat) + "_state.png")
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
