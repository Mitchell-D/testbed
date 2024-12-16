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
    performance_dir = Path("data/performance/partial-new")

    ## Specify a subset of Evaluator pkls to plot based on their name fields:
    ## eval_{data_source}_{md.name}_{eval_feat}_{et}_{bias|abs-err}.pkl
    plot_data_sources = ["test"]
    plot_models_contain = [
            "accfnn",
            "accrnn",
            "acclstm",
            "lstm-rsm",
            ]
    plot_eval_feats = ["rsm-10", "rsm-40", "rsm-100"]
    plot_eval_type = [
            "horizon",
            "temporal",
            "static-combos",
            "hist-true-pred",
            "hist-saturation-error",
            "hist-state-increment",
            "hist-humidity-temp",
            "hist-infiltration",
            ]
    plot_error_type = ["bias", "abs-err"]

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

    ## group horizons by unique (data,model) so they can be plotted together
    horizons = {}
    for h,ht in filter(lambda p:p[1][4]=="horizon", eval_pkls):
        _,dataset,model = ht[:3]
        tmp_key = (dataset, model)
        if tmp_key not in horizons.keys():
            horizons[tmp_key] = [(h,ht)]
        else:
            horizons[tmp_key].append((h,ht))
    for k in horizons.keys():
        horizons[k] = sorted(horizons[k], key=lambda p:p[1][3])

    ## group temporal error by unique (data, model, error_type)
    temporal = {}
    for p,pt in filter(lambda p:p[1][4]=="temporal", eval_pkls):
        _,dataset,model,_,_,error_type = pt
        tmp_key = (dataset, model, error_type)
        if tmp_key not in temporal.keys():
            temporal[tmp_key] = [(p,pt)]
        else:
            temporal[tmp_key].append((p,pt))
    for k in temporal.keys():
        temporal[k] = sorted(temporal[k], key=lambda p:p[1][3])

    ## plot static combination matrices
    for p,pt in filter(lambda p:p[1][4]=="static-combos", eval_pkls):
        ev = EvalStatic()
        ev.from_pkl(p)
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
                        "vmax":.0005,
                        }
                    )
            ev.plot(
                    state_or_res="state",
                    fig_path=state_fig_path,
                    plot_index=ix,
                    plot_spec={
                        "title":f"{model} state {new_feat} " + \
                                f"{error_type} {data_source}",
                        "vmax":.05,
                        }
                    )
