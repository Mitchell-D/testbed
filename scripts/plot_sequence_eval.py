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
from testbed.list_feats import units_names_mapping
from testbed import plotting

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/testbed")
    fig_dir = proj_root_dir.joinpath("figures/performance-partial")
    #performance_dir = proj_root_dir.joinpath("data/eval_sequence_pkls")
    performance_dir = proj_root_dir.joinpath("data/eval_rr-rmb_pkls")

    ## Specify a subset of Evaluator pkls to plot based on their name fields:
    ## eval_{data_source}_{md.name}_{eval_feat}_{et}_{na|bias|abs-err}.pkl

    ##  datasets to evaluate (2nd name field)
    plot_data_sources = [
            "test",
            #"thsub-freeze",
            #"thsub-hot",
            ]
    ## models to evaluate (3rd name field)
    plot_models_named = [
            #"lstm-rsm-9",
            ## Basic spread of "best" models
            #"accfnn-rsm-8", "lstm-20", "lstm-rsm-9",
            #"accrnn-rsm-2", "acclstm-rsm-4",

            ## initial accfnn-rsm models w/o loss func increment norming
            #"accfnn-rsm-0", "accfnn-rsm-1", "accfnn-rsm-2", "accfnn-rsm-3",
            #"accfnn-rsm-4", "accfnn-rsm-5", "accfnn-rsm-6", "accfnn-rsm-7",
            #"accfnn-rsm-8", "accfnn-rsm-9",

            ## initial lstm models predicting soilm at 4 levels + snow
            #"lstm-1", "lstm-2", "lstm-3", "lstm-4", "lstm-8", "lstm-9",
            #"lstm-10", "lstm-11", "lstm-12", "lstm-13", "lstm-14", "lstm-15",
            #"lstm-16",
            ## coarsened models... lstm-18 is bizzarely good, but ignore now?
            #"lstm-17", "lstm-18", "lstm-19",
            #"lstm-20", "lstm-21",
            #"lstm-22", "lstm-23", "lstm-24", "lstm-25", "lstm-26", "lstm-27",

            ## initial lstm-rsm models without loss function increment norming
            #"lstm-rsm-0", "lstm-rsm-2", "lstm-rsm-3", "lstm-rsm-5",
            #"lstm-rsm-6", "lstm-rsm-9", "lstm-rsm-10", "lstm-rsm-11",
            #"lstm-rsm-12", "lstm-rsm-19", "lstm-rsm-20",

            ## initial acclstm-rsm models w/o loss func increment norming
            #"acclstm-rsm-0", "acclstm-rsm-1", "acclstm-rsm-2",
            #"acclstm-rsm-3", "acclstm-rsm-4", "acclstm-rsm-5",
            #"acclstm-rsm-6", "acclstm-rsm-7", "acclstm-rsm-8",
            #"acclstm-rsm-9", "acclstm-rsm-10", "acclstm-rsm-11",
            #"acclstm-rsm-12",

            ## acclstm-rsm-9 variations w/ inc norming in loss function
            #"acclstm-rsm-9", "acclstm-rsm-4", "acclstm-rsm-14",
            #"acclstm-rsm-15", "acclstm-rsm-16", "acclstm-rsm-17",
            #"acclstm-rsm-18", "acclstm-rsm-19", "acclstm-rsm-20",

            ## lstm-rsm-9 variations including increment norm in loss function
            #"lstm-rsm-9", "lstm-rsm-21", "lstm-rsm-22", "lstm-rsm-23",
            #"lstm-rsm-24", "lstm-rsm-26", "lstm-rsm-27", "lstm-rsm-28",
            #"lstm-rsm-29", "lstm-rsm-30", "lstm-rsm-31",

            ## acclstm-rsm-4 variations w/o norming in loss function
            #"acclstm-rsm-4", "acclstm-rsm-9", "acclstm-rsm-21",
            #"acclstm-rsm-22", "acclstm-rsm-23", "acclstm-rsm-25",
            #"acclstm-rsm-26", "acclstm-rsm-27", "acclstm-rsm-28",
            #"acclstm-rsm-29", "acclstm-rsm-30", "acclstm-rsm-31",
            #"acclstm-rsm-32", "acclstm-rsm-33",

            ## RNNs without intermediate layer propagation
            #"accrnn-rsm-9", "accrnn-rsm-11",

            ## Feature variations on lstm-rsm-9
            #"lstm-rsm-34", "lstm-rsm-35", "lstm-rsm-36", "lstm-rsm-37",
            #"lstm-rsm-38", "lstm-rsm-39", "lstm-rsm-40", "lstm-rsm-41",
            #"lstm-rsm-42", "lstm-rsm-43", "lstm-rsm-44", "lstm-rsm-45",

            ## residual magnitude bias variations on lstm-rsm-9
            "lstm-rsm-51", "lstm-rsm-50", "lstm-rsm-48", "lstm-rsm-49",

            ## residual ratio variations on lstm-rsm-9
            "lstm-rsm-53", "lstm-rsm-54", "lstm-rsm-55",

            ## mean squared error loss variation on lstm-rsm-9
            "lstm-rsm-56"
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
            #"efficiency",
            #"hist-infiltration",
            ]
    ## Types of error to include (6th name field)
    plot_error_type = [
            "na",
            "bias",
            "abs-err"
            ]

    ## Since efficiency bar plots group multiple models, a label must be
    ## specified that summarizes the grouping
    #efficiency_plot_group_label = "initial-best"
    #efficiency_plot_group_title = "Best Models Per Category"
    #efficiency_plot_group_label = "initial-accfnn-rsm"
    #efficiency_plot_group_title = "Initial Runs of accfnn-rsm"
    #efficiency_plot_group_label = "initial-lstm-soilm"
    #efficiency_plot_group_title = "Initial Runs of lstm (predicting soilm)"
    #efficiency_plot_group_label = "initial-lstm-rsm"
    #efficiency_plot_group_title = "Initial Runs of lstm-rsm"
    #efficiency_plot_group_label = "initial-acclstm-rsm"
    #efficiency_plot_group_title = "Initial Runs of acclstm-rsm"
    #efficiency_plot_group_label = "variations-acclstm-rsm-9"
    #efficiency_plot_group_title = "Model Variations on acclstm-rsm-9"
    #efficiency_plot_group_label = "variations-lstm-rsm-9"
    #efficiency_plot_group_title = "Model Variations on lstm-rsm-9"
    #efficiency_plot_group_label = "variations-acclstm-rsm-4"
    #efficiency_plot_group_title = "Model Variations on acclstm-rsm-4"
    #efficiency_plot_group_label = "variations-feat-lstm-rsm-9"
    #efficiency_plot_group_title = "Feature Variations on lstm-rsm-9"
    #efficiency_plot_group_label = "variations-rmb-lstm-rsm-9"
    #efficiency_plot_group_title = "Loss Magnitude Bias Variations on lstm-rsm-9"
    #efficiency_plot_group_label = "variations-rr-lstm-rsm-9"
    #efficiency_plot_group_title = "Loss Increment Ratio Variations on lstm-rsm-9"
    #efficiency_plot_group_label = "variations-mse-lstm-rsm-9"
    #efficiency_plot_group_title = "Mean Squared Error Variation on lstm-rsm-9"

    ## ---- ( end evaluator pkl selection config ) ----

    hist_plot_specs = {
            "hist-true-pred":{
                "na":{
                    "title":"Increment RSM True/Pred Joint Histogram " + \
                            "\n{model_name} {eval_feat} {domain}",
                    "xlabel":"Predicted change in RSM (%)",
                    "ylabel":"Actual change in RSM (%)",
                    "aspect":1,
                    "norm":"log",
                    "text_size":14,
                    }
                },
            "hist-saturation-error":{
                "na":{
                    "title":"Increment RSM Error Bias wrt Saturation " + \
                            "Percentage\n{model_name} {eval_feat} {domain}",
                    "xlabel":"Absolute error in hourly change in RSM",
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
                    "xlabel":"Hourly increment change in {eval_feat} (%)",
                    "ylabel":"Soil saturation in RSM (%)",
                    "norm":"log",
                    "cov_norm":"log",
                    #"cov_vmin":0,
                    "cov_vmax":{
                        "rsm-10":.05,
                        "rsm-40":.01,
                        "rsm-100":.005,
                        },
                    "cb_size":.8,
                    "cov_cmap":"jet",
                    "aspect":1,
                    "fig_size":(18,8),
                    },
                "bias":{
                    "title":"Hourly RSM Bias wrt Saturation and Increment " + \
                            "Change in RSM\n{model_name} {eval_feat} {domain}",
                    "xlabel":"Hourly increment change in {eval_feat} (%)",
                    "ylabel":"Soil saturation in RSM (%)",
                    "norm":"log",
                    "cov_vmin":{
                        "rsm-10":-.05,
                        "rsm-40":-.01,
                        "rsm-100":-.005,
                        },
                    "cov_vmax":{
                        "rsm-10":.05,
                        "rsm-40":.01,
                        "rsm-100":.005,
                        },
                    "cb_size":.8,
                    "cov_cmap":"seismic_r",
                    "cov_norm":"symlog",
                    "aspect":1,
                    "fig_size":(18,8),
                    },
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
                    "cov_title":"Absolute Error in Histogram Bins",
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
                    "cov_title":"Error Bias in Histogram Bins",
                    }
                },
            "hist-infiltration":{
                    "na":{
                        "title":"Infiltration Validation and Mean Layer " + \
                                "Water Content (kg/m^2) " + \
                                "{model_name} {eval_feat} {domain} ",
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
            and any(s==pt[2] for s in plot_models_named)
            and pt[3] in plot_eval_feats
            and pt[4] in plot_eval_type
            and (len(pt)==5 or pt[5] in plot_error_type)
            ]

    eff_pkls = list(filter(lambda p:"efficiency" in p[1][4], eval_pkls))
    eff_metrics = {
            "mae":"Mean Absolute Error",
            "mse":"Root Mean Squared Error",
            "cc":"Pearson Correlation Coefficient",
            "kge":"Kling-Gupta Efficiency",
            "nse":"Nash-Sutcliffe Efficiency",
            "nnse":"Normalized Nash-Sutcliffe Efficiency",
            }
    s_metric_ylims = {
            "mae":(0,.1),
            #"mse":(0,.015),
            "mse":(0,.1),
            "cc":(0,1.2),
            "kge":(-5,1),
            "nse":(-1e9,1e5),
            "nnse":(0, 1),
            }
    r_metric_ylims = {
            "mae":(0,.0015),
            #"mse":(0,1e-5),
            "mse":(0,3e-3),
            "cc":(0,1.2),
            "kge":(-5,1),
            "nse":(-5e5,1e5),
            "nnse":(0, 1),
            }
    if len(eff_pkls):
        eff_evs = [(evaluators.EvalEfficiency().from_pkl(p),pt)
                for p,pt in eff_pkls]
        _,datasets,models,eval_feats,_,_ = zip(*[pt for _,pt in eff_pkls])
        dataset = datasets[0]
        for tmp_metric in eff_metrics.keys():
            unq_models = list(set(pt[2] for _,pt in eff_evs))
            unq_feats = list(set(pt[3] for _,pt in eff_evs))
            ## Make dicts for state and residual data of this metric type
            s_eff_dict = {
                    m:{f:[None, None] for f in unq_feats}
                    for m in unq_models}
            r_eff_dict = {
                    m:{f:[None, None] for f in unq_feats}
                    for m in unq_models}
            exp = .5 if tmp_metric=="mse" else 1
            for ev,pt in eff_evs:
                _,_,tmp_model,tmp_feat,_,_ = pt
                s_eff_dict[tmp_model][tmp_feat][0] = \
                        ev.get_mean("s", tmp_metric)**exp
                s_eff_dict[tmp_model][tmp_feat][1] = \
                        ev.get_var("s", tmp_metric)**exp#**(1/2)
                r_eff_dict[tmp_model][tmp_feat][0] = \
                        ev.get_mean("r", tmp_metric)**exp
                r_eff_dict[tmp_model][tmp_feat][1] = \
                        ev.get_var("r", tmp_metric)**exp#**(1/2)
            epgl = efficiency_plot_group_label
            epgt = efficiency_plot_group_title
            state_path = fig_dir.joinpath(
                    f"eval_{dataset}_efficiency_{epgl}_{tmp_metric}_state.png")
            res_path = fig_dir.joinpath(
                    f"eval_{dataset}_efficiency_{epgl}_{tmp_metric}_res.png")
            plotting.plot_nested_bars(
                    data_dict=s_eff_dict,
                    labels={k:v[1] for k,v in units_names_mapping.items()},
                    plot_error_bars=True,
                    bar_order=["rsm-10","rsm-40","rsm-100"],
                    group_order=plot_models_named,
                    plot_spec={
                        "title":f"{epgt} State {eff_metrics[tmp_metric]}",
                        "ylabel":eff_metrics[tmp_metric],
                        "xlabel":"Model Instance",
                        "ylim":s_metric_ylims[tmp_metric],
                        "bar_spacing":.5,
                        "figsize":(24,12),
                        "xtick_rotation":30,
                        "title_fontsize":32,
                        "xtick_fontsize":24,
                        "legend_fontsize":20,
                        "label_fontsize":24,
                        },
                    bar_colors=["xkcd:forest green",
                        "xkcd:bright blue", "xkcd:light brown"],
                    fig_path=state_path,
                    )
            print(f"Generated {state_path}")
            plotting.plot_nested_bars(
                    data_dict=r_eff_dict,
                    labels={k:v[1] for k,v in units_names_mapping.items()},
                    plot_error_bars=True,
                    bar_order=["rsm-10","rsm-40","rsm-100"],
                    group_order=plot_models_named,
                    plot_spec={
                        "title":f"{epgt} Inc. {eff_metrics[tmp_metric]}",
                        "ylabel":eff_metrics[tmp_metric],
                        "xlabel":"Model Instance",
                        "ylim":r_metric_ylims[tmp_metric],
                        "bar_spacing":.5,
                        "figsize":(24,12),
                        "xtick_rotation":30,
                        "title_fontsize":32,
                        "xtick_fontsize":24,
                        "legend_fontsize":20,
                        "label_fontsize":24,
                        },
                    bar_colors=["xkcd:forest green",
                        "xkcd:bright blue", "xkcd:light brown"],
                    fig_path=res_path,
                    )
            print(f"Generated {res_path}")

    for p,pt in filter(lambda p:p[1][4]=="horizon", eval_pkls):
        print(f"Loading horizon pkl {p.name}")
        ev = evaluators.EvalHorizon().from_pkl(p)
        _,data_source,model,eval_feat,eval_type,error_type = pt
        feat_labels = [
                "-".join((eval_feat, f.split("-")[-1]))
                for f in ev.attrs["model_config"]["feats"]["pred_feats"]
                if (f != "weasd") and (f != "soilm-200")
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
                    "ylabel":f"MAE in State ({eval_feat.upper()})",
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
                    "ylabel":f"MAE in Increment ({eval_feat.upper()})",
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
                        "title":f"Increment {_elm[error_type]} per Static " + \
                                f"Combo\n{model} {new_feat} {data_source}",
                        "cmap":{
                            "bias":"seismic_r",
                            "abs-err":"gnuplot2"
                            }[error_type],
                        "vmin":{
                            "bias":-.00015,
                            "abs-err":0
                            }[error_type],
                        "vmax":{
                            "bias":.00015,
                            "abs-err":{
                                "rsm-10":.0015,
                                "rsm-40":.0006,
                                "rsm-100":.0004,
                                "rsm-200":.0002,
                                }[new_feat],
                            }[error_type],
                        },
                    )
            print(f"Generated {res_fig_path.name}")
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
                            "bias":-.015,
                            "abs-err":0
                            }[error_type],
                        "vmax":{
                            "bias":.015,
                            "abs-err":{
                                "rsm-10":.06,
                                "rsm-40":.03,
                                "rsm-100":.03,
                                "rsm-200":.03,
                                }[new_feat],
                            }[error_type],
                        },
                    )
            print(f"Generated {state_fig_path.name}")
