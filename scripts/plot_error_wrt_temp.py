""" """
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
    fig_dir = proj_root.joinpath("figures/performance-partial")
    #fig_dir = proj_root.joinpath("figures/eval_grid_slope-tiles")
    #eval_pkl_dir = proj_root.joinpath("data/eval_grid_pkls")
    eval_pkl_dir = proj_root.joinpath("data/eval_rr-rmb_pkls")

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
            #"full",
            "test",
            #"lt-north-michigan",
            #"lt-high-plains",
            #"lt-cascades",
            #"lt-fourcorners",
            #"lt-miss_alluvial",
            #"lt-atlanta",
            ]
    ## substrings of model names to plot (3rd field of file name)
    plot_models = [
            #"lstm-rsm-51",
            "lstm-rsm-9",
            #"lstm-rsm-50",
            #"lstm-rsm-48",
            #"lstm-rsm-49",
            #"lstm-rsm-53",
            #"lstm-rsm-54",
            #"lstm-rsm-55",
            #"lstm-rsm-56",
            ]
    ## evlauated features to plot (4th field of file name)
    plot_eval_feats = [
            #"rsm",
            #"rsm-10",
            "rsm-40",
            #"rsm-100",
            #"soilm"
            #"soilm-10"
            #"soilm-40"
            #"soilm-100"
            #"soilm-200"
            ]
    ## Evaluator instance types to include (5th field of file name)
    plot_eval_type = [
            #"horizon",
            #"temporal",
            #"static-combos",
            #"hist-true-pred",
            #"hist-saturation-error",
            "hist-state-increment",
            #"hist-humidity-temp",
            #"hist-infiltration",
            #"spatial-stats",
            ]
    ## error types of evaluators to plot (6th field of file name)
    plot_error_type = [
            "na",
            "bias",
            "abs-err",
            ]

    ## subset available pkls according to selection string configuration
    eval_pkls = [
            (p,pt) for p,pt in map(
                lambda f:(f,f.stem.split("_")),
                sorted(eval_pkl_dir.iterdir()))
            if pt[0] == "eval"
            and pt[1] in plot_domains
            and any(s==pt[2] for s in plot_models)
            and pt[3] in plot_eval_feats
            and pt[4] in plot_eval_type
            and (len(pt)==5 or pt[5] in plot_error_type)
            and "PARTIAL" not in pt
            ]

    print(f"Found {len(eval_pkls)} matching eval pkls:")
    print("\n".join([p[0].name for p in eval_pkls]))

    state_increment_hists = {}
    ## plot error wrt forecast horizon
    for p,pt in filter(lambda p:p[1][4]=="hist-state-increment", eval_pkls):
        ev = evaluators.EvalJointHist().from_pkl(p)
        _,data_source,model,eval_feat,eval_type,error_type = pt
        if eval_feat not in state_increment_hists:
            state_increment_hists[eval_feat] = []
        state_increment_hists[eval_feat].append({
            "model":model,
            "hparams":(ev._ax1_args[-1], ev._ax2_args[-1]),
            "use_abs_err":ev.absolute_error,
            "cov_feat":ev._cov_feat,
            "counts":ev._counts,
            "cov_sum":ev._cov_sum
            })

    feat_maps = {
            "rsm-10":"0-10cm RSM",
            "rsm-40":"10-40cm RSM",
            "rsm-100":"40-100cm RSM",
            }
    model_maps = {
            "lstm-rsm-9":"$\gamma=10$",
            "lstm-rsm-51":"$\gamma=0$",
            "lstm-rsm-50":"$\gamma=50$",
            "lstm-rsm-48":"$\gamma=100$",
            "lstm-rsm-49":"$\gamma=500$",
            "lstm-rsm-53":"$\gamma=10$",
            "lstm-rsm-54":"$\gamma=10$",
            "lstm-rsm-55":"$\gamma=10$",
            "lstm-rsm-56":"MSE Loss",
            }
    base_plot_spec = {
            "zero_axis":True,
            "zero_yaxis":True,
            "fig_size":(12,8),
            "legend_ncols":1,
            "label_size":18,
            "title_size":24,
            "legend_font_size":18,
            }
    for  fk in state_increment_hists.keys():
        tmp_hists = state_increment_hists[fk]

        ## absolute error
        domain,codomain,labels = map(list,zip(*sorted([(
            np.linspace(*h["hparams"][1]),
            np.sum(h["cov_sum"], axis=0) / np.sum(h["counts"], axis=0),
            h["model"]
            ) for h in tmp_hists if h["use_abs_err"]
            ], key=lambda t:plot_models.index(t[-1])
            )))
        fig_path = fig_dir.joinpath("_".join([
            "eval", data_source, fk, "increment-error-1d", "abs-err"
            ]) + ".png")
        for i,l in enumerate(labels):
            if l in model_maps.keys():
                labels[i] += f" {model_maps[l]}"
        plotting.plot_lines(
                domain=domain,
                ylines=codomain,
                fig_path=fig_path,
                labels=labels,
                multi_domain=True,
                plot_spec={
                    "title":f"Mean Absolute Error in {feat_maps[fk]} wrt " + \
                            "Noah-LSM Increment Change",
                    "xlabel":"True Hourly Increment Change in RSM",
                    "ylabel":"Average RSM Error",
                    "yrange":{
                        "rsm-10":(0.,.03),
                        "rsm-40":(0.,.025),
                        "rsm-100":(0.,.01),
                        }[fk],
                    **base_plot_spec,
                    },
                )
        print(f"Generated {fig_path.as_posix()}")

        ## bias
        domain,codomain,labels = map(list,zip(*sorted([(
            np.linspace(*h["hparams"][1]),
            np.sum(h["cov_sum"], axis=0) / np.sum(h["counts"], axis=0),
            h["model"]
            ) for h in tmp_hists if not h["use_abs_err"]
            ], key=lambda t:plot_models.index(t[-1])
            )))
        fig_path = fig_dir.joinpath("_".join([
            "eval", data_source, fk, "increment-error-1d", "bias"
            ]) + ".png")
        for i,l in enumerate(labels):
            if l in model_maps.keys():
                labels[i] += f" {model_maps[l]}"
        plotting.plot_lines(
                domain=domain,
                ylines=codomain,
                labels=labels,
                fig_path=fig_path,
                multi_domain=True,
                plot_spec={
                    "title":f"Mean Bias in {feat_maps[fk]} wrt " + \
                            "Noah-LSM Increment Change",
                    "xlabel":"True Hourly Increment Change in RSM",
                    "ylabel":"Average Increment RSM Bias",
                    "yrange":{
                        "rsm-10":(-.03,.03),
                        "rsm-40":(-.025,.025),
                        "rsm-100":(-.01,.01),
                        }[fk],
                    **base_plot_spec,
                    },
                )
        print(f"Generated {fig_path.as_posix()}")


        ## histograms
        domain,codomain,labels = zip(*[(
            np.linspace(*h["hparams"][1]),
            np.sum(h["counts"], axis=0),
            h["model"]
            ) for h in tmp_hists if not h["use_abs_err"]
            ])
        fig_path = fig_dir.joinpath("_".join([
            "eval", data_source, fk, "increment-error-1d", "hist"
            ]) + ".png")
        plotting.plot_lines(
                domain=domain[0],
                ylines=[codomain[0]],
                #labels=labels,
                fig_path=fig_path,
                multi_domain=False,
                plot_spec={
                    "title":"Histogram of Increment Change in " + \
                            f"{feat_maps[fk]} from Noah-LSM",
                    "xlabel":"True Hourly Increment Change in RSM",
                    "ylabel":"Average Increment RSM Bias",
                    "xrange":{
                        "rsm-10":(-.04, .04),
                        "rsm-40":(-.015, .02),
                        "rsm-100":(-.005, .01),
                        }[fk],
                    #"yrange":(0,400000)
                    **base_plot_spec,
                    },
                )
        print(f"Generated {fig_path.as_posix()}")

        ## focused absolute error
        domain,codomain,labels = map(list,zip(*sorted([(
            np.linspace(*h["hparams"][1]),
            np.sum(h["cov_sum"], axis=0) / np.sum(h["counts"], axis=0),
            h["model"]
            ) for h in tmp_hists if h["use_abs_err"]
            ], key=lambda t:plot_models.index(t[-1])
            )))
        fig_path = fig_dir.joinpath("_".join([
            "eval", data_source, fk, "increment-error-1d-focus", "abs-err"
            ]) + ".png")
        for i,l in enumerate(labels):
            if l in model_maps.keys():
                labels[i] += f" {model_maps[l]}"
        plotting.plot_lines(
                domain=domain,
                ylines=codomain,
                fig_path=fig_path,
                labels=labels,
                multi_domain=True,
                plot_spec={
                    "title":f"Mean Absolute Error in {feat_maps[fk]} wrt " + \
                            "Noah-LSM Increment Change (focused)",
                    "xlabel":"True Hourly Increment Change in RSM",
                    "ylabel":"Average RSM Error",
                    "xrange":{
                        "rsm-10":(-.04, .04),
                        "rsm-40":(-.015, .02),
                        "rsm-100":(-.005, .01),
                        }[fk],
                    "yrange":{
                        "rsm-10":(0., .004),
                        "rsm-40":(0., .0025),
                        "rsm-100":(0., .002),
                        }[fk],
                    **base_plot_spec,
                    },
                )
        print(f"Generated {fig_path.as_posix()}")

        ## focused bias
        domain,codomain,labels = map(list,zip(*sorted([(
            np.linspace(*h["hparams"][1]),
            np.sum(h["cov_sum"], axis=0) / np.sum(h["counts"], axis=0),
            h["model"]
            ) for h in tmp_hists if not h["use_abs_err"]
            ], key=lambda t:plot_models.index(t[-1])
            )))
        fig_path = fig_dir.joinpath("_".join([
            "eval", data_source, fk, "increment-error-1d-focus", "bias"
            ]) + ".png")
        for i,l in enumerate(labels):
            if l in model_maps.keys():
                labels[i] += f" {model_maps[l]}"
        plotting.plot_lines(
                domain=domain,
                ylines=codomain,
                labels=labels,
                fig_path=fig_path,
                multi_domain=True,
                plot_spec={
                    "title":f"Mean Bias in {feat_maps[fk]} wrt " + \
                            "Noah-LSM Increment Change (focused)",
                    "xlabel":"True Hourly Increment Change in RSM",
                    "ylabel":"Average Increment RSM Bias",
                    "xrange":{
                        "rsm-10":(-.04, .04),
                        "rsm-40":(-.015, .02),
                        "rsm-100":(-.005, .01),
                        }[fk],
                    "yrange":{
                        "rsm-10":(-.004, .004),
                        "rsm-40":(-.0018, .0018),
                        "rsm-100":(-.001, .002),
                        }[fk],
                    **base_plot_spec,
                    },
                )
        print(f"Generated {fig_path.as_posix()}")
