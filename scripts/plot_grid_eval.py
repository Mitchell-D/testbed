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
    eval_pkl_dir = proj_root.joinpath("data/eval_grid_pkls")

    ## Specify a subset of grid Evaluator pkls to plot based on name fields:
    ## eval-grid_{domain}_{md.name}_{eval_feat}_{et}_{na|bias|abs-err}.pkl

    ## Spatiotemporal domains to plot (2nd field of file name)
    plot_domains = [
            #"kentucky-flood",
            "high-sierra",
            ]
    ## substrings of model names to plot (3rd field of file name)
    plot_models_contain = [
            #"accfnn",
            #"accrnn",
            #"lstm-rsm",
            #"acclstm-rsm-1",
            "lstm-rsm-9",
            "accfnn-rsm-8",
            #"accrnn-rsm-2",
            "accfnn-rsm-5",
            #"lstm-20",
            "acclstm-rsm-4",
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
    plot_eval_type = [
            #"horizon",
            #"temporal",
            #"static-combos",
            #"hist-true-pred",
            #"hist-saturation-error",
            #"hist-state-increment",
            #"hist-humidity-temp",
            #"hist-infiltration",
            "spatial-stats"
            ]
    ## error types of evaluators to plot (6th field of file name)
    plot_error_type = [
            "na",
            "bias",
            "abs-err"
            ]

    ## Select which 4-panel configurations to plot (from plot_spatial_stats)
    plot_spatial_stats = [
            "res-err-bias-mean",
            "res-err-bias-stdev",
            "state-err-abs-mean",
            "state-err-abs-stdev",
            "temp-spfh-apcp-mean",
            "temp-spfh-apcp-stdev",
            ]

    ## --------( END BASIC CONFIGURATION )--------

    ## Specify 4-panel figure configurations of spatial statistics data
    common_spatial_plot_spec = {
            }
    spatial_plot_info = {
            "res-err-bias-mean":{
                "feats":[
                    ("err_res", "rsm-10", "mean"),
                    ("err_res", "rsm-40", "mean"),
                    ("err_res", "rsm-100", "mean"),
                    ],
                "error_type":"bias",
                "plot_spec":{
                    **common_spatial_plot_spec,
                    },
                },
            "res-err-bias-stdev":{
                "feats":[
                    ("err_res", "rsm-10", "stdev"),
                    ("err_res", "rsm-40", "stdev"),
                    ("err_res", "rsm-100", "stdev"),
                    ],
                "error_type":"bias",
                "plot_spec":{
                    **common_spatial_plot_spec,
                    },
                },
            "state-err-abs-mean":{
                "feats":[
                    ("err_state", "rsm-10", "mean"),
                    ("err_state", "rsm-40", "mean"),
                    ("err_state", "rsm-100", "mean"),
                    ],
                "error_type":"abs-err",
                "plot_spec":{
                    **common_spatial_plot_spec,
                    },
                },
            "state-err-abs-stdev":{
                "feats":[
                    ("err_state", "rsm-10", "stdev"),
                    ("err_state", "rsm-40", "stdev"),
                    ("err_state", "rsm-100", "stdev"),
                    ],
                "error_type":"abs-err",
                "plot_spec":{
                    **common_spatial_plot_spec,
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
                    **common_spatial_plot_spec,
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
                    **common_spatial_plot_spec,
                    },
                },
            }

    hist_plot_info = {
            "hist-true-pred":{
                "na":{
                    "title":"{model_name} {eval_feat} {domain} " + \
                            "increment validation joint histogram",
                    "xlabel":"Predicted change in RSM (%)",
                    "ylabel":"Actual change in RSM (%)",
                    "aspect":1,
                    "norm":"log",
                    }
                },
            "hist-saturation-error":{
                "na":{
                    "title":"{model_name} {eval_feat} {domain} increment " + \
                            "error bias wrt saturation percentage",
                    "xlabel":"Absolute error in hourly change in RSM",
                    "ylabel":"Relative soil moisture ({eval_feat})",
                    "aspect":1,
                    "norm":"log",
                    },
                },
            "hist-state-increment":{
                "abs-err":{
                    "title":"{model_name} {eval_feat} {domain} hourly " + \
                            "MAE wrt saturation and increment change in RSM",
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
                    "title":"{model_name} {eval_feat} {domain} hourly bias" + \
                            "wrt saturation and increment change in RSM",
                    "xlabel":"Hourly increment change in {eval_feat} (%)",
                    "ylabel":"Soil saturation in RSM (%)",
                    "norm":"log",
                    "cov_vmin":-.05,
                    "cov_vmax":.05,
                    "cov_cmap":"seismic",
                    "cov_norm":"symlog",
                    "aspect":1,
                    "fig_size":(18,8),
                    },
                },
            "hist-humidity-temp":{
                "abs-err":{
                    "title":"{model_name} {eval_feat} {domain} hourly MAE " + \
                            "wrt humidity and temp distribution",
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
                    "title":"{model_name} {eval_feat} {domain} hourly bias" + \
                            " wrt humidity and temp distribution",
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
                        "title":"{model_name} {eval_feat} {domain} " + \
                                "infiltration validation and mean layer " + \
                                "water content (kg/m^2)",
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
                    "title":"{model} {eval_feat} {data_source} state MAE " + \
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
                    "title":"{model} {eval_feat} {data_source} increment " + \
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
        _,data_source,model,eval_feat,_,error_type = pt
        for ix,pf in enumerate(pred_feats):
            try:
                new_feat = eval_feat.split("-")[0] + "-" + pf.split("-")[1]
            except:
                continue
            res_fig_path = fig_dir.joinpath(
                    p.stem.replace(eval_feat, new_feat) + "_res.png")
            state_fig_path = fig_dir.joinpath(
                    p.stem.replace(eval_feat, new_feat) + "_state.png")
            ev.plot(
                    state_or_res="res",
                    fig_path=res_fig_path,
                    plot_index=ix,
                    plot_spec={
                        "title":f"{model} {new_feat} {data_source} " + \
                                f"increment {error_type}",
                        "vmax":.005,
                        "vmin":{"bias":-.005,"abs-err":0}[error_type],
                        }
                    )
            ev.plot(
                    state_or_res="state",
                    fig_path=state_fig_path,
                    plot_index=ix,
                    plot_spec={
                        "title":f"{model} {new_feat} {data_source} " + \
                                f"state {error_type}",
                        "vmax":.1,
                        "vmin":{"bias":-.005,"abs-err":-.1}[error_type],
                        }
                    )
    ## plot 4-panel spatial statistics
    for p,pt in filter(lambda p:p[1][4]=="spatial-stats", eval_pkls):
        ev = evaluators.EvalGridAxes().from_pkl(p)
        _,data_source,model,eval_feat,_,error_type = pt

        ## Gotta do this since indeces are concatenated along along axis 1
        ## with EvalGridAxis concatenation. Probably need to just keep a list.
        idx_zero_splits = list(np.where(np.all(ev.indeces == 0, axis=1))[0])
        idx_zero_splits.append(ev.indeces.shape[0])
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
                    {"mean":ev.average[...,ix], "stdev":ev.variance[...,ix]}[m]
                    for ix,m in zip(fidxs, (c[2] for c in tmp_cfg["feats"]))
                    ]
            feats = np.stack(feats, axis=-1)

            ## independently grid each of the tiles
            gridded_feats = []
            for ll,tl,slc in tiles_info:
                tmp_tile_shape = (*ll.shape[:2], len(tmp_cfg["feats"]))
                tmp_tile_feats = np.full(tmp_tile_shape, np.nan)
                tmp_tile_feats.shape
                ix = ev.indeces[slc]
                ## Batch and sequence axes should be size 1 (marginalized)
                tmp_tile_feats[ix[:,0], ix[:,1],:] = feats[0,slc,0,:]
                gridded_feats.append(tmp_tile_feats)

            ## plot each of the requested spatial plots
            xt = ev.attrs["domain"].mosaic_shape[-1]
            tile_arrays = [ev.attrs["latlon"], gridded_feats]
            for i,ta in enumerate(tile_arrays):
                rows = [ta[i:i + xt] for i in range(0,len(ta),xt)]
                tile_arrays[i] = np.concatenate(
                        [np.concatenate(x, axis=1) for x in rows], axis=0)
            latlon,feats = tile_arrays

            plotting.geo_quad_plot(
                    data=[feats[...,i] for i in range(feats.shape[-1])],
                    flabels=[" ".join(fl) for fl in tmp_cfg["feats"]],
                    latitude=latlon[...,0],
                    longitude=latlon[...,1],
                    plot_spec={},
                    fig_path=fig_dir.joinpath("_".join(pt)+f"_{spt}.png"),
                    )
