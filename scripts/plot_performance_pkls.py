"""
Methods for plotting bulk statistics from the old kind of "performance pkls"
which were generated by methods in eval_models using prediction hdf5 files.

Now the main way I do model evaluation is with Evaluator objects applied
batch-wise in eval_sequences. The plotting methods here are shared with
those, but the actual plotting is handled in plot_sequences.
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from testbed.plotting import plot_static_error,plot_stats_1d,plot_heatmap
from testbed.plotting import plot_lr_func,plot_lines

if __name__=="__main__":
    eval_dir = Path(f"data/performance")
    fig_dir = Path("figures/performance")
    horizons_pkl = eval_dir.joinpath("error_horizons.pkl")
    temporal_pkl = eval_dir.joinpath("temporal_absolute.pkl")
    hists_pkl = eval_dir.joinpath("validation_hists_7d.pkl")
    static_error_pkl = Path(f"data/performance/static_error.pkl")
    sequence_h5_dir = Path("data/sequences/")
    pred_h5_dir = Path("data/predictions")

    plot_regions = ("ne", "nc", "nw", "se", "sc", "sw")
    plot_seasons = ("warm", "cold")
    #plot_periods = ("2018-2023",)
    plot_periods = ("2018-2021", "2021-2024")
    #plot_models = ("lstm-17-235",)
    #plot_models = ("lstm-16-505",)
    #plot_models = ("lstm-19-191", "lstm-20-353")
    #plot_models = ("lstm-19-191", "lstm-20-353")
    #plot_models = ("lstm-21-522", "lstm-22-339", "lstm-23-217")
    #plot_models = ("lstm-24-401", "lstm-25-624")
    #plot_models = ("snow-7-069")
    #plot_models = ("lstm-rsm-9-231")
    plot_models = ("lstm-rsm-9-231")

    ## Collect (region, season, period, model) keys to hash performance dicts
    rspm_keys = [
            tuple(pt[1:])
            for st in map(
                lambda f:f.stem.split("_"),
                sequence_h5_dir.iterdir())
            for pt in map(
                lambda f:f.stem.split("_"),
                pred_h5_dir.iterdir())
            if st[0] == "sequences"
            and pt[0] == "pred"
            and pt[-1] in plot_models
            and st[1:4] == pt[1:4]
            and st[1] in plot_regions
            and st[2] in plot_seasons
            and st[3] in plot_periods
            ]
    ## alternative manual selection
    rspm_keys = [("all", "all", "2018-2024", "lstm-rsm-9-231")]

    '''
    """
    Loop through individual (region, period, model) combos, combining warm
    and cold season pairs for day-of-year state and residual error rates
    """
    completed = []
    for tup_key in rspm_keys:
        r0,_,p0,m0 = tup_key
        ## Check if the other season for this combo has already been plotted
        if (r0,p0,m0) in completed:
            continue
        pair = None
        for cand_key in rspm_keys:
            if cand_key == tup_key:
                continue
            r1,_,p1,m1 = cand_key
            if all((r0==r1, p0==p1, m0==m1)):
                pair = (tup_key, cand_key)
                break
        if pair is None:
            raise ValueError(f"No seasonal pair found for {tup_key}")
        completed.append((r1,p1,m1))

        with open(temporal_pkl, "rb") as temporal_file:
            temporal = pkl.load(temporal_file)
        doy_state = temporal[pair[0]]["doy_state"] + \
                temporal[pair[1]]["doy_state"]
        doy_res = temporal[pair[0]]["doy_residual"] + \
                temporal[pair[1]]["doy_residual"]
        doy_counts = temporal[pair[0]]["doy_counts"] + \
                temporal[pair[1]]["doy_counts"]
        doy_state /= doy_counts
        doy_res /= doy_counts

        ## Plot the season-merged (region, period, model) doy state error
        plot_lines(
                domain=np.arange(doy_state.shape[0]),
                ylines=[doy_state[:,i] for i in range(doy_state.shape[-1])],
                labels=temporal[pair[0]]["feats"],
                plot_spec={
                    "colors":["red", "orange", "green", "blue", "purple"],
                    "title":"State error wrt DoY; " + \
                            ' '.join(completed[-1]),
                    },
                fig_path=fig_dir.joinpath(
                    f"doy-state_{'_'.join(completed[-1])}.png")
                )
        ## Plot the season-merged (region, period, model) doy residual error
        plot_lines(
                domain=np.arange(doy_res.shape[0]),
                ylines=[doy_res[:,i] for i in range(doy_res.shape[-1])],
                labels=temporal[pair[0]]["feats"],
                plot_spec={
                    "colors":["red", "orange", "green", "blue", "purple"],
                    "title":"RSM increment error wrt DoY; " + \
                            ' '.join(completed[-1]),
                    },
                fig_path=fig_dir.joinpath(
                    f"doy-res_{'_'.join(completed[-1])}.png")
                )
    '''

    """
    Iterate over selected (region, season, period, model) tuple keys,
    which must have been loaded using the methods in eval_models
    """
    for tup_key in rspm_keys:
        ## Plot mean error wrt soil and vegetation types
        #'''
        np.seterr(divide="ignore")
        with open(static_error_pkl, "rb") as static_error_file:
            serr = pkl.load(static_error_file)
        for i,f in enumerate(serr[tup_key]["feats"]):
            serr_state = serr[tup_key]["err_state"][:,:,i] \
                    / serr[tup_key]["counts"]
            serr_res = serr[tup_key]["err_residual"][:,:,i] \
                    / serr[tup_key]["counts"]
            #serr_state[np.logical_not(np.isfinite(serr_state))] = 0
            #serr_res[np.logical_not(np.isfinite(serr_res))] = 0

            plot_static_error(
                    static_error=serr_state,
                    fig_path=fig_dir.joinpath(
                        f"static-state_{'_'.join(tup_key)}_{f}.png"),
                    plot_spec={
                        "title":f"{f} state error wrt static params " + \
                                " ".join(tup_key),
                        "vmax":.01,
                        "cmap":"turbo"
                        },
                    )
            plot_static_error(
                    static_error=serr_res,
                    fig_path=fig_dir.joinpath(
                        f"static-res_{'_'.join(tup_key)}_{f}.png"),
                    plot_spec={
                        "title":f"{f} error increment wrt static params " + \
                                " ".join(tup_key),
                        #"vmax":.2,
                        "vmax":.0002,
                        "cmap":"turbo"
                        },
                    )
        np.seterr(divide=None)
        #'''

        ## Plot state and residual error with respect to time of day
        '''
        with open(temporal_pkl, "rb") as temporal_file:
            temporal = pkl.load(temporal_file)
        subdict = temporal[tup_key]
        plot_lines(
                domain=np.arange(subdict["tod_state"].shape[0]),
                ylines=[
                    subdict["tod_state"][:,i]/subdict["tod_counts"][:,i]
                    for i in range(subdict["doy_state"].shape[-1])],
                labels=subdict["feats"],
                plot_spec={
                    "colors":["red", "orange", "green", "blue", "purple"],
                    "title":"State error wrt ToD; "+" ".join(tup_key),
                    "xlabel":"Time of Day (UTC)",
                    "ylabel":"Mean error in RSM (fraction)",
                    },
                fig_path=fig_dir.joinpath(
                    f"tod-state_{'_'.join(tup_key)}.png")
                )
        plot_lines(
                domain=np.arange(subdict["tod_residual"].shape[0]),
                ylines=[
                    subdict["tod_residual"][:,i]/subdict["tod_counts"][:,i]
                    for i in range(subdict["doy_residual"].shape[-1])],
                labels=subdict["feats"],
                plot_spec={
                    "colors":["red", "orange", "green", "blue", "purple"],
                    "title":"RSM increment error wrt ToD; "+" ".join(tup_key),
                    "xlabel":"Time of Day (UTC)",
                    "ylabel":"Mean error in RSM increment (fraction/hour)",
                    },
                fig_path=fig_dir.joinpath(
                    f"tod-res_{'_'.join(tup_key)}.png")
                )
        '''

        ## Plot true/predicted residual and state joint histograms for feats
        #'''
        with open(hists_pkl, "rb") as hists_file:
            hists = pkl.load(hists_file)
        for fidx in range(hists[tup_key]["residual_hist"].shape[-1]):
            tmp_feat = hists[tup_key]["feats"][fidx]
            plot_heatmap(
                    heatmap=hists[tup_key]["residual_hist"][:,:,fidx],
                    fig_path=fig_dir.joinpath(
                        f"hist-res_{'_'.join(tup_key)}_{tmp_feat}.png"),
                    plot_spec={
                        "imshow_norm":"log",
                        "imshow_extent":(
                            hists[tup_key]["residual_bounds"][0][fidx],
                            hists[tup_key]["residual_bounds"][1][fidx],
                            hists[tup_key]["residual_bounds"][0][fidx],
                            hists[tup_key]["residual_bounds"][1][fidx],
                            ),
                        "title":f"RSM increment joint hists {tmp_feat} " + \
                                " ".join(tup_key)
                        }
                    )
            plot_heatmap(
                    heatmap=hists[tup_key]["state_hist"][:,:,fidx],
                    fig_path=fig_dir.joinpath(
                        f"hist-state_{'_'.join(tup_key)}_{tmp_feat}.png"),
                    plot_spec={
                        "imshow_norm":"log",
                        "imshow_extent":(
                            hists[tup_key]["state_bounds"][0][fidx],
                            hists[tup_key]["state_bounds"][1][fidx],
                            hists[tup_key]["state_bounds"][0][fidx],
                            hists[tup_key]["state_bounds"][1][fidx],
                            ),
                        "title":f"State joint hists {tmp_feat} " + \
                                " ".join(tup_key)
                        }
                    )
        #'''

        ## Plot error with respect to horizon distance
        #'''
        with open(horizons_pkl, "rb") as horizons_file:
            horizons = pkl.load(horizons_file)
        subdict = horizons[tup_key]
        domain = np.arange(subdict["state_avg"].shape[0]) * \
                subdict["pred_coarseness"]
        plot_stats_1d(
                data_dict={
                    f:{"means":subdict["residual_avg"][:,i],
                        "stdevs":subdict["residual_var"][:,i]**(1/2)}
                    for i,f in enumerate(subdict["feats"])
                    },
                fig_path=fig_dir.joinpath(
                    f"hzn-res_{'_'.join(tup_key)}.png"),
                fill_between=True,
                fill_sigma=1/4,
                class_space=1,
                bar_sigma=1/4,
                yscale="linear",
                plot_spec={
                    "title":"RSM increment error wrt horizon; " + \
                            f"{' '.join(tup_key)}",
                    "xlabel":"Forecast distance (hours)",
                    "ylabel":"Mean error in RSM (fraction)",
                    "alpha":.6,
                    "line_width":2,
                    "error_line_width":.5,
                    "error_every":4,
                    "fill_alpha":.25,
                    #"yrange":(0,1),
                    #"yrange":(0,.15),
                    "yrange":(0,.002),
                    "xticks":domain,
                    }
                )

        plot_stats_1d(
                data_dict={
                    f:{"means":subdict["state_avg"][:,i],
                        "stdevs":subdict["state_var"][:,i]**(1/2)}
                    for i,f in enumerate(subdict["feats"])
                    },
                fig_path=fig_dir.joinpath(
                    f"hzn-state_{'_'.join(tup_key)}.png"),
                fill_between=True,
                fill_sigma=1/4,
                class_space=1,
                bar_sigma=1/4,
                yscale="linear",
                plot_spec={
                    "title":"Relative soil moisture error wrt horizon; " + \
                            f"{' '.join(tup_key)}",
                    "xlabel":"Forecast distance (hours)",
                    "ylabel":"Mean error in RSM (fraction)",
                    "alpha":.6,
                    "line_width":2,
                    "error_line_width":.5,
                    "error_every":4,
                    "fill_alpha":.25,
                    #"yrange":(0,20),
                    #"yrange":(0,10),
                    "yrange":(0,.04),
                    "xticks":domain,
                    }
                )
        #'''

    ## Plot a learning rate curve
    '''
    from model_methods import get_cyclical_lr
    plot_lr_func(
            lr_func=get_cyclical_lr(
                lr_min=1e-4,
                lr_max=5e-3,
                inc_epochs=3,
                dec_epochs=12,
                decay=.025,
                log_scale=True,
                ),
            num_epochs=128,
            init_lr=7.5e-3,
            show=False,
            #fig_path=Path("figures/cyclical_lr_linear.png"),
            fig_path=Path("figures/cyclical_lr_logarithmic.png"),
            plot_spec={
                "title":"Log-cyclical learning rate with decay",
                #"yscale":"log",
                #"ylabel":"learning rate (log scale)",
                }
            )
    '''