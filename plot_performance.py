import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

def plot_static_error(static_error, fig_path:Path, plot_spec={}):
    """
    Plot error rates wrt unique combination of vegetation and soil  parameters
    """
    old_ps = {"cmap":"magma", "norm":"linear", "xlabel":"Soil type",
            "ylabel":"Surface type"}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    soils = ["other", "sand", "loamy-sand", "sandy-loam", "silty-loam", "silt",
            "loam", "sandy-clay-loam", "silty-clay-loam", "clay-loam",
            "sandy-clay", "silty-clay", "clay"]
    vegs = ["water", "evergreen-needleleaf", "evergreen_broadleaf",
            "deciduous-needleleaf", "deciduous-broadleaf", "mixed-cover",
            "woodland", "wooded-grassland", "closed-shrubland",
            "open-shrubland", "grassland", "cropland", "bare", "urban"]

    fig,ax = plt.subplots()
    cb = ax.imshow(static_error, cmap=plot_spec.get("cmap"),
            vmax=plot_spec.get("vmax"), norm=plot_spec.get("norm"))
    fig.colorbar(cb)
    ax.set_xlabel(plot_spec.get("xlabel"),
                  fontsize=plot_spec.get("label_size"))
    ax.set_ylabel(plot_spec.get("ylabel"),
                  fontsize=plot_spec.get("label_size"))
    ax.set_title(plot_spec.get("title"),
            fontsize=plot_spec.get("title_size"))

    # Adding labels to the matrix
    ax.set_yticks(range(len(vegs)), vegs)
    ax.set_xticks(range(len(soils)), soils, rotation=45, ha='right',)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return

def plot_stats_1d(data_dict:dict, x_labels:list=None, fig_path:Path=None,
             show:bool=False, class_space:float=.2, bar_sigma:float=1,
             fill_sigma:float=1/3, yscale="linear", fill_between=False,
             plot_spec:dict={}):
    """
    Plot the mean and standard deviation of multiple classes on the same
    X axis. Means and standard deviations for each band in each class must be
    provided as a dictionary with class labels as keys mapping to a dictionary
    with "means" and "stdevs" keys mapping to lists each with N members for
    N bands. Labels for each of the N bands must be provided separately.

    data_dict = {
        "Class 1":{"means":[9,8,7], "stdevs":[1,2,3]}
        "Class 2":{"means":[9,8,7], "stdevs":[1,2,3]}
        "Class 3":{"means":[9,8,7], "stdevs":[1,2,3]}
        }

    :@param yscale: linear by default, but logit may be good for reflectance.
    :@param class_space: directs spacing between class data points/error bars
    :@param bar_sigma: determines the size of error bars in terms of a
        constant multiple on the class' standard deviation.
    :@param fill_sigma: the shaded region of the error bar is typically
        smaller than the bar sigma.
    """
    cat_labels = list(data_dict.keys())
    if x_labels is None:
        x_labels = list(range(len(tuple(data_dict.values())[0]["means"])))

    # Merge provided plot_spec with un-provided default values
    old_ps = {}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    fig, ax = plt.subplots()
    transforms = [
            Affine2D().translate(n, 0.)+ax.transData
            for n in np.linspace(
                -.5*class_space, .5*class_space, num=len(cat_labels))
            ]
    ax.set_yscale(yscale)
    if not plot_spec.get("yrange") is None:
        ax.set_ylim(plot_spec.get("yrange"))
    if not plot_spec.get("xrange") is None:
        ax.set_xlim(plot_spec.get("xrange"))
    if not plot_spec.get("x_ticks") is None:
        ax.set_xticks(plot_spec.get("x_ticks"))
    if not plot_spec.get("y_ticks") is None:
        ax.set_yticks(plot_spec.get("y_ticks"))
    for i in range(len(cat_labels)):
        cat = cat_labels[i]
        ax.errorbar(
                x_labels,
                data_dict[cat]["means"],
                yerr=data_dict[cat]["stdevs"]*bar_sigma,
                marker=plot_spec.get("marker"),
                label=cat_labels[i],
                linestyle="-",
                transform=transforms[i],
                linewidth=plot_spec.get("line_width"),
                alpha=plot_spec.get("alpha"),
                elinewidth=plot_spec.get("error_line_width"),
                errorevery=plot_spec.get("error_every", 1),
                )
        if fill_between:
            under_bars = [
                    m-s*fill_sigma
                    for m,s in zip(
                        data_dict[cat]["means"],
                        data_dict[cat]["stdevs"])]
            over_bars = [
                    m+s*fill_sigma
                    for m,s in zip(
                        data_dict[cat]["means"],
                        data_dict[cat]["stdevs"])]
            ax.fill_between(
                x=x_labels,
                y1=under_bars,
                y2=over_bars,
                alpha=plot_spec.get("fill_alpha"),
                transform=transforms[i]
                )
        ax.grid(visible=plot_spec.get("grid"))
        ax.set_xlabel(plot_spec.get("xlabel"),
                      fontsize=plot_spec.get("label_size"))
        ax.set_ylabel(plot_spec.get("ylabel"),
                      fontsize=plot_spec.get("label_size"))
        ax.set_title(plot_spec.get("title"),
                fontsize=plot_spec.get("title_size"))
        ax.legend(fontsize=plot_spec.get("legend_font_size"))

    fig.tight_layout()
    if not plot_spec.get("fig_size") is None:
        fig.set_size_inches(*plot_spec.get("fig_size"))
    if show:
        plt.show()
    if fig_path:
        fig.savefig(fig_path.as_posix())
    plt.close()
    return

def plot_heatmap(heatmap:np.ndarray, fig_path:Path=None, show=False,
                 show_ticks=True, plot_diagonal:bool=False, plot_spec:dict={}):
    """
    Plot an integer heatmap, with [0,0] indexing the lower left corner
    """
    # Merge provided plot_spec with un-provided default values
    old_ps = { "cmap":"nipy_spectral", "cb_size":1, "cb_orient":"vertical",
            "imshow_norm":"linear"}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    fig, ax = plt.subplots()

    if plot_diagonal:
        ax.plot((0,heatmap.shape[1]-1), (0,heatmap.shape[0]-1),
                linewidth=plot_spec.get("line_width"))
    im = ax.imshow(
            heatmap,
            cmap=plot_spec.get("cmap"),
            vmax=plot_spec.get("vmax"),
            extent=plot_spec.get("imshow_extent"),
            norm=plot_spec.get("imshow_norm"),
            origin="lower",
            aspect=plot_spec.get("imshow_aspect")
            )
    cbar = fig.colorbar(
            im, orientation=plot_spec.get("cb_orient"),
            label=plot_spec.get("cb_label"), shrink=plot_spec.get("cb_size")
            )
    if not show_ticks:
        plt.tick_params(axis="x", which="both", bottom=False,
                        top=False, labelbottom=False)
        plt.tick_params(axis="y", which="both", bottom=False,
                        top=False, labelbottom=False)
    if plot_spec.get("imshow_extent"):
        extent = plot_spec.get("imshow_extent")
        assert len(extent)==4, extent
        plt.xlim(extent[:2])
        plt.ylim(extent[2:])

    #fig.suptitle(plot_spec.get("title"))
    ax.set_title(plot_spec.get("title"))
    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    if not plot_spec.get("x_ticks") is None:
        ax.set_xticks(plot_spec.get("x_ticks"))
    if not plot_spec.get("y_ticks") is None:
        ax.set_yticks(plot_spec.get("y_ticks"))
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), dpi=plot_spec.get("dpi"),
                    bbox_inches="tight")
    plt.close()
    return

def plot_lr_func(lr_func, num_epochs:int, init_lr:float,
        show:bool=False, fig_path:Path=None, plot_spec:dict={}):
    """ """
    def_ps = {"title":"Learning rate", "xlabel":"epoch",
            "ylabel":"learning rate"}
    ps = {**def_ps, **plot_spec}
    lr = [init_lr]
    for cur_epoch in range(num_epochs):
        lr.append(lr_func(cur_epoch, lr[-1]))

    fig,ax = plt.subplots()
    ax.plot(list(range(num_epochs+1)), lr)
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))
    ax.set_title(ps.get("title"))
    ax.set_xscale(plot_spec.get("xscale", "linear"))
    ax.set_yscale(plot_spec.get("yscale", "linear"))
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), dpi=plot_spec.get("dpi"),
                bbox_inches="tight")
    plt.close()
    return lr

def plot_lines(domain:list, ylines:list, fig_path:Path=None,
               labels:list=[], plot_spec={}, show:bool=False):
    """
    Plot a list of 1-d lines that share a domain and codomain.

    :@param domain: 1-d numpy array describing the common domain
    :@param ylines: list of at least 1 1-d array of data values to plot, which
            must be the same size as the domain array.
    :@param fig_path: Path to the location to store the figure. If None,
            doesn't store an image.
    :@param labels: list of string labels to include in a legend describing
            each line. If fewer labels than lines are provided, the labels
            will apply to the first of the lines.
    :@param plot_spec: Dictionary of plot options see the geo_plot module
            for plot_spec options, but the defaults are safe.
    :@param show: if True, shows the image in the matplotlib Agg client.
    """
    plt.clf()
    # Merge provided plot_spec with un-provided default values
    old_ps = {"xscale":"linear", "legend_font_size":8, "legend_ncols":1}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    # Plot each
    domain = np.asarray(domain)
    fig, ax = plt.subplots()
    colors = plot_spec.get("colors")
    if colors:
        assert len(ylines)<=len(colors)
    for i in range(len(ylines)):
        ax.plot(domain, ylines[i],
                label=labels[i] if len(labels) else "",
                linewidth=plot_spec.get("line_width"),
                color=None if not colors else colors[i])

    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    ax.set_title(plot_spec.get("title"))
    ax.set_ylim(plot_spec.get("yrange"))
    ax.set_xlim(plot_spec.get("xrange"))
    ax.set_xscale(plot_spec.get("xscale"))

    if plot_spec.get("xtick_rotation"):
        plt.tick_params(axis="x", **{"labelrotation":plot_spec.get(
            "xtick_rotation")})
    if plot_spec.get("ytick_rotation"):
        plt.tick_params(axis="y", **{"labelrotation":plot_spec.get(
            "ytick_rotation")})

    if len(labels):
        plt.legend(fontsize=plot_spec.get("legend_font_size"),
                   ncol=plot_spec.get("legend_ncols"))
    if plot_spec.get("grid"):
        plt.grid()
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))
    plt.close()
    return

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
