import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

def stats_1d(data_dict:dict, x_labels:list=None, fig_path:Path=None,
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

if __name__=="__main__":
    eval_dir = Path(f"data/performance")
    fig_dir = Path("figures/performance")
    horizons_pkl = eval_dir.joinpath("error_horizons_new.pkl")
    temporal_pkl = eval_dir.joinpath("error_temporal.pkl")
    hists_pkl = eval_dir.joinpath("validation_hists_7d.pkl")

    '''
    """ Plot histograms """
    hists = pkl.load(hists_pkl.open("rb"))
    tmp_key = ('nc', 'cold', '2018-2023', 'lstm-8-091')
    feat_idx = 2
    plot_heatmap(
            heatmap=hists[tmp_key]["residual_hist"][:,:,feat_idx],
            fig_path=fig_dir.joinpath(f"hist_{'_'.join(tmp_key)}.png"),
            plot_spec={
                "imshow_norm":"log",
                "imshow_extent":(
                    hists[tmp_key]["residual_bounds"][0][feat_idx],
                    hists[tmp_key]["residual_bounds"][1][feat_idx],
                    hists[tmp_key]["residual_bounds"][0][feat_idx],
                    hists[tmp_key]["residual_bounds"][1][feat_idx],
                    ),
                }
            )
    '''

    '''
    """ Plot error with respect to horizon distance """
    horizons = pkl.load(horizons_pkl.open("rb"))
    tmp_key = ('nc', 'cold', '2018-2023', 'lstm-8-091')
    subdict = horizons[tmp_key]
    stats_1d(
            data_dict={
                f:{"means":subdict["residual_avg"][:,i],
                    "stdevs":subdict["residual_var"][:,i]**(1/2)}
                for i,f in enumerate(subdict["feats"])
                },
            fig_path=fig_dir.joinpath(f"hzn-residual_{'_'.join(tmp_key)}.png"),
            fill_between=True,
            fill_sigma=1/4,
            class_space=1,
            bar_sigma=1/4,
            yscale="linear",
            plot_spec={
                "alpha":.6,
                "line_width":2,
                "error_line_width":.5,
                "error_every":4,
                "fill_alpha":.25,
                "yrange":(0,1),
                }
            )

    stats_1d(
            data_dict={
                f:{"means":subdict["state_avg"][:,i],
                    "stdevs":subdict["state_var"][:,i]**(1/2)}
                for i,f in enumerate(subdict["feats"])
                },
            fig_path=fig_dir.joinpath(f"hzn-state_{'_'.join(tmp_key)}.png"),
            fill_between=True,
            fill_sigma=1/4,
            class_space=1,
            bar_sigma=1/4,
            yscale="linear",
            plot_spec={
                "alpha":.6,
                "line_width":2,
                "error_line_width":.5,
                "error_every":4,
                "fill_alpha":.25,
                "yrange":(0,20),
                }
            )
    '''

    '''
    """ Plot error with respect to time of day and day of year """
    tmp_key = ('nc', 'cold', '2018-2023', 'lstm-8-091')
    temporal = pkl.load(temporal_pkl.open("rb"))
    '''

