import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import generators

def plot_sample(window:np.array, horizon:np.array, predictions:np.array,
        feat_labels:list, feat_colors:list=None, image_path:Path=None,
        plot_spec={}, show:bool=False):
    """
    Plot a list of 1-d lines that share a domain and codomain.

    :@param window: (S_w, F_p) sequence of pred features in the window range
    :@param horizon: (S_h, F_p) sequence of true pred features in horizon
    :@param predictions: (S_h, F_p) sequence of predicted outputs in horizon
    :@param feat_labels: list of string labels to include in a legend
        describing each line. If fewer labels than lines are provided, the
        labels will apply to the first of the lines.
    :@param image_path: Path to the location to store the figure. If None,
        doesn't store an image.
    :@param plot_spec: Dictionary of plot options see the geo_plot module
        for plot_spec options, but the defaults are safe.
    :@param show: if True, shows the image in the matplotlib Agg client.
    """
    assert horizon.shape == predictions.shape
    # Merge provided plot_spec with un-provided default values
    old_ps = {}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    w_range = np.arange(-1*window.shape[0],0)
    h_range = np.arange(horizon.shape[0])

    plt.clf()
    fig, ax = plt.subplots()
    if feat_colors:
        assert window.shape[-1]==len(feat_colors),(window.shape,feat_colors)
    for i in range(window.shape[-1]):
        ax.plot(w_range, window[:,i],
                linewidth=plot_spec.get("line_width"), linestyle="-",
                color=None if feat_colors is None else feat_colors[i],)
        ax.plot(h_range, horizon[:,i], label=feat_labels[i]+" true",
                linewidth=plot_spec.get("line_width"), linestyle="-",
                color=None if feat_colors is None else feat_colors[i])
        ax.plot(h_range, predictions[:,i], label=feat_labels[i]+" predicted",
                linewidth=plot_spec.get("line_width"), linestyle=":",
                color=None if feat_colors is None else feat_colors[i])

    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    ax.set_title(plot_spec.get("title"))
    ax.set_ylim(plot_spec.get("yrange"))
    ax.set_xlim(plot_spec.get("xrange"))
    ax.set_xscale(plot_spec.get("xscale", "linear"))
    ax.set_yscale(plot_spec.get("yscale", "linear"))

    if plot_spec.get("xtick_rotation"):
        plt.tick_params(axis="x", **{"labelrotation":plot_spec.get(
            "xtick_rotation")})
    if plot_spec.get("ytick_rotation"):
        plt.tick_params(axis="y", **{"labelrotation":plot_spec.get(
            "ytick_rotation")})

    if len(feat_labels):
        plt.legend(fontsize=plot_spec.get("legend_font_size", 12),
                   ncol=plot_spec.get("legend_ncols", 1))
    if plot_spec.get("grid"):
        plt.grid()
    if show:
        plt.show()
    if not image_path is None:
        fig.savefig(image_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))

if __name__=="__main__":
    timegrid_dir = Path("data/timegrids/")
    fig_dir = Path("figures/performance")
    sequence_h5_dir = Path("data/sequences/")
    pred_h5_dir = Path("data/predictions/")

    eval_regions = ("ne", "nc", "nw", "se", "sc", "sw")
    eval_seasons = ("warm", "cold")
    eval_periods = ("2018-2023",)
    seq_pred_files = [
            (s,p,tuple(pt[1:]))
            for s,st in map(
                lambda f:(f,f.stem.split("_")),
                sequence_h5_dir.iterdir())
            for p,pt in map(
                lambda f:(f,f.stem.split("_")),
                pred_h5_dir.iterdir())
            if st[0] == "sequences"
            and pt[0] == "pred"
            and st[1:4] == pt[1:4]
            and st[1] in eval_regions
            and st[2] in eval_seasons
            and st[3] in eval_periods
            ]

    for s,p,t in seq_pred_files:
        label = "_".join(t)
        gen = generators.gen_sequence_prediction_combos(
                seq_h5=s,
                pred_h5=p,
                batch_size=8,
                buf_size_mb=128,
                gen_times=True,
                gen_static=True,
                gen_window=True,
                shuffle=True,
                #seed=200007221700,
                seed=None,
                )
        param_dict = generators.parse_sequence_params(s)
        pred_idxs = tuple(
                param_dict["window_feats"].index(l)
                for l in param_dict["pred_feats"])
        for ((w,_,s,si,(_,pt)), (ys, pr)) in gen:
            ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
            yr = ys[:,1:]-ys[:,:-1]
            time = datetime.fromtimestamp(int(pt[0,0]))
            plot_sample(
                    window=w[0][:,pred_idxs],
                    horizon=ys[0,1:],
                    predictions=ps[0],
                    feat_labels=param_dict["pred_feats"],
                    feat_colors=["red", "orange", "green", "blue", "purple"],
                    image_path=fig_dir.joinpath(
                        f"samples/samples-state_{label}.png"),
                    plot_spec={
                        "legend_font_size":6,
                        "legend_ncols":2,
                        "yrange":(-5,500),
                        "xlabel":"Horizon distance (hours)",
                        },
                    show=False
                    )
            plt.clf()
            plot_sample(
                    window=w[0][:,pred_idxs],
                    horizon=yr[0],
                    predictions=pr[0],
                    feat_labels=param_dict["pred_feats"],
                    feat_colors=["red", "orange", "green", "blue", "purple"],
                    image_path=fig_dir.joinpath(
                        f"samples/samples-residual_{label}.png"),
                    plot_spec={
                        "legend_font_size":6,
                        "legend_ncols":2,
                        "yrange":(-2,2),
                        "yscale":"symlog",
                        "xlabel":"Horizon distance (hours)",
                        },
                    show=False
                    )
            exit(0)
            input()
