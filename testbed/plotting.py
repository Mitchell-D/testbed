"""
General methods for plotting all kinds of data
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def geo_quad_plot(data, flabels:list, latitude, longitude,
        geo_bounds=None, plot_spec={}, show=False, fig_path=None):
    """
    Plot a gridded scalar value on a geodetic domain, using cartopy for borders
    """
    ps = {"xlabel":"", "ylabel":"", "marker_size":4,
          "cmap":"jet_r", "text_size":12, "title":"", "map_linewidth":2,
          "norm":None,"figsize":None, "marker":"o", "cbar_shrink":1.,
          "xtick_freq":None, "ytick_freq":None, ## pixels btw included ticks
          "idx_ticks":False, ## if True, use tick indeces instead of lat/lon
          "gridlines":False,
          }
    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    fig,ax = plt.subplots(2, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    if geo_bounds is None:
        geo_bounds = [np.amin(longitude), np.amax(longitude),
                  np.amin(latitude), np.amax(latitude)]
    for n in range(4):
        i = n // 2
        j = n % 2
        ## Remove any unused axis panes
        if n == len(flabels):
            fig.delaxes(ax[i,j])
            break
        ax[i,j].set_extent(geo_bounds, crs=ccrs.PlateCarree())
        ax[i,j].add_feature(
                cfeature.LAND,
                linewidth=ps.get("map_linewidth")
                )
        ax[i,j].set_title(flabels[n])

        if not ps.get("xtick_freq") is None:
            xspace = np.linspace(*geo_bounds[:2], data[n].shape[1])
            ax[i,j].set_xticks(xspace[::ps.get("xtick_freq")])
            if ps.get("idx_ticks"):
                xidxs = list(map(str,range(data[n].shape[1])))
                ax[i,j].set_xticklabels(xidxs[::ps.get("xtick_freq")])
        if not ps.get("ytick_freq") is None:
            yspace = np.linspace(*geo_bounds[2:], data[n].shape[0])
            ax[i,j].set_yticks(yspace[::ps.get("ytick_freq")])
            if ps.get("idx_ticks"):
                yidxs = list(map(str,range(data[n].shape[0])))
                ax[i,j].set_yticklabels(yidxs[::ps.get("ytick_freq")][::-1])

        contour = ax[i,j].contourf(
                longitude,
                latitude,
                data[n],
                cmap=ps.get("cmap")
                )
        ax[i,j].add_feature(
                cfeature.BORDERS,
                linewidth=ps.get("map_linewidth"),
                zorder=120
                )
        ax[i,j].add_feature(
                cfeature.STATES,
                linewidth=ps.get("map_linewidth"),
                zorder=120
                )
        ax[i,j].coastlines()
        fig.colorbar(contour, ax=ax[i,j], shrink=ps.get("cbar_shrink"))

    fig.suptitle(ps.get("title"))

    if ps.get("gridlines"):
        plt.grid()

    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
        print(f"Generated image at {fig_path.as_posix()}")
    if show:
        plt.show()
    plt.close()
    return

def plot_geo_rgb(rgb:np.ndarray, lat_range:tuple, lon_range:tuple,
        plot_spec:dict={}, fig_path=None, show=False):
    """
    """
    ps = {"title":"", "figsize":(16,12), "border_linewidth":2,
            "title_size":12 }
    ps.update(plot_spec)
    fig = plt.figure(figsize=ps.get("figsize"))

    pc = ccrs.PlateCarree()

    ax = fig.add_subplot(1, 1, 1, projection=pc)
    extent = [*lon_range, *lat_range]
    ax.set_extent(extent, crs=pc)

    ax.imshow(rgb, extent=extent, transform=pc)

    ax.coastlines(color='black', linewidth=ps.get("border_linewidth"))
    ax.add_feature( ccrs.cartopy.feature.STATES,
            linewidth=ps.get("border_linewidth"))

    plt.title(ps.get("title"), fontweight='bold',
            fontsize=ps.get("title_size"))

    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=80)
    if show:
        plt.show()
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

def plot_quad_sequence(
        pred_array, fig_path=None, true_array=None, pred_coarseness=1,
        plot_spec={}, show=False):
    """
    Plot a series of true and predicted sequences in a 4-panel plot,
    each panel containing the data from a single feature

    :@param true_array:(N,S,4) shaped array with N sequence samples, each of
        length S and having 4 features corresponding to feat_labels
    :@param pred_array:(N,S,4) shaped array with N sequence samples, each of
        length S and having 4 features corresponding to feat_labels
    """
    if not true_array is None:
        assert true_array.shape==pred_array.shape, \
                (true_array.shape, pred_array.shape)
    ps = {
            "true_linewidth":1, "pred_linewidth":1,
            "true_linestyle":"-", "pred_linestyle":"-", "main_title":"",
            "quad_titles":["", "", "", ""], "figsize":(12,12), "dpi":100,
            "yscale":"linear", "lines_rgb":None, "grid":False,
            }
    ps.update(plot_spec)
    seq_range = np.arange(pred_array.shape[1]) * pred_coarseness
    plt.clf()
    fig,ax = plt.subplots(2, 2)
    num_px = pred_array.shape[0]
    cm = matplotlib.cm.get_cmap("hsv", num_px)

    has_legend = False
    for n in range(4):
        i = n // 2
        j = n % 2
        ## If fewer than 4 features are included, stop plotting
        if pred_array.shape[-1]-1 < n:
            break
        for px in range(pred_array.shape[0]):
            if not ps.get("lines_rgb") is None:
                color_true = ps["lines_rgb"][px]
                color_pred = ps["lines_rgb"][px]
            else:
                color_true = ps.get("true_color", cm(px))
                color_pred = ps.get("pred_color", cm(px))


            if not true_array is None:
                tmp_ax_true, = ax[i,j].plot(
                        seq_range,
                        true_array[px,:,n],
                        color=color_true,
                        linewidth=ps.get("true_linewidth"),
                        alpha=ps.get("line_opacity"),
                        linestyle=ps.get("true_linestyle", "-")
                        )
            tmp_ax_pred, = ax[i,j].plot(
                    seq_range,
                    pred_array[px,:,n],
                    color=color_pred,
                    linewidth=ps.get("pred_linewidth"),
                    alpha=ps.get("line_opacity"),
                    linestyle=ps.get("pred_linestyle", "-")
                    )

            ## Add a legend if it is requested but hasn't been added yet
            if not has_legend and not ps.get("pred_legend_label") is None:
                if true_array is None:
                    fig_legend = fig.legend(
                            (tmp_ax_pred,),
                            (ps.get("pred_legend_label"),),
                            loc=ps.get("legend_location", "upper left")
                            )
                else:
                    fig_legend = fig.legend(
                            (tmp_ax_pred, tmp_ax_true),
                            (ps.get("pred_legend_label"),
                                ps.get("true_legend_label")),
                            loc=ps.get("legend_location", "upper left"),
                            prop={"size": ps.get("legend_size",12)},
                            )

            ax[i,j].set_title(ps["quad_titles"][n],
                    fontsize=ps.get("quad_title_size",12))
            if plot_spec.get("yrange"):
                ax[i,j].set_ylim(plot_spec.get("yrange"))
            if plot_spec.get("xrange"):
                ax[i,j].set_xlim(plot_spec.get("xrange"))
            ax[i,j].set_xscale(plot_spec.get("xscale", "linear"))
            ax[i,j].set_yscale(plot_spec.get("yscale", "linear"))

    fig.supxlabel(plot_spec.get("xlabel"), fontsize=ps.get("xlabel_size", 16))
    fig.supylabel(plot_spec.get("ylabel"), fontsize=ps.get("ylabel_size", 16))

    if ps.get("grid"):
        plt.grid()
    if ps["main_title"] != "":
        fig.suptitle(ps["main_title"], fontsize=ps.get("main_title_size", 16))
    plt.tight_layout()
    if not fig_path is None:
        print(f"Saving {fig_path.as_posix()}")
        if ps.get("figsize"):
            fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",
                dpi=ps.get("dpi"))
    if show:
        plt.show()
    plt.close()
    return

if __name__=="__main__":
    pass
