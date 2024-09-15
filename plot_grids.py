import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from eval_timegrid import geo_plot
from eval_grids import gen_bulk_grid_stats,parse_bulk_grid_params

def geo_quad_plot(data, flabels:list, latitude, longitude,
        value_bounds:list=None, geo_bounds=None, plot_spec={}, show=False,
        fig_path=None):
    """
    Plot a gridded scalar value on a geodetic domain, using cartopy for borders
    """
    ps = {"xlabel":"", "ylabel":"", "marker_size":4,
          "cmap":"jet_r", "text_size":12, "title":"", "map_linewidth":2,
          "norm":None,"figsize":(32,16), "marker":"o", "cbar_shrink":1.,
          "xtick_freq":None, "ytick_freq":None, ## pixels btw included ticks
          "idx_ticks":False, ## if True, use tick indeces instead of lat/lon
          }
    assert len(flabels) == 4
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

    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
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

if __name__=="__main__":
    fig_dir = Path(f"figures/grid_error")
    #eval_dir = Path(f"data/pred_grids")
    eval_dir = Path(f"/rtmp/mdodson/pred_grids")
    ## Load each region's latitude and longitude from a pkl dict, since
    ## the static values aren't stored alongside the gridded predictions
    region_latlons_path = Path("data/static/regional_latlons.pkl")
    region_latlons = pkl.load(region_latlons_path.open("rb"))
    bulk_grids = [
            eval_dir.joinpath("bulk-grid_nc_20180101_20211216_lstm-20-353.h5"),
            eval_dir.joinpath("bulk-grid_ne_20180101_20211216_lstm-20-353.h5"),
            eval_dir.joinpath("bulk-grid_nw_20180101_20211216_lstm-20-353.h5"),
            eval_dir.joinpath("bulk-grid_sc_20180101_20211216_lstm-20-353.h5"),
            eval_dir.joinpath("bulk-grid_se_20180101_20211216_lstm-20-353.h5"),
            eval_dir.joinpath("bulk-grid_sw_20180101_20211216_lstm-20-353.h5"),
            ]
    stats_to_plot = [
            "state_error_max",
            #"state_error_mean",
            #"state_error_stdev",
            "state_bias_final",
            "res_error_max",
            "res_error_mean",
            #"res_error_stdev",
            ]
    feats_to_plot = [
            "soilm-10",
            "soilm-40",
            "soilm-100",
            "soilm-200",
            ]
    replace_existing = False
    for p in bulk_grids:
        _,region,_,_,model = p.stem.split("_")
        grid_shape,model_config,gen_args,stat_labels = \
                parse_bulk_grid_params(p)
        ## Declare a per-timestep generator for the bulk statistics data
        gen = gen_bulk_grid_stats(
                bulk_grid_path=p,
                init_time=None,
                final_time=None,
                buf_size_mb=256,
                )
        ## Get indeces for selected data and statistic features
        idxs_feats = tuple([
            gen_args["pred_feats"].index(l)
            for l in feats_to_plot
            ])
        idxs_stats = tuple([stat_labels.index(l) for l in stats_to_plot])
        ## Get the latlon array of this region from the static dictionary
        latlon = region_latlons[region]
        for stats,idxs,ptime in gen:
            tmp_time = datetime.fromtimestamp(int(ptime)).strftime("%Y%m%d-%H")
            idxs = idxs.astype(int)
            ## declare a 2d array to store the stats to be plotted
            tmp_shape = (
                    *latlon.shape[:2],
                    len(stats_to_plot),
                    len(feats_to_plot)
                    )

            ## subset to only the requested features and statistics
            stats = stats[:,idxs_stats][:,:,idxs_feats]
            ## fill valid pixels with corresponding statistics.
            X = np.full(tmp_shape, np.nan)
            X[idxs[:,0],idxs[:,1]] = stats
            for i in range(len(stats_to_plot)):
                fig_name = f"{stats_to_plot[i].replace('_','-')}_{model}" + \
                        f"_{region}_{tmp_time}.png"
                fig_path = fig_dir.joinpath(fig_name)
                if fig_path.exists() and not replace_existing:
                    continue
                geo_quad_plot(
                        [X[:,:,i,j] for j in range(len(feats_to_plot))],
                        flabels=feats_to_plot,
                        latitude=latlon[...,0],
                        longitude=latlon[...,1],
                        geo_bounds=None,
                        plot_spec={
                            "title":f"{region} {stats_to_plot[i]} {tmp_time}",
                            "cbar_shrink":.8,
                            "text_size":18,
                            "xtick_freq":10,
                            "ytick_freq":5,
                            "idx_ticks":True,
                            },
                        show=False,
                        fig_path=fig_path,
                        )
