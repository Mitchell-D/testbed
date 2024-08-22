import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from eval_timegrid import geo_plot
from eval_grids import gen_bulk_grid_stats,parse_bulk_grid_params

def geo_quad_plot(data, flabels:list, latitude, longitude,
        geo_bounds=None, plot_spec={}, show=False, fig_path=None):
    """
    Plot a gridded scalar value on a geodetic domain, using cartopy for borders
    """
    ps = {"xlabel":"", "ylabel":"", "marker_size":4,
          "cmap":"jet_r", "text_size":12, "title":"",
          "norm":None,"figsize":(12,32), "marker":"o", "cbar_shrink":1.,
          "map_linewidth":2}
    assert len(flabels) == 4
    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    fig,ax = plt.subplots(2, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    if geo_bounds is None:
        geo_bounds = [np.amin(longitude), np.amax(longitude),
                  np.amin(latitude), np.amax(latitude)]
    for i in range(2):
        for j in range(2):
            ax[i,j].set_extent(geo_bounds, crs=ccrs.PlateCarree())
            ax[i,j].add_feature(
                    cfeature.LAND,
                    linewidth=ps.get("map_linewidth")
                    )
            ax[i,j].set_title(flabels[i+j])

            contour = ax[i,j].contourf(
                    longitude,
                    latitude,
                    data[i+j],
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

if __name__=="__main__":
    fig_dir = Path(f"figures/grid_error")
    eval_dir = Path(f"data/pred_grids")
    ## Load each region's latitude and longitude from a pkl dict, since
    ## the static values aren't stored alongside the gridded predictions
    region_latlons_path = Path("data/static/regional_latlons.pkl")
    region_latlons = pkl.load(region_latlons_path.open("rb"))
    bulk_grids = [
            eval_dir.joinpath("bulk-grid_nw_20180101_20211216_lstm-16-505.h5"),
            eval_dir.joinpath("bulk-grid_nc_20180101_20211216_lstm-16-505.h5"),
            eval_dir.joinpath("bulk-grid_ne_20180101_20211216_lstm-16-505.h5"),
            eval_dir.joinpath("bulk-grid_sc_20180101_20211216_lstm-16-505.h5"),
            eval_dir.joinpath("bulk-grid_se_20180101_20211216_lstm-16-505.h5"),
            eval_dir.joinpath("bulk-grid_sw_20180101_20211216_lstm-16-505.h5"),
            ]
    stats_to_plot = [
            #"state_error_max",
            #"state_error_mean",
            #"state_error_stdev",
            "state_error_final",
            #"res_error_max",
            "res_error_mean",
            #"res_error_stdev",
            ]
    feats_to_plot = [
            "soilm-10",
            "soilm-40",
            "soilm-100",
            "soilm-200",
            ]
    for p in bulk_grids:
        _,region,_,_,model = p.stem.split("_")
        grid_shape,gen_args,stat_labels = parse_bulk_grid_params(p)
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
            X = np.full(tmp_shape, np.nan)
            ## subset to only the requested features and statistics
            stats = stats[:,idxs_stats][:,:,idxs_feats]
            X[idxs[:,0],idxs[:,1]] = stats
            for i in range(len(stats_to_plot)):
                fig_name = f"bulk-error_{region}_{model}_" + \
                        f"{stats_to_plot[i].replace('_','-')}_{tmp_time}.png"
                geo_quad_plot(
                        [X[:,:,i,j] for j in range(len(feats_to_plot))],
                        flabels=feats_to_plot,
                        latitude=latlon[...,0],
                        longitude=latlon[...,1],
                        geo_bounds=None,
                        plot_spec={
                            "title":f"{region} {stats_to_plot[i]} {tmp_time}",
                            "cbar_shrink":.7,
                            "text_size":16,
                            },
                        show=False,
                        fig_path=fig_dir.joinpath(fig_name)
                        )
