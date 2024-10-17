"""
Methods for interacting with 'timegrid' style HDF5s, which each cover 1/6 of
CONUS over a 3 month period, and store their data as a (T,P,Q,F) dynamic grid
with (P,Q,F) static grids and (T,1) timestamps
"""
import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from list_feats import nldas_record_mapping,noahlsm_record_mapping

def plot_hist(counts:list, labels:list, bin_bounds:list,
        plot_spec:dict={}, show=False, fig_path=None):
    """
    Plot one or more histograms on a single pane

    :@param counts: List of 1D arrays representing the binned counts
    :@param labels: List of string labels corresponding to each histogram
    :@param bin_mins: List of 2-tuple (min, max) data coordinate values for
        each histogram. The minimum should be the minimum value of the first
        bin, and the maximum should be the upper value of the last bin.
    :@param plot_spec: Dict of configuration options for the plot
    """
    ps = {"xlabel":"", "ylabel":"", "linewidth":2, "text_size":12,
            "title":"", "dpi":80, "norm":None,"figsize":(12,12),}
    ps.update(plot_spec)
    fig,ax = plt.subplots()
    for carr,label,(bmin,bmax) in zip(counts, labels, bin_bounds):
        assert len(carr.shape) == 1, "counts array must be 1D"
        bins = (np.arange(carr.size)+.5)/carr.size * (bmax-bmin) + bmin
        ax.plot(bins, carr, label=label, linewidth=ps.get("linewidth"))

    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))
    ax.set_title(ps.get("title"))
    fig.legend()

    if show:
        plt.show()
    if fig_path:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(),bbox_inches="tight",dpi=ps.get("dpi"))
    return

def plot_geo_scalar(data, latitude, longitude, bounds=None, plot_spec={},
             show=False, fig_path=None):
    """
    Plot a gridded scalar value on a geodetic domain, using cartopy for borders
    """
    ps = {"xlabel":"", "ylabel":"", "marker_size":4,
          "cmap":"jet_r", "text_size":12, "title":"",
          "norm":None,"figsize":(12,12), "marker":"o", "cbar_shrink":1.,
          "map_linewidth":2}
    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    ax = plt.axes(projection=ccrs.PlateCarree())
    fig = plt.gcf()
    if bounds is None:
        bounds = [np.amin(longitude), np.amax(longitude),
                  np.amin(latitude), np.amax(latitude)]
    ax.set_extent(bounds, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, linewidth=ps.get("map_linewidth"))
    #ax.add_feature(cfeature.LAKES, linewidth=ps.get("map_linewidth"))
    #ax.add_feature(cfeature.RIVERS, linewidth=ps.get("map_linewidth"))

    ax.set_title(ps.get("title"))
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))

    scat = ax.contourf(longitude, latitude, data, cmap=ps.get("cmap"))

    ax.add_feature(cfeature.BORDERS, linewidth=ps.get("map_linewidth"),
                   zorder=120)
    ax.add_feature(cfeature.STATES, linewidth=ps.get("map_linewidth"),
                   zorder=120)
    ax.coastlines()
    fig.colorbar(scat, ax=ax, shrink=ps.get("cbar_shrink"))

    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    if show:
        plt.show()

if __name__=="__main__":
    data_dir = Path("data")
    tg_dir = data_dir.joinpath("timegrids")
    static_pkl_path = data_dir.joinpath("static/nldas_static_cropped.pkl")
    gridstat_dir = Path("data/gridstats")
    gridstat_fig_dir = Path("figures/gridstats")

    ## Plot histograms from the aggregate gristats file
    #'''
    full_gs_file = gridstat_dir.joinpath(
            "gridstats_2012-1_2023-12_full-grid.h5")
    gsf = h5py.File(full_gs_file, "r")
    dlabels = json.loads(gsf["data"].attrs["dlabels"])
    slabels = json.loads(gsf["data"].attrs["slabels"])
    hparams = json.loads(gsf["data"].attrs["hist_params"])
    for i,dl in enumerate(dlabels):
        ## reduce the histogram over the monthly and spatial axes
        tmp_hist = np.sum(gsf["/data/histograms"][:,:,:,i,:], axis=(0,1,2))
        file_name = "_".join(
                ["gridstat-hist", dl] + full_gs_file.stem.split("_")[1:]
                ) + ".png"
        plot_hist(
                counts=[tmp_hist],
                labels=[dl],
                bin_bounds=[hparams["hist_bounds"][dl]],
                plot_spec={
                    "title":f"{dl} value histogram 2012-2023",
                    },
                show=False,
                fig_path=gridstat_fig_dir.joinpath(file_name),
                )
    #'''

    ## Plot gridded statistics on a CONUS map
    '''
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))
    _,nl_labels = map(list,zip(*nldas_record_mapping))
    _,no_labels = map(list,zip(*noahlsm_record_mapping))
    flabels = nl_labels+no_labels
    avgs = np.load(Path("data/gridstats/gridstats_avg.npy"))

    avgs[sdata[slabels.index("m_9999")]] = np.nan

    soilm_labels = ("soilm-10","soilm-40","soilm-100","soilm-200",)
    soilm = avgs[..., tuple(flabels.index(s) for s in soilm_labels), 2]
    tsoil_labels = ("tsoil-10","tsoil-40","tsoil-100","tsoil-200",)
    tsoil = avgs[..., tuple(flabels.index(s) for s in tsoil_labels), 2]
    plot_geo_scalar(
            #data=avgs[...,flabels.index("apcp"),2],
            data=avgs[...,flabels.index("veg"),2],
            #data=np.average(tsoil, axis=-1),
            #data=np.sum(soilm, axis=-1),
            latitude=sdata[slabels.index("lat")],
            longitude=sdata[slabels.index("lon")],
            plot_spec={
                #"title":"2012-2022 Mean Full-Column Soil Moisture (kg/m^3)"
                #"title":"2012-2022 Mean Full-Column Soil Temperature (K)"
                #"title":"2012-2022 Mean hourly precipitation (kg/m^2)"
                "title":"Mean vegetation fraction (%)"
                },
            show=True,
            fig_path=None
            )
    #'''
