
import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

from krttdkit.visualize import geoplot as gp

def plot_hist(hist_pkl:Path, fig_dir:Path=None, show:bool=False, title:str=""):
    """ """
    depth_labels = ["0-10 cm", "10-40 cm", "40-100 cm", "100-200 cm"]
    phist,thist,all_min,all_max = pkl.load(hist_pkl.open("rb"))
    domain = np.linspace(all_min,all_max,phist.shape[1])
    cmap = ["red", "blue", "green", "orange"]

    fig,ax = plt.subplots(2,2)

    fig.suptitle(title,y=.96)
    fig.supxlabel("Soil Moisture ($kg / m^3$)",y=.04)
    fig.supylabel("Frequency")
    axids = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(len(axids)):
        c = cmap[i]
        ax[*axids[i]].plot(
                domain[:,i], thist[i], color=c, linewidth=1)
        ax[*axids[i]].plot(
                domain[:,i], phist[i], color=c, linestyle="dashed", linewidth=1)
        ax[*axids[i]].set_title("SOILM " + depth_labels[i])
        ax[*axids[i]].tick_params(
                axis="y",left=False, right=False,labelleft=False)
    fig.tight_layout()
    if fig_dir:
        plt.savefig(fig_dir.joinpath(hist_pkl.stem+".png"), dpi=800)
    if show:
        plt.show()

def plot_grid_error(
        error_file:Path, fig_dir:Path=None, show:bool=False, title:str=""):
    """ """
    depth_labels = ["0-10 cm", "10-40 cm", "40-100 cm", "100-200 cm"]
    error = np.load(error_file)

    fig,ax = plt.subplots(2,2)

    fig.suptitle(title,y=.96)
    axids = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(len(axids)):
        cmap = matplotlib.cm.gist_rainbow
        cmap.set_bad("black",1.)
        arr = np.ma.array(error[...,i], mask=np.isnan(error[...,i]))
        im = ax[*axids[i]].imshow(arr, cmap=cmap)
        div = make_axes_locatable(ax[*axids[i]])
        cax = div.append_axes("bottom",size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="horizontal")
        ax[*axids[i]].set_title("SOILM " + depth_labels[i])
        ax[*axids[i]].axes.xaxis.set_visible(False)
        ax[*axids[i]].axes.yaxis.set_visible(False)
    fig.tight_layout()
    #plt.subplot
    if fig_dir:
        plt.savefig(fig_dir.joinpath(error_file.stem+".png"), dpi=800)
    if show:
        plt.show()

if __name__ == "__main__":
    data_dir = Path("data")
    fig_dir = Path("figures")

    #error_file = data_dir.joinpath("grid_error/grid_mae_dense-1.npy")
    #error_file = data_dir.joinpath("grid_error/grid_mae_lstm-rec-1.npy")
    #error_file = data_dir.joinpath("grid_error/grid_mae_lstm-s2s-2.npy")
    #error_file = data_dir.joinpath("grid_error/grid_mae_lstm-s2s-5.npy")
    error_file = data_dir.joinpath("grid_error/grid_mae_tcn-1.npy")
    model_name = error_file.stem.split("_")[-1]
    plot_grid_error(
            error_file=error_file,
            title=f"{model_name} Mean Absolute Error (test data)",
            fig_dir=fig_dir.joinpath("grid_error"),
            show=False,
            )

    '''
    """ Plot depth-wise value histograms """
    #hist_pkl = data_dir.joinpath("hists/hist_2018_dense-1.pkl")
    #hist_pkl = data_dir.joinpath("hists/hist_2018_lstm-rec-1.pkl")
    #hist_pkl = data_dir.joinpath("hists/hist_2018_lstm-s2s-2.pkl")
    #hist_pkl = data_dir.joinpath("hists/hist_2018_lstm-s2s-5.pkl")
    hist_pkl = data_dir.joinpath("hists/hist_2018_tcn-1.pkl")

    model_name = hist_pkl.stem.split("_")[-1]
    plot_hist(
            hist_pkl=hist_pkl,
            title=f"{model_name} distribution (test data)",
            fig_dir=fig_dir.joinpath("hists"),
            )
    '''
