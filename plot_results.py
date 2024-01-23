
import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pprint import pprint

#from krttdkit.visualize import geoplot as gp
from model_evaluate import gen_pred_seqs

## labels shared by multiple plotting methods
depth_labels = ["0-10 cm", "10-40 cm", "40-100 cm", "100-200 cm"]

def plot_hist(hist_pkl:Path, fig_dir:Path=None, show:bool=False, title:str=""):
    """ """
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
        ax[axids[0],axids[1]].plot(
                domain[:,i], thist[i], color=c, linewidth=1)
        ax[axids[0],axids[1]].plot(
                domain[:,i], phist[i], color=c, linestyle="dashed", linewidth=1)
        ax[axids[0],axids[1]].set_title("SOILM " + depth_labels[i])
        ax[axids[0],axids[1]].tick_params(
                axis="y",left=False, right=False,labelleft=False)
    fig.tight_layout()
    if fig_dir:
        plt.savefig(fig_dir.joinpath(hist_pkl.stem+".png"), dpi=800)
    if show:
        plt.show()

def plot_grid_error(
        error_file:Path, fig_dir:Path=None, show:bool=False, title:str=""):
    """ """
    error = np.load(error_file)

    fig,ax = plt.subplots(2,2)

    fig.suptitle(title,y=.96)
    axids = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(len(axids)):
        cmap = matplotlib.cm.gist_rainbow
        cmap.set_bad("black",1.)
        arr = np.ma.array(error[...,i], mask=np.isnan(error[...,i]))
        im = ax[axids[0],axids[1]].imshow(arr, cmap=cmap)
        div = make_axes_locatable(ax[axids[0],axids[1]])
        cax = div.append_axes("bottom",size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="horizontal")
        ax[axids[0],axids[1]].set_title("SOILM " + depth_labels[i])
        ax[axids[0],axids[1]].axes.xaxis.set_visible(False)
        ax[axids[0],axids[1]].axes.yaxis.set_visible(False)
    fig.tight_layout()
    #plt.subplot
    if fig_dir:
        plt.savefig(fig_dir.joinpath(error_file.stem+".png"), dpi=800)
    if show:
        plt.show()

def load_csv_prog(model_dir):
    """
    Load the per-epoch metrics from a tensorflow CSVLogger file as a dict.
    """
    cfg = mm.load_config(model_dir)
    csv_path = model_dir.joinpath(f"{cfg['model_name']}_prog.csv")
    csv_lines = csv_path.open("r").readlines()
    csv_lines = list(map(lambda l:l.strip().split(","), csv_lines))
    csv_labels = csv_lines.pop(0)
    csv_cols = list(map(
        lambda l:np.asarray([float(v) for v in l]),
        zip(*csv_lines)))

def plot_error_horizons(error_dict, title="", fig_dir:Path=None, show=False):
    """
    Plot the error rates wrt horizon distance for each model given a

    :@param: dict mapping a string model name to a (H,D) shaped numpy array
        for H horizons at D depth levels.
    :@param title: String for the plot title; "{key}" fstring is exposed to
        the keys in the corresponding dictionary.
    """
    for k in error_dict.keys():
        fig,ax = plt.subplots()
        X = error_dict[k].T
        for i in range(X.shape[0]):
            ax.plot(range(X.shape[1]), X[i], label=depth_labels[i])
        ax.legend()
        ax.set_title(title.format(key=k))
        ax.set_xlabel("Horizon Distance (Hours)")
        ax.set_ylabel("Mean Absolute Error ($kg/m^3$)")
        if show:
            plt.show()
        if not fig_dir is None:
            assert fig_dir.exists()
            plt.savefig(fig_dir.joinpath(f"error_horizons_{k}.png"), dpi=800)
        plt.clf()

def show_prediction_sequences(seq_gen, num_seqs=100):
    """
    """
    for i in range(num_seqs):
        sdict = next(seq_gen)
        ## Get a time domain with window values depicted as negative
        full_range = np.arange(-sdict["window"].shape[0],
                               sdict["true"].shape[0])
        fig,ax = plt.subplots(2,2)
        axids = [(0,0), (0,1), (1,0), (1,1)]
        print(ax)
        for j in range(len(axids)):
            ## Plot the window, true, and predicted features
            ax[axids[j][0],axids[j][1]].plot(
                    full_range[:sdict["window"].shape[0]],
                    sdict["window"],
                    color="blue",
                    )
            ax[axids[j][0],axids[j][1]].plot(
                    full_range[sdict["window"].shape[0]:],
                    sdict["true"],
                    color="blue",
                    linestyle="dashed",
                    )
            ax[axids[j][0],axids[j][1]].plot(
                    full_range[sdict["window"].shape[0]:],
                    sdict["prediction"],
                    color="red"
                    )
            ax[axids[j][0],axids[j][1]].set_title("SOILM " + depth_labels[j])
            ax[axids[j][0],axids[j][1]].tick_params(
                    axis="y",left=False, right=False,labelleft=False)
        fig.tight_layout()
        fig.suptitle("".format(time=sdict["time"]))
        fig.supxlabel("Forecast Hour",y=.04)
        plt.show()

if __name__ == "__main__":
    data_dir = Path("data")
    fig_dir = Path("figures")

    '''
    """
    Plot bulk error rates wrt horizon distance from a pkl
    """
    mae_pkl = data_dir.joinpath("mae.pkl")
    mae_bulk,mae_horizons = pkl.load(mae_pkl.open("rb"))
    plot_error_horizons(
            error_dict=mae_horizons,
            title="{key} MAE wrt Prediction Horizon (test data)",
            fig_dir=fig_dir.joinpath("horizon_error"),
            )
    pprint(mae_bulk)
    '''

    sgen = gen_pred_seqs(
            sample_h5=data_dir.joinpath("shuffle_2018.h5"),
            #pred_h5=data_dir.joinpath("pred_2018_dense-1.h5"),
            #pred_h5=data_dir.joinpath("pred_2018_lstm-rec-1.h5"),
            #pred_h5=data_dir.joinpath("pred_2018_lstm-s2s-2.h5"),
            #pred_h5=data_dir.joinpath("pred_2018_lstm-s2s-5.h5"),
            pred_h5=data_dir.joinpath("pred/pred_2018_tcn-1.h5"),
            pred_feats=['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200'],
            window_size=12
            )
    show_prediction_sequences(
            seq_gen=sgen,
            )


    '''
    """
    Plot gridded error files at 4 depth level from npy files generated by
    model_evaluate.get_grid_mae using a consecutive sample and prediction file
    """
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

    '''
    """
    Plot depth-wise value histograms from pkl files generated by
    model_evaluate.get_histograms using a prediction file
    """
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
