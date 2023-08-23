"""
This script is the primary means of generating datasets of pixel time series
from a TimeGrid-style directory of ".npy" grids.

  --- :: Valid static datasets :: ---
"lai", "frac_veg", "params", "params_info", "veg_type_ints",
"soil_type_ints", "soil_comp", "geo"

  --- :: Valid NLDAS-2 Forcing datasets :: ---
"TMP", "SPFH", "PRES", "UGRD", "VGRD", "DLWRF", "NCRAIN",
"CAPE", "PEVAP", "APCP", "DSWRF"

  --- :: Valid NLDAS Noah-LSM model variables datasets :: ---
'SOILM-0-100', 'SOILM-0-10', 'SOILM-10-40', 'SOILM-40-100', 'SOILM-100-200',
'LSOIL-0-10', 'LSOIL-10-40', 'LSOIL-40-100', 'LSOIL-100-200']
"""
from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime
import multiprocessing as mp

from krttdkit.products import TimeGrid
from krttdkit.visualize import guitools as gt
from krttdkit.operate import enhance as enh
from krttdkit.operate import preprocess as pp


def get_forcings(subgrid_dir, pixels, features, init_time=None,
                 final_time=None, data_label:str="nldas-all", nworkers:int=4):
    """
    High-level method using TimeGrid objects to extract NLDAS2 forcing and
    Noah-LSM outputs from a series of .npy files within provided parameters.

    This method assumes that both forcings and LSM outputs are in a single
    numpy file corresponding to each time step, with a "YYYYmmdd-HH" style
    time string as the second underscore-separated field.

    Each pixel is extracted into a member of a list as a (T,F) shaped array
    with T timesteps between init_time and final_time and F features in the
    order of supplied features label list.
    """
    nldas_paths = [p for p in subgrid_dir.iterdir()
                   if p.stem.split("_")[0]==data_label]

    # Extract a timeseries for each selected pixels as a (T,F) shaped array
    # for T timesteps in range (init_time, final_time) and F features.
    # The hard-coded labels are records 1-11 from the NLDAS forcings, and
    # records 25-33 from the NLDAS/Noah-LSM model run.
    tg = TimeGrid(
            time_file_tuples = [
                (datetime.strptime(p.stem.split("_")[1], "%Y%m%d-%H"),p)
                for p in nldas_paths],
            labels = [
                'TMP', 'SPFH', 'PRES', 'UGRD', 'VGRD', 'DLWRF', 'NCRAIN',
                'CAPE', 'PEVAP', 'APCP', 'DSWRF', 'SOILM-0-100', 'SOILM-0-10',
                'SOILM-10-40', 'SOILM-40-100', 'SOILM-100-200', 'LSOIL-0-10',
                'LSOIL-10-40', 'LSOIL-40-100', 'LSOIL-100-200']
            ).subset(init_time,final_time)
    px_arrays = tg.extract_timeseries(pixels, features, nworkers=nworkers)
    return px_arrays, tg.times

if __name__=="__main__":
    data_dir = Path("data")
    static_pkl = data_dir.joinpath("static/nldas2_static_all.pkl")
    static_dict = pkl.load(static_pkl.open("rb"))

    # Dimensions of the .npy file subgrids wrt original nldas grid
    subgrid_dir = data_dir.joinpath("timegrid_y64-192_x200-328")
    # The provided TimeGrid is assumed to already be sliced to these bounds;
    # they are only here as a notation for posterity.
    yrange, xrange = slice(64, 192), slice(200,328)

    # Feature labels of the datasets to extract.
    y_feats = ['SOILM-0-10','SOILM-10-40', 'SOILM-40-100','SOILM-100-200']
    x_feats = ["TMP","PRES","NCRAIN","SPFH","DLWRF","DSWRF","PEVAP"] + y_feats
    feats = sorted(list(set(x_feats+y_feats)))

    # Initial and final time of the extracted time series (None for full range)
    init_time = datetime(2021, 1, 1)
    final_time = datetime(2021, 12, 31)

    # Output pickled dictionary
    out_pkl = Path("data/buffer/tmp.pkl")

    #'''
    """
    Use a GUI to select and extract a set of pixels.

    Data is stored as a list of (T,F) shaped arrays for T timesteps and F
    features, with each member of the list corresponding to the same-index
    pixel that was selected.
    """
    # Use the soil composition RGB for pixel selection
    pixels = gt.get_category(static_dict["soil_comp"][yrange,xrange])
    data_pixels, times = get_forcings(
            subgrid_dir=subgrid_dir,
            pixels=pixels,
            features=feats,
            init_time=init_time,
            final_time=final_time
            )

    dataset = {
            # List of (T,F) time series corresponding to each pixel
            "data":data_pixels,
            # Feature labels corresponding to 2nd axis members of pixel arrays
            "data_labels":feats,
            # Datetimes corresponding to 1st axis members of pixel arrays
            "times":times,
            # Pixel slices indicating the subgrid ranges of the TimeGrid
            "subgrid_slices":(yrange, xrange),
            # Indeces of each selected pixel ON THE SUBGRID
            "pixel_idx":pixels,
            }
    with out_pkl.open("wb") as pklfp:
        pkl.dump(dataset, pklfp)
    #'''

    #'''
    with out_pkl.open("rb") as pklfp:
        dataset = pkl.load(pklfp)
    #'''
    data_pixels = dataset["data"]
    times = dataset["times"]

    """ Preprocess extracted pixel data """
    # Outputs are the forward-difference of each timestep,
    Y = [np.diff(A[:,[feats.index(f) for f in y_feats]], axis=0)
         for A in data_pixels]
    # Inputs don't include the last timestep due to differencing.
    X = [A[:,[feats.index(f) for f in x_feats]][:-1]
         for A in data_pixels]
    times = times[:-1]
    M, means, stdevs = pp.gauss_norm(X[0], axis=1)
    print(type(M), type(means), type(stdevs))

    print(X[0].shape, Y[0].shape)
    X, Y, times = pp.double_window_slide(
            X=X[0],
            Y=Y[0],
            look_back=12,
            look_forward=8,
            times=times,
            )
    print(X.shape, Y.shape, len(times))
