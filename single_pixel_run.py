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

from generators import gen_timegrid_samples
from list_feats import nldas_record_mapping,noahlsm_record_mapping
from list_feats import umd_veg_classes, statsgo_textures

def get_soil_veg_combo_masks(veg_ints:np.ndarray, soil_ints:np.ndarray,
        print_combos:bool=False):
    """
    Given integer-labeled class masks for soil and vegetation, calculates and
    returns a grid of boolean masks associated with each distinct combination,
    which is useful for stratified and uniform sampling techniques.

    The classes are expected to be assigned with the integer order defined by
    the corresponding indeces in the label lists list_feats.umd_veg_classes
    and list_feats.statsgo_textures

    :@param veg_ints: Integer array of vegetation classes
    :@param soil_ints: Same-shape integer array of soil classes
    :@param print_combos: If True, prints the labels and comma-separated
        counts for each combination

    :@return: 2-tuple of arrays (combos, combo_masks) where combos is a (N,2)
        shaped array of N valid combinations of (vegetation, soil) classes
        identified by their integers, and combo_masks is a (P,Q,N) shaped array
        of boolean masks over the (P,Q) grid identifying the positions of
        samples of each combination.
    """
    ## Create a (N,2) array of each of the N combinations of (vegetation, soil)
    combos = np.reshape(np.stack(
            np.meshgrid(np.unique(veg_ints), np.unique(soil_ints)),
            axis=-1,
            ), (-1, 2))

    ## Get a (P,Q,N) shaped boolean grid setting samples matching each of the
    ## N possible combinations to True.
    combo_masks = np.stack([
            np.logical_and((veg_ints==combos[i,0]),(soil_ints==combos[i,1]))
            for i in range(combos.shape[0])
            ], axis=-1)

    if print_combos:
        for i in range(combos.shape[0]):
            tmp_veg_label = umd_veg_classes[combos[i,0]]
            tmp_soil_label = statsgo_textures[combos[i,1]]
            tmp_num_samples = np.count_nonzero(combo_masks[...,i])
            print(", ".join(
                (tmp_veg_label, tmp_soil_label, str(tmp_num_samples))
                ))
    return combos,combo_masks

def plot_soil_veg_matrix(combos, combo_masks, fig_path:Path,
        vmax=10000, cmap="magma", norm="linear"):
    unq_veg = tuple(np.unique(combos[:,0]))
    unq_soil = tuple(np.unique(combos[:,1]))

    matrix = np.zeros((len(unq_veg), len(unq_soil)))
    for i in range(combos.shape[0]):
        tmp_veg_idx = unq_veg.index(combos[i,0])
        tmp_soil_idx = unq_soil.index(combos[i,1])
        matrix[tmp_veg_idx,tmp_soil_idx] = np.count_nonzero(combo_masks[...,i])

    fig,ax = plt.subplots()
    cb = ax.imshow(matrix, cmap=cmap, vmax=vmax, norm=norm)
    fig.colorbar(cb)

    # Adding labels to the matrix
    ax.set_yticks(
            range(len(unq_veg)),
            [umd_veg_classes[u] for u in unq_veg],
            )
    ax.set_xticks(
            range(len(unq_soil)),
            [statsgo_textures[u] for u in unq_soil],
            rotation=45,
            ha='right',
            )

    fig.savefig(fig_path)
    return matrix

if __name__=="__main__":
    gridstat_dir = Path("data/grid_stats")
    static_pkl_path = Path("data/static/nldas_static_cropped.pkl")

    """ Generate pixel masks for each veg/soil class combination """
    ## Load the full-CONUS static pixel grid
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))
    ## Get the integer-identified soil texture and vegetation class grids
    int_veg = sdata[slabels.index("int_veg")]
    int_soil = sdata[slabels.index("int_soil")]
    ## Get masks identifying all unique combinations of veg/soil classes
    combos,combo_masks = get_soil_veg_combo_masks(
            veg_ints=int_veg,
            soil_ints=int_soil,
            print_combos=False,
            )
    '''
    ## Make a grid plot of the number of samples within each combination.
    plot_soil_veg_matrix(
            combos=combos,
            combo_masks=combo_masks,
            fig_path=Path("figures/static/veg_soil_combos.png"),
            cmap="magma_r",
            norm="log",
            vmax=3000,
            )
    '''

    timegrid_dir = Path("/rstor/mdodson/thesis/timegrids")
    timegrids = [p.as_posix() for p in timegrid_dir.iterdir()]

    window_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "dswrf", "apcp",
            "soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd",
            ]
    horizon_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "dswrf", "apcp",
            ]
    pred_feats = [ 'soilm-10', 'soilm-40', 'soilm-100', 'soilm-200', "weasd" ]
    static_feats = [ "pct_sand", "pct_silt", "pct_clay", "elev", "elev_std" ]
    int_feats = [ "int_veg" ]

    ## (P,Q,F_d,4) array of statistics for dynamic feats F_d on the (P,Q) grid.
    ## The final dimension indexes the (min, max, mean, stdev) of each feature.
    gridstats = np.load(gridstat_dir.joinpath("gridstats_avg.npy"))
    ## Calculate full-domain averages of all dynamic feature statistics
    gmin,gmax,gmean,gstdev = map(np.squeeze,np.split(np.mean(
            gridstats, axis=(0,1),), 4, axis=-1))
    _,gslabels = map(tuple,zip(*nldas_record_mapping, *noahlsm_record_mapping))

    g = gen_timegrid_samples(
            timegrid_paths=timegrids,
            window_size=24,
            horizon_size=24*14,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            static_int_feats=[("int_veg",14)],
            static_conditions=[
                #("int_veg", lambda a: np.any(np.stack(
                #    [a==v for v in (7,8,9,10,11)], axis=-1
                #    ), axis=-1)),
                #("pct_silt", lambda a: a>.2),
                #("m_conus", lambda a: a==1.),
                ],
            num_procs=6,
            deterministic=False,
            block_size=16,
            buf_size_mb=4096,
            samples_per_timegrid=1024,
            max_offset=23,
            sample_separation=53,
            include_init_state_in_predictors=True,
            load_full_grid=False,
            seed=200007221750,
            )
