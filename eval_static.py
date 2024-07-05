"""
Generate masks of each unique class combination of vegetation and soil pixels,
and plot a grid showing the number of pixels in each combination category.
"""
import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt

from list_feats import umd_veg_classes,statsgo_textures

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

    fig.savefig(fig_path, bbox_inches="tight")
    return matrix

if __name__=="__main__":
    gridstat_dir = Path("data/grid_stats")
    static_pkl_path = Path("data/static/nldas_static_cropped.pkl")

    #'''
    """ Generate pixel masks for each veg/soil class combination """
    ## Load the full-CONUS static pixel grid
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))
    ## Get the integer-identified soil texture and vegetation class grids
    int_veg = sdata[slabels.index("int_veg")]
    int_soil = sdata[slabels.index("int_soil")]
    m_valid = sdata[slabels.index("m_valid")].astype(bool)

    ## Get masks identifying all unique combinations of veg/soil classes
    combos,combo_masks = get_soil_veg_combo_masks(
            veg_ints=int_veg,
            soil_ints=int_soil,
            print_combos=False,
            )

    print(combo_masks.shape)

    ## Restrict counting to valid pixels
    combo_masks = np.logical_and(m_valid[...,np.newaxis], combo_masks)

    ## Make a grid plot of the number of samples within each combination.
    plot_soil_veg_matrix(
            combos=combos,
            combo_masks=combo_masks,
            fig_path=Path("figures/static/veg_soil_combos.png"),
            #cmap="magma_r",
            cmap="plasma",
            norm="linear",
            vmax=3000,
            )
    #'''
