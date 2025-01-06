"""
Generate masks of each unique class combination of vegetation and soil pixels,
and plot a grid showing the number of pixels in each combination category.
"""
import numpy as np
import pickle as pkl
import h5py
import json
from pathlib import Path
import matplotlib.pyplot as plt

from testbed.list_feats import umd_veg_classes,statsgo_textures

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
            np.meshgrid(
                np.unique(veg_ints.astype(int)),
                np.unique(soil_ints.astype(int))),
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

    '''
    gridstat_dir = Path("data/grid_stats")
    static_pkl_path = Path("data/static/nldas_static_cropped.pkl")

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
    '''

    #'''
    """ Collect data and make plots for each region independently """
    from eval_timegrid import parse_timegrid_path
    from krttdkit.visualize import geoplot as gp
    timegrid_dir = Path("data/timegrids/")
    times,yranges,xranges = zip(*map(
        parse_timegrid_path,
        sorted(timegrid_dir.iterdir())
        ))
    times,yranges,xranges,paths = zip(*sorted(
        (*parse_timegrid_path(p), p)
        for p in timegrid_dir.iterdir()
        ))
    xranges,yranges = set(xranges),set(yranges)

    unique_regions = []
    for x0,xf in sorted(list(set(xranges))):
        for y0,yf in sorted(list(set(yranges))):
            substr = f"y{y0:03}-{yf:03}_x{x0:03}-{xf:03}"
            unique_regions.append((
                substr,
                next(p for p in timegrid_dir.iterdir() if substr in p.name)
                ))

    ## Collect data for each region independently
    for substr,tmp_path in unique_regions:
        with h5py.File(tmp_path, mode="r") as tmp_file:
            tmp_file = h5py.File(tmp_path, mode="r")
            tmp_static = tmp_file["/data/static"][...]
            tmp_slabels = json.loads(
                    tmp_file["data"].attrs["static"]
                    )["flabels"]
            print(f"\n{tmp_path}")
            tmp_combos,tmp_masks = get_soil_veg_combo_masks(
                    veg_ints=tmp_static[...,tmp_slabels.index("int_veg")],
                    soil_ints=tmp_static[...,tmp_slabels.index("int_soil")],
                    print_combos=True,
                    )

            m_valid = tmp_static[...,tmp_slabels.index("m_valid")].astype(bool)
            tmp_masks = np.logical_and(m_valid[...,np.newaxis], tmp_masks)

            ## Plot a vegetation/soil combination matrix
            plot_soil_veg_matrix(
                    combos=tmp_combos,
                    combo_masks=tmp_masks,
                    fig_path=Path(f"figures/static/combos_{substr}.png"),
                    cmap="plasma",
                    norm="linear",
                    vmax="1000",
                    )
            ## Plot a RGB of soil texture percentages
            rgb_soil = np.stack([
                tmp_static[...,tmp_slabels.index("pct_sand")],
                tmp_static[...,tmp_slabels.index("pct_silt")],
                tmp_static[...,tmp_slabels.index("pct_clay")],
                ], axis=-1)
            rgb_soil = (255*rgb_soil).astype(np.uint8)
            rgb_soil[np.logical_not(m_valid)] = 0
            gp.generate_raw_image(
                    rgb_soil,
                    Path(f"figures/static/texture_{substr}.png")
                    )

        #'''
