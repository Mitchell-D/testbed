from pathlib import Path
import random as rand
import numpy as np
import pickle as pkl
import h5py

from list_feats import noahlsm_record_mapping, nldas_record_mapping

def gen_batch(h5_paths:list, static_data:np.array, in_idxs:list, out_idxs:list,
        window:int, horizon:int, seed:int, domain_mask:np.array=None):
    """
    Given a collection of hdf5 paths, gridded static inputs,

    :@param h5_paths: List of hdf5 files containing continuous equal-interval
        arrays shaped like (time, lat, lon, feature)
    :@param static_data: (lat, lon, static_feature) shaped array of data which
        are consistent over time corresponding to each of the grid cells.
    :@param in_idxs: input feature indeces in the order they should be provided
        to the model
    """
    ## Open a mem map of hdf5 files with (time, lat, lon, feat) datasets
    assert all(f.exists() for f in h5_paths)
    feats = [h5py.File(f.as_posix(), "r")["/data/feats"] for f in h5_paths]
    ## All dataset shapes except the first dimension must be uniform shaped
    grid_shape = feats[0].shape[1:]
    assert all(s.shape[1:]==grid_shape for s in feats[1:])
    ## lat/lon components of static data shape must match those of the features
    assert static_data.shape[:2] == grid_shape[:2]
    ## If no domain mask is provided, assume the full grid has valid samples
    if domain_mask is None:
        domain_mask = np.full(grid_shape[:2], True)
    ## Domain mask must be (lat, lon) shaped, matching the feats & static data
    assert domain_mask.shape == grid_shape[:2]

    ## !!! Shuffle pivot idx and grid point selection prior to loop !!!

    rand.seed(seed)
    while True:
        ## Choose the dataset to use
        h5_choice = feats[rand.randrange(len(feats))]
        ## Index of the final observable features (last window time)
        pivot_idx = rand.randrange(window, h5_choice.shape[2]-horizon)


if __name__=="__main__":
    data_dir = Path("data")
    in_feats = ["lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf"]
    out_feats = ['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200']
    static_path = data_dir.joinpath("nldas2_static_all.pkl")
    h5_paths = [data_dir.joinpath(f"feats_{y}.hdf5") for y in range(2015,2022)]
    _,feat_order = zip(*nldas_record_mapping, *noahlsm_record_mapping)

    in_idxs = [feat_order.index(f) for f in in_feats]
    out_idxs = [feat_order.index(f) for f in out_feats]

    static_dict = pkl.load(static_path.open("rb"))
    static = static_dict["soil_comp"]

    g = gen_batch(
            h5_paths=h5_paths,
            static_data=static,
            in_idxs=in_idxs,
            out_idxs=out_idxs,
            window=18,
            horizon=18,
            seed=20231228,
            )
