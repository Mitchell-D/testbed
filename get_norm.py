from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pickle as pkl
import h5py

from list_feats import noahlsm_record_mapping, nldas_record_mapping
from model_methods import gen_hdf5_sample

def collect_norm_coeffs(h5_paths:list, nsamples:int, mask_valid,
        workers=1, seed:int=None):
    """ """
    ## Open a mem map of hdf5 files with (time, lat, lon, feat) datasets
    assert all(f.exists() for f in h5_paths)
    feats = [h5py.File(f.as_posix(), "r")["/data/feats"] for f in h5_paths]
    ## All dataset shapes except the first dimension must be uniform shaped
    grid_shape = feats[0].shape[1:]
    assert all(s.shape[1:]==grid_shape for s in feats[1:])

    ### Make a randomized vector of valid indeces
    rng = np.random.default_rng(seed=seed)
    idxs = np.vstack(np.where(mask_valid))
    rng.shuffle(idxs, axis=1)
    rng.shuffle(idxs, axis=1)
    rng.shuffle(idxs, axis=1)
    rng.shuffle(idxs, axis=1)
    assert idxs.shape[1] >= nsamples
    idxs = np.vstack((idxs[:,:nsamples],
        rng.integers(0,len(h5_paths),nsamples))).T

    ctr = 0
    args = [(idxs[i][0], idxs[i][1], h5_paths[idxs[i][2]].as_posix())
            for i in range(idxs.shape[0])]
    means = np.zeros(shape=(nsamples,grid_shape[-1]))
    stdevs = np.zeros(shape=(nsamples,grid_shape[-1]))
    mins = np.zeros(shape=(nsamples,grid_shape[-1]))
    maxs = np.zeros(shape=(nsamples,grid_shape[-1]))
    with Pool(workers) as pool:
        for tmp_mean,tmp_stdev,tmp_min,tmp_max \
                in pool.imap(_collect_norm_coeffs, args):
            means[ctr] = tmp_mean
            stdevs[ctr] = tmp_stdev
            mins[ctr] = tmp_min
            maxs[ctr] = tmp_max
            print(args[ctr])
            ctr += 1
    print(np.average(means, axis=0))
    print(np.average(stdevs, axis=0))
    print(np.average(mins, axis=0))
    print(np.average(maxs, axis=0))
    return (means, stdevs, mins, maxs)

def curate_samples(h5_path:Path, nsamples:int, times:list, ):
    pass

def _collect_norm_coeffs(args):
    """
    args := (vidx:int, hidx:int, h5_path:Path)

    :@return: 2-tuple (mean, stdev) each as (feats,) shaped ndarrays
    """
    vidx,hidx,h5_path = args
    f = h5py.File(h5_path, "r")["/data/feats"]
    X = f[:,vidx,hidx,:]
    return (np.average(X, axis=0),np.std(X,axis=0),
            np.amin(X,axis=0),np.amax(X,axis=0))

def get_value_mask(h5_path:Path, feat_idx=0, val_to_mask=9999):
    """
    Returns a boolean mask of the 1 & 2 dimensions of a feature hdf5,
    with any occurences of the value set to True
    """
    f = h5py.File(h5_path, "r")["/data/feats"]
    return f[0,...,0] == val_to_mask

if __name__=="__main__":
    data_dir = Path("data")
    window_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf",
            "soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    horizon_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf"]
    pred_feats = ['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200']
    static_feats = ["pct_sand", "pct_silt", "pct_clay", "elev", "elev_std"]

    static_path = data_dir.joinpath("static/nldas_static.pkl")
    ## Mask values derived directly from NLDAS data (instead of static netCDFs)
    invalid_path = data_dir.joinpath("static/mask_9999.npy")
    h5_paths = [data_dir.joinpath(f"feats/feats_{y}.hdf5")
            for y in range(2015,2022)]
    _,feat_order = zip(*nldas_record_mapping, *noahlsm_record_mapping)

    window_feat_idxs = [feat_order.index(f) for f in window_feats]
    horizon_feat_idxs = [feat_order.index(f) for f in horizon_feats]
    pred_feat_idxs = [feat_order.index(f) for f in pred_feats]

    """ Load information from static data """
    slabels,sdata = pkl.load(static_path.open("rb"))
    ## Load the static data that goes directly to the model
    static = np.dstack([sdata[slabels.index(l)] for l in static_feats])
    ## Load ancillary static datasets
    lat = sdata[slabels.index("lat")]
    lon = sdata[slabels.index("lon")]
    int_veg = sdata[slabels.index("int_veg")]
    m_conus = sdata[slabels.index("m_conus")]

    '''
    """ Add the 9999 value mask to a .npy file """
    m_9999 = get_value_mask(h5_paths[0], feat_idx=0, val_to_mask=9999.)
    assert not invalid_path.exists()
    np.save(invalid_path, m_9999)
    '''

    """ Construct a geographic mask setting valid data points to True"""
    ## Geographically constrain to the South East
    '''
    m_lon = np.logical_and(-100<=lon,lon<=-80)
    m_lat = np.logical_and(30<=lat,lat<=40)
    m_geo = np.logical_and(m_lon, m_lat)
    '''
    ## Restrict dominant surface type to soil surfaces
    m_water = (int_veg == 0)
    m_bare = (int_veg == 12)
    m_urban = (int_veg == 13)
    m_soil = np.logical_not(np.logical_or(np.logical_or(
        m_water, m_urban), m_bare))
    ## Make sure all values are valid by default
    m_not9999 = np.logical_not(np.load(invalid_path))
    ## Make a collected mask with all valid points set to True
    m_valid = np.logical_and(np.logical_and(m_soil, m_not9999), m_conus)

    means,stdevs,mins,maxs = collect_norm_coeffs(
            h5_paths=h5_paths,
            mask_valid=m_valid,
            nsamples=1000,
            workers=23,
            )
    pkl.dump(
        (means,stdevs,mins,maxs),
        data_dir.joinpath("feat_coeffs.pkl").open("wb")
        )
