from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pickle as pkl
import h5py
from datetime import datetime

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

def _curate_samples(args):
    """

    args := (h5_path, times, static, valid_mask, sample_length,
            feat_idxs, chunk_shape, seed, chunk_idx)
    chunks := integer coordinates of the starting point of all assigned chunks
    """
    CACHE_SIZE = 256 * 1024**2 ## default 256MB cache
    h5_path,times,static,valid_mask,sample_length, \
            feat_idxs,chunk_shape,seed,chunk_idx = args
    print(h5_path)
    feats = h5py.File(
            h5_path.as_posix(),
            mode="r",
            rdcc_nbytes=CACHE_SIZE,
            )["/data/feats"]
    y_slice = (chunk_idx[0],chunk_idx[0]+chunk_shape[0])
    x_slice = (chunk_idx[1],chunk_idx[1]+chunk_shape[1])
    ## Check if any valid points are in range. If not, return None
    chunk_mask = valid_mask[y_slice[0]:y_slice[1],x_slice[0]:x_slice[1]]
    if not np.any(chunk_mask):
        return None,None

    ## Otherwise, select the full time range within the valid chunk
    chunk = feats[:,y_slice[0]:y_slice[1],x_slice[0]:x_slice[1],:]
    static_chunk = static[y_slice[0]:y_slice[1],x_slice[0]:x_slice[1],:]
    print(chunk.shape)
    ## Transpose the chunk so shape is (y_chunk, x_chunk, time, feature)
    chunk = np.transpose(chunk, (1,2,0,3))
    print(chunk.shape)
    ## Apply mask so shape is (valid_pixel, time, feature)
    chunk = chunk[chunk_mask][...,feat_idxs]
    static_chunk = static_chunk[chunk_mask]
    ## Partition the chunk into equal-interval sequences(pixel,sequence)
    chunk = chunk[:,:sample_length*(times.size//sample_length)]
    chunk = chunk.reshape((chunk.shape[0],-1,sample_length,chunk.shape[-1]))
    times = times.reshape((-1,sample_length))
    #times = times[:sample_length*(times.size//sample_length)]
    static_chunk = np.stack([static_chunk for i in range(chunk.shape[1])],axis=1)
    print(chunk.shape, static_chunk.shape, times.size)
    chunk = chunk.reshape((
        chunk.shape[0]*chunk.shape[1], chunk.shape[2],chunk.shape[3]
        ))
    static_chunk = static_chunk.reshape((
        static_chunk.shape[0]*static_chunk.shape[1],static_chunk.shape[2]
        ))
    print(chunk.shape, static_chunk.shape, times.shape)
    return None,None

def curate_samples(
        h5_path:Path, times:np.array, static:np.array, valid_mask:np.array,
        sample_length:int, feat_idxs, chunk_shape:tuple,
        seed=None, workers=1):
    """
    :@param h5_path: Path to a hdf5 file containing a (T,M,N,F) shaped array of
        T consecutive times, M latitude and N longitude points, and F features.
    :@param times: 1d numpy array of T integer timesteps labeling the the first
        axis of the h5 file, which are stored alongside static data per sample.
    :@param static: (M,N,Q) shaped array of Q static features on the (M,N) grid
    :@param valid_mask: (M,N) shaped boolean masks setting valid values to True
    :@param sample_length: Number of consecutive feature state observations
        to include in a single sample (granularity of partitions in time seq).
    :@param chunk_shape: (M,N) geographic shape of h5 chunks (2nd & 3rd axes)
    :@param chunks_per_worker:
    """
    assert h5_path.exists()
    rng = np.random.default_rng(seed=seed)
    feats = h5py.File(h5_path.as_posix(), "r")["/data/feats"]
    ## (T,) timestamps must match the size of the time axis
    assert feats.shape[0] == times.size
    ## (M,N,Q) static array must match the geographic shape of the domain
    assert static.shape[:2]==feats.shape[1:3]
    dy,dx = chunk_shape
    M,N = feats.shape[1],feats.shape[2]
    idxs = np.indices((M//dy,N//dx))*np.expand_dims(np.array([dy,dx]),(1,2))
    idxs = idxs.reshape(2,-1).T
    rng.shuffle(idxs)
    rng.shuffle(idxs)
    args = [
        (h5_path,times,static,valid_mask,sample_length,
            feat_idxs,chunk_shape,seed,idxs[i])
        for i in range(idxs.shape[0])
        ]

    with Pool(workers) as pool:
        for samples,static in pool.imap_unordered(_curate_samples,args):
            if samples==None or static==None:
                continue
            print(samples,static)

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

    ## Get the initial time for the year
    t_0 = int(datetime(year=2015, month=1, day=1).strftime("%s"))
    curate_samples(
            h5_path=h5_paths[0],
            times=np.array([t_0+60*60*i for i in range(24*365)]),
            static=static,
            valid_mask=m_valid,
            sample_length=72,
            feat_idxs=window_feat_idxs,
            chunk_shape=(16,16),
            workers=1
            )

    '''
    """ Iterate through annualized samples to get normalization averages. """
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
    '''
