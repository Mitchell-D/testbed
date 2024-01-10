from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pickle as pkl
import random as rand
import h5py
from datetime import datetime

from list_feats import noahlsm_record_mapping, nldas_record_mapping
#from model_methods import gen_hdf5_sample

def collect_norm_coeffs_OLD(h5_paths:list, nsamples:int, mask_valid,
        workers=1, seed:int=None):
    """
    Deprecated since it's really inefficient to iterate sparsely over the full
    grid domain.
    """
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

#def _sum_chunk(chunk_slice):
#    np.sum(np.sum(X,axis=0),axis=0)
#    pass

def collect_norm_coeffs(sample_h5, chunk_depth:int=2048, chunks_per_calc=1000):
    """
    """
    G = h5py.File(sample_h5.as_posix())["data"]
    D = G["dynamic"]
    S = G["static"]
    l_D = G.attrs["flabels"]
    l_S = G.attrs["slabels"]

    N = D.shape[0]*D.shape[1] ## samples in the entire array
    feat_var = np.zeros(D.shape[-1], dtype=np.float64)
    averages = np.zeros(D.shape[-1], dtype=np.float64)
    for cslice in D.iter_chunks():
        chunk = D[cslice]
        z = (chunk==0.0)
        invalid_timesteps = np.count_nonzero(np.all(z, axis=-1))
        print(f"times with all zero: {invalid_timesteps}")
        averages += np.sum(np.sum(chunk,axis=0),axis=0)/N
    for cslice in D.iter_chunks():
        feat_var += np.sum(np.sum((D[cslice]-averages)**2, axis=0), axis=0)
    stdevs = (feat_var/N)**(1/2)
    return averages,stdevs

def _curate_samples(args):
    """

    args := (h5_path, times, static, valid_mask, sample_length,
            feat_idxs, chunk_shape, seed, chunk_idx, pivot_idx)

    h5_path := (t,y,x,f) shaped hdf5 features dataset to draw from
    times := (t,) shaped array of timestamp integers labeling the hdf5
    static := (y,x,q) shaped array providing q static features on the grid
    valid_mask := (y,x) shaped boolean mask marking valid values to True
    sample_length := number of consecutive timesteps to include in each sample
    feat_idxs := order index subset of features to include with each sample
    chunk_shape := (y,x) shaped array giving the spatial size of the chunk
    seed := random number seed to use while shuffling returned data
    chunk_idx := starting index of the chunk assigned to this  process
    pivot_idx := index within each sample that labels the time for that sample.
        This is typically the first index after the window, by convention
        (ie the initial index of the horizon)
    """
    CACHE_SIZE = 256 * 1024**2 ## default 256MB cache
    h5_path,times,static,valid_mask,sample_length,feat_idxs, \
            chunk_shape,seed,chunk_idx,pivot_idx = args
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
        return None,None,None
    ## Otherwise, select the full time range within the valid chunk
    chunk = feats[:,y_slice[0]:y_slice[1],x_slice[0]:x_slice[1],:]
    static_chunk = static[y_slice[0]:y_slice[1],x_slice[0]:x_slice[1],:]
    ## Transpose the chunk so shape is (y_chunk, x_chunk, time, feature)
    chunk = np.transpose(chunk, (1,2,0,3))
    ## Apply mask so shape is (valid_pixel, time, feature)
    chunk = chunk[chunk_mask][...,feat_idxs]
    static_chunk = static_chunk[chunk_mask]
    ## The number of timesteps included must divide the sample count evenly.
    ## Subset the times to an even number of samples, divide into contiguous
    ## samples, and keep only timesteps at the "pivot time" of samples.
    ## Copy it over the valid pixels axis so it acts like a static feature.
    subset_length = sample_length*(times.size//sample_length)
    times = times[:subset_length]
    times = times.reshape((-1,sample_length))
    times = np.vstack([times[:,pivot_idx] for i in range(chunk.shape[0])])
    ## Partition the chunk into equal-interval sequences(pixel,sequence) so
    ## chunk:(p,s,w+h,f) static_chunk:(p,s,q) time:(p,s)
    ## for p individual pixels, s samples per year, w+h timesteps per sample,
    ## f dynamic features, and q static features. Static data is copied along
    ## the s axis per pixel, and timestamps copied along the p axis per sample.
    chunk = chunk[:,:subset_length]
    chunk = chunk.reshape((chunk.shape[0],-1,sample_length,chunk.shape[-1]))
    static_chunk = np.stack([
        static_chunk for i in range(chunk.shape[1])],axis=1)
    ## Combine the pixel and timestep dimensions so the first dimension
    ## contains samples from all times or places in the chunk
    chunk = chunk.reshape((
        chunk.shape[0]*chunk.shape[1], chunk.shape[2],chunk.shape[3]))
    static_chunk = static_chunk.reshape((
        static_chunk.shape[0]*static_chunk.shape[1],static_chunk.shape[2]))
    times = times.reshape((times.shape[0]*times.shape[1]))
    print(chunk.shape, static_chunk.shape, times.shape)
    is9999 = (chunk==9999.)
    if np.any(is9999):
        is9999 = np.all(np.any(is9999, axis=2), axis=1)
        print(f"9999 at {np.unique(times[is9999]),static[is9999,0,-2:]}")
    return chunk,static_chunk,times

def curate_samples(
        feat_h5_path:Path, out_h5_path:Path, times:np.array, static:np.array,
        valid_mask:np.array, sample_length:int, feat_idxs, feat_labels,
        static_labels, chunk_shape:tuple, pivot_idx:int,
        seed=None, workers=1, new_chunk_depth=2048):
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
    :@param pivot_idx: index within a sample which determines the sample's time
    :@param seed: Integer random seed to use for shuffling data
    :@param workers: Number of parallel processes to use in processing chunks.
    :@param new_chunk_depth: The native chunk shape of the created hdf5 file
        is (new_chunk_depth, sample_len, len(feat_idxs)), so this parameter
        sets the number of samples in a hdf5 chunk for the generated file.

    Generated hdf5 file has 1 group "data" with 3 datasets:

    "dynamic":(num_samples,sample_len,len(feat_idxs))
        Contains continuous time samples of features from valid pixels
    "static":(num_samples,static_labels)
        Contains per-sample static features (which don't vary wrt sample_len)
    "time":(num_samples)
        Contains integer epoch time of the element of each sample at pivot_idx
    """
    assert feat_h5_path.exists()
    rng = np.random.default_rng(seed=seed)
    feats = h5py.File(feat_h5_path.as_posix(), "r")["/data/feats"]
    ## (T,) timestamps must match the size of the time axis
    assert feats.shape[0] == times.size
    ## (M,N,Q) static array must match the geographic shape of the domain
    assert static.shape[:2]==feats.shape[1:3]
    assert len(feat_idxs)==len(feat_labels)
    assert len(static_labels)==static.shape[-1]

    ## Calculate the number of samples in the final array
    samples_per_px = times.size//sample_length
    subset_length = sample_length*samples_per_px
    num_samples = samples_per_px*np.count_nonzero(valid_mask)

    ## Create a new hdf5 file
    F = h5py.File(out_h5_path.as_posix(), "w-", rdcc_nbytes=512*1024**2)
    G = F.create_group("/data")
    D = G.create_dataset(
            name="dynamic",
            shape=(num_samples,sample_length,len(feat_idxs)),
            chunks=(new_chunk_depth, sample_length, len(feat_idxs)),
            compression="gzip"
            )
    S = G.create_dataset(
            name="static",
            shape=(num_samples, static.shape[-1]),
            chunks=(new_chunk_depth, static.shape[-1]),
            compression="gzip",
            )
    T = G.create_dataset(name="time", shape=(num_samples,))
    G.attrs["flabels"] = feat_labels
    G.attrs["slabels"] = static_labels

    ## Determine the starting indeces of all h5 feature array chunks
    ## and shuffle them before providing each to a subprocess
    dy,dx = chunk_shape
    M,N = feats.shape[1],feats.shape[2]
    idxs = np.indices((M//dy,N//dx))*np.expand_dims(np.array([dy,dx]),(1,2))
    idxs = idxs.reshape(2,-1).T
    rng.shuffle(idxs)
    rng.shuffle(idxs)

    ## Construct arguments for per-chunk workers
    args = [
        (feat_h5_path,times,static,valid_mask,sample_length,
            feat_idxs,chunk_shape,seed,idxs[i],pivot_idx)
        for i in range(idxs.shape[0])
        ]
    samples,static,times = [],[],[]
    cur_chunk = 0
    ## Spawn subprocesses to parse a single chunk for all contiguous samples
    ## in valid pixels, and load them into a data array shaped like:
    ## (num_samples, sample_length, feats)
    with Pool(workers) as pool:
        for out in pool.imap_unordered(_curate_samples,args):
            tmp_samples,tmp_static,tmp_times = out
            if tmp_samples is None or tmp_static is None or tmp_times is None:
                continue
            samples.append(tmp_samples)
            static.append(tmp_static)
            times.append(tmp_times)
            ## If there are enough entries for new chunks can be loaded into
            ## the h5, do so.
            new_len = sum([X.shape[0] for X in samples])
            new_chunks = new_len//new_chunk_depth
            if new_chunks > 0:
                Xd = np.concatenate(samples, axis=0)
                Xs = np.concatenate(static, axis=0)
                Xt = np.concatenate(times, axis=0)
                for i in range(new_chunks):
                    tmp_slc = slice(i*new_chunk_depth,(i+1)*new_chunk_depth)
                    h5_slc = slice((i+cur_chunk)*new_chunk_depth,
                            (i+cur_chunk+1)*new_chunk_depth)
                    D[h5_slc] = Xd[tmp_slc]
                    S[h5_slc] = Xs[tmp_slc]
                    T[h5_slc] = Xt[tmp_slc]
                samples = [Xd[new_chunks*new_chunk_depth:]]
                static = [Xs[new_chunks*new_chunk_depth:]]
                times = [Xt[new_chunks*new_chunk_depth:]]
                cur_chunk += new_chunks
                print(f"Added {new_chunks}; {samples[0].shape[0]} left over")
        D[cur_chunk*new_chunk_depth:] = np.concatenate(samples, axis=0)
        S[cur_chunk*new_chunk_depth:] = np.concatenate(static, axis=0)
        T[cur_chunk*new_chunk_depth:] = np.concatenate(times, axis=0)
    print(D.shape, cur_chunk*new_chunk_depth, samples[0].shape)
    F.close()


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

def get_h5_subset(h5_path:Path, sample_idx=None):
    """
    Returns a subset of all the fields of a sample-style h5
    """
    F = h5py.File(h5_path, "r")
    G = F["data"]
    r = {**{k:G[k][sample_idx] for k in ("dynamic", "static", "time")},
            **G.attrs}
    F.close()
    return r


def get_value_mask(h5_path:Path, feat_idx=0, val_to_mask=9999):
    """
    Returns a boolean mask of the 1 & 2 dimensions of a feature hdf5,
    with any occurences of the value set to True
    """
    f = h5py.File(h5_path, "r")["/data/feats"]
    return f[0,...,0] == val_to_mask

def shuffle_samples(in_samples_path:Path, out_samples_path:Path,
        batch_depth=2048, seed=None):
    """
    Shuffle the provided samples along the first axis, and store the result
    as a new hdf5 file.
    """
    ## Load the datasets
    G = h5py.File(in_samples_path, "r", rdcc_nbytes=2000*1024**2)["data"]
    D,S,T = tuple(G[k] for k in ("dynamic", "static", "time"))

    ## Create a new hdf5 file with the same structure
    s_F = h5py.File(out_samples_path.as_posix(), "w-", rdcc_nbytes=2000*1024**2)
    s_G = s_F.create_group("/data")
    s_D = s_G.create_dataset(
            name="dynamic",
            shape=D.shape,
            chunks=(batch_depth, *D.shape[1:]),
            compression="gzip"
            )
    s_S = s_G.create_dataset(
            name="static",
            shape=S.shape,
            chunks=(batch_depth, S.shape[-1]),
            compression="gzip",
            )
    s_T = s_G.create_dataset(name="time", shape=(S.shape[0],))
    s_G.attrs["flabels"] = G.attrs["flabels"]
    s_G.attrs["slabels"] = G.attrs["slabels"]

    ## Shuffle the chunks and group them as defined by chunk_entropy
    rng = np.random.default_rng(seed=seed)
    chunk_entropy = 48 ## num chunks shuffled together
    chunk_slices = [c for c,_,_  in list(D.iter_chunks())]
    rand.shuffle(chunk_slices)
    out_chunk_idx = 0
    ## Shuffle samples between the chunks in each group and write to new file
    while len(chunk_slices)>0:
        cur_chunks = chunk_slices[:chunk_entropy]
        del chunk_slices[:chunk_entropy]

        cur_dynamic = np.vstack([D[s] for s in cur_chunks])
        cur_static = np.vstack([S[s] for s in cur_chunks])
        cur_times = np.concatenate([T[s] for s in cur_chunks],axis=0)

        print(cur_dynamic.shape,cur_static.shape,cur_times.shape)

        cur_idxs = np.arange(cur_dynamic.shape[0])
        rng.shuffle(cur_idxs)

        out_slice = slice(out_chunk_idx,out_chunk_idx+cur_idxs.size)
        s_D[out_slice,...] = cur_dynamic[cur_idxs,...]
        s_S[out_slice,...] = cur_static[cur_idxs,:]
        s_T[out_slice,...] = cur_times[cur_idxs]
        out_chunk_idx += cur_idxs.size
        s_F.flush()
        del cur_dynamic,cur_static,cur_times
        print(f"out chunk idx: {out_chunk_idx}")
    ## Close the files
    s_F.close()

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
    static_feats = ["pct_sand", "pct_silt", "pct_clay",
            "elev", "elev_std", "vidx", "hidx"]

    static_path = data_dir.joinpath("static/nldas_static.pkl")
    ## Mask values derived directly from NLDAS data (instead of static netCDFs)
    invalid_path = data_dir.joinpath("static/mask_9999.npy")
    ## Path where a small sample of the hdf5 feature grid is stored.
    sample_path = data_dir.joinpath("sample/feature_sample.npy")
    years,h5_paths,init_epochs = zip(*[(
        y,
        data_dir.joinpath(f"feats/feats_{y}.hdf5"),
        int(datetime(year=y, month=1, day=1, hour=0).strftime("%s"))
        ) for y in range(2015,2022)])
    _,feat_order = zip(*nldas_record_mapping, *noahlsm_record_mapping)

    window_feat_idxs = [feat_order.index(f) for f in window_feats]
    horizon_feat_idxs = [feat_order.index(f) for f in horizon_feats]
    pred_feat_idxs = [feat_order.index(f) for f in pred_feats]

    """ Load information from static data """
    slabels,sdata = pkl.load(static_path.open("rb"))
    ## Load the static data that goes directly to the model
    static = np.dstack([sdata[slabels.index(l)] for l in static_feats])
    static_labels = [slabels[slabels.index(l)] for l in static_feats]
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

    '''
    """ """
    hdf5_sample = get_sample(h5_paths[3], tuple(range(24)))
    np.save(sample_path, hdf5_sample)
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
    m_not9999 = np.logical_not(np.load(invalid_path))[::-1] ## vertically flip
    ## Make a collected mask with all valid points set to True
    m_valid = np.logical_and(np.logical_and(m_soil, m_not9999), m_conus)
    #np.save(Path("data/static/mask_valid.npy"), m_valid)

    ## Get the initial time for the year
    #t_0 = int(datetime(year=2015, month=1, day=1, hour=0).strftime("%s"))
    run_year = 2021
    run_idx = years.index(run_year)
    ## Separate hdf5 files for unshuffled and shuffled samples each year
    run_sample_h5 = Path(f"/rstor/mdodson/thesis/samples_{run_year}.h5")
    run_shuffle_h5 = Path(f"/rstor/mdodson/thesis/shuffle_{run_year}.h5")

    '''
    curate_samples(
            feat_h5_path=h5_paths[run_idx],
            out_h5_path=run_sample_h5,
            times=np.array([ ## Get epoch of every hour in the year
                init_epochs[run_idx]+60*60*i
                for i in range(24*(365,366)[not run_year%4])]),
            static=static[::-1], ## feat h5 is flipped vertically
            valid_mask=m_valid[::-1], ## feat h5 is flipped vertically
            sample_length=72,
            feat_idxs=window_feat_idxs,
            feat_labels=window_feats,
            static_labels=static_labels,
            chunk_shape=(16,16),
            pivot_idx=36,
            workers=6,
            new_chunk_depth=2048,
            )
    '''

    '''
    shuffle_samples(
            in_samples_path=run_sample_h5,
            out_samples_path=run_shuffle_h5,
            batch_depth=2048,
            )
    '''

    #'''
    """ New method for iterating over an entire array for mean/stdev """
    avgs,stdevs = collect_norm_coeffs(
            #sample_h5=run_sample_h5,
            sample_h5=run_shuffle_h5,
            chunk_depth=2048,
            chunks_per_calc=1000,
            )
    print(f"averages = {avgs}")
    print(f"std devia = {stdevs}")
    #'''

    '''
    """
    Select a number of random pixels to contribute feature mean/stdev
    over a full year. Returned statistics are the averaged values
    from nsamples full-year pixels.
    """
    means,stdevs,mins,maxs = collect_norm_coeffs_OLD(
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
