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

from testbed.list_feats import nldas_record_mapping,noahlsm_record_mapping

def parse_timegrid_path(timegrid_path:Path):
    """
    Parse the timegrid file naming scheme

    timegrid naming template:
    timegrid_{YYYY}q{Q}_y{start_y}-{end_y}_x{start_x}-{end_x}.h5

    :@param timegrid_path: Path to a timegrid-style file.
    :@return: Well-ordered 4-tuple like ((year, quarter), (y0,yf), (x0,xf))
        where (year,quarter) describes the time period of the file, and the
        spatial (y0,yf) and (x0,xf) describe the extent of this regional tile
        in the full array (after the cropping from extract_timegrid)
    """
    _,year_quarter,y_range,x_range = timegrid_path.stem.split("_")
    year,quarter = map(int, year_quarter.split("q"))
    y0,yf = map(int, y_range[1:].split("-"))
    x0,xf = map(int, x_range[1:].split("-"))
    return ((year, quarter), (y0,yf), (x0,xf))

def make_gridstat_hdf5(timegrids:list, out_file:Path, derived_feats:dict=None,
        calculate_hists:bool=False, hist_bins=32, hist_bounds={}, debug=False):
    """
    Calculate pixel-wise monthly min, max, mean, and standard deviation of each
    stored dynamic and derived feature in the timegrids and store the
    statistics alongside static data in a new hdf5 file.

    :@param timegrids: List of timegrids that cover the same spatial domain,
        which will all be incorporated into the pixelwise monthly calculations.
    :@param out_file: Path to a non-existing file to write gridstats to.
    :@param derived_feats: Provide a dict mapping NEW feature labels to a
        3-tuple (dynamic_args, static_args, lambda_str) where the args are
        each tuples of existing dynamic/static labels, and lambda_str contains
        a string-encoded function taking 2 arguments (dynamic,static) of tuples
        containing the corresponding arrays, and returns the subsequent new
        feature after calculating it based on the arguments. These will be
        invoked if the new derived feature label appears in one of the window,
        horizon, or pred feature lists.
    :@param calculate_hists: If True, creates a dataset of pixelwise histograms
        for each feature with hist_bins resolution between the corresponding
        hist_bounds. The bounds MUST be specified for each feature if True.
    :@param hist_bins: Value bins between each feature's bounds for histograms.
    :@param hist_bounds: Dict mapping feature names to 2-tuple (min,max) values
        for calculating histograms. The dict must contain an entry for every
        stored and derived feature when calculate_hists is True.
    """
    ## Collect labels and static data from the timegrids
    tg_shape,tg_dlabels,tg_slabels,tg_static,m_valid = None,None,None,None,None
    tgs_months = []
    ## verify that all provided timegrids are on the same grid and have
    ## uniform features and valid pixel masks.
    for tg in timegrids:
        ## 128MB cache with 256 slots; each chunk is a little over 1/3 MB
        tg_open = h5py.File(tg, "r", rdcc_nbytes=128*1024**2, rdcc_nslots=256)
        if tg_shape is None:
            tg_shape = tg_open["/data/dynamic"].shape
            ## collect dynamic and static feature labels
            tg_dlabels = json.loads(
                    tg_open["data"].attrs["dynamic"])["flabels"]
            tg_slabels = json.loads(
                    tg_open["data"].attrs["static"])["flabels"]
            tg_static = tg_open["/data/static"][...]
            m_valid = tg_static[...,tg_slabels.index("m_valid")]
        else:
            assert tg_shape[1:] == tg_open["/data/dynamic"].shape[1:], \
                    "Timegrid grid shapes & feature count must be uniform"
            tmp_vld = tg_open["/data/static"][...,tg_slabels.index("m_valid")]
            assert np.all(m_valid == tmp_vld)

        tmp_months = np.array([
                datetime.fromtimestamp(int(t)).month
                for t in tuple(tg_open["/data/time"][...])
                ], dtype=np.uint8)
        ## keep the open timegrid file alongside its months mask so that its
        ## buffer persists during iteration over features
        tgs_months.append((tg_open, tmp_months))

    ## convert to a boolean mask
    m_valid = (m_valid == 1.)
    ## collect all feature labels, whether stored  or derived.
    ## process derived features first since they are most likely to fail
    all_flabels =  list(derived_feats.keys()) + tg_dlabels
    F = h5py.File(name=out_file.as_posix(), mode="w-", rdcc_nbytes=256*1024**2)
    ## stats shape for 12 months on (P,Q,F) grid with 4 stats per feature
    stats_shape = (12, *tg_shape[1:3], len(all_flabels), 4)
    ## create chunked hdf5 datasets for gridstats
    G = F.create_dataset(
            name="/data/gridstats",
            shape=stats_shape,
            maxshape=stats_shape,
            chunks=(12,32,32,8,4),
            compression="gzip",
            )
    ## Create and load the static datasets
    S = F.create_dataset(name="/data/static", shape=tg_static.shape)
    S[...] = tg_static

    if calculate_hists:
        hist_shape = (*stats_shape[:4], hist_bins)
        assert all(l in hist_bounds.keys() for l in all_flabels), \
                "Not all derived and stored feature labels were provided " + \
                f"histogram bounds\n{hist_bounds.keys() =}\n{all_flabels =}"
        H = F.create_dataset(
                name="/data/histograms",
                shape=hist_shape,
                maxshape=hist_shape,
                chunks=(12,32,32,8,4),
                dtype="uint64",
                )
        F["data"].attrs["hist_params"] = json.dumps({
            "hist_bounds":hist_bounds,
            "hist_bins":hist_bins
            })

    ## Save labels, derived feats, and source timegrids as attributes
    F["data"].attrs["dlabels"] = json.dumps(all_flabels)
    F["data"].attrs["slabels"] = json.dumps(tg_slabels)
    F["data"].attrs["derived_feats"] = json.dumps(derived_feats)
    F["data"].attrs["timegrids"] = json.dumps([p.name for p in timegrids])

    print("starting to extract months...")
    for fidx,flabel in enumerate(all_flabels):
        if debug:
            print(f"Extracting {flabel}")
        tmp_stats = np.zeros((*stats_shape[:3], 4))
        ## Collect monthly data from all timegrids for each feature.
        ## monthly sub-arrays are (hours,pixels) shaped for each px in m_valid
        month_arrays = [[] for j in range(12)]
        if calculate_hists:
            ## (month, lat, lon, bins) hist array for this feature
            tmp_hists_grid = np.zeros((*hist_shape[:3], hist_bins))
            month_hists = [None for j in range(12)]
        ## iterate over input timegrids
        for tgo,m_months in tgs_months:
            ## iterate over months in this timegrid
            for m in np.unique(m_months):
                m_match = (m_months==m)
                if flabel in tg_dlabels:
                    ix = tg_dlabels.index(flabel)
                    tmp_subarr = tgo["/data/dynamic"][m_match,...,ix]
                    tmp_subarr = tmp_subarr[:,m_valid]
                ## if not a stored feature, key must be a derived feature
                else:
                    ## extract arguments for and evaluate derived features
                    sd_labels,ss_labels,fun = derived_feats[flabel]
                    sd_idxs = tuple(tg_dlabels.index(k) for k in sd_labels)
                    ss_idxs = tuple(tg_slabels.index(k) for k in ss_labels)
                    tmp_subarr = tgo["/data/dynamic"][m_match,...][:,m_valid]
                    sd_args = tuple(tmp_subarr[...,j] for j in sd_idxs)
                    tmp_static = tg_static[m_valid]
                    ss_args = tuple(tmp_static[...,j] for j in ss_idxs)
                    tmp_subarr = eval(fun)(sd_args, ss_args)
                if debug:
                    print(f"{m} {tmp_subarr.shape = }")
                ## subarrays are (times,pixels) shaped for this month
                month_arrays[m-1].append(tmp_subarr)
                if calculate_hists:
                    ## extract histogram bounds and determine bin boundaries
                    ## (including the upper bound bin that cannot be exceeded)
                    hmin,hmax = hist_bounds[flabel]
                    ## Discretize the feature subarray to ints in the bin edges
                    ## and saturate above and below maximum bin sizes
                    hidxs = hist_bins*(tmp_subarr-hmin)/(hmax-hmin)
                    if debug:
                        v_over = np.count_nonzero(hidxs>=hist_bins)
                        v_under = np.count_nonzero(hidxs<0)
                        print(f"saturating {v_under = }, {v_over = }")
                    hidxs = np.clip(np.floor(hidxs), 0, hist_bins-1)
                    hidxs = hidxs.astype(np.uint64)
                    ## (pixels, bins) histogram array for this month/feature
                    tmp_hist = np.zeros((tmp_subarr.shape[1], hist_bins))
                    ## Accumulate the bins to the histogram array
                    for i,ix in enumerate(np.unique(hidxs)):
                        tmp_hist[:,i] = np.sum(hidxs==ix, axis=0)
                    ## Add this histogram to the aggregate monthly totals
                    if month_hists[m-1] is None:
                        month_hists[m-1] = tmp_hist
                    else:
                        month_hists[m-1] += tmp_hist

        ## Collect the monthly data and calculate bulk statistics
        for m,ma in enumerate(month_arrays, 1):
            ma = np.concatenate(ma, axis=0)
            tmp_stats[m-1,m_valid,:] = np.stack((
                np.amin(ma, axis=0),
                np.amax(ma, axis=0),
                np.average(ma, axis=0),
                np.std(ma, axis=0),
                ), axis=-1)
        if debug:
            print(f"Loading {flabel}")
            print(tmp_stats.shape)
        ## Load the bulk statistics to the new hdf5
        G[:,:,:,fidx,:] = tmp_stats
        ## Collect and load the histograms to the new hdf5
        if calculate_hists:
            tmp_hists_grid[:,m_valid,:] = np.stack(month_hists, axis=0)
            H[:,:,:,fidx,:] = tmp_hists_grid
        F.flush()
    for tgo,_ in tgs_months:
        tgo.close()
    return

def collect_gridstats_hdf5s(gridstat_hdf5_paths:list, gridstat_slices:list,
        new_hdf5_path:Path, include_hists=True):
    """
    Concatenate gridstats from each region into a single array over the CONUS
    domain, which aids in determining bulk normalization coefficients and hists

    :@param gridstat_hdf5_paths: List of paths to valid gridstat hdf5s. Each
        file should represent a distinct region of the full grid.
    :@param gridstat_slices: List of slices assigning a pixel range of the
        corresponding gridstat file with respect to the full grid. The ranges
        are typically stored as part of the source timegrid file name, and
        propagated to the subsequent regional gridstat files, and can be parsed
        as such before invoking this method.
    :@param new_hdf5_path: Path to a non-existent new hdf5 file where the
        concatenated output file will be generated.
    :@param include_hists: If True, the histogram datasets of each  gridstat
        file will be chunked and concatentated in the new file as well.
    """
    assert len(gridstat_hdf5_paths)==len(gridstat_slices)
    print(sorted(gridstat_slices))
    ## identify the maximum grid extent for the output dataset
    ystop,xstop = map(max, zip(*[(y.stop, x.stop) for y,x in gridstat_slices]))
    gs_open = [h5py.File(gs, "r") for gs in gridstat_hdf5_paths]
    ## Verify that all input files have the same labels and stats/features
    slabels = tuple(json.loads(gs_open[0]["data"].attrs["slabels"]))
    dlabels = tuple(json.loads(gs_open[0]["data"].attrs["dlabels"]))
    derived_feats = gs_open[0]["data"].attrs["derived_feats"]
    nfeats,nstats = gs_open[0]["/data/gridstats"].shape[-2:]
    assert all(tuple(json.loads(gso["data"].attrs["slabels"]))==slabels
            for gso in gs_open[1:]), "Not all gridstats' static labels match"
    assert all(tuple(json.loads(gso["data"].attrs["dlabels"]))==dlabels
            for gso in gs_open[1:]), "Not all gridstats' dynamic labels match"
    assert all(gso["/data/gridstats"].shape[-2:]==(nfeats,nstats)
            for gso in gs_open[1:]), "Not all gridstats have the same # feats"
    all_timegrids = []
    hbounds,hbins = None,None
    for gso in gs_open:
        ## Compile a list of all source timegrids
        all_timegrids += list(json.loads(gso["data"].attrs["timegrids"]))
        ## Verify histogram bins uniformity if hists are to be extracted
        if include_hists:
            if hbounds is None:
                hparams = json.loads(gso["data"].attrs["hist_params"])
                hbounds = hparams["hist_bounds"]
                hbins = hparams["hist_bins"]
            else:
                tmp_hp = json.loads(gso["data"].attrs["hist_params"])
                assert sorted(tmp_hp["hist_bounds"].items()) \
                        == sorted(hbounds.items()), "hist bounds not uniform"
                assert tmp_hp["hist_bins"] == hbins

    ## Initialize the new hdf5 file
    F = h5py.File(
            name=new_hdf5_path,
            mode="w-",
            rdcc_nbytes=128*1024**2, ## use a 128MB cache
            )
    ## Declare a dataset for the compiled gridstats
    G = F.create_dataset(
            name="/data/gridstats",
            shape=(12, ystop, xstop, nfeats, nstats),
            chunks=(12, 32, 32, 8, 4),
            compression="gzip",
            )
    ## Declare a dataset for pixelwise histograms if requested
    H = None
    if include_hists:
        H = F.create_dataset(
                name="/data/histograms",
                shape=(12, ystop, xstop, nfeats, hbins),
                maxshape=(12, ystop, xstop, nfeats, hbins),
                chunks=(12, 32, 32, 8, hbins),
                dtype="uint64",
                )
        F["data"].attrs["hist_params"] = json.dumps({
            "hist_bounds":hbounds, "hist_bins":hbins
            })
    ## Declare a static data array
    S = F.create_dataset(name="/data/static", shape=(ystop,xstop,len(slabels)))

    ## Add the relevant attributes in the same fashion as the input gridstats
    F["data"].attrs["slabels"] = json.dumps(slabels)
    F["data"].attrs["dlabels"] = json.dumps(dlabels)
    F["data"].attrs["derived_feats"] = derived_feats
    ## Keeping source timegrids pushes attribute size over the limit
    #F["data"].attrs["timegrids"] = all_timegrids

    ## Load the each region's gridstat, static, and (optionally) histogram
    ## data into the new total hdf5 in their individual slice ranges.
    for gso,s in zip(gs_open, gridstat_slices):
        G[:,*s,:,:] = gso["/data/gridstats"][...]
        S[*s,:] = gso["/data/static"][...]
        if include_hists:
            H[:,*s,:,:] = gso["/data/histograms"][...]
        gso.close()
    return new_hdf5_path

def make_monthly_stats_pkl(timegrid:Path):
    """
    Calculate monthly min, max, mean, and stdev of each dynamic feature on the
    grid as a series of (P,Q,F,4) arrays of year/month combinations on the PxQ
    grid of F features, each having 4 stats (min, max, mean, stdev).

    :@param timegrid: Path to an hdf5 timegrid generated by extract_timegrid
    :@return: length M list of 2-tuples (year_months, stats) where year_months
        are 2-tuples (year, month) and the corresponding 'stats' is a (P,Q,F,4)
        array as specified above.
    """
    print(f"Opening {timegrid.name}")
    F = h5py.File(timegrid)
    D = F["/data/dynamic"]
    T = [datetime.fromtimestamp(int(t)) for t in tuple(F["/data/time"][...])]
    ## well-ordered tuples of the year/month combination of each timestep
    all_year_months = [(t.year,t.month) for t in T]
    ## Unique year/month combinations
    unq_year_months = tuple(set(all_year_months))
    mins,maxs,means,stdevs = [],[],[],[]
    stats = []
    for unq_ym in unq_year_months:
        ## Extract grids corresponding to this year/month combination
        print(f"Extracting {unq_ym}")
        m_ym = np.array([ym == unq_ym for ym in all_year_months])
        X = D[m_ym,...]
        tmp_stats = np.zeros((*D.shape[1:], 4))
        tmp_stats[...,0] = np.amin(X, axis=0)
        tmp_stats[...,1] = np.amax(X, axis=0)
        tmp_stats[...,2] = np.average(X, axis=0)
        tmp_stats[...,3] = np.std(X, axis=0)
        stats.append(tmp_stats)
    return list(zip(unq_year_months,stats))

def collect_monthly_stats_pkls(stats_pkl_paths, gridstat_slices,
        new_h5_path, static_pkl_path, feat_labels, chunk_shape=None):
    """
    Quick-and-dirty method to convert regional 'gridstat' files from those
    generated by make_monthly_stats_pkl into a single (T,Y,X,F,4) hdf5 grid
    for T months on a Y,X grid having F features described in terms of by 4
    stats (min, max, mean, stdev).

    Note that the hdf5 spatial grid shape is inferred from the static pkl,
    so make sure that the provided static pkl arrays are on the same grid
    as the combined parent domain of the regional gridstat files

    :@param stats_pkl_paths: List of paths to gridstat hdf5 files
    :@param gridstat_slices: List of 2-tuples containing  slice objects
        corresponding to the y and x position (respectively) of each gridstat
        file's grid on the 2d (Y,X) spatial grid.
    :@param new_h5_path: Path to the hdf5 file that will be generated.
    :@param static_pkl_path: Path to an existing pkl of static values and
        labels (generated by get_static_data) which are stored in a dataset
        alongside the grid stats. The spatial dimensions of the resulting hdf5
        are determined from these grid shapes.
    :@param feat_labels: list of string labels corresponding to the nldas/noah
        features present in the gridstat files (inhereted from timegrids).
    :@param chunk_shape: Shape of hdf5 chunks
    """
    assert len(stats_pkl_paths)==len(gridstat_slices)
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))
    sdata = np.stack(sdata, axis=-1)

    fg_dict_gridstat = {
            "clabels":("month","lat","lon","feat"),
            "flabels":["min", "max", "mean", "stdev"],
            "meta":{
                "nldas_flabels":feat_labels,
                }
            }
    fg_dict_static = {
            "clabels":("lat","lon"),
            "flabels":tuple(slabels),
            "meta":{}
            }

    F = None
    for p,s in zip(stats_pkl_paths,gridstat_slices):
        years_months,stats = pkl.load(p.open("rb"))
        if F is None:
            F = h5py.File(
                    name=new_h5_path,
                    mode="w-",
                    rdcc_nbytes=128*1024**2, ## use a 128MB cache
                    )
            G = F.create_group("/data")
            og_years_months = years_months
            fg_dict_gridstat["meta"]["years_months"] = years_months
            G.attrs["gridstats"] = json.dumps(fg_dict_gridstat)
            G.attrs["static"] = json.dumps(fg_dict_static)
            D = G.create_dataset(
                    name="gridstats",
                    shape=(stats.shape[0],*sdata.shape[:2],*stats.shape[-2:]),
                    chunks=chunk_shape,
                    compression="gzip",
                    )
            S = G.create_dataset(name="static", shape=sdata.shape)
            S[...] = sdata
        assert years_months == og_years_months, \
                "regional gridstat time frames must be identical"
        D[:,*s,:,:] = stats
    F.close()

if __name__=="__main__":
    data_dir = Path("data")
    tg_dir = data_dir.joinpath("timegrids")
    static_pkl_path = data_dir.joinpath("static/nldas_static_cropped.pkl")
    gridstat_dir = Path("data/gridstats/")

    ## Create regional gridstat hdf5 files, which include derived features,
    ## and aggregate monthly data for all years in the provided domain.
    #'''
    from list_feats import derived_feats,hist_bounds
    substr = "y000-098_x000-154" ## NW
    #substr = "y000-098_x154-308" ## NC
    #substr = "y000-098_x308-462" ## NE
    #substr = "y098-195_x000-154" ## SW
    #substr = "y098-195_x154-308" ## SC
    #substr = "y098-195_x308-462" ## SE

    timegrids = sorted([p for p in tg_dir.iterdir() if substr in p.name])

    ## Generate gridstats file over a single region
    '''
    print(timegrids)
    make_gridstat_hdf5(
            timegrids=timegrids,
            out_file=gridstat_dir.joinpath(
                f"gridstats_2012-1_2023-12_{substr}.h5"),
            derived_feats=derived_feats,
            calculate_hists=True,
            hist_bounds=hist_bounds,
            hist_bins=48,
            debug=True,
            )

    exit(0)
    '''

    ## Print out gridstat hdf5 information as a sanity check
    '''
    #substr = "2012-1"
    substr = "gridstats-full"
    gridstat_paths = [p for p in gridstat_dir.iterdir() if substr in p.stem]
    for gsp in gridstat_paths:
        with h5py.File(gsp, "r") as gsf:
            slabels = json.loads(gsf["data"].attrs["slabels"])
            dlabels = json.loads(gsf["data"].attrs["dlabels"])
            ## (M, P, Q, F, 4)
            gstats = gsf["/data/gridstats"][...]
            gstatic = gsf["/data/static"][...]
            m_valid = (gstatic[...,slabels.index("m_valid")]).astype(bool)
            print(f"\n{gsp.stem}")
            print(12*" "+f"{'min min':<14} {'max max':<14} " + \
                    f"{'mean mean':<14} {'mean std':<14}")
            aggstats = np.stack([
                np.amin(gstats[:,m_valid,:,0], axis=(0,1)),
                np.amax(gstats[:,m_valid,:,1], axis=(0,1)),
                np.average(gstats[:,m_valid,:,2], axis=(0,1)),
                np.average(gstats[:,m_valid,:,3], axis=(0,1)),
                ], axis=-1)
            print(aggstats.shape)
            for i,l in enumerate(dlabels):
                print(f"{l:<10}  "+" ".join(
                    [f"{v:<14.3f}" for v in aggstats[i]]))
    exit(0)
    '''

    ## Collect the regional gridstats into a single file
    '''
    gs_paths = [p for p in gridstat_dir.iterdir()
            if "gridstats" in p.name and p.suffix==".h5"]
    ## Parse the slice bounds from the region gridstat file path standard name
    gs_slices = [
            tuple(map(lambda s:slice(*tuple(map(int,s[1:].split("-")))),t))
            for t in [p.stem.split("_")[-2:] for p in gs_paths]]
    collect_gridstats_hdf5s(
            gridstat_hdf5_paths=gs_paths,
            gridstat_slices=gs_slices,
            new_hdf5_path=gridstat_dir.joinpath(
                "gridstats-full_2012-1_2023-12_y000-195_x000-462.h5"),
            include_hists=True,
            )
    '''

    ## Multiprocess over collecting pkl-based gridstats
    '''
    workers = 4
    with Pool(workers) as pool:
        results = []
        for r in pool.imap_unordered(make_monthly_stats_pkl, timegrids):
            print(f"Finished {[t[0] for t in r]}")
            results += r
    '''

    ## Create regional gridstat pickle files, which aggregate monthly
    ## statistics (min, max, mean, stdev) for each feature in a timegrid
    ## (this approach doesn't calculate statistics for derived features).
    '''
    #substr = "y000-098_x000-154"
    #substr = "y000-098_x154-308"
    #substr = "y000-098_x308-462"
    #substr = "y098-195_x000-154"
    #substr = "y098-195_x154-308"
    substr = "y098-195_x308-462"
    workers = 4
    timegrids = [p for p in tg_dir.iterdir() if substr in p.name]
    with Pool(workers) as pool:
        results = []
        for r in pool.imap_unordered(make_monthly_stats_pkl, timegrids):
            print(f"Finished {[t[0] for t in r]}")
            results += r
    years_months,stats = zip(*list(sorted(results, key=lambda r:r[0])))
    stats = np.stack(stats, axis=0)
    pkl_name = f"gridstats_{'-'.join(map(str,years_months[0]))}_" + \
            f"{'-'.join(map(str,years_months[-1]))}_{substr}.pkl"
    pkl.dump((years_months,stats), gridstat_dir.joinpath(pkl_name).open("wb"))
    '''

    ## Aggregate regional gridstat pkl files into a single hdf5
    '''
    gs_paths = [
            p for p in gridstat_dir.iterdir()
            if "gridstats" in p.name and p.suffix==".pkl"]
    ## Parse the slice bounds from the region gridstat file path standard name
    gs_slices = [
            tuple(map(lambda s:slice(*tuple(map(int,s[1:].split("-")))),t))
            for t in [p.stem.split("_")[-2:] for p in gs_paths]]
    ## Assume all labels specified in list_feats are present
    _,nl_labels = map(list,zip(*nldas_record_mapping))
    _,no_labels = map(list,zip(*noahlsm_record_mapping))
    collect_monthly_stats_pkls(
            stats_pkl_paths=gs_paths,
            gridstat_slices=gs_slices,
            new_h5_path=gridstat_dir.joinpath("full_grid_stats.h5"),
            static_pkl_path=static_pkl_path,
            feat_labels=nl_labels+no_labels,
            chunk_shape=(3,64,64,8,4),
            )
    '''

    ## Save overall average values as a numpy array
    '''
    F = h5py.File(gridstat_dir.joinpath("gridstats_full.h5"))
    D = F["/data/gridstats"][...]
    S = F["/data/static"]
    D = np.average(D, axis=0)
    np.save(Path("data/gridstats/gridstats_avg.npy"), D)
    '''

    ## Load a gridstat full-domain average file, reduce its data to a (F_d, 4)
    ## array of mean valid pixel stats (min, max, mean, stdev) for each feat.
    '''
    """ Generate pixel masks for each veg/soil class combination """
    ## Load the full-CONUS static pixel grid
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))
    ## Get the integer-identified soil texture and vegetation class grids
    int_veg = sdata[slabels.index("int_veg")]
    int_soil = sdata[slabels.index("int_soil")]
    m_valid = sdata[slabels.index("m_valid")].astype(bool)

    ## (P,Q,F_d,4) array of statistics for dynamic feats F_d on the (P,Q) grid.
    ## The final dimension indexes the (min, max, mean, stdev) of each feature.
    gridstats = np.load(gridstat_dir.joinpath("gridstats_avg.npy"))
    ## Calculate full-domain averages of all dynamic feature statistics
    gmean,gstdev = map(np.squeeze,np.split(np.mean(
        gridstats[m_valid,:,2:], axis=0), 2, axis=-1))
    gmin = np.amin(gridstats[m_valid,:,0], axis=0)
    gmax = np.amax(gridstats[m_valid,:,1], axis=0)
    _,gslabels = map(tuple,zip(*nldas_record_mapping, *noahlsm_record_mapping))

    for f in set((*window_feats, *horizon_feats, *pred_feats)):
        tmp_idx = gslabels.index(f)
        tmp_min = gmin[tmp_idx]
        tmp_max = gmax[tmp_idx]
        tmp_mean = gmean[tmp_idx]
        tmp_stdev = gstdev[tmp_idx]
        print(f"('{f}', ({tmp_min}, {tmp_max}, {tmp_mean}, {tmp_stdev})),")
    '''

    ## Generate basic scalar RGBs of particular features
    '''
    from krttdkit.visualize import guitools as gt
    from krttdkit.visualize import geoplot as gp
    tmp = D[0,:,:,17,:]
    mask = (tmp[...,1] == 9999.)
    tmp[mask] = 0.
    for i in range(tmp.shape[-1]):
        tmp_rgb = gt.scal_to_rgb(tmp[...,i])
        tmp_rgb[mask] = 0
        gp.generate_raw_image(
                (tmp_rgb*255).astype(np.uint8),
                Path(f"figures/tmp_{i}.png")
                )
    '''

    ## Extract ranges describing the timegrids from their file names, and sort
    ## them by time range, y domain position, then x domain position
    '''
    timegrid_paths = tuple(tg_dir.iterdir())
    timegrid_info,timegrid_paths = tuple(zip(*sorted(zip(
        tuple(map(parse_timegrid_path, timegrid_paths)),
        timegrid_paths
        ))))
    '''
