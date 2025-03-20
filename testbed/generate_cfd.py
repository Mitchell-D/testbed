import h5py
import pickle as pkl
import json
import numpy as np
from datetime import datetime,timedelta
from pathlib import Path
from pprint import pprint as ppt

from testbed import generators

def gen_timegrid_series(
        timegrid_paths, dynamic_feats, static_feats,
        init_epoch:float, final_epoch:float, frequency:int,
        steps_per_batch:int, m_valid=None, include_residual=True,
        derived_feats:dict={}, buf_size_mb=128, max_delta_hours=2, **kwargs):
    """
    Extracts a time series for a set of pixels given contiguous timegrid-style
    hdf5s, time bounds and sequence length, and optionally a user-provided
    valid pixel mask.

    yields gridded samples in chronological order as a 2-tuple like:

    (D, S, T, IX)

    D  : (N, P, F_d)    Dynamic array with N timesteps for P pixels, F_d feats
    S  : (P, F_s)       Static array with P pixels having F_s feats
    T  : (N,)           N epoch float values over the current time range
    IX : (P, 2)         2d integer pixel indeces of each of the P points

    :@param timegrid_paths: List of paths constituting temporally contiguous
        timegrid files uniformly covering the same domain
    :@param dynamic_feats: list of dynamic features to extract, in order.
    :@param static_feats: list of static features to extract, in order.
    :@param init_epoch: First inclusive timestep to extract.
    :@param final_epoch: Exclusive timestep after the last one extracted.
    :@param frequency: Number of timesteps between the initial times of batches
    :@param steps_per_batch: Number of timesteps to include per batch returned.
        If include_residual, this number may be greater than the final batch.
    :@param include_residual: If True, every timestep in [initial, final) will
        be returned even if that means the final batch has fewer elements.
    :@param buf_size_mb: hdf5 buffer size for each timegrid file in MB
    :@param max_delta_hours: Maximum amount of time in hours that samples may
        be separated before an error is raised. This is used for making sure
        consecutive timegrid files are close enough to be sampled across.
    :@param include_init_state_in_predictors: if True, the horizon features for
        the last observed state are prepended to the horizon array, which is
        necessary for forward-differencing if the network is predicting
        residual changes rather than absolute magnitudes.
    """
    timegrid_paths = list(map(Path, timegrid_paths))
    assert all(p.exists() for p in timegrid_paths)
    times = []
    static_dicts = []
    dynamic_dicts = []
    yslice,xslice = None,None
    ## Parse info dicts and timestamps from each file
    for p in timegrid_paths:
        assert p.exists(), p
        with h5py.File(p, mode="r") as tmpf:
            tmp_time = tmpf["/data/time"][...]
            tmp_static = json.loads(tmpf["/data"].attrs["static"])
            tmp_dynamic = json.loads(tmpf["/data"].attrs["dynamic"])
            dshape = tmpf["/data/dynamic"].shape
            if m_valid is None:
                m_valid = np.full(dshape[1:3], True)
            else:
                ## m_valid will be sliced to match the maximum bounds of yslice
                ## and xslice, so only check size on first pass
                if yslice is None:
                    assert m_valid.shape == dshape[1:3]
            if yslice is None:
                ix = np.stack(np.where(m_valid), axis=1)
                ymin,xmin = np.amin(ix, axis=0)
                ymax,xmax = np.amax(ix, axis=0)
                yslice = slice(ymin, ymax+1)
                xslice = slice(xmin, xmax+1)
                m_valid = m_valid[yslice,xslice]
        times.append(tmp_time)
        static_dicts.append(tmp_static)
        dynamic_dicts.append(tmp_dynamic)

    ## All of the timegrids' feature ordering must be uniform
    dynamic_labels = tuple(dynamic_dicts[0]["flabels"])
    static_labels = tuple(static_dicts[0]["flabels"])
    assert all(tuple(d["flabels"])==dynamic_labels for d in dynamic_dicts[1:])
    assert all(tuple(d["flabels"])==static_labels for d in static_dicts[1:])
    if static_feats is None:
        static_feats = static_labels
    if dynamic_feats is None:
        dynamic_feats = dynamic_labels

    ## Determine the index ordering of requested features in the timegrids
    ## and get derived feature information
    fidx,derived,_ = generators._parse_feat_idxs(
        out_feats=dynamic_feats,
        src_feats=dynamic_labels,
        static_feats=static_labels,
        derived_feats=derived_feats,
        )

    sidxs = tuple(static_labels.index(f) for f in static_feats)

    ## Collect each path in order with its valid range, sorted by init time
    time_ranges,times = list(zip(*sorted(
            [((p,t[0],t[-1]),t) for t,p in zip(times,timegrid_paths)],
            key=lambda tr:tr[0][1],
            )))
    conc_times = np.concatenate(times, axis=0)

    ## Make sure provided files are appropriately chronological
    file_diffs = list(zip(time_ranges[:-1], time_ranges[1:]))
    for p0,pf,dt in [(p0,pf,(i-f)) for (p0,_,f),(pf,i,_) in file_diffs]:
        if dt<0:
            raise ValueError(
                    "timegrid files must be ordered chronologically;",
                    f"currently {p0} followed by {pf}")
        if dt>max_delta_hours*60*60:
            raise ValueError(
                    "timegrid files must be adjacent in time; currently",
                    f"{pf} starts {dt} seconds the last time in {p0}")

    ## Only include files with time ranges intersecting the requested bounds.
    init_idx = np.argmin(np.abs(conc_times-init_epoch))
    final_idx = np.argmin(np.abs(conc_times-final_epoch))
    if init_idx < 0:
        raise ValueError(
                "Timegrids must include data before the initial provided time",
                datetime.fromtimestamp(int(conc_times[0])),
                " for the first window."
                )
    if final_idx >= conc_times.size:
        raise ValueError(
                "Timegrids must include data after the final provided time",
                datetime.fromtimestamp(int(conc_times[-1])),
                " for the last horizon."
                )
    init_epoch = conc_times[init_idx]
    final_epoch = conc_times[final_idx]

    ## Reassign the times arrays to only include valid files
    valid_files,times = zip(*[
            (p,t) for (p,t0,tf),t in zip(time_ranges,times)
            if not tf < init_epoch and not t0 >= final_epoch
            ])
    conc_times = np.concatenate(times, axis=0)
    init_idx = np.argmin(np.abs(conc_times - init_epoch))
    final_idx = np.argmin(np.abs(conc_times - final_epoch))

    ## Determine the index boundaries of each sample in the domain of times
    ## only including files overlapping the requested period
    total_size = (final_idx-init_idx)
    if total_size <= frequency:
        num_samples = 1
    else:
        num_samples = total_size // int(frequency)
    ## includes partial steps
    init_step_idxs = np.arange(num_samples) * int(frequency) + init_idx
    final_step_idxs = np.clip(init_step_idxs + steps_per_batch, 0, final_idx)
    if not include_residual:
        m_res = (final_step_idxs-init_step_idxs) < steps_per_batch
        init_step_idxs = init_step_idxs[~m_res]
        final_step_idxs = final_step_idxs[~m_res]

    ## Get a list of files and their index bounds in terms of full time array
    idx_accum = 0
    files_idx_bounds = []
    for f,t in zip(valid_files, times):
        files_idx_bounds.append((f,idx_accum,idx_accum+t.size))
        idx_accum += t.size

    ## Make a list of slices and corresponding files for each of the samples;
    ## some samples may span multiple files.
    grid_slices = []
    cur_file_idx = 0
    for idx0,idxf in zip(*map(list,(init_step_idxs,final_step_idxs))):
        cur_slices = []
        for f,fidx0,fidxf in files_idx_bounds:
            ## Start index within this file's index range
            if fidx0<=idx0<fidxf:
                ## end index also within this file's range
                if idxf<=fidxf:
                    cur_slices.append((f,slice(idx0-fidx0,idxf-fidx0)))
                ## end index beyond this file's range
                else:
                    cur_slices.append((f,slice(idx0-fidx0,fidxf-fidx0)))
                continue
            ## Start and end index ranges surround the entire file range
            elif idx0<=fidx0<idxf and idx0<=fidxf<idxf:
                cur_slices.append((f,slice(0,fidxf-fidx0)))
            ## End index within this file's index range, but not start index
            elif fidx0<idxf<=fidxf:
                cur_slices.append((f,slice(0,idxf-fidx0)))
        grid_slices.append(cur_slices)

    ## Extract subgrids from the timegrid files in chronological order,
    ## according to the sample format, concatenating across files if needed
    open_files = {}
    static_grid = None
    for sample in grid_slices:
        ## Close files that are no longer in use and remove from the dict
        del_keys = []
        for k in open_files.keys():
            if k not in [p for p,_ in sample]:
                open_files[k].close()
                del_keys.append(k)
        for k in del_keys:
            del open_files[k]
        ## Open new files and add them to the dict
        open_files.update({
            tmp_path:h5py.File(
                tmp_path, mode="r",
                rdcc_nbytes=buf_size_mb*1024**2,
                rdcc_nslots=buf_size_mb*15
                )
            for tmp_path,_ in sample
            if tmp_path not in open_files.keys()
            })

        ## Extract the full dynamic grid associated with this sample
        d = np.concatenate(
                [open_files[f]["/data/dynamic"][s,yslice,xslice]
                    for f,s in sample],
                axis=0)[:,m_valid]
        t = np.concatenate(
                [open_files[f]["/data/time"][s] for f,s in sample],
                axis=0)

        if static_grid is None:
            tmp_static = open_files[sample[0][0]]["/data/static"]
            tmp_static = tmp_static[yslice,xslice][m_valid]
            ## extract numeric static values
            s = tmp_static[...,sidxs]

        d = generators._calc_feat_array(
                src_array=d,
                static_array=tmp_static,
                stored_feat_idxs=fidx,
                derived_data=derived,
                )

        yield (d,s,t,ix)

if __name__=="__main__":
    from testbed import eval_gridstats
    from testbed.list_feats import derived_feats

    proj_root_dir = Path("/rhome/mdodson/testbed")
    timegrid_dir = Path("/rstor/mdodson/thesis/timegrids")
    static_pkl_path = proj_root_dir.joinpath(
            "data/static/nldas_static_cropped.pkl")
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))

    ## define which regions and years to provide to the generator
    region_substrs = [
            "y000-098_x000-154", ## nw
            "y000-098_x154-308", ## nc
            "y000-098_x308-462", ## ne
            "y098-195_x000-154", ## sw
            "y098-195_x154-308", ## sc
            "y098-195_x308-462", ## se
            ]
    year_range = (1992, 2012)
    init_time = datetime(1992,12,1,0)
    final_time = datetime(2022,7,1,0)
    #final_time = datetime(2009,7,1,0)

    ## get a dict of valid timegrids per region
    timegrid_paths = {
            rs:[(*eval_gridstats.parse_timegrid_path(p),p)
                for p in timegrid_dir.iterdir() if rs in p.stem]
            for rs in region_substrs
            }

    ## Iterate over each region requested to be included.
    for reg,tgs in timegrid_paths.items():
        ## restrict to timegrids within the year range
        tgs = list(filter(lambda tg:tg[0][0] in range(*year_range), tgs))

        ## extract full-grid slices from timegrids and ensure they are uniform
        _,tg_yranges,tg_xranges,tgs = zip(*tgs)
        assert all(yr==tg_yranges[0] for yr in tg_yranges[1:])
        assert all(xr==tg_xranges[0] for xr in tg_xranges[1:])
        reg_slice = (slice(*tg_yranges[0]), slice(*tg_xranges[0]))

        ## extract the valid mask from the external full-grid static data
        ## and apply the regional slice.
        m_valid = sdata[slabels.index("m_valid")][*reg_slice].astype(bool)
        gen = gen_timegrid_series(
                timegrid_paths=tgs,
                dynamic_feats=[
                    "soilm-10","soilm-40","soilm-100","soilm-200",
                    "rsm-10","rsm-40","rsm-100","rsm-200",
                    "ssrun", "bgrun", "weasd", "apcp"
                    ],
                static_feats=None,
                init_epoch=float(init_time.strftime("%s")),
                final_epoch=float(final_time.strftime("%s")),
                m_valid=m_valid,
                frequency=24,
                steps_per_batch=24,
                include_residual=False,
                derived_feats=derived_feats,
                max_delta_hours=2,
                )
        for (d,s,t,ix) in gen:
            print(datetime.fromtimestamp(int(t[0])))
        break
