import h5py
import pickle as pkl
import json
import numpy as np
from datetime import datetime,timedelta
from pathlib import Path
from pprint import pprint as ppt

from testbed import generators
from testbed import eval_gridstats
from testbed.list_feats import derived_feats

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/testbed")
    timegrid_dir = Path("/rstor/mdodson/thesis/timegrids")
    static_pkl_path = proj_root_dir.joinpath(
            "data/static/nldas_static_cropped.pkl")
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))
    frames_dir = Path("/rstor/mdodson/timegrid_frames/daily")

    dynamic_feats = [
            "soilm-10","soilm-40","soilm-100","soilm-200",
            "rsm-10","rsm-40","rsm-100","rsm-200",
            "ssrun", "bgrun", "weasd", "apcp"
            ]
    ## specify which features to capture the discrete integral rather than
    ## the overall difference between initial and final times.
    record_sum = ["ssrun", "bgrun", "apcp"]
    ## define which regions and years to provide to the generator
    region_substrs = [
            "y000-098_x000-154", ## nw
            "y000-098_x154-308", ## nc
            "y000-098_x308-462", ## ne
            "y098-195_x000-154", ## sw
            "y098-195_x154-308", ## sc
            "y098-195_x308-462", ## se
            ]
    year_range = (1992, 2024)
    #init_time = datetime(1992,12,0,0)
    #final_time = datetime(2022,7,0,0)
    #final_time = datetime(2009,7,1,0)
    init_time = datetime(2012,1,1,0)
    final_time = datetime(2018,1,1,0)

    ## get a dict of valid timegrids per region
    timegrid_paths = {
            rs:[(*eval_gridstats.parse_timegrid_path(p),p)
                for p in timegrid_dir.iterdir() if rs in p.stem]
            for rs in region_substrs
            }

    ## Iterate over each region requested to be included.
    gens = []
    region_slices = []
    for reg,tgs in timegrid_paths.items():
        ## restrict to timegrids within the year range
        tgs = list(filter(lambda tg:tg[0][0] in range(*year_range), tgs))

        ## extract full-grid slices from timegrids and ensure they are uniform
        _,tg_yranges,tg_xranges,tgs = zip(*tgs)
        assert all(yr==tg_yranges[0] for yr in tg_yranges[1:])
        assert all(xr==tg_xranges[0] for xr in tg_xranges[1:])
        reg_slice = (slice(*tg_yranges[0]), slice(*tg_xranges[0]))
        region_slices.append(reg_slice)

        ## extract the valid mask from the external full-grid static data
        ## and apply the regional slice.
        m_valid = sdata[slabels.index("m_valid")][*reg_slice].astype(bool)
        gens.append(
            generators.gen_timegrid_series(
                timegrid_paths=tgs,
                dynamic_feats=dynamic_feats,
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
            )
    idxs = []
    static = []
    agg_labels = ["mean", "min", "max", "stdev", "sum_or_diff"]
    first_loop = True
    print(gens)
    while True:
        try:
            cur_batches = [next(g) for g in gens]
        except StopIteration as e:
            break
        tmp_darrays = []
        for slc,(d,s,t,ix) in zip(region_slices,cur_batches):
            ## only collect indices and static data once since invariant
            if first_loop:
                static.append(s)
                ix += np.array([slc[0].start, slc[1].start])[np.newaxis]
                idxs.append(ix)
            sum_or_diff = np.stack([
                    np.sum(d[...,i], axis=0)
                    if l in record_sum
                    else d[-1,...,i]-d[0,...,i]
                    for i,l in enumerate(dynamic_feats)
                    ], axis=-1)
            print(sum_or_diff.shape, d.shape)
            tmp_darrays.append(np.stack([
                np.average(d, axis=0),
                np.amin(d, axis=0),
                np.amax(d, axis=0),
                np.std(d, axis=0),
                sum_or_diff,
                ], axis=-1).astype(np.float32))
        if first_loop:
            static = np.concatenate(static, axis=0)
            idxs = np.concatenate(idxs, axis=0)
            first_loop = False
        darray = np.concatenate(tmp_darrays, axis=0)
        t0 = datetime.fromtimestamp(int(t[0])).strftime("%Y%m%d-%H%M")
        tmp_pkl_path = frames_dir.joinpath(f"tgframe_{t0}.pkl")
        labels = (dynamic_feats, agg_labels, record_sum)
        print(tmp_pkl_path.name, darray.shape)
        pkl_data = (labels, darray, static, idxs)
        pkl.dump(pkl_data, tmp_pkl_path.open("wb"))
