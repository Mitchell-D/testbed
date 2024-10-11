"""
Method to extract data from a continuous series of hourly NLDAS forcing and
Noah model outputs as a 'timegrid' style hdf5 file, which serves as an
intermediate data format facilitating efficient sampling and analysis.
"""
import numpy as np
import pickle as pkl
import json
import multiprocessing as mp
import h5py
from pathlib import Path
from datetime import datetime
from itertools import chain
from pprint import pprint as ppt

from krttdkit.acquire import grib_tools, gesdisc

from list_feats import nldas_record_mapping, noahlsm_record_mapping

def extract_timegrid(nldas_grib_paths:Path, noahlsm_grib_paths:Path,
        static_pkl_path:Path, out_h5_dir:Path, out_path_prepend_str:str,
        nldas_labels:list, noahlsm_labels:list, subgrid_x=32, subgrid_y=32,
        time_chunk=128, space_chunk=16, feat_chunk=8, wgrib_bin="wgrib",
        crop_y=(0,0), crop_x=(0,0), valid_mask=None, fill_value=9999.,
        workers=1):
    """
    Multiprocessed method for converting directories of NLDAS2 and Noah-LSM
    grib1 files (acquired from the GES DISC DAAC) into a single big hdf5 file
    containing only the records specified above in noahlsm_record_mapping and
    nldas_record_mapping, adhering to the order that they appear in the lists.

    The files in each directory must be uniformly separated in time (ie same
    dt between each consecutive file), and each NLDAS2 file must directly
    correspond to a simultaneous NoahLSM file.

    :@param nldas_grib_dir: Dir containing NLDAS2 data from GES DISC
    :@param noahlsm_grib_dir: Dir containing NLDAS2 NoahLSM data from GES DISC
    :@param out_h5_dir: Directory where extracted hdf5 files are placed
    :@param out_path_template: String out file stem prepended to generated
        file namees. This method appends '_yNNN_xMMM.h5' to indicate the bounds
        of each subgrid.
    :@param subgrid_x: Number of horizontal grid points per subgrid file
    :@param subgrid_y: Number of vertical grid points per subgrid file
    :@param time_chunk: Number of hourly files to chunk in each generated file
    :@param wgrib_bin: Binary file to wgrib (used to extract metadata)
    :@param workers:
    """
    ## Refuse to overwrite an existing hdf5 output file.
    #assert out_h5_dir.exists()
    #assert not out_file.exists()

    ## Find matching-time pairs of NLDAS2 and NoahLSM grib files, and store
    ## as a list of 3-tuples like (datetime, nldas_path, noahlsm_path).
    nldas_files = list(sorted(
        [(gesdisc.nldas2_to_time(f),f) for f in nldas_grib_paths],
        key=lambda t:t[0]
        ))
    noahlsm_files = list(sorted(
        [(gesdisc.nldas2_to_time(f),f) for f in noahlsm_grib_paths],
        key=lambda t:t[0]
        ))
    assert len(nldas_files)==len(noahlsm_files)

    ## Pair files by the acquisition time reported in the file name
    file_pairs = [
            (nldas_files[i][0],nldas_files[i][1],noahlsm_files[i][1])
            for i in range(len(nldas_files))
            if nldas_files[i][0] == noahlsm_files[i][0]
            ]

    ## Verify that the time steps between frames are consistent according
    ## to the acquisition time in the file name
    times = [t[0] for t in file_pairs]
    dt = times[1]-times[0]
    assert all(b-a==dt for b,a in zip(times[1:],times[:-1]))

    ## Extract a sample grid for setting hdf5 shape
    tmp_time,tmp_nldas,tmp_noah = file_pairs[0]
    nldas_data,nldas_info,_ = grib_tools.get_grib1_data(
            tmp_nldas, wgrib_bin=wgrib_bin)
    noah_data,noah_info,_ = grib_tools.get_grib1_data(
            tmp_noah, wgrib_bin=wgrib_bin)

    ## Determine the total shape of all provided files
    crop_y0,crop_yf = crop_y
    crop_x0,crop_xf = crop_x
    ## Make a spatial slice tuple for sub-gridding dynamic and static data
    crop_slice = (
            slice(crop_y0,nldas_data[0].shape[0]-crop_yf),
            slice(crop_x0,nldas_data[0].shape[1]-crop_xf)
            )
    full_shape = (
            len(times),
            nldas_data[0].shape[0]-crop_y0-crop_yf,
            nldas_data[0].shape[1]-crop_x0-crop_xf,
            len(nldas_labels) + len(noahlsm_labels),
            )

    print(f"hdf5 feature shape: {full_shape}")

    ## establish slices over the spatial dimensions that describe each file
    y_bins = list(subgrid_y*np.arange(full_shape[1]//subgrid_y+1))
    y_offset = full_shape[1]%subgrid_y
    y_bins += [[],[y_bins[-1]+y_offset]][y_offset>0]

    x_bins = list(subgrid_x*np.arange(full_shape[2]//subgrid_x+1))
    x_offset = full_shape[2]%subgrid_x
    x_bins += [[],[x_bins[-1]+x_offset]][x_offset>0]

    y_slices = [slice(ya,yb) for ya,yb in zip(y_bins[:-1],y_bins[1:])]
    x_slices = [slice(xa,xb) for xa,xb in zip(x_bins[:-1],x_bins[1:])]
    out_slices = [
            (y_slices[j], x_slices[i])
            for j in range(len(y_slices))
            for i in range(len(x_slices))
            ]

    ## Extract static data from the pkl made by get_static_data
    static_labels,static_data = pkl.load(static_pkl_path.open("rb"))
    static_data = np.stack(static_data, axis=-1)
    static_data = static_data[*crop_slice]
    chunk_shape = (time_chunk, space_chunk, space_chunk, feat_chunk)
    fg_dict_dynamic = {
            "clabels":("time","lat","lon"),
            "flabels":tuple(nldas_labels+noahlsm_labels),
            "meta":{ ## extract relevant info parsed from wgrib
                "nldas":[(d["name"], d["param_pds"], d["lvl_str"])
                    for d in nldas_info],
                "noah":[(d["name"], d["param_pds"], d["lvl_str"])
                    for d in noah_info],
                }
            }
    fg_dict_static = {
            "clabels":("lat","lon"),
            "flabels":tuple(static_labels),
            "meta":{}
            }
    out_paths = []
    out_h5s = []
    ## initialize the hdf5 files with datasets for static, dynamic, and time
    ## data formatted so that they can initialize FeatureGrid classes.
    for s in out_slices: #zip(out_h5s,out_paths,out_slices):
        sgstr = f"_y{s[0].start:03}-{s[0].stop:03}" +  \
                f"_x{s[1].start:03}-{s[1].stop:03}.h5"
        new_h5_path = out_h5_dir.joinpath(out_path_prepend_str+sgstr)
        out_paths.append(new_h5_path)
        f = h5py.File(
                name=new_h5_path,
                mode="w-",
                rdcc_nbytes=128*1024**2, ## use a 128MB cache
                )
        out_h5s.append(f)
        dynamic_shape = (
                len(times),
                s[0].stop-s[0].start,
                s[1].stop-s[1].start,
                len(nldas_labels)+len(noahlsm_labels),
                )
        g = f.create_group("/data")
        ## create datasets for dynamic, static, and timestep data
        d_dynamic = g.create_dataset(
                name="dynamic",
                shape=dynamic_shape,
                chunks=chunk_shape,
                compression="gzip"
                )
        d_static = g.create_dataset(
                name="static",
                shape=(*dynamic_shape[1:3],len(static_labels)),
                compression="gzip"
                )
        d_times = g.create_dataset(name="time", shape=(len(times),))
        ## add the FeatureGrid-like json dictionaries to the attributes
        g.attrs["dynamic"] = json.dumps(fg_dict_dynamic)
        g.attrs["static"] = json.dumps(fg_dict_static)
        ## load the static data corresponding to this slice
        d_static[...] = static_data[*s]
        ## load the epoch int timesteps associated with this set of grib files
        d_times[...] = np.array([int(t.strftime("%s")) for t in times])
        print(f"Initialized {new_h5_path.as_posix()}")

    nl_rec_dict = dict(t[::-1] for t in nldas_record_mapping)
    no_rec_dict = dict(t[::-1] for t in noahlsm_record_mapping)
    nldas_records = [nl_rec_dict[k] for k in nldas_labels]
    noahlsm_records = [no_rec_dict[k] for k in noahlsm_labels]

    cur_chunk = []
    fill_count = 0
    chunk_idx = 0
    with mp.Pool(workers) as pool:
        args = [(nl,no,nldas_info,noah_info,nldas_records,noahlsm_records)
                for t,nl,no in file_pairs]
        for r in pool.imap(_parse_file, args):
            ## If a mask of valid grid values is provided, fill the invalid
            if not valid_mask is None:
                r[np.logical_not(valid_mask)] = fill_value
            ## Crop according to the user-provided boundaries, which are
            ## applied AFTER flipping to the proper vertical orientation
            r = r[*crop_slice]
            cur_chunk.append(r)
            if len(cur_chunk)==time_chunk:
                cur_chunk = np.stack(cur_chunk, axis=0)
                print(f"Loading chunk with shape: {cur_chunk.shape}")
                for i in range(len(out_h5s)):
                    ds = out_h5s[i]["/data/dynamic"]
                    ds[chunk_idx:chunk_idx+time_chunk,...] = \
                            cur_chunk[:,*out_slices[i],:]
                chunk_idx += time_chunk
                cur_chunk = []
            print(f"Completed timestep {times[fill_count]}")
            fill_count += 1
        ## After extracting all files, load any remaining partial chunks
        if len(cur_chunk) != 0:
            cur_chunk = np.stack(cur_chunk, axis=0)
            for i in range(len(out_h5s)):
                ds = out_h5s[i]["/data/dynamic"]
                ds[chunk_idx:chunk_idx+cur_chunk.shape[0]] = \
                        cur_chunk[:,*out_slices[i],:]
    for f in out_h5s:
        f.close()

def _parse_file(args:tuple):
    """
    Extract the specified nldas and noahlsm grib1 files and return the
    requested records as a uniform-size 3d array for each timestep like
    (lat, lon, feature) in the same order as the provided records, with
    all nldas records coming first, then noahlsm records.

    Note that this method only returns the consequent numpy array, so the
    record labels, geolocation, info dicts need to be kept track of externally
    (although the record entry in the info dict is used to order the array).

    args := (nldas_path, noahlsm_path,
             nldas_info, noahlsm_info,
             nldas_records, noahlsm_records)
    """
    all_data = []
    nldas_path,noahlsm_path,nldas_info,noahlsm_info, \
            nldas_records, noahlsm_records = args
    ## extract all the data from the files
    nldas_data = np.stack(grib_tools.get_grib1_grid(nldas_path), axis=-1)
    noahlsm_data = np.stack(grib_tools.get_grib1_grid(noahlsm_path), axis=-1)

    ## Extract record numbers in the order they appear in the file, then
    ## make a list of file record indexes in the user-requested order.
    nldas_file_records = tuple(nl["record"] for nl in nldas_info)
    noah_file_records = tuple(no["record"] for no in noahlsm_info)
    nldas_idxs = tuple(nldas_file_records.index(r) for r in nldas_records)
    noahlsm_idxs = tuple(noah_file_records.index(r) for r in noahlsm_records)

    nldas_data = nldas_data[...,nldas_idxs]
    noahlsm_data = noahlsm_data[...,noahlsm_idxs]
    ## flip zonally since the grib files are flipped for some reason.
    all_data = np.concatenate((nldas_data, noahlsm_data), axis=-1)[::-1]
    return all_data

if __name__=="__main__":
    ## Directories should contain only files that should be loaded to the hdf5
    #data_dir = Path("data")
    wgrib_bin = "/nas/rhome/mdodson/.micromamba/envs/learn/bin/wgrib"
    data_dir = Path("data")
    static_pkl = data_dir.joinpath("static/nldas_static.pkl")
    out_dir = data_dir.joinpath("timegrids_new/")
    nldas_grib_dirs = data_dir.joinpath(f"nldas2").iterdir()
    noahlsm_grib_dirs = data_dir.joinpath(f"noahlsm").iterdir()

    static_labels,static_data = pkl.load(static_pkl.open("rb"))
    m_valid = static_data[static_labels.index("m_valid")]

    extract_years = [2023]
    #extract_years = list(range(2013,2022))
    ## Separate months into quarters
    extract_months = (
            (1,(1,2,3)),
            (2,(4,5,6)),
            (3,(7,8,9)),
            (4,(10,11,12)),
            )
    ## maximum zonal and meridional per-file subgrid tile size
    subgrid_x,subgrid_y = 154,98

    ## Collect the analysis time associated with each file
    nldas_paths_times = [
            (p,gesdisc.nldas2_to_time(p))
            for p in chain(*map(lambda p:p.iterdir(), nldas_grib_dirs))
            ]
    noahlsm_paths_times = [
            (p,gesdisc.nldas2_to_time(p))
            for p in chain(*map(lambda p:p.iterdir(), noahlsm_grib_dirs))
            ]

    ## use the features and ordering specified in list_feats.py
    _,nl_labels = map(list,zip(*nldas_record_mapping))
    _,no_labels = map(list,zip(*noahlsm_record_mapping))

    ## Extract features for each requested year/month combination as a
    ## 'timegrid' shaped like (T,P,Q,F) for T timesteps per month on a PxQ
    ## spatial subgrid (maximum bounds from subgrid_x/subgrid_y) with F feats.
    for y,(q,ms) in [(y,m) for m in extract_months for y in extract_years]:
        nldas_paths = [
                p for p,t in nldas_paths_times
                if t.year==y and t.month in ms
                ]
        noahlsm_paths = [
                p for p,t in noahlsm_paths_times
                if t.year==y and t.month in ms
                ]
        extract_timegrid(
                nldas_grib_paths=nldas_paths,
                noahlsm_grib_paths=noahlsm_paths,
                static_pkl_path=static_pkl,
                out_h5_dir=out_dir,
                subgrid_x=subgrid_x,
                subgrid_y=subgrid_y,
                out_path_prepend_str=f"timegrid_{y}q{q}",
                nldas_labels=nl_labels,
                noahlsm_labels=no_labels,
                time_chunk=256,
                space_chunk=32,
                feat_chunk=8,
                workers=11,
                crop_y=(29,0), ## 29 pixels North of first CONUS pixel
                crop_x=(2,0),  ## 2 pixels West of first CONUS pixel
                valid_mask=m_valid,
                wgrib_bin=wgrib_bin,
                )
