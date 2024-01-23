from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime
import multiprocessing as mp
from pprint import pprint as ppt
from krttdkit.acquire import grib_tools, gesdisc
import h5py

from list_feats import nldas_record_mapping, noahlsm_record_mapping

def extract_feats(nldas_grib_dir:Path, noahlsm_grib_dir:Path, out_h5file:Path,
                  wgrib_bin="wgrib", workers=1):
    """
    Multiprocessed method for converting directories of NLDAS2 and Noah-LSM
    grib1 files (acquired from the GES DISC DAAC) into a single big hdf5 file
    containing only the records specified above in noahlsm_record_mapping and
    nldas_record_mapping, adhering to the order that they appear in the lists.

    The files in each directory must be uniformly separated in time (ie same
    dt between each consecutive file), and each NLDAS2 file must directly
    correspond to a simultaneous NoahLSM file.
    """
    ## Refuse to overwrite an existing hdf5 output file.
    assert not out_file.exists()
    ## Find matching-time pairs of NLDAS2 and NoahLSM grib files, and store
    ## as a list of 3-tuples like (datetime, nldas_path, noahlsm_path).
    nldas_files = list(sorted(
        [(gesdisc.nldas2_to_time(f),f) for f in nldas_grib_dir.iterdir()],
        key=lambda t:t[0]
        ))
    noahlsm_files = list(sorted(
        [(gesdisc.nldas2_to_time(f),f) for f in noahlsm_grib_dir.iterdir()],
        key=lambda t:t[0]
        ))
    assert len(nldas_files)==len(noahlsm_files)
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

    ## Extract a sample grid
    tmp_time,tmp_nldas,tmp_noah = file_pairs[0]
    nldas_data,nldas_info,nldas_geo = grib_tools.get_grib1_data(
            tmp_nldas, wgrib_bin=wgrib_bin)
    noah_data,noah_info,noah_geo = grib_tools.get_grib1_data(
            tmp_noah, wgrib_bin=wgrib_bin)

    h5_shape = (
            len(times),
            *nldas_data[0].shape,
            len(nldas_record_mapping) + len(noahlsm_record_mapping),
            )
    nl_recs,nl_labels = map(list,zip(*nldas_record_mapping))
    no_recs,no_labels = map(list,zip(*noahlsm_record_mapping))

    print(f"hdf5 feature shape: {h5_shape}")

    ## Use a ~64 MB cache
    csize = 64*1024**2
    with h5py.File(out_file.as_posix(), "w-", rdcc_nbytes=csize) as f:
        g = f.create_group("/data")
        d = g.create_dataset(
                name="feats",
                shape=h5_shape,
                chunks=(24,16,16,h5_shape[-1]),
                compression="gzip",
                )
        d.dims[0].label = "time"
        d.dims[1].label = "lat"
        d.dims[2].label = "lon"
        d.attrs["labels"] = nl_labels + no_labels

        fill_count = 0

        with mp.Pool(workers) as pool:
            args = [(nl,no,nldas_info,noah_info,nl_recs,no_recs)
                    for t,nl,no in file_pairs]
            for r in pool.imap(_parse_file, args):
                d[fill_count:fill_count+1,:,:,:] = r
                print(f"Completed timestep {times[fill_count]}")
                fill_count += 1

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
    nldas_path, noahlsm_path, nldas_info, noahlsm_info, \
            nldas_records, noahlsm_records = args
    try:
        ## append requested records in order
        tmp_data = grib_tools.get_grib1_grid(nldas_path)
        for r in nldas_records:
            all_data.append(next(
                tmp_data[i].data for i in range(len(tmp_data))
                if nldas_info[i]["record"]==r
                ))
    except Exception as e:
        print(f"FAILED: {nldas_path.name}")
        raise e
    try:
        tmp_data = grib_tools.get_grib1_grid(noahlsm_path)
        for r in noahlsm_records:
            all_data.append(next(
                tmp_data[i].data for i in range(len(tmp_data))
                if noahlsm_info[i]["record"]==r
                ))
        #print(f"Extracted from {noahlsm_path.name}")
    except Exception as e:
        print(f"FAILED: {noahlsm_path.name}")
        raise e
    ## Return the combined features as a (1, y, x, f) array
    return np.expand_dims(np.dstack(all_data), axis=0)

if __name__=="__main__":
    ## Directories should contain only files that should be loaded to the hdf5
    #data_dir = Path("data")
    year = 2020
    data_dir = Path("/rstor/mdodson/thesis/")
    out_dir = Path("data")
    nldas_grib_dir = data_dir.joinpath(f"nldas2/{year}")
    noahlsm_grib_dir = data_dir.joinpath(f"noahlsm/{year}")
    out_file = out_dir.joinpath(f"feats_{year}.hdf5")

    ## from old domain
    #v_bounds = slice(64,192)
    #h_bounds = slice(200,328)

    extract_feats(
            nldas_grib_dir=nldas_grib_dir,
            noahlsm_grib_dir=noahlsm_grib_dir,
            out_h5file=out_file,
            workers=7,
            wgrib_bin="/rhome/mdodson/.micromamba/bin/wgrib",
            )
