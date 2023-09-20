from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime
import multiprocessing as mp

from grib_tools import get_grib1_data

def _extract_nldas_subgrid(args:tuple):
    """
    Extract a subgrid of the specified nldas grib1 file storing a series of
    records, and save it to a .npy binary file in the provided directory.

    args = [grib1_path, file_time, vert_slice, horiz_slice, output_dir,
            record_list, data_label]
    """
    try:
        ftype = args[6] #args[0].name.split("_")[1]
        time = args[1].strftime("%Y%m%d-%H")
        new_path = Path(args[4].joinpath(f"{ftype}_{time}.npy"))
        if new_path.exists():
            print(f"Skipping {new_path.as_posix()}; exists!")
            return
        alldata,info,_ = get_grib1_data(args[0])
        data = []
        # append records in order
        for r in args[5]:
            data.append(next(
                alldata[i].data for i in range(len(alldata))
                if info[i]["record"]==r
                ))
        data = np.dstack(data)[args[2],args[3]]
        np.save(new_path, data)
    except Exception as e:
        #print(f"FAILED: {args[0]}")
        print(e)

def mp_extract_nldas_subgrid(grib_files:list, file_times:list, v_bounds:slice,
                             h_bounds:slice, out_dir:Path, records:list,
                             data_label:str, nworkers:int=1):
    """
    Multiprocessed method to extract a pixel subgrid of NLDAS2-grid grib1
    forcing files, which includes data from the NLDAS run of the Noah-LSM.
    Extracted array subgrids are stored as ".npy" serial files

    :@param grib_files: List of valid grib1 files to extract arrays from.
    :@param file_times: List of datetimes associated with each grib1 file.
    :@param v_bounds: Slice of vertical coordinates to subset the array.
    :@param h_bounds: Slice of horizontal coordinates to subset the arrays.
    :@param out_dir: Directory to deposit new ".npy" files in
    :@param records: List of record numbers from wgrib for data to keep in
        the serial files.
    :@param data_label: Identifying string for the dataset being extracted.
        This is the first underscore-separated field in the generated ".npy"
        arrays, followed by the date in YYYYmmdd-HH format.
    :@param nworkers: Number of subprocesses to spawn concurrently in order
        to extract files.
    """
    assert len(file_times)==len(grib_files)
    with mp.Pool(nworkers) as pool:
        args = [(Path(grib_files[i]), file_times[i], v_bounds, h_bounds,
                 out_dir, records, data_label)
                for i in range(len(grib_files))]
        results = pool.map(_extract_nldas_subgrid, args)

def merge_grids(label_a:str, label_b:str, new_label:str, data_dir:Path):
    """
    Simple and high-level method to merge collections of TimeGrid-style ".npy"
    arrays along the feature axis, saving them as new grids.

    This method assumes...
    - both array collections are in the same directory
    - both collections conform to underscore-separated naming like
      <data label>_<YYYYmmdd-HH time>.npy
    - both collections have identical timesteps within the directory
    - both collections have identically-shaped first and second axes

    If label_a refers to (M,N,A) shaped data with 'A' features and label_b to
    (M,N,B) data with 'B' features, the final array will be (M,N,A+B) shaped,
    such that the 'A' dataset's features are indexed first, extended by 'B'.

    :@param label_a: Unique string equivalent to the first underscore-separated
        field of every ".npy" file in the 'A' dataset.
    :@param label_b: Unique string equivalent to the first underscore-separated
        field of every ".npy" file in the 'B' dataset.
    :@param new_label: New unique string to use as the first field of ".npy"
        files generated as a combination of datasets 'A' and 'B'
    :@param data_dir: directory both where array files are retrieved and where
        new combined data arrays are serialized and deposited.
    """
    paths_a = [p.name for p in data_dir.iterdir() if label_a in p.name]
    paths = [(data_dir.joinpath(p),
              data_dir.joinpath(p.replace(label_a,label_b)),
              datetime.strptime(p.split("_")[1],"%Y%m%d-%H.npy"))
             for p in paths_a]
    assert all(p[0].exists() and p[1].exists() for p in paths)
    for a,b,t in paths:
        X = np.dstack((np.load(a),np.load(b)))
        new_path = data_dir.joinpath(f"{new_label}_{t.strftime('%Y%m%d-%H')}")
        print(f"Saving {new_path.as_posix()}")
        #np.save(new_path,X)

if __name__=="__main__":
    data_dir = Path("data")
    out_dir = data_dir.joinpath("subgrids")
    grib_dir = data_dir.joinpath("noahlsm_2021")
    v_bounds = slice(64,192)
    h_bounds = slice(200,328)
    # records = list(range(1,12)) # nldas2 (all)
    records = list(range(25,34)) # noahlsm (SOILM + LSOIL fields)

    #'''
    import gesdisc
    grib_files, file_times = zip(*[
        (f,gesdisc.nldas2_to_time(f)) for f in grib_dir.iterdir()
         if "NOAH0125" in f.name])

    mp_extract_nldas_subgrid(
            grib_files=grib_files,
            file_times=file_times,
            v_bounds=v_bounds,
            h_bounds=h_bounds,
            out_dir=out_dir,
            #records=list(range(1,12)), # nldas2 (all)
            records=list(range(25,34)), # noahlsm (SOILM + LSOIL)
            nworkers=4,
            )
    #'''
    #merge_grids("FORA0125","NOAH0125","newlabel",Path("data/old_tg"))
