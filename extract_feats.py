from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime
import multiprocessing as mp
from pprint import pprint as ppt
from krttdkit.acquire import grib_tools, gesdisc
import h5py

noahlsm_record_mapping = (
        (1,"nswrs"),        ## net shortwave at surface
        (2,"nlwrs"),        ## net longwave at surface
        (3,"lhtfl"),        ## latent heat flux
        (4,"shtfl"),        ## sensible heat flux
        (5,"gflux"),        ## ground heat flux
        (10,"arain"),       ## liquid precipitation
        (11,"evp"),         ## evapotranspiration
        (12,"ssrun"),       ## surface runoff
        (13,"bgrun"),       ## sub-surface runoff
        (19,"tsoil-10"),    ## depth-wise soil temperature
        (20,"tsoil-40"),
        (21,"tsoil-100"),
        (22,"tsoil-200"),
        (26,"soilm-10"),    ## depth-wise soil moisture content
        (27,"soilm-40"),
        (28,"soilm-100"),
        (29,"soilm-200"),
        (30,"lsoil-10"),    ## depth-wise liquid soil moisture
        (31,"lsoil-40"),
        (32,"lsoil-100"),
        (33,"lsoil-200"),
        (34,"mstav-200"),   ## moisture availability 0-200cm
        (35,"mstav-100"),   ## moisture availability 0-100cm
        (36,"evcw"),        ## canopy water evaporation
        (37,"trans"),       ## transpiration
        (38,"evbs"),        ## bare soil evaporation
        (49,"rsmin"),       ## minimal stomatal resistance
        (50,"lai"),         ## leaf area index
        (51,"veg"),         ## vegetation fraction
        )
nldas_record_mapping = (
        (1,"tmp"),          ## 2m temperature
        (2,"spfh"),         ## 2m specific humidity
        (3,"pres"),         ## surface pressure
        (4,"ugrd"),         ## 10m zonal wind speed
        (5,"vgrd"),         ## 10m meridional wind speed
        (6,"dlwrf"),        ## downward longwave radiative flux
        (7,"ncrain"),       ##
        (8,"cape"),         ## convective available potential energy
        (9,"pevap"),        ## hourly potential evaporation
        (10,"apcp"),        ## hourly precip total
        (11,"dswrf"),       ## downward shortwave radiative flux
        )

'''
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
'''

def extract_feats(nldas_grib_dir:Path, noahlsm_grib_dir:Path, out_h5file:Path,
                  workers=1):
    ## Find matching-time pairs of NLDAS2 and NoahLSM grib files, and store
    ## as a list of 3-tuples like (datetime, nldas_path, noahlsm_path).
    ## Also verify that the time steps between frames are consistent according
    ## to the acquisition time in the file name
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
    times = [t[0] for t in file_pairs]
    dt = times[1]-times[0]
    assert all(b-a==dt for b,a in zip(times[1:],times[:-1]))

    ## Extract a sample grid
    tmp_time,tmp_nldas,tmp_noah = file_pairs[0]
    nldas_data,nldas_info,nldas_geo = grib_tools.get_grib1_data(tmp_nldas)
    noah_data,noah_info,noah_geo = grib_tools.get_grib1_data(tmp_noah)

    h5_shape = (
            len(times),
            *nldas_data[0].shape,
            len(nldas_record_mapping) + len(noahlsm_record_mapping),
            )
    nl_recs,nl_labels = map(list,zip(*nldas_record_mapping))
    no_recs,no_labels = map(list,zip(*noahlsm_record_mapping))

    f = h5py.File(out_file.as_posix(), "w-")
    g = f.create_group("/data")
    d = g.create_dataset(
            name="feats",
            shape=h5_shape,
            chunks=(24,1,1,h5_shape[-1]),
            compression="gzip",
            )
    d.dims[0].label = "time"
    d.dims[1].label = "lat"
    d.dims[2].label = "lon"
    d.attrs["labels"] = nl_labels + no_labels

    fill_count = 0

    with mp.Pool(workers) as pool:
        args = [(nl,no,nl_recs,no_recs)
                for t,nl,no in file_pairs]
        for r in pool.imap(_parse_file, args):
            d[fill_count] = r
            print(f"extracted {times[fill_count]}")
            print(r.shape)
            fill_count += 1

def _parse_file(args:tuple):
    """
    Extract the specified nldas grib1 file and return the requested records
    as a uniform-size 3d array for each timestep like (heigh, width, feature).

    The

    args := (nldas_path, noahlsm_path, nldas_records, noahlsm_records)
    """
    all_data = []
    nldas_path, noahlsm_path, nldas_records, noahlsm_records = args
    try:
        # append requested records in order
        tmp_data,tmp_info,_ = grib_tools.get_grib1_data(nldas_path)
        for r in nldas_records:
            all_data.append(next(
                tmp_data[i].data for i in range(len(tmp_data))
                if tmp_info[i]["record"]==r
                ))
        tmp_data,tmp_info,_ = grib_tools.get_grib1_data(noahlsm_path)
        for r in noahlsm_records:
            all_data.append(next(
                tmp_data[i].data for i in range(len(tmp_data))
                if tmp_info[i]["record"]==r
                ))
    except Exception as e:
        #print(f"FAILED: {args[0]}")
        print(e)
    return np.dstack(all_data)

if __name__=="__main__":
    ## Directories should contain only files that should be loaded to the hdf5
    data_dir = Path("data")
    nldas_grib_dir = data_dir.joinpath("nldas2_2021")
    noahlsm_grib_dir = data_dir.joinpath("noahlsm_2021")
    out_file = data_dir.joinpath("feats_2021.hdf5")

    ## from old domain
    #v_bounds = slice(64,192)
    #h_bounds = slice(200,328)

    extract_feats(
            nldas_grib_dir=nldas_grib_dir,
            noahlsm_grib_dir=noahlsm_grib_dir,
            out_h5file=out_file,
            workers=4,
            )
