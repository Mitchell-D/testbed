from pathlib import Path
import numpy as np
import pickle as pkl
from datetime import datetime
import multiprocessing as mp

from grib_tools import get_grib1_data
import gesdisc

from aes670hw2 import guitools as gt
from aes670hw2 import enhance as enh

def _extract_nldas_grid(args:tuple):
    """
    Extract a subgrid of the specified nldas grib1 file storing a series of
    records, and save it to a .npy binary file in the provided directory

    args = [ grib1_path, vert_slice, horiz_slice, output_dir, record_list ]
    """
    try:
        ftype = args[0].name.split("_")[1]
        time = gesdisc.nldas2_to_time(args[0]).strftime("%Y%m%d-%H")
        new_path = Path(args[3].joinpath(f"{ftype}_{time}.npy"))
        if new_path.exists():
            print(f"Skipping {new_path.as_posix()}; exists!")
            return
        alldata,info,_ = get_grib1_data(args[0])
        data = []
        # append records in order
        for r in args[4]:
            data.append(next(
                alldata[i].data for i in range(len(alldata))
                if info[i]["record"]==r
                ))
        data = np.dstack(data)[args[1],args[2]]
        np.save(new_path, data)
    except Exception as e:
        #print(f"FAILED: {args[0]}")
        print(e)

def mp_extract_nldas_grid(grib_files:list, v_bounds:slice, h_bounds:slice,
                          out_dir:Path, records:list, nworkers:int=1):
    with mp.Pool(nworkers) as pool:
        args = [(Path(f),v_bounds,h_bounds,out_dir,records)
                for f in grib_files]
        results = pool.map(_extract_nldas_grid, args)

if __name__=="__main__":
    data_dir = Path("data")
    static_pkl = data_dir.joinpath("static/nldas2_static_all.pkl")
    subgrid_dir = data_dir.joinpath("subgrids")
    #static = pkl.load(static_pkl.open("rb"))
    v_bounds = slice(64,192)
    h_bounds = slice(200,328)

    #grib_dir = data_dir.joinpath("nldas2_2021")
    grib_dir = data_dir.joinpath("noahlsm_2021")

    mp_extract_nldas_grid(
            grib_files=list(grib_dir.iterdir()),
            v_bounds=v_bounds,
            h_bounds=h_bounds,
            out_dir=subgrid_dir,
            #records=list(range(1,12)), # nldas2 (all)
            records=list(range(25,34)), # noahlsm (SOILM + LSOIL)
            nworkers=4,
            )
    #gt.quick_render(static["soil_comp"][64:192,200:328])
