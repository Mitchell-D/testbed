""" General methods for interfacing with the grib format """

import pygrib
import numpy as np
from pathlib import Path
import multiprocessing as mp
import subprocess
import shlex
#from http.cookiejar import CookieJar
#import urllib

def _parse_pixels(args:tuple, debug=False):
    """
    :@param args: tuple of args like (file path, list of 2d index tuples)
    :@return: (N,B(,..)) shaped array for N pixels each with B records.
    """
    if debug:
        print(f"Opening {args[0]}")
    try:
        data,_,_ = get_grib1_data(args[0])
        return np.dstack(data)[tuple(zip(*args[1]))]
    except Exception as e:
        raise Exception(f"Issue with file {args[0]}:\n{e}")


def grib_parse_pixels(pixels:list, grib1_files:list, chunk_size:int=1,
                      workers:int=None,debug=False):
    """
    Opens a grib1 file and extracts the values of pixels at a list of
    indeces like [ (j1,i1), (j2,i2), (j3,i3), ], returning a list of
    tuples like [ (fname, [val1,val2,val3]), (fname, [val1,val2,val3]), ]

    This method uses a multiprocessing Pool by default in order to parallelize
    the pixel parsing without having many arrays open simultaneously.

    Assumes the provided grib1 file paths are valid, and that each of the
    records in the file are on a uniform grid indexed in the first 2 dims
    by the provided list of pixels.

    :@param pixels: list of 2-tuple indeces for the first 2 dimensions of all
        valid grib1 arrays, for each of the requested components.
    :@param grib1_files: list of string file paths to grib1 files containing
        records on equally-sized grids. No geographic coordinates are
        retrieved, so they are assumed to be consistent or otherwise available.
    :@param chunk_size: Number of values assigned to each thread at a time
    """
    # Using imap, extract values from each 1st and 2nd dim index location
    with mp.Pool(workers) as pool:
        args = [(g,pixels,debug) for g in grib1_files]
        results = list(pool.imap(_parse_pixels, args))
    return results

def wgrib_tuples(grb1:Path):
    """
    Calls wgrib on the provided file as a subprocess and returns the result
    as a list of tuples corresponding to each record, the tuples having string
    elements corresponding to the available fields in the grib1 file.
    """
    wgrib_command = f"wgrib {grb1.as_posix()}"
    out = subprocess.run(shlex.split(wgrib_command), capture_output=True)
    return [tuple(o.split(":")) for o in out.stdout.decode().split("\n")[:-1]]

def wgrib(grb1:Path):
    """
    Parses wgrib fields for a grib1 file into a dict of descriptive values.
    See: https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib/readme
    """
    return [{"record":int(wg[0]),
             "name":wg[3],
             "lvl_str":wg[11], # depth level
             "mdl_type":wg[12], # Model type; anl or fcst
             "date":wg[2].split("=")[-1],
             "byte":int(wg[1]),
             "param_pds":int(wg[4].split("=")[-1]), # parameter/units
             "type_pds":int(wg[5].split("=")[-1]), # layer/level type
             "vert_pds":int(wg[6].split("=")[-1]), # Vertical coordinate
             "dt_pds":int(wg[7].split("=")[-1]),
             "t0_pds":int(wg[8].split("=")[-1]),
             "tf_pds":int(wg[9].split("=")[-1]),
             "fcst_pds":int(wg[10].split("=")[-1]), # Forecast id
             "navg":int(wg[13].split("=")[-1]), # Number of grid points in avg
             } for wg in wgrib_tuples(grb1)]

def get_grib1_data(grb1_path:Path):
    """
    Parses grib1 file into a series of scalar arrays of the variables,
    geographic coordinate reference grids, and information about the dataset.

    :@param grb1_path: Path of an existing grb1 file file with all scalar
        records on uniform latlon grids.
    :@return: (data, info, geo) such that:
        data -> list of uniform-shaped 2d scalar arrays for each record
        info -> list of dict wgrib results for each record, in order.
        geo  -> 2-tuple (lat,lon) of reference grid, assumed to be uniform
                for all 2d record arrays.
    """
    f = grb1_path
    assert f.exists()
    gf = pygrib.open(f.as_posix())
    geo = gf[1].latlons()
    gf.seek(0)
    # Only the first entry in data is valid for FORA0125 files, the other
    # two being the (uniform) lat/lon grid. Not sure how general this is.
    data = [ d.data()[0] for d in gf ]
    return (data, wgrib(f), geo)
