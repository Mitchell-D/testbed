"""
Quick script for plotting the slopetype field from a GDAS parameter file
"""
import json
import numpy as np
import pickle as pkl
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from datetime import datetime,timedelta

#from testbed import plotting

def mp_tgframes_to_tiffs(args):
    return tgframes_to_tiffs(**args)

def tgframes_to_tiffs(tgframe_path:Path, imgs_dir:Path, plot_feats,
        plot_metrics, save_latlon=False, replace_existing=False):
    """
    Given a timegrid series style tgframe file, extract each timestep as a tiff
    """
    labels,dynamic,static,idxs,times = pkl.load(tgframe_path.open("rb"))
    dlabels,slabels,mlabels,uses_sum = labels
    yix_max,xix_max = np.amax(idxs[:,0]),np.amax(idxs[:,1])
    yix_min,xix_min = np.amin(idxs[:,0]),np.amin(idxs[:,1])
    yix_range = yix_max-yix_min
    xix_range = xix_max-xix_min
    _,dataset,*_ = tgframe_path.stem.split("_")
    times = np.average(times, axis=1)

    if save_latlon:
        latlon_name = f"tgframe_{dataset}_latlon.npy"
        lat = static[...,slabels.index("lat")]
        lon = static[...,slabels.index("lon")]
        latlon =  np.full((yix_range+1, xix_range+1, 2), np.nan)
        latlon[idxs[:,0]-yix_min, idxs[:,1]-xix_min] = \
                np.stack((lat,lon),axis=-1)
        np.save(imgs_dir.joinpath(latlon_name), latlon)

    for i,t in enumerate(times):
        for f in plot_feats:
            for m in plot_metrics:
                tstr = datetime.fromtimestamp(int(t)).strftime("%Y%m%d")
                img_name = f"tgframe_{dataset}_{tstr}_{f}_{m}.tiff"
                img_path = imgs_dir.joinpath(img_name)
                if img_path.exists() and not replace_existing:
                    continue
                X = np.full((yix_range+1, xix_range+1), np.nan)
                tmpd = dynamic[i,:,dlabels.index(f),mlabels.index(m)]
                X[idxs[:,0]-yix_min, idxs[:,1]-xix_min] = tmpd
                im = Image.fromarray(X)
                im.save(img_path.as_posix(), "TIFF")

if __name__=="__main__":
    tgframe_dir = Path("/rstor/mdodson/timegrid_frames/soilm-200")
    imgs_dir = Path("/rstor/mdodson/timegrid_frames/tiffs_soilm-200")

    dynamic_feats = [
            "soilm-200",
            ]

    def_args = {
            "imgs_dir":imgs_dir,
            "plot_feats":["soilm-200"],
            "plot_metrics":["mean"],
            "replace_existing":False,
            "save_latlon":True,
            }

    args = [{"tgframe_path":tgp, **def_args} for tgp in tgframe_dir.iterdir()]

    ## use multiple processors to generate images
    with Pool(5) as pool:
        pool.map(mp_tgframes_to_tiffs, args)

