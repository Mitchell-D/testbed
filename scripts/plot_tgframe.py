"""
Quick script for plotting the slopetype field from a GDAS parameter file
"""
import json
import numpy as np
import pickle as pkl
import imageio
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint

#from testbed import plotting

def _mp_raw_imgs_from_tgframe(kwargs):
    """ """
    return raw_imgs_from_tgframe(**kwargs)

def raw_imgs_from_tgframe(tgframe_pkl:Path, imgs_dir:Path, gen_pairs:list=None,
        norm_bounds:dict={}, cmap="nipy_spectral", cmap_variations={},
        use_alpha=False, debug=False):
    """
    Generate a series of raw images based on the pixel data in a timegrid frame
    pkl file create by extract_timegrid_frame.py

    :@param tgframe_pkl: Path to a valid timegrid frame pickle file
    :@param gen_pairs: Optional list of 2-tuples (feature, metric) indicating
        a subset of images to plot.
    :@param norm_bounds: 2-level nested dict mapping dynamic feature names to
        subdicts of supported metrics, which are mapped to 2-tuple
        (lower,upper) bounds for norming. If no data is provided, each array
        is normalized within its min/max.
    :@param cmap: Default matplotlib color map to use for conversion to png
    :@param cmap_variations: If you would like for some image types to have a
        color map other than the one specified as `cmap`, provide a 2-level
        nested dict mapping the dynamic feat label to the metric label, then
        the metric label to the matplotlib cmap string.
    :@param use_alpha: If True, the PNG will be transparent where no data is
        provided.
    """
    labels,dynamic,static,idxs = pkl.load(tgframe_pkl.open("rb"))
    dlabels,slabels,mlabels,uses_sum = labels
    yix_max,xix_max = np.amax(idxs[:,0]),np.amax(idxs[:,1])

    if gen_pairs is None:
        gen_pairs = [(df,mf) for df in dlabels for mf in mlabels]

    tmp_time = tgframe_pkl.stem.split("_")[-1]

    for tmp_feat,tmp_metric in gen_pairs:
        X = np.full((yix_max+1, xix_max+1), np.nan)
        tmpd = dynamic[:,dlabels.index(tmp_feat),mlabels.index(tmp_metric)]
        X[idxs[:,0], idxs[:,1]] = tmpd

        ## get normalization bounds
        vmin,vmax = None,None
        if tmp_feat in norm_bounds.keys():
            if tmp_metric in norm_bounds[tmp_feat].keys():
                vmin,vmax = norm_bounds[tmp_feat][tmp_metric]
        if vmin is None:
            vmin = np.nanmin(X)
        if vmax is None:
            vmax = np.nanmax(X)
        ## select the color map
        tmp_cmap = None
        if tmp_feat in cmap_variations.keys():
            if tmp_metric in cmap_variations[tmp_feat].keys():
                tmp_cmap = cmap_variations[tmp_feat][tmp_metric]
        if tmp_cmap is None:
            tmp_cmap = cmap

        ## apply the colormap to get a uint8 image array
        cm = plt.get_cmap(tmp_cmap)
        rgb = cm(np.clip((X-vmin)/(vmax-vmin),0,1), bytes=True)
        if not use_alpha:
            rgb[np.where(~np.isfinite(x))] = 255
            rgb = rgb[...,:-1]
        ## Save the image as a png
        img_name = f"tgframe_{tmp_time}_{tmp_feat}_{tmp_metric}.png"
        imageio.v3.imwrite(uri=imgs_dir.joinpath(img_name), image=rgb)
        if debug:
            print(f"Generated {img_name}")

if __name__=="__main__":
    tgframe_dir = Path("/rstor/mdodson/timegrid_frames/daily2")
    imgs_dir = Path("/rstor/mdodson/timegrid_frames/rgbs")
    norms = json.load(Path(
        "/rhome/mdodson/water-insight-web/datafeed/cmap_default_norms.json"
        ).open("r"))

    dynamic_feats = [
            "soilm-10","soilm-40","soilm-100","soilm-200",
            "rsm-10","rsm-40","rsm-100","rsm-200",
            "ssrun", "bgrun", "weasd", "apcp"
            ]

    def_args = {
            "imgs_dir":imgs_dir,
            #"gen_pairs":None,
            "gen_pairs":[(fl,"sum-or-diff") for fl in dynamic_feats],
            "norm_bounds":norms,
            "cmap":"nipy_spectral",
            "cmap_variations":{},
            "use_alpha":True,
            "debug":False,
            }

    ## use multiple processors to generate images
    #'''
    workers = 30
    args = [{"tgframe_pkl":tgp, **def_args} for tgp in tgframe_dir.iterdir()]
    with Pool(workers) as pool:
        pool.map(_mp_raw_imgs_from_tgframe, args)
    #'''

    ## use a single thread to generate images
    '''
    for tgp in tgframe_dir.iterdir():
        raw_imgs_from_tgframe(
                tgframe_pkl=tgp,
                **def_args,
                )
    '''

    ## collect normalization coefficients.
    ## This could be done in a (Fd,M) array for broadcasting, but
    ## implemented here as a dict to retain easy access to string labels
    '''
    mins,maxs = {},{}
    for tgp in tgframe_dir.iterdir():
        labels,dynamic,static,idxs = pkl.load(tgp.open("rb"))
        dlabels,mlabels,uses_sum = labels
        for j,dl,i,ml in [(j,dl,i,ml)
                for j,dl in enumerate(dlabels)
                for i,ml in enumerate(mlabels)]:
            ## initialize the subdicts if a dynamic label isn't found
            if dl not in mins.keys():
                mins[dl],maxs[dl] = {},{}
                for k,tmp_ml in enumerate(mlabels):
                    mins[dl][tmp_ml] = np.amin(dynamic[:,j,k])
                    maxs[dl][tmp_ml] = np.amax(dynamic[:,j,k])
                continue
            tmp_min = np.amin(dynamic[:,j,i])
            tmp_max = np.amax(dynamic[:,j,i])
            if tmp_min < mins[dl][ml]:
                mins[dl][ml] = tmp_min
            if tmp_max > maxs[dl][ml]:
                maxs[dl][ml] = tmp_max
    pprint(mins)
    pprint(maxs)
    '''
