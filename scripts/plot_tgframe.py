"""
Quick script for plotting the slopetype field from a GDAS parameter file
"""
import numpy as np
import pickle as pkl
import imageio
from pathlib import Path

#from testbed import plotting

def _mp_raw_img_from_tgframe(kwargs):
    """ """
    return raw_img_from_tgframe(**kwargs)

def raw_img_from_tgframe(tgframe_pkl, norm_bounds:dict={}, use_alpha=False):
    """ """
    pass


if __name__=="__main__":
    root_dir = Path("/rhome/mdodson/testbed")
    tgframe_dir = Path("/rstor/mdodson/timegrid_frames/daily")

