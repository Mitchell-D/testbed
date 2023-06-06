#!/Users/mtdodson/opt/anaconda3/envs/aes_osx/bin/python

from aes670hw2 import guitools as gt
from aes670hw2 import enhance as enh
import pickle as pkl
import sys
from pathlib import Path
import numpy as np

def pick_pixels(X:np.ndarray, replace_val:bool=None):
    """
    Choose a subset of pixels from a 2d array or 3d RGB, return their indeces.

    :@param X: (M,N) or (M,N,3) array or masked array to select pixels from.
        If X is 2D, the greyscale array will be rendered as an RGB using a
        default hue range of (0,.66) in HSV space.
    :@param replace_val: if X is a masked array and replace_val is a valid
        scalar value of the same type as X, replaces any masked values with
        the number prior to converting to RGB color space.
    """
    Y = enh.norm_to_uint(X, 256, np.uint8)
    if type(X)==np.ma.MaskedArray and not replace_val is None:
        Y = Y.data
        Y[np.where(X.mask)] = replace_val
    if len(Y.shape)==2:
        Y = gt.scal_to_rgb(Y, hue_range=(0,.66))
    return gt.get_category(Y, fill_color=(0,0,0), show_pool=False)

if __name__=="__main__":
    if len(sys.argv)==1:
        print("No pkl file argument provided")
    pklfile = Path(sys.argv[1])
    assert pklfile.suffix==".pkl" and pklfile.exists()
    print(pick_pixels(pkl.load(pklfile.open("rb"))[::-1], -3))
