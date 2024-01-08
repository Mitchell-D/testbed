
import numpy as np
import pickle as pkl
from pathlib import Path

from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.operate import enhance as enh


def m_9999(path:Path, img_path:Path=None):
    """
    Load the 9999-value mask derived from a hdf5 sample timestep, and
    optionally generate an image representing True values as white.
    """
    m_9999 = np.load(path)
    if img_path:
        mask_img = np.full_like(m_9999, 0, dtype=np.uint8)
        mask_img[m_9999] = 255
        gp.generate_raw_image(mask_img, img_path)
    return m_9999

def static(path:Path, print_info=True, img_dir:Path=None):
    slabels,sdata = pkl.load(path.open("rb"))
    for l,s in zip(slabels, sdata):
        if print_info:
            print(l,s.shape,s.dtype)
        if img_dir:
            nanmask = np.isnan(s)
            s[nanmask] = np.nanmean(s)
            rgb = enh.norm_to_uint(
                    gt.scal_to_rgb(enh.linear_gamma_stretch(
                        s.astype(np.float32))),
                    256, np.uint8)
            rgb[nanmask] = 0
            img_path = img_dir.joinpath(f"{l}.png")
            if img_path.exists():
                print(f"Skipping {img_path.as_posix()}; already exists.")
                continue
            gp.generate_raw_image(rgb, img_path)
    return slabels,sdata


if __name__=="__main__":
    m_9999 = m_9999(
            path=Path("data/static/mask_9999.npy"),
            img_path=Path("figures/static/mask_9999.png")
            )
    static = static(
            path=Path("data/static/nldas_static.pkl"),
            img_dir=Path("figures/static")
            )
