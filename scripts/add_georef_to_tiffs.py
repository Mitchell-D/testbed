from pathlib import Path
import numpy as np
import rasterio as rio
from rasterio.transform import Affine

def add_georef_to_tiff(tiff_path:Path, transform,
        target_path=None, crs="EPSG:4326"):
    """
    """
    array = rio.open(tiff_path).read(1)
    print(array.shape)

    if target_path is None:
        target_path = tiff_path

    with rio.open(
            target_path,
            mode="w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=array.dtype,
            crs=crs,
            transform=transform
            ) as new_tiff:
        new_tiff.write(array, 1)

if __name__=="__main__":
    #tiff_dir = Path("/rstor/mdodson/timegrid_frames/tiffs_soilm-200")
    tiff_dir = Path("/rhome/mdodson/testbed/tmp")
    latlon = np.load(tiff_dir.joinpath("tgframe_shae_latlon.npy"))
    tiff_paths = [p for p in tiff_dir.iterdir() if ".tiff" in p.name]

    lats = np.average(latlon[...,0], axis=1)
    lons = np.average(latlon[...,1], axis=0)
    xres = (lons[-1] - lons[0]) / len(lons)
    yres = (lats[-1] - lats[0]) / len(lats)
    transform = Affine.translation(lons[0] - xres / 2, lats[0] - yres / 2) \
            * Affine.scale(xres, yres)

    for p in tiff_paths:
        add_georef_to_tiff(
                tiff_path=p,
                transform=transform,
                target_path=p.parent.joinpath("geo-"+p.name),
                crs="EPSG:4326"
                )
