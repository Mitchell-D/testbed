from pathlib import Path
import numpy as np
import pickle as pkl

from list_feats import noahlsm_record_mapping, nldas_record_mapping
from model_methods import gen_hdf5_sample

if __name__=="__main__":
    data_dir = Path("data")
    window_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf",
            "soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    horizon_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf"]
    pred_feats = ['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200']

    static_path = data_dir.joinpath("nldas2_static_all.pkl")
    h5_paths = [data_dir.joinpath(f"feats/feats_{y}.hdf5")
            for y in range(2015,2022)]
    _,feat_order = zip(*nldas_record_mapping, *noahlsm_record_mapping)

    window_feat_idxs = [feat_order.index(f) for f in window_feats]
    horizon_feat_idxs = [feat_order.index(f) for f in horizon_feats]
    pred_feat_idxs = [feat_order.index(f) for f in pred_feats]

    """ Load information from static data """
    static_dict = pkl.load(static_path.open("rb"))
    static = static_dict["soil_comp"]
    lon,lat = static_dict["geo"]

    """ Construct a geographic mask setting valid data points to True"""
    ## Geographically constrain to the South East
    print(np.amin(lon),np.amax(lon))
    print(np.amin(lat),np.amax(lat))
    m_lon = np.logical_and(-100<=lon,lon<=-80)
    m_lat = np.logical_and(30<=lat,lat<=40)
    ## Don't consider water or urban surfaces
    m_water = static_dict["soil_type_ints"] == 0
    m_urban = static_dict["soil_type_ints"] == 13
    m_sfc = np.logical_not(np.logical_or(m_water, m_urban))
    m_geo = np.logical_and(m_lon, m_lat)
    m_valid = np.logical_and(m_sfc, m_geo)

    g = gen_hdf5_sample(
            h5_paths=h5_paths,
            static_data=static,
            window_feat_idxs=window_feat_idxs,
            horizon_feat_idxs=horizon_feat_idxs,
            pred_feat_idxs=pred_feat_idxs,
            window=18,
            horizon=18,
            seed=20231228,
            domain_mask=m_valid,
            )

    for j in range(32):
        tmp_x, tmp_y = next(g)
        print(tmp_x["window"].shape, tmp_x["horizon"].shape,
                tmp_x["static"].shape, tmp_y.shape)
