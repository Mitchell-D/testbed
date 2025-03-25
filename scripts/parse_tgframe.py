"""
This script contains information on the structure and some examples on how to
parse the tgframe pickle

:  -------------------------  :file organization:  -------------------------  :

(labels, dynamic_arrray, static_array, idxs, times)

labels: (dynamic_labels, metric_labels, uses_sum)
    dynamic_labels: unique strings corresponding to elements along the second
        to last axis of the dynamic array, which name the data features.
    metric_labels: unique strings for elements of the last axis of the dynamic
        array which name the operations used to coarsen the data from hourly.
    uses_sum: One of the metric labels is "sum_or_diff". Data for this metric
        is integrated to the coarser resolution if the corrresponding dynamic
        label appears in this list. Otherwise, its total change is stored.

dynamic_array: (T, P, Fd, M)
    T: Time axis, each element corresponding to an aggregated timestep.
    P: Pixel axis, each element corresponding to a unique pixel location
    Fd: Dynamic feature axis, each element associated with a data feature
    M: Metric axis, each element associated with a function used to coarsen
        the data from hourly resolution to the resolution of the tgframe.

static_array: (P, Fs)
    P: Pixel axis, as above
    Fs: Static feature axis, each element associated with a static feature.

idxs: (P,2)
    P: Pixel axis, as above.
    2: (vertical, horizontal) integer indices

times: (T,N)
    T: Time axis for coarsened steps, as above
    N: Time axis for points within each step before aggregation
        (ie N==24 for daily coarsening of hourly data)
"""
import numpy as np
import pickle as pkl
import imageio
from pathlib import Path
from datetime import datetime

## dict mapping dynamic feature labels to longer name strings and units.
feat_info = {
        "soilm-10":{
            "name":"0-10cm Soil Moisture Area Density", "unit":"kg/m^2" },
        "soilm-40":{
            "name":"10-40cm Soil Moisture Area Density", "unit":"kg/m^2" },
        "soilm-100":{
            "name":"40-100cm Soil Moisture Area Density", "unit":"kg/m^2" },
        "soilm-200":{
            "name":"100-200cm Soil Moisture Area Density", "unit":"kg/m^2" },
        "rsm-10":{
            "name":"0-10cm Relative Soil Moisture", "unit":"%" },
        "rsm-40":{
            "name":"10-40cm Relative Soil Moisture", "unit":"%" },
        "rsm-100":{
            "name":"40-100cm Relative Soil Moisture", "unit":"%" },
        "rsm-200":{
            "name":"100-200cm Relative Soil Moisture", "unit":"%" },
        "ssrun":{
            "name":"Surface Runoff", "unit":"kg/m^2" },
        "bgrun":{
            "name":"Sub-surface Runoff", "unit":"kg/m^2" },
        "weasd":{
            "name":"Accumulated Snow Water Equivalent", "unit":"kg/m^2" },
        "apcp":{
            "name":"Precipitation (Rain and Snow)", "unit":"kg/m^2" },
        }

if __name__=="__main__":
    tgframe_dir = Path("/rstor/mdodson/timegrid_frames/1d/")
    tgframe_path = tgframe_dir.joinpath(
            "tgframe_shae_19920101_20231231_daily.pkl")

    """ Open the pkl file and parse its contents according to above format """
    (labels, dynamic, static, idxs, times) = pkl.load(tgframe_path.open("rb"))
    (dlabels, slabels, mlabels, uses_sum) = labels
    print(f"{dlabels = }\n{mlabels = }\n{slabels = }\n{uses_sum = }")
    print(f"{dynamic.shape = }\n{static.shape = }")
    print(f"{idxs.shape = }\n{times.shape = }")

    """ Calculate mean full-column volumetric soil moisture """
    soilm_labels = ["soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    soilm_idxs = [dlabels.index(f) for f in soilm_labels]
    ## sum to (T,P) array of total water content (kg/m^2) per 2m deep column
    vsm_fc = np.sum(dynamic[:, :, soilm_idxs, mlabels.index("mean")], axis=2)
    vsm_fc /= 2. ## now vsm (kg/m^3) in the 2m deep column

    """ Get the mean time of each timestep as a readable string """
    times_str = [datetime.fromtimestamp(int(t)).strftime("%Y%m%d")
            for t in np.average(times, axis=1)]

    """ get a list of masks and timesteps with nonzero surface runoff """
    ## develop a mask of >0 vaues along the subsurface runoff maximum array
    m_runoff = (dynamic[:,:,dlabels.index("ssrun"),mlabels.index("max")] > 0)
    ## extract times and dynamic array data where surface runoff is happening
    runoff_masks,runoff_times,runoff_dynamic = zip(*[
        (m_runoff[:,i], times[m_runoff[:,i]], dynamic[m_runoff[:,i]])
        for i in range(m_runoff.shape[1])
        ])
    print([rt.shape[0] for rt in runoff_times])
