"""
Abstract class for
"""
import pickle as pkl
import numpy as np
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from typing import Callable

recipes = {}

class GeoTimeSeries:
    """
    Abstract class for organizing and interfacing with multiple independent
    and dependent data variables in a 1-dimensional time series.

    Each 1d time series, static data value, or recipe is referenced with a
    unique string label that allows a copy to be returned and enables the data
    to be invoked in recipes.
    """
    def __init__(self, times:np.ndarray):
        """
        Initialize a GeoTimeSeries with an array of times used to index all
        data added with the GeoTimeSeries.add() object method.
        """
        self._data = {}
        self._static = {}
        self._info = {}
        self._labels = []
        self._recipes = recipes

        # Make sure the timesteps are monotonic
        assert all([t0<t1 for t0,t1 in zip(times[:-1],times[1:])])
        self._time = times

    @property
    def labels(self):
        return list(self._data.keys()) + list(self._recipes.keys()) + \
                list(self._static.keys())

    def _validate_label(self, label:str):
        """
        Verify that the provided string label is not already either a static
        dataset key in the data dictionary or a recipe label.
        """
        if label in self._data.keys():
            raise ValueError(f"{label} is already a dataset key")
        if label in self._recipes.keys():
            raise ValueError(f"{label} is already a recipe key")
        if label in self._static.keys():
            raise ValueError(f"{label} is already a static dataset key")
        return label

    def add_data(self, label:str, data:np.ndarray, info:dict=None):
        """
        Add a 1d dataset to the collection with a string label and a 1d array
        with the same size as this GeoTimeSeries' number of timesteps.

        :@param label: Unique string label to assign to the new data
        :@param data: 1-dimensional numpy scalar array with the same number of
            elements as timesteps in this GeoTimeSeries.
        """
        assert len(data.shape)==1 and data.size==len(self._time)
        self._data[self._validate_label(label)] = np.copy(data)

    def add_static(self, label:str, static, info:dict=None):
        """
        Add a static dataset with a unique label, which may be any
        serializable value such as a number or a vector.
        """
        self._static[self._validate_label(label)] = static

    def add_recipe(self, label:str, recipe:Callable, params:list,
                   info:dict=None):
        """
        Add a callable recipe

        (!) Need to determine whether to use string labels of callable
            parameters as an argument... If so, all inputs will need to be
            pre-loaded into the GTS, which causes redundancies if an external
            array needs to be applied to many GTS instances.
            Otherwise, there seems to be little difference from just applying
            the method externally and referencing the data from the GTS inst.
        """
        self._recipes[self._validate_label(label)] = recipe
        return None

    def subset(self,label:str, t0:datetime, tf:datetime):
        """
        Returns a 1d array of the labeled time series between the provided
        initial and final times.
        """
        assert t0<tf
        assert label in self._data
        init_idx = None
        final_idx = None
        for i in range(len(self._time)):
            if t0<=self._time[i] and not init_idx:
                init_idx = i
            # Non-inclusive final timestep, like python array slicing
            if tf<=self._time[i]:
                final_idx = i
        return self._data[label][init_idx:final_idx]

if __name__=="__main__":
    debug = True
    data_dir = Path("data/1D")

    noahlsm_pkl = data_dir.joinpath("silty-loam_noahlsm_all-fields_2019.pkl")
    nldas2_pkl = data_dir.joinpath("silty-loam_nldas2_all-forcings_2019.pkl")

    #noahlsm_pkl = data_dir.joinpath("sl2_noahlsm_all-fields_2019.pkl")
    #nldas2_pkl = data_dir.joinpath("sl2_nldas2_all-forcings_2019.pkl")

    noahlsm,_,noahlsm_info = pkl.load(noahlsm_pkl.open("rb"))
    nldas2,_,nldas2_info = pkl.load(nldas2_pkl.open("rb"))

    # For silty-loam only
    t0 = datetime(year=2019, month=1, day=1, hour=0)
    tf = datetime(year=2020, month=1, day=1, hour=0)

    timesteps = [t0+timedelta(hours=hours) for hours in
                 range(int((tf-t0).total_seconds() // 3600 ))]

    gts = GeoTimeSeries(timesteps)
    for i in range(noahlsm.shape[2]):
        if noahlsm_info[i]["name"] in ("SOILM","LSOIL"):
            tmp_name = noahlsm_info[i]["name"] + "_" + \
                    noahlsm_info[i]["lvl_str"].split(" ")[0]
            gts.add_data(tmp_name, noahlsm[:,0,i], noahlsm_info)

    for i in range(nldas2.shape[2]):
        gts.add_data(nldas2_info[i]["name"], nldas2[:,0,i], nldas2_info)

    print(gts.labels)
    print(gts.subset(
        "SOILM_0-100",
        datetime(year=2019,month=4,day=1),
        datetime(year=2019,month=10,day=1)
        ))
