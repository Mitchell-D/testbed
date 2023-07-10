import pickle as pkl
import numpy as np
import string
import re
from pathlib import Path
from datetime import datetime
from datetime import timedelta

class GeoTimeSeries:
    """
    GeoTimeSeries is a basic class representing a 1d time series of a scalar
    variable corresponding to a grid point on a larger domain. GeoTimeSeries
    objects enforce uniqueness using 4 attributes: time range, index, flabel
    (feature label), and mlabel (modification label)
    """
    @staticmethod
    def load(gts_file:Path):
        gts_file = Path(gts_file)
        timetuple, idx, flabel, mlabel = GeoTimeSeries.file_to_fields(gts_file)
        t0, dt, size = timetuple
        return GeoTimeSeries(np.load(gts_file.as_posix()),
                             t0, dt, idx, flabel, mlabel)

    @staticmethod
    def file_to_fields(gts_file:Path):
        """
        Strips values from GeoTimeSeries numpy serial file names per the
        standard naming scheme, formatted as follows:

        {flabel}_{mlabel}_i{initial time}_d{time interval}_s{data size}_ \
                y{vert index}_x{horiz index}.npy

        For initial time and time step fields, decimals are replaced with
        dashes in the file name.

        Information in these fields is sufficient for describing a unique
        continuous equal-interval time series on a larger grid for a single
        dataset with optional derived values specified by the mlabel.

        :@param gts_file: Path to a GeoTimeSeries .npy serial file which
                conforms to the standard naming scheme above. The file
                should have been created by the .save() object attribute.
        :@return: 4-tuple with fields like ((t0,dt,size), idx, flabel, mlabel)
                where times is a list of datetime objects, idx is a 2-tuple
                of ints, and flabel/mlabel are strings. Note that the returned
                tuple doesn't have the same order as the fields in the file
                name; instead it's ordered like the GeoTimeSeries init params.
                If the provided file path doesn't match the standard naming
                scheme, returns None.
        """
        gts_file = Path(gts_file)
        exp = r"([\w-]+)_([\w-]+)_i([\d-]+)_d([\d-]+)_s(\d+)_y(\d+)_x(\d+).npy"
        match = re.search(exp, gts_file.name)
        if not match:
            return None
        flabel, mlabel, t0, dt, size, yidx, xidx = match.groups()
        t0 = datetime.fromtimestamp(float(t0.replace("-",".")))
        dt = timedelta(seconds=float(dt.replace("-",".")))
        #times = [ t0+i*dt for i in range(int(size)) ]
        idx = int(yidx), int(xidx)
        return ((t0, dt, size), idx, flabel, mlabel)

    @staticmethod
    def _validate_label(label:str):
        valid = list(string.ascii_lowercase + string.ascii_uppercase)
        valid += list(map(str, range(10))) + ["-"]
        if not all([c in valid for c in label]):
            raise ValueError("All characters in a label string must be " + \
                    f"alphanumeric or a dash. Invalid: {label}")
        return label

    def __init__(self, data1d:np.array, t0:datetime, dt:timedelta, idx:tuple,
                 flabel:str, mlabel:str="default"):
        """
        Initialize a GeoTimeSeries with a 1d data array.

        :@param data1d: 1-dimensional numpy array for a scalar time series
        :@param t0: datetime of the first observation in data1d
        :@param dt: time interval between observations in data1d
        :@param idx: 2-tuple integer index of the point on a larger grid from
                which this time series was extracted
        :@param flabel: "feature" label, used to identify a unique type of
                data. For example: pressure, soilm-10cm, solar-flux, etc.
                Only alphanumeric characters and dashes are allowed.
        :@param mlabel: "modification" label, used to distinguish time series
                of the same feature type that have been modified in some way.
                For example, if I make a 1-week percentile of a normalized
                time series with flabel=tsoil, the user may set the mlabel to
                something descriptive like "pct1wk-norm".
        """
        # Make sure the dataset is indeed 1-dimensional
        assert len(data1d.shape)==1
        # Timesteps must be uniform-interval and monotonic
        #dt = times[1]-times[0]
        #assert all([ t1-t0==dt for t0,t1 in zip(times[:-1],times[1:]) ])
        self._data = data1d
        #self._times = times
        self._t0 = t0
        self._dt = dt
        self._idx = idx
        self._flabel = GeoTimeSeries._validate_label(flabel)
        self._mlabel = GeoTimeSeries._validate_label(mlabel)

    @property
    def mlabel(self):
        return self._mlabel

    @mlabel.setter
    def mlabel(self, mlabel):
        """
        Only the mlabel attribute can be changed by the user, which is useful
        for storing modified versions of a single time series.
        """
        self._mlabel = GeoTimeSeries._validate_label(mlabel)

    @property
    def flabel(self):
        return self._flabel

    @property
    def idx(self):
        return self._idx

    @property
    def data(self):
        return self._data

    def save(self, data_dir:Path, replace:bool=False):
        """
        Save the data array to a .npy file in the provided directory using
        the standard file naming scheme, which enables this GeoTimeSeries to
        be fully re-loaded with GeoTimeSeries.load()

        :@param data_dir: Directory to generate the new serial array
                representing this GeoTimeSeries
        :@param replace: If True and a file matching all attributes exists in
                the provided data directory, overwrites it. Raises ValueError
                otherwise. There shouldn't be name collisions unless you're
                actually re-generating identical time series.
        :@return: Path object to the newly created .npy file.
        """
        data_dir = Path(data_dir)
        assert data_dir.exists() and data_dir.is_dir()
        fname  = data_dir.joinpath(self.get_file_name())
        if fname.exists() and not replace:
            raise ValueError(f"{fname.as_posix()} already exists!")
        np.save(fname.as_posix(), self._data)
        return fname

    def __repr__(self):
        #return f"{self.flabel}, {self.mlabel}, {self.idx}"
        return f"{self.idx}"

    def get_file_name(self):
        """
        Format a numpy serial file path that enables retrieval of
        all GeoTimeSeries attributes without storing meta-information

        :@return str: standard-format GeoTimeSeries file name.
        """
        # first 2 fields are the flabel and mlabel
        fname = f"{self.flabel}_{self.mlabel}_"
        # 3rd field is the initial time in epoch seconds
        fname += f"i{str(self._t0.timestamp()).replace('.','-')}_"
        # 4th field is the temporal resolution in seconds
        fname += f"d{str(self._dt.total_seconds()).replace('.','-')}_"
        # 5th field is the number of timesteps in total
        fname += f"s{self._data.size}_"
        # 6th and 7th fields are the 1st and 2nd (vertical and horizontal) idx
        fname += f"y{self._idx[0]}_x{self._idx[1]}.npy"
        return fname

    '''
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
    '''

if __name__=="__main__":
    debug = True
    data_dir = Path("data/GTS")
    buf_dir = Path("data/buffer")
    import gesdisc
    label = "rockies"
    times = [gesdisc.nldas2_to_time(f) for f in
             sorted(Path("data/nldas2_20180401-20180931").iterdir())]
    nldas, pixels, _ = pkl.load(Path(
        f"data/1D/{label}_nldas2_all-forcings_2018.pkl").open("rb"))
    noahlsm, _, noahlsm_info = pkl.load(Path(
        f"data/1D/{label}_noahlsm_all-fields_2018.pkl").open("rb"))

    t0 = times[0]
    dt = times[1]-dt
    for px in range(noahlsm.shape[1]):
        for b in range(len(noahlsm_info)):
            if noahlsm_info[b]["name"] not in ("LSOIL", "SOILM"):
                continue
            flabel = noahlsm_info[b]["name"] + "-" + \
                    noahlsm_info[b]["lvl_str"].split(" ")[0]
            tmp_gts = GeoTimeSeries(
                    noahlsm[:,px,b].data, t0, dt, pixels[px], flabel)
            tmp_gts.save(data_dir, replace=True)
        for b in range(len(nldas_info)):
            tmp_gts = GeoTimeSeries(nldas[:,px,b].data, t0, dt, pixels[px],
                                    nldas_info[b]["name"])
            tmp_gts.save(data_dir, replace=True)
