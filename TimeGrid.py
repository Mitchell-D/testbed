from pathlib import Path
from datetime import datetime
import numpy as np
import pickle as pkl
import multiprocessing as mp

class TimeGrid:
    """
    TimeGrid is a class that abstracts a list of uniform-shaped numpy arrays
    stored serial files, which constitute a time series. The time series
    members should be (M,N,F)-shaped for M vertical coordinates, N horizontal
    coordinates, and F labeled features. The class also supports (M,N)-shaped
    static arrays that assign a scalar to each point in the member grids.

    Ultimately, the goal of this class is to provide a generalized way of
    extracting subsets of the time series given an ordered list of desired
    features, a time range, and a list of indeces.

    Currently the storage medium is simple ".npy" files for each time step
    since ".npz" files increase access time in exchange for disc space
    (since decompression takes time). Same principle with structured arrays.
    I'm assuming disc space isn't a limiting factor, and that file size only
    really matters when transferring data, in which case archive compression
    algorithms like tar/gzip
    """
    def __init__(self):
        self._datasets = {}

    def validate_dataset(self, dataset:str):
        """
        Validates that the arrays contained in all ".npy" files of a dataset
        are uniformly-shaped and have the same number of features (third axis)
        as feature labels. This is a computationally costly operation.
        """
        # Make sure the asked-for dataset key is already registered
        assert dataset in self._datasets.keys()
        paths = self._datasets[dataset]["paths"]
        shape = np.load(paths[0], mmap_mode="r").shape
        # All arrays in the dataset must have the same shape
        if not all([np.load(p, mmap_mode="r").shape==shape for p in paths]):
            raise ValueError(
                    f"Not all arrays in datset {dataset} have the same shape!")
        # Arrays must have the same number of features as feature labels
        nlabels = len(self._datasets[dataset]["flabels"])
        if not shape[2]==nlabels:
            raise ValueError(
                    f"Dataset {dataset} must have the same number of "
                    f"features as feature labels ({shape[2]} != {nlabels})")

    def get_feature_labels(self, dataset:str):
        """
        Returns all feature labels associated with a registered dataset

        :@param dataset: string label of a loaded dataset
        """
        # Make sure the asked-for dataset key is already registered
        assert dataset in self._datasets.keys()
        return list(self._datasets[dataset]["flabels"])

    def get_paths(self, dataset:str):
        """
        Returns all .npy file Path objects associated with a registered dataset

        :@param dataset: string label of a loaded dataset
        """
        # Make sure the asked-for dataset key is already registered
        assert dataset in self._datasets.keys()
        return list(self._datasets[dataset]["paths"])

    def extract_timeseries(self, dataset_label:str, pixels:list):
        """
        """
        pass

    def register_files(self, dataset_label:str, files:list,
                       feature_labels:list):
        """
        Register a list of file corresponding to a dataset with the TimeGrid.
        This method does not load the data, but ensures that the provided files
        constitute a continuous and uniform-interval time series per the
        datetime object provided with each file.

        This method doesn't validate shape constraints of the arrays contained
        in each file, namely that each array must have the same shape and the
        length of the third axis must match the number of provided feature
        labels. For a (computationally expensive) sanity check, see the
        validate_dataset() object method.

        :@param dataset_label: String identifying the data contained in each
            file in the list. Individual time step files may contain multiple
            features for different variables; this label generalizes the
            collection of features.
        :@param files: list of 2-tuples like (file_datetime, file_path) for
            each file in a unique dataset's time series. file_datetime must
            be a datetime object, and file_path must be a Path object to a
            valid file. Furthermore, each file must be a ".npy" serial binary,
            each with a (M,N,F) shape for M vertical coordinates, N horizontal
            coordinates, and F features (see feature_labels)
        :feature_labels: Ordered list of unique string label corresponding to
            each of the features specified in the third array axis of each
            ".npy" grid.
        """
        # All feature labels must be unique to this dataset
        assert len(set(feature_labels)) == len(feature_labels)
        # The dataset label must be unique
        assert dataset_label not in self._datasets.keys()
        # All file list members must be 2-tuples
        assert all(type(t)==tuple and len(t)==2 for t in files)
        # Sort files by their datetime component
        files.sort(key=lambda t:t[0])
        times, paths = zip(*files)
        # All file paths must be existing ".npy" file Path objects
        for p in paths:
            if not type(p)==Path and p.exists() and p.suffix=="npy":
                raise ValueError(f"Invalid Path: {p}")
        dt = times[1]-times[0]
        for t0,t1 in zip(times[:-1],times[1:]):
            if not t1-t0==dt:
                raise ValueError(
                        f"Default time step ({dt}) not abided by ({t0},{t1})")
        # All timesteps must be equal-interval
        assert all([t1-t0==dt ])
        # Update the datasets attribute with the new dataset dictionary
        dataset_dict = {"flabels":feature_labels, "paths":paths}
        self._datasets[dataset_label] = dataset_dict

if __name__=="__main__":
    tg_dir = Path("data/subgrids/")
    nldas_paths = [p for p in tg_dir.iterdir()
                   if p.stem.split("_")[0]=="FORA0125"]
    nldas = [(datetime.strptime(p.stem.split("_")[1], "%Y%m%d-%H"),p)
             for p in nldas_paths ]
    TG = TimeGrid()
    TG.register_files(
            dataset_label="nldas",
            files=nldas,
            feature_labels=[
                'TMP', 'SPFH', 'PRES', 'UGRD', 'VGRD', 'DLWRF',
                'NCRAIN', 'CAPE', 'PEVAP', 'APCP', 'DSWRF']
            )
    #TG.validate_dataset("nldas")
