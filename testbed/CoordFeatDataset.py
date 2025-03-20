import numpy as np
import h5py
import json
from datetime import datetime,timedelta
from pathlib import Path

class CoordFeatArray:
    def __init__(self, flabels:list, clabels:list, cshape=None,
            self._data=None, self._cdata={},
            ):
        """
        Framework for labeling and interacting with an Nd array-like datum
        structured like (C1,...,CN,F) for N coordinate axes C1-CN, and F
        data features. Provides methods for indexing along the coord and
        feat axes, calculating derived feats, etc.
        """
        self._flabels = tuple(flabels)
        self._clabels = tuple(clabels if clabels else ())
        self._cshape = cshape if not cshape is None \
                else tuple(None for i in range(len(self._clabels)))
        assert len(self._cshape)==len(self._clabels)
        self._data = None
        if not self._data is None:
            assert self._data.shape[:-1] == self._cshape
            assert self._data.shape[-1] == len(self._flabels)

    @staticmethod
    def _chunk_slices(axis_size:int, chunk_size:int,
            include_residual=True, as_array=False):
        """
        Returns a list of 2-tuples of integers corresponding to the index
        boundaries of each chunk of size chunk_size. Optionally include or
        exclude the "residual" partial chunk if chunk_size doesn't divide
        axis_size.

        :@param axis_size: Number of elements in the axis.
        :@param chunk_size: Maximum number of elements per chunk.
        :@param include_residual: If True, the final "partial" chunk with
            fewer elements will be included in the returned list.
        :@param as_array: If True, return the data as a (N, 2) shaped array
            for N chunks with an upper and lower index (2 integers) each. This
            can be handy for applying math operations to offset etc afterwards.
        """
        m = axis_size // chunk_size
        r = axis_size % chunk_size
        if r==0:
            s = tuple((i*chunk_size, (i+1)*chunk_size) for i in range(m))
        else:
            s = [(i*chunk_size, (i+1)*chunk_size) for i in range(m))] + \
                [[],[(axis_size, axis_size + r)]][include_residual]
        if as_array:
            return np.array(s)
        return s

    def flabels(self, idxs=None):
        """
        Return the feature labels in order for this CFA. Optionally provide
        1 or more indeces determining the order in which labels are returned.

        :@param idxs: Optionally provide 1 or more integer indeces for labels
            to return in the corresponding order. If idxs is an integer, only
            the string label will be returned. Otherwise if an iterable is
            provided, a tuple of string labels is returned.
        """
        if idxs is None:
            return tuple(self._flabels)
        elif type(idxs) == int:
            return self._flabels[idxs]
        elif hasattr(idxs, "__iter__"):
            return tuple(self._flabels[ix] for ix in idxs)
        raise ValueError(
                f"'idxs' must be an integer or iterable, not {type(idxs) = }"
                )

    def clabels(self, idxs=None):
        """
        Return coordinate labels in order for this CFA. Optionally provide
        1 or more indeces determining the order in which labels are returned.

        :@param idxs: Optionally provide 1 or more integer indeces
            referencing the placement of coordinate dimensions in this CFA
            (ie the index along the shape tuple). If a single index is
            provided, only the corresponding string label is returned.
            Otherwise if an iterable of indeces are provided, return the
            corresponding string labels as a tuple ordered accordingly
        """
        if idxs is None:
            return tuple(self._clabels)
        elif type(idxs) == int:
            return self._clabels[idxs]
        elif hasattr(idxs, "__iter__"):
            return tuple(self._clabels[ix] for ix in idxs)
        raise ValueError(
                f"'idxs' must be an integer or iterable, not {type(idxs) = }"
                )

    def fidxs(self, labels=None):
        """
        Return indeces of string feature labels. If no argument is provided,
        this will just be a tuple of integers in the range of the final axis.
        Othewise if a single string or iterable of string labels is provided,
        the corresponding indeces will be returned in the provided order.

        :@param labels: Optional string label or list of labels in the order
            of feature indeces to retrieve
        """
        if labels is None:
            return tuple(range(len(self._flabels)))
        elif type(labels) == str:
            return self._flabels.index(labels)
        elif hasattr(labels, "__iter__"):
            return tuple(self._flabels.index(l) for l in labels)

    def cidxs(self, labels=None, get_idx_arrays=False):
        """
        Return a list of indeces corresponding to the axis position of the
        corresponding coordinate labels. Otherwise return an array of integers
        indexing the corresponding coordinate axes.

        :@param labels: String label or list of labels for coordinate axis
            indeces to return. If a single label is provided, only one index
            (or array) is returned, otherwise a tuple of indeces (or arrays)
            are returned in the provided order of labels.
        """
        if labels is None:
            if axis_indeces:
                return tuple(np.arange(c) for c in self._cshape)
            return tuple(range(len(self._clabels)))
        elif type(labels) is str:
            if axis_indeces:
                return np.arange(self._cshape[self._clabels.index(labels)])
            return self._cshape[self._clabels.index(labels)]
        elif hasattr(labels, "__iter__"):
            if axis_indeces:
                return tuple(np.arange(self._cshape[self._clabels.index(l)])
                        for l in labels)
        else:
            raise ValueError(
                    f"Must provide label, list of labels, or ")

    @property
    def cshape(self):
        """
        Returns the shape of only coordinate axes, excluding the feature axis,
        which is always the final axis.
        """
        if self._data is None:
            return self._cshape
        return self._data.shape[:-1]

    @property
    def shape(self):
        if self._data is None:
            return tuple(*self.cshape, len(self._flabels))

    def data(self, flabels=None, **kwargs):
        if self._data is None:
            return None
        if flabels is None and len(kwargs)==0:
            return self._data
        if flabels is None:
            fslice =


class CoordFeatDataset:
    """
    Collection of CoordFeatArray objects supporting higher-level
    data generators with derived features from multiple domains.
    """
    def __init__(self, cfarrays:list):
        """
        """
        self._cfas = cfarrays
