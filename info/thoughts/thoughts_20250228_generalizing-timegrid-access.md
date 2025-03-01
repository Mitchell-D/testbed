# generalizing timegrid access

Timegrids should ultimately support any number of n-dimensional
arrays, keeping the convention that the final axis is the feature
axis. Subsequent generator methods should use this profile to create
batch-wise a set of data matching a new custom output profile.

For example, the sequence generators currently take the 4d
(t,y,x,F\_d) dynamic data array, 3d (y,x,F\_s)  static data, and 1d
(t,) arrays from a timegrid (using a chunk sampling method they
individually define) and produce batches of 4d ()

The timegrid could be accessed in general by

timegrid generating steps:
1. generator recieves timegrid(s) path(s)
2. open timegrids, extract feature and coordinate labels for each
   array, create dict of array sizes, coordinates, and labels.

need to:
 - generalize access across array types.
 - better define CoordinateFeatureArray (`cfa`) profile
    - ndims, shape, feature, and labels are minimum
    - optionally include array-like data under `cfa.data`, which
      could be hdf5 instance, numpy array, dask array, etc
    - `flabels`, `coords`, `fidxs`, `cidxs`
    - `flabels`: returns full tuple of labels by default, and feature
      labels indexed by any integer or tuple of integers
    - `fidxs`: returns an index or tuple of indeces corresponding to
      feature label(s) provided. Vacuously returns
      `tuple(range(Fd))` for Fd features if no arguments provided
    - `coords`: returns
    - `cidxs`: returns integer tuples of the full range along
      provided axis (or axes). If an integer is provided,
    - determine how to generalize to CoordinateFeatureDataset, which
      collects CoordinateFeatureArray instances, providing a
      directory-like access hierarchy and multi-array derived feats
