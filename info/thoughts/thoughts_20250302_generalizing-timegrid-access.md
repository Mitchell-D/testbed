# generalizing timegrid access

 - **CoordFeatDataset** (CFD) object provides exact feature
   and coordinate configuration and shapes for each CoordFeatArray
   (CFA), as well as arrays for the coordinate metrics.

 - CFDs can be saved as hdf5 files such that each CFA is a data
   array, the collection of CFA configs and other file and dataset
   information is serialized to JSON through the cumulative CFD
   config, and the metric arrays are hdf5 data arrays having fewer
   attributes.

 - generators implement access and sampling policies for extracting
   data from one or more CFD-style HDF5 files. They may need to
   handle coordinate concatenation and sampling across files, apply
   conditions along an axis, etc in order to determine what they
   yield. The data shape of a yielded batch may be described by a
   CFD if they allow for coordinate axes of undefined length.

 - generators' sampling policy may involve setting axes of iteration.
    - consists of a reference point defining where the extracted data
      volume begins at each iteration step
    - (ix0, ixf, s) with remainder inclusion/exclusion policy
       - ix0/ixf: initial/final indeces:
       - s: stride between initial
       - defaults: (0, N, 0), where zero stride implies one sample
    - (dix0, dixf) for chunks extracted from the reference point

 -
