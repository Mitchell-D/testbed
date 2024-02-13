# testbed

### nldas\_static\_netcdf.py

Generate a pickle file containing static data including vegetation
classes, soil texture components, land mask, lat/lon, and elevation,
stored as a 2-tuple like `(labels:list[str], data:list[np.array])`

### get\_nldas\_noahlsm.py

Simple script for generating gesdisc URLs to valid files and calling
a `curl` command as a subprocess to download them. Also includes a
simple loop to check if downloaded files exist and have the
appropriate size, since curl will sometimes silently download an
https error report instead of the file.

I should probably look into handling skipped files in the download
method by checking https return codes, but for now I just remove
failed-download files and re-run the method to re-download them.

### extract\_feats.py

Given separate directories of NLDAS2 NoahLSM grib files, each of
which have files corresponding to the same timesteps separated by
uniform intervals, extracts all the records requested in the
`nldas_record_mapping` and `noahlsm_record_mapping` lists from
`list_feats.py`, and combines them into a (time, lat, lon, feature)
shaped hdf5 file (where the feature axis has the same ordering as
the record mapping lists).

### curate\_samples.py

Multiprocessed method for converting the (T,M,N,F) shaped data grid
(and its (M,N,F) static features) into a well-shuffled (P,L,F) array
of P continuous samples having length L and F features per point, and
corresponding (P,S) and (P,) shaped arrays for S static features and
an initial time corresponding to each point. Each sample point should
have a random starting time of day. The domain of extracted samples
can be restricted by providing a (M,N) boolean mask setting invalid
values to False.

During the extraction process, a FeatureGrid file like (T,M,N,F) is
chunked across the (M,N) dimension. A user-specified number of random
chunks is provided to each subprocess, which partition the T axis of
each pixel in every chunk to length L contiguous timesteps after
applying a random offset of up to size L. The subprocesses then
shuffle the samples from each chunk and return the (P,L,F) result.

Also includes a method `shuffle_samples` that further mixes the time
axis, and a method `collect_norm_coeffs` which iterates over the
sample file to calculate the feature-wise means and standard devia
(in order to normalize prior to training).

### model\_methods.py

Module of general methods for the model training process, including:

 1. Functions for creating model sub-modules including LSTM or
    feed-forward stacks and TCN blocks.
 2. Generator for extracting model-ready samples from a sample file,
    and methods for returning separate training and validation
    TensorFlow Dataset generators.
 3. Helper methods for loading model configurations from the model
    directory, keras CSV progress files as a dict, and normalization
    coefficients as numpy arrays.

### model\_predict.py


