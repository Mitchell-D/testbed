# testbed

## Model Inputs and Outputs

<p align="center">
      <img src="https://github.com/Mitchell-D/testbed/blob/main/proposal/figs/abstract_rnn?raw=true" />
</p>

Each of the models recieves 4 distinct input types that are provided
as a tuple of tensors like `(window, horizon, static, static_int)`.
With B equal to the batch size, S indicating a sequence length, and
F giving the feature elements, the input shapes, data types, and
purposes are defined as follows:

__window__: (B, S<sub>w</sub>, F<sub>w</sub>) `np.float32`

The window consists of F<sub>w</sub> time-varying features used to
initialize the model, including atmospheric forcings, land surface
states, and plant phenology. The S<sub>w</sub> sequence elements
correspond to observations leading up to the first predicted step.

__horizon__: (B, S<sub>h</sub>, F<sub>h</sub>) `np.float32`

The horizon tensor contains F<sub>h</sub> time-varying covariates
associated with each prediction in the S<sub>h</sub> output sequence.
This includes atmospheric forcings and vegetation phenology.

__static__: (B, F<sub>s</sub>) `np.float32`

Static features are those which are consistent throughout time for
any individual pixel, only varying spatially, and which have physical
quantities with magnitudes that (in principle) vary continuously.
This includes soil component percentages, elevation, and hydraulic
properties.

__static_int__: (B, F<sub>si</sub>) `np.uint`

Static integer features are time-invariant like regular static
features, however they represent categorical data types not defined
in terms of a physical metric (ie nearby integers don't necessarily
correspond to similar properties, and vice-versa). Here, the feature
dimension F<sub>si</sub> contains one or more one-hot encoded vectors
concatenated along their feature axis. These will be embedded to
a lower-diensional representation by a learned matrix, with a
final size determined by the model configuration.

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


