# testbed

This repository contians the code for my master's thesis. The goal of this project is to emulate the soil moisture dynamics of the NLDAS-2 run of the Noah Land Surface Model using neural networks for time series prediction.

## software / data pipeline

<p align="center">
  <img src="https://github.com/Mitchell-D/testbed/blob/main/figures/software_arch/acquisition-and-stats.png?raw=true" width="620"/>
</p>
<p align="center">
  Figure 1: Data acquisition and statistic analysis
</p>

NLDAS-2 and Noah-LSM GRIB files are acquired from the GES DISC DAAC,
then aggregated alongside static data and descriptive attributes into
__timegrid__ style hdf5 files by `extract_timegrid.extract_timegrid`.
Timegrids typically each contain 1/6 of the CONUS grid over 1/4 of
the year.

In order to verify data integrity, identify reasonable histogram
bounds, and determine gaussian normalization coefficients, monthly
pixel-wise minimum, maximum, mean, and standard deviation of each
feature may be calculated and stored with methods in eval\_timegrids.

`eval_timegrid.pixelwise_stats` returns the above statistics
calculated independently for each unique year/month combination,
which may be stored as a series of regional pkl files. These pkl
files can be collected into an hdf5 representing the full grid using
`eval_timegrid.collect_gridstats`.

`eval_timegrid.make_gridstat_hdf5` offers similar functionality,
except that the monthly statistics are calculated using the aggregate
data from a series of provided timegrids, which is useful for
calculating true bulk values over many years of data. Furthermore,
statistics can be calculated over derived features in addition to
stored features.

While `make_gridstat_hdf5` is slower than `pixelwise_stats`, the
former is the preferred method for determining normalization
coefficients since aggregating multiple years' data makes the
returned values less vulnerable to short-term fluctuations in data,
and because of its ability to handle derived feature values.

<p align="center">
  <img src="https://github.com/Mitchell-D/testbed/blob/main/figures/software_arch/model-training.png?raw=true" width="620"/>
</p>

<p align="center">
  Figure 2: Model training pipeline
</p>

Models are trained on data that are structured according to the
"sequence" (window,horizon,static,static\_int,pred) format defined in
the __model inputs and outputs__ section below. This kind of data may
be extracted, preprocessed, and generated directly from timegrid
hdf5s using `generators.timegrid_sequence_dataset`, however it is
generally more efficient to train models using data generated from
__sequence__ hdf5 files, which contain samples that are already
preprocessed.

Sequence hdf5s are advantageous during training because they
speed up access to sample data that has been thoroughly
spatially and temporally shuffled, and reformatted to the
sequence array style. The exact file format defining a sequence hdf5
is outlined below under __custom file types__.

The model training pipeline utilizes my [tracktrain][1] framework.
Before dispatching each model, the user modifies a configuration
dictionary that is sufficient to fully specify the model
architecture, training and validation data sources, training
conditions, loss function, metrics, and other parameters.

The configuration is used to initialize a `tracktrain.ModelDir`
object, which creates a unique directory for the current model run,
and stores the configuration, architecture diagrams, and other
useful information inside of it. After this, the model training
sequence is dispatched, and trained weights are stored back in the
model's directory.

The `tracktrain` framework makes it easy to systematically re-load
models alongside the information needed to appropriately structure
and normalize their input data, training history and parameters,
model architecture diagrams, and manual notes provided upon dispatch.

<p align="center">
  <img src="https://github.com/Mitchell-D/testbed/blob/main/figures/software_arch/evaluation.png?raw=true" width="620"/>
</p>
<p align="center">
  Figure 3: Model evaluation systems
</p>

After training, model performance can be evaluated in bulk using
shuffled sequence hdf5s, or on a spatially and temporally consistent
domain using grids generated from timegrid hdf5s.

The left pipeline in the above figure shows the procedure for
determining bulk statistics. Models are evaluated over sequence
hdf5s by `eval_models.sequence_preds_to_hdf5`, and their outputs are
stored in the same order as the inputs in prediction hdf5s.

The predictions and true values are subsequently yielded alongside
each other by `eval_models.gen_sequence_prediction_combos` within
other methods in `eval_grids` in order to update pkls containing
error with respect to forecast distance, truth/prediction joint
histograms, error with respect to time of day and time of year,
and error with respect to the soil and vegetation static parameter
classes.

The right pipeline in the figure shows the gridded evaluation system,
which evaluates the model over the gridded domain given uniform
initialization times and frequencies.
`eval_grids.gen_gridded_predictions` makes predictions based on data
extracted directly from timegrid hdf5s by utilizing the
`generators.gen_timegrid_subgrids` generator. These predictions
(and optionally forcings, static inputs, and true values) are stored
in __subgrid__ style hdf5 files.

## custom file types

### timegrid hdf5

__Generated by__: `extract_timegrid.extract_timegrid`

Timegrid hdf5s are the basic format for storing data extracted from
the source grib1 files, and are necessary in order to efficiently
parse sparse chunked and memory-mapped data. They shouldn't contain
data that is normalized or broken up into model input or output
constituents; only a continuous hourly time series stored as a
4D data block per file, alongside a time-invariant static grid.

#### datasets:

 - __/data/dynamic__: (T, Y, X, F<sub>d</sub>) \
   Time-varying NLDAS-2 forcings and Noah-LSM outputs.
 - __/data/static__: (Y, X, F<sub>s</sub>) \
   Static parameters like soil texture, elevation, geodesy
 - __/data/time__: (T,) \
   Epoch times corresponding to each of the first-axis elements in
   the dynamic array

#### attributes:

 - __/data/dynamic__: (JSON string) \
   Coordinate labels corresponding to the dynamic array dimensions,
   and feature labels for data indexed by the final axis. Additional
   meta-info is provided by the wgrib output describing each feature.
 - __/data/static__: (JSON string) \
   Coordinate labels corresponding to the static array dimensions,
   and feature labels for data indexed by the final axis.

__T__: time axis, __Y__: vertical axis, __X__: horizontal axis,
__F<sub>d</sub>__: dynamic feature axis, __F<sub>s</sub>__:
static feature axis

### sequence hdf5

__Generated by__: `generators.make_sequence_hdf5`

Sequence hdf5 data is stored in the format generated by a
`generators.timegrid_sequence_datset`, which is organized into
thoroughly shuffled samples for each input and output category.
Except for normalization (which is typically handled on-demand by
generators) the data stored in sequence hdf5 files is fully prepared
for model training or inference.

#### datasets:

 - __/data/window__: (B, S<sub>w</sub>, F<sub>w</sub>) \
   Data prior to the first prediction, used to initialize the model.
 - __/data/horizon__: (B, S<sub>h</sub>, F<sub>h</sub>) \
   Covariate forcings corresponding to each prediction step.
 - __/data/static__: (B, F<sub>s</sub>) \
   Time-invariant input quantities like slope and soil information.
 - __/data/static_int__: (B, F<sub>si</sub>) \
   One-hot encoded time-invariant inputs like surface categories.
 - __/data/pred__: (B, S<sub>h</sub>, F<sub>p</sub>) \
   Dynamic variables that are predicted by the model at each horizon.

#### attributes:
 - __/data/gen_params__ (JSON string) \
   Dictionary of arguments to `generators.timegrid_sequence_dataset`

__B__: batch axis (samples), __S__<sub>w</sub>: window sequence
steps, __F__<sub>w</sub>: window input features, __S__<sub>h</sub>:
horizon sequence steps, __F__<sub>h</sub>: horizon input features,
__F__<sub>s</sub>: static input features, __F__<sub>si</sub>: integer
static input features, __F__<sub>p</sub>: predicted features.

### subgrid hdf5

__Generated by__: `eval_grids.grid_preds_to_hdf5`

Subgrid hdf5 files are separated into model input/output categories
similar to sequence hdf5s, however their gridded form is maintained
by keeping an array of integer indeces corresponding to each valid
pixel in a boundary. A subgrid hdf5 may contain data initialized at
multiple times, and typically includes outputs from a specific model.

#### datasets:

 - __/data/truth__: (N, P, S<sub>h</sub>, F<sub>p</sub>) \
   Dynamic variables that are predicted by the model at each horizon.
 - __/data/preds__: (N, P, S<sub>h</sub>, F<sub>p</sub>) \
   Model outputs corresponding to the "truth" values
 - __/data/idxs__: (P, 2) \
   (y,x) indeces of each pixel in the subgrid, referring to a grid
   with the shape described by the /data/grid\_shape attribute
 - __/data/time__: (N, S<sub>p</sub>) \
   Epoch times associated with each output sequence step.
 - __/data/window__: (N, P, S<sub>w</sub>, F<sub>w</sub>) \
   Data prior to the first prediction, used to initialize the model.
 - __/data/horizon__: (N, P, S<sub>h</sub>, F<sub>h</sub>) \
   Covariate forcings corresponding to each prediction step.
 - __/data/static__: (P, F<sub>s</sub>) \
   Time-invariant input quantities like slope and soil information.
 - __/data/static_int__: (P, F<sub>si</sub>) \
   One-hot encoded time-invariant inputs like surface categories.

#### attributes:

 - __/data/gen_args__: (JSON string) \
   Dictionary of arguments to `generators.gen_timegrid_subgrid`
 - __/data/grid_shape__: (2-tuple of integers) \
   Vertical/horizontal grid shape of the subgrid.
 - __/data/model_config__ (JSON string) \
   Configuration dictionary for a `tracktrain.ModelDir` instance,
   which contains enough information to fully re-initialize a model.

__N__: Unique initialization time axis, __P__: grid pixel axis,
__S__<sub>w</sub>: window sequence steps, __F__<sub>w</sub>: window
input features, __S__<sub>h</sub>: horizon sequence steps,
__F__<sub>h</sub>: horizon input features, __F__<sub>s</sub>: static
input features, __F__<sub>si</sub>: integer static input features,
__F__<sub>p</sub>: predicted features.

## model inputs and outputs

<p align="center">
  <img src="https://github.com/Mitchell-D/testbed/blob/main/proposal/figs/abstract_rnn.png?raw=true" width="620"/>
</p>
<p align="center">
  Figure 4: General sequence prediction structure
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

## module descriptions

#### nldas\_static\_netcdf.py

Generate a pickle file containing static data including vegetation
classes, soil texture components, land mask, lat/lon, and elevation,
stored as a 2-tuple like `(labels:list[str], data:list[np.array])`

#### get\_nldas\_noahlsm.py

Simple script for generating gesdisc URLs to valid files and calling
a `curl` command as a subprocess to download them. Also includes a
simple loop to check if downloaded files exist and have the
appropriate size, since curl will sometimes silently download an
https error report instead of the file.

I should probably look into handling skipped files in the download
method by checking https return codes, but for now I just remove
failed-download files and re-run the method to re-download them.

#### extract\_feats.py

Given separate directories of NLDAS2 NoahLSM grib files, each of
which have files corresponding to the same timesteps separated by
uniform intervals, extracts all the records requested in the
`nldas_record_mapping` and `noahlsm_record_mapping` lists from
`list_feats.py`, and combines them into a (time, lat, lon, feature)
shaped hdf5 file (where the feature axis has the same ordering as
the record mapping lists).

#### curate\_samples.py

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

#### model\_methods.py

Module of general methods for the model training process, including:

 1. Functions for creating model sub-modules including LSTM or
    feed-forward stacks and TCN blocks.
 2. Generator for extracting model-ready samples from a sample file,
    and methods for returning separate training and validation
    TensorFlow Dataset generators.
 3. Helper methods for loading model configurations from the model
    directory, keras CSV progress files as a dict, and normalization
    coefficients as numpy arrays.

#### model\_predict.py


[1]:https://github.com/Mitchell-D/tracktrain
