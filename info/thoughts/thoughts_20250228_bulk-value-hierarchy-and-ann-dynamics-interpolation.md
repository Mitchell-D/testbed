# 20250228

One of the fundamental problems in near real-time and climatological
data storage and analysis is that the volume of data is too large to
handle and analyze without significant dedicated computational
resources.

Stakeholders need to access information at hourly, daily, monthly,
and annual resolution, as well as in forms combining time resolutions
(ie percentiles by different spatial designations).

Interpolating daily-scale land surface states and forcings to hourly
is a challenge that Dr. Hain has brought up, which some time series
neural networks may be capable of. Aside from that, many applications
can still derive utility from only having daily-scale bulk values
(at least for past data, which can contextualize current data).

For example, if precip > 6 mm/hr is expected for sandy soil with
rsm > .84, a hierarchy of bulk statistic values from the past can be
used to rapidly find analogous situations, and determine the surface
runoff amonts expected for proxy cases.

Timegrids (without some of the energy fluxes or lsoil) are
24.86 Gb/yr at hourly resolution, so about 1.04 Gb/yr at daily
resolution over 50,875 valid pixels. Ignoring the overhead from
strongly-compressed zeroed invalid pixels, this corresponds to about
.51 MB/px/yr, or 16.95 KB/px/yr/feat that there are 30 dynamic feats.

With a 27,245,580 px NDLAS3 domain, each year stored over the full
domain at hourly resolution would be 13852.3 Tb, or
461.744 Gb/year/feat over the full new grid.

Neural networks can plausibly imitate many of the statistical
properties of that data at the finer resolution, which potentially
allows for dramatic data volume reduction by using coarsened bulk
values and a small number of covariate features as inputs for time
series dynamics prediction problems. This will be an interesting
avenue to explore in the future, however for now, it seems more
pressing that we make available some data products based on
historical and near-real-time data.

Nobody will expect that the full hourly climatology of NLDAS-3 data
will be made available for visualized data exploration. In fact, the
hourly model outputs may not be made available at all; only forcings.
As such, for now I need to consider ways we can develop visual
products based on bulk values which do not require too much data to
be available for analysis through the webserver, and based on
analysis that will not require too much server or client processing.

Given about 18 features having 4 metrics, and 10 features having
1 metric (total change) at the daily resolution, as outlined below,
there are a total of 82 feature values per day, which is probably
too many if we keep the full NLDAS-3 grid.

It would be more reasonable to keep values over the full grid at 1km
for monthly resolution, and coarsen the grid to perhaps 10km @ daily.

With this information, one could rapidly collect a large amount of
information on the relative position of near real time data wrt
monthly, annual, and climatological statistics, and find analogs
in the historical data for the region, soil type, state values, etc.

Given the analogs, you could see for example how many experienced
drought conditions within a month by extracting a daily time series.

could also consider applying matrix profiling techniques for time
series motif discovery... not sure best way to integrate with visuals
http://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf

## bulk values to collect:

### daily resolution:

metric: mean, stdev, min, max
feat: temp, precip, humidity, pressure, tsoil (all), soilm (all),
    gwstor, ssrun, bgrun, weasd, lhtfl, shtfl

metric: total change
feat: precip, soilm (all), weasd, tsoil (?)

### monthly resolution:

mean, stdev, min, max
*all available forcings and model states*

histograms
temp, precip, humidity, pressure, tsoil (all), soilm (all)

### annual resolution:

mean, stdev, min, max
*all available forcings and model states*

histograms
temp, precip, humidity, pressure, tsoil (all), soilm (all)
