# conditional evaluation

samples indexed by the batch axis can by processed by a generator
if they meet a condition defined by a boolean function. The boolean
function should based on the their CoordinateFeatureDataset (CFD),
which includes features from all available arrays, and the metrics
from each coordinate.

## Role of the generator

 - Although both generators and DerivedFeature (DF) functionals have
   expected input and output configurations defined by CFDs,
   (1) generators create data batch-wise, and thus may accumulate
   into a different CFD configuration than they convey per yield, and
   (2) generators are generally initialized to ingest data from one
   or more files, and thus will need to adapt to the particular
   implementation of each file (ie feat order, coord shapes, etc).
 - Generator-returning function wraps within the generator states
   which are immutable once the generator starts processing.
    - The sufficiency of a series of files can be checked when the
      generators are initialized based on the files' attributes.
    - Parameters of the sampling policy are wrapped in the generator
    - DFs may add features to CFA objects, add CFA objects to the
      returned CFD
