# coordinate feature dataset

CoordinateFeatureDataset (CFD) objects should define the input
and output shapes of any DerivedFeat functional, so that the
order and shapes of all inputs and outputs can be verified before
data has begun processing.

DerivedFeat instances may include coordinate transformations that
allow for output cfa to have different numbers of elements on any
coordinate axis, and corresponding new coordinate metrics that
default to the integer range of the axis, but may also be calculated
by an arbitrary function of the prior coordinate metrics.

When a coordinate-transforming DerivedFeat is applied, it will need
to explicitly provide a way of resolving the new coord metrics as a
function of the input cfd's arrays or metrics

Generators could be defined in the same way, except that they
institute the "batch", which is the conceptual grouping of the
sub-groups implied by the iteration of the generator.
of iteration. The evaluator, training process, axis of shuffling,
etc iterate or accumulate over this axes.

CFDs must facilitate multiple CFAs with the same number of dimensions
since they cannot be combined along the feat axis when their
coordinate dims are incompatible (ie different coord metrics, sizes).

CFD profiles (without CFA.data) must be easy to initialize with a
current CFD instance (same coordinate, etc), but with some
modifications, as a generator might collapse a dimension or something

