# Final library name TK

This library implements methods to facilitate minimally lossy matrix decomposition for sparse nonnegative matrices, under the
paradigm described in Saul (2022).

In the problem setting, given a sparse nonnegative matrix **X**, we would like to return a low-rank matrix **L** of known
target rank *r*. Applying a
[ReLU nonlinearity](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) to **L** allows lossless recovery of **X**.

Three methods of estimating the low-rank representation **L** are currently offered:
 - a model-free method, which returns only the low-rank approximation
 - a method which returns the means-and-variance parameters (**L**, *v*) of a Gaussian model, as described in Saul (2022)
 - an extension of the Gaussian-model method which uses a different variance parameter per row

All methods operate in an iterative fashion; the model-based methods are particularly analogous to expectation-maximization,
in that they iteratively refine a model's parameters and recompute the posterior probability under the new parameters.


## Getting Started

The library is accessed through Python calls (interactive REPL and scripts have been tested; Jupyter has not, but should work).

- Ensure that you have a package environment in your environment manager of choice (e.g. [Conda](https://conda.io/projects/conda/en/latest/index.html),
[venv](https://docs.python.org/3/library/venv.html))
- Required packages:
  - numpy 1.25+
  - scipy 1.11.1+
  - scikit-learn 1.3.0+

Only numpy is used extensively, but the other libraries offer more convenient implementations of some of the statistical operations.

As the `fi_nomad` package is not yet published, you'll need to install it in local mode. The easiest way to do this is to:
- Clone this repository and change into it
- Ensure your appropriate environment is active
- `pip install -e .` to make `fi_nomad` an installed package for the environment

We'll publish to `pypi` once we're out of alpha and have picked a good name.

## Example

Load the observed non-negative matrix `X` as a numpy array:

```python
from fi_nomad.decompose import decompose
from fi_nomad.types import KernelStrategy
import numpy as np
import logging

# "info"-level log messages in the library won't be displayed unless the
# caller explicitly allows them--this is by design, since it's bad to let
# library code override its caller's logging strategy!
logging.basicConfig(level=logging.INFO)

# NOTE: Ensure that this uses a float dtype!
# If you try to use this method on an integer array, it
# isn't going to work very well!
nonnegative_matrix_X = np.array([...]) # or load from file, etc.

target_rank = 5         # as per your domain expertise

result_data = decompose(
    nonnegative_matrix_X,
    target_rank,
    kernel_strategy=KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE,
    verbose=True
)
model_means_L = result_data.reconstruction
model_variance = result_data.variance

# use model_means_L as appropriate for your application

# To visualize recovery of X, assuming target_rank was high enough to do so:
#   First improve readability of printed numpy arrays:
np.set_printoptions(precision=5, linewidth=150)

#   then pass the low-rank estimate through a ReLU nonlinearity and compare
#   its result to the input sparse matrix:
relu_L = np.copy(model_means_L)
relu_L[relu_L < 0] = 0
print(relu_L)
```

### Additional Features

The main entry point for the model-based low-rank matrix estimation is `fi_nomad.decompose.decompose`.
Three parameters are required:
- the sparse nonnegative matrix
- the target rank
- the "kernel strategy," which specifies which of the family of algorithms to use. Currently supported
  kernel strategies are:
  - `KernelStrategy.BASE_MODEL_FREE` -- a naive approach that just iteratively applies SVD, with no
    underlying statistical model
  - `KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE` -- a Gaussian model as described in Saul (2022)
  - `KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE` -- a Gaussian model similar to that described in
    Saul (2022), but which computes a different variance value for each row instead of using a single
    global/mean variance

Additionally, the following options are exposed:

- `svd_strategy`: Strategy to use for SVD step during iteration. Supported options:
  - `SVDStrategy.FULL`: Full, deterministic decomposition.
  - `SVDStrategy.EXACT_TRUNCATED`: Deterministic/exact decomposition, but will not attempt to recover beyond `target_rank` values.
  - `SVDStrategy.RANDOM_TRUNCATED`: Uses a randomized algorithm for truncated SVD. The default, with much better performance.
- `initialization`: Initialization strategy for the first guess at the low-rank matrix.
  - `InitializationStrategy.BROADCAST_MEAN`: the default for the model-based algorithm; creates a first-guess low-rank matrix
  where each element is the mean of the elements of `X`.
  - `InitializationStrategy.ROWWISE_MEAN`: creates a first-guess low-rank matrix where each element is the mean of the
  elements of `X` from the corresponding row. This is intuitively appealing for the per-row-variance Gaussian model algorithm.
  - `InitializationStrategy.COPY`: Uses a copy of `X` as the first guess for the low-rank matrix. The naive EM algorithm is
  currently restricted to this, though that may change in the future.
  - `InitializationStrategy.KNOWN_MATRIX`: When used along with a value for the `initial_guess_matrix` parameter,
  allows the caller to use any appropriately-sized matrix for the current low-rank estimate. (This is to facilitate
  checkpointing and warm starts.)
- `tolerance`: If set, the algorithm will stop early once the loss (defined as the Frobenius norm of the
difference between `X` and the current estimate) is below this value.
- `manual_max_iterations`: If set, the algorithm will use this number as the maximum number of iterations
(instead of running for 100 * target_rank iterations).
- `verbose`: if set to `True` (and the caller's logging config allows it), will log a running record of estimation performance.

Regardless of settings, the estimator will generate a warning if the iteration-over-iteration likelihood was observed
to decrease during the course of estimation. (This is quite common due to numerics noise once a good estimate has been
reached.)


## References

Lawrence K Saul (2022), "A Nonlinear Matrix Decomposition for Mining the Zeros of Sparse Data"
[https://doi.org/10.1137/21M1405769](https://doi.org/10.1137/21M1405769)
(Preprint: https://cseweb.ucsd.edu/~saul/papers/preprints/simods22_preprint.pdf)

