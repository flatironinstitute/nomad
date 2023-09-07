# Final library name TK

This library implements minimally lossy matrix decomposition for sparse nonnegative matrices, as described in Saul (2022). Given a sparse
nonnegative matrix **X**, it returns a low-rank matrix **L** of known target rank *r*. Applying a
[ReLU nonlinearity](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) to **L** allows lossless recovery of **X**.

Two methods of estimating the low-rank reprsentation **L** are currently offered:
a straightforward model-free method, and a method based on modeling **X** as a nonnegative realization of a latent Gaussian model.
Both methods operate in an iterative fashion analogous to expectation-maximization or *k*-means clustering.

MORE DESCRIPTION TK
<!-- a basic iterative method that alternates between constructing a utility matrix **Z** 
[Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) to the  -->

## Getting Started

The library is currently accessed through Python calls (interactive REPL, scripts, or probably Jupyter, although the latter has not been tested).

- Ensure that you have a package environment in your environment manager of choice (e.g. [Conda](https://conda.io/projects/conda/en/latest/index.html),
[venv](https://docs.python.org/3/library/venv.html))
- Required packages:
  - numpy 1.25+
  - scipy 1.11.1+
  - scikit-learn 1.3.0+

Only numpy is used extensively, but the other libraries offer more convenient implementations of some of the statistical operations.

As the `lzcompression` package is not yet published, you'll need to install it in local mode. The easiest way to do this is to:
- Clone this repository and change into it
- Ensure your appropriate environment is active
- `pip install -e .` to make `lzcompression` an installed package for the environment

We'll publish to `pypi` once we're out of alpha and have picked a good name.

## Example

Load the observed non-negative matrix `X` as a numpy array:

```python
from lzcompression.gauss_model import compress_sparse_matrix_probabilistic
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

(low_rank_L, model_variance) = compress_sparse_matrix_probabilistic(
    nonnegative_matrix_X,
    target_rank,
    verbose=True
)

# use low_rank_L as appropriate for your application

# To visualize recovery of X, assuming target_rank was high enough to do so:
#   First improve readability of printed numpy arrays:
np.set_printoptions(precision=5, linewidth=150)

#   then pass the low-rank estimate through a ReLU nonlinearity and compare
#   its result to the input sparse matrix:
relu_L = np.copy(low_rank_L)
relu_L[relu_L < 0] = 0
print(relu_L)
```

### Additional Features

The main entry point for the model-based low-rank matrix estimation is `lzcompression.gauss_model.compress_sparse_matrix_probabilistic`.
In addition to the two required parameters (the sparse nonnegative matrix and the target rank), the following options are exposed:

- `svd_strategy`: Strategy to use for SVD step during iteration. Supported options:
  - `SVDStrategy.FULL`: Full, deterministic decomposition.
  - `SVDStrategy.EXACT_TRUNCATED`: Deterministic/exact decomposition, but will not attempt to recover beyond `target_rank` values.
  - `SVDStrategy.RANDOM_TRUNCATED`: Uses a randomized algorithm for truncated SVD. The default, with much better performance.
- `initialization`: Initialization strategy for the first guess at the low-rank matrix.
  - `InitializationStrategy.BROADCAST_MEAN`: the default for the model-based algorithm; creates a first-guess low-rank matrix
  where each element is the mean of the elements of `X`.
  - `InitializationStrategy.COPY`: Uses a copy of `X` as the first guess for the low-rank matrix.
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

