from typing import cast
import numpy as np
from lzcompression.decompose import decompose

from lzcompression.types import FloatArrayType, KernelStrategy
from lzcompression.util.util import compute_loss

# fmt: off

eleven_matrix = np.array(
    [
        [5., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 4., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 3., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 2., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 2., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 3., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 4., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 5.] 
    ]
)
eleven_matrix_target_rank = 6
# fmt: on

# TODO: These tests are slow, use more sophisticated pytest config to not always run them


# TODO: make this a proper test fixture
random_matrix_tolerance = 0.01
random_matrix_target_rank = 10
random_matrix_n_dimension = 250
random_matrix_m_dimension = 300
random_matrix_generator = np.random.default_rng()  # can pass optional seed
random_matrix_N = random_matrix_generator.normal(
    size=(random_matrix_n_dimension, random_matrix_target_rank)
)
random_matrix_M = random_matrix_generator.normal(
    size=(random_matrix_target_rank, random_matrix_m_dimension)
)

random_base_matrix = random_matrix_N @ random_matrix_M
random_matrix_sparse = np.copy(random_base_matrix)
random_matrix_sparse[random_matrix_sparse < 0] = 0


def test_base_model_free_kernel_random_matrix_recovery() -> None:
    low_rank = decompose(
        random_matrix_sparse,
        random_matrix_target_rank,
        KernelStrategy.BASE_MODEL_FREE,
        tolerance=random_matrix_tolerance,
    )
    low_rank = cast(FloatArrayType, low_rank)
    low_rank_relu = np.copy(low_rank)
    low_rank_relu[low_rank_relu < 0] = 0
    sparse_loss = compute_loss(low_rank_relu, random_matrix_sparse)
    assert sparse_loss < random_matrix_tolerance


def test_single_variance_gauss_kernel_random_matrix_recovery() -> None:
    result = decompose(
        random_matrix_sparse,
        random_matrix_target_rank,
        KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE,
        tolerance=random_matrix_tolerance,
    )
    low_rank = result[0]
    low_rank_relu = np.copy(low_rank)
    low_rank_relu[low_rank_relu < 0] = 0
    sparse_loss = compute_loss(low_rank_relu, random_matrix_sparse)
    assert sparse_loss < random_matrix_tolerance


def test_rowwise_variance_gauss_kernel_random_matrix_recovery() -> None:
    result = decompose(
        random_matrix_sparse,
        random_matrix_target_rank,
        KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE,
        tolerance=random_matrix_tolerance,
    )
    low_rank = result[0]
    low_rank_relu = np.copy(low_rank)
    low_rank_relu[low_rank_relu < 0] = 0
    sparse_loss = compute_loss(low_rank_relu, random_matrix_sparse)
    assert sparse_loss < random_matrix_tolerance


## TODO: Add appropriate tolerance and do an elevens-recovery test for naive method


def test_single_variance_gauss_kernel_elevens_matrix_recovery() -> None:
    result = decompose(
        eleven_matrix,
        eleven_matrix_target_rank,
        KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE,
    )
    relu_l = np.copy(result[0])
    relu_l[relu_l < 0] = 0
    assert np.allclose(relu_l, eleven_matrix)


def test_rowwise_variance_gauss_kernel_elevens_matrix_recovery() -> None:
    result = decompose(
        eleven_matrix,
        eleven_matrix_target_rank,
        KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE,
    )
    relu_l = np.copy(result[0])
    relu_l[relu_l < 0] = 0
    assert np.allclose(relu_l, eleven_matrix)
