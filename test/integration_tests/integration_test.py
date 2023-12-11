import numpy as np
import pytest
from lzcompression.decompose import decompose
from lzcompression.types import KernelStrategy
from lzcompression.util.util import compute_loss

pytestmark = pytest.mark.integration

all_kernels = [
    KernelStrategy.BASE_MODEL_FREE,
    KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE,
    KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE,
]

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


@pytest.mark.parametrize("kernel", all_kernels)
def test_random_matrix_recovery(kernel: KernelStrategy) -> None:
    result = decompose(
        random_matrix_sparse,
        random_matrix_target_rank,
        kernel,
        tolerance=random_matrix_tolerance,
    )
    low_rank = result.reconstruction
    low_rank_relu = np.copy(low_rank)
    low_rank_relu[low_rank_relu < 0] = 0
    sparse_loss = compute_loss(low_rank_relu, random_matrix_sparse)
    assert sparse_loss < random_matrix_tolerance


@pytest.mark.parametrize("kernel", all_kernels)
def test_model_kernel_elevens_matrix_recovery(kernel: KernelStrategy) -> None:
    result = decompose(eleven_matrix, eleven_matrix_target_rank, kernel)
    relu_l = np.copy(result.reconstruction)
    relu_l[relu_l < 0] = 0
    if kernel == KernelStrategy.BASE_MODEL_FREE:
        assert np.allclose(relu_l, eleven_matrix, atol=0.002)
    else:
        assert np.allclose(relu_l, eleven_matrix)
