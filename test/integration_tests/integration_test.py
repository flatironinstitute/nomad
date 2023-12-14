import numpy as np
import pytest
from fi_nomad import decompose
from fi_nomad.types import KernelStrategy
from fi_nomad.util import compute_loss

pytestmark = pytest.mark.integration

all_kernels = [
    KernelStrategy.BASE_MODEL_FREE,
    KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE,
    KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE,
]
BASE_MODEL_FREE_ELEVENS_ATOL = 0.002

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
    low_rank = result.factors[0] @ result.factors[1]
    low_rank_relu = np.copy(low_rank)
    low_rank_relu[low_rank_relu < 0] = 0
    sparse_loss = compute_loss(low_rank_relu, random_matrix_sparse)
    assert sparse_loss < random_matrix_tolerance
    lowrank_factored = result.factors[0] @ result.factors[1]
    np.testing.assert_allclose(lowrank_factored, low_rank)


@pytest.mark.parametrize("kernel", all_kernels)
def test_model_kernel_elevens_matrix_recovery(kernel: KernelStrategy) -> None:
    result = decompose(eleven_matrix, eleven_matrix_target_rank, kernel)
    low_rank = result.factors[0] @ result.factors[1]
    relu_l = np.copy(low_rank)
    relu_l[relu_l < 0] = 0
    if kernel == KernelStrategy.BASE_MODEL_FREE:
        assert np.allclose(relu_l, eleven_matrix, atol=BASE_MODEL_FREE_ELEVENS_ATOL)
    else:
        assert np.allclose(relu_l, eleven_matrix)
