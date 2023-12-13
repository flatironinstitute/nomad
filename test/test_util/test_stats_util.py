import numpy as np
from pytest import approx

from fi_nomad.util.stats_util import pdf_to_cdf_ratio_psi


def test_pdf_to_cdf_ratio_psi() -> None:
    crossover = -0.30263083
    # float: example -0.30263083,
    # pdf: 0.3810856, cdf: 0.381056
    res1 = pdf_to_cdf_ratio_psi(crossover)
    assert approx(res1) == 1.0
    # Try with arrays as well
    # 0: cdf = 0.5, pdf = 0.3989, ratio 0.79788456...
    expected_scalar = 0.79788456
    expected = np.array([[1.0, expected_scalar], [expected_scalar, 1.0]])
    matrix = np.eye(2) * crossover
    res2 = pdf_to_cdf_ratio_psi(matrix)
    np.testing.assert_allclose(res2, expected)
