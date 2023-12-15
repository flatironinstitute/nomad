from unittest.mock import Mock, patch
from pytest import raises

from fi_nomad.types.enums import KernelStrategy


from fi_nomad.util.factory_util import instantiate_kernel

PKG = "fi_nomad.util.factory_util"


def test_instantiate_kernel_throws_on_unsupported_kernel() -> None:
    mock_data_in = Mock()
    with raises(ValueError, match="Unsupported"):
        _ = instantiate_kernel(KernelStrategy.TEST, mock_data_in)


def test_instantiate_kernel_throws_on_unsupported_feature() -> None:
    mock_data_in = Mock()
    some_non_null_value = 0.5
    with raises(NotImplementedError):
        _ = instantiate_kernel(
            KernelStrategy.BASE_MODEL_FREE, mock_data_in, some_non_null_value
        )


@patch(f"{PKG}.RowwiseVarianceGaussianModelKernel", autospec=True)
@patch(f"{PKG}.SingleVarianceGaussianModelKernel", autospec=True)
@patch(f"{PKG}.BaseModelFree", autospec=True)
def test_instantiate_kernel_honors_strategy_selection(
    mock_base: Mock, mock_svgauss: Mock, mock_rwgauss: Mock
) -> None:
    mock_data_in = Mock()
    _ = instantiate_kernel(KernelStrategy.BASE_MODEL_FREE, mock_data_in)
    mock_base.assert_called_once()
    _ = instantiate_kernel(KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE, mock_data_in)
    mock_svgauss.assert_called_once()
    _ = instantiate_kernel(KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE, mock_data_in)
    mock_rwgauss.assert_called_once()
