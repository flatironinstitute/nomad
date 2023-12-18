from typing import Tuple, cast
from unittest.mock import Mock, patch
from pytest import raises

from fi_nomad.types.enums import KernelStrategy, DiagnosticLevel
from fi_nomad.types.types import DiagnosticDataConfig
from fi_nomad.kernels.kernel_base import KernelBase


from fi_nomad.util.factory_util import instantiate_kernel, get_diagnostic_fn

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


def test_get_diagnostic_fn_returns_null_fn_on_no_config() -> None:
    mock_kernel = Mock()
    res = get_diagnostic_fn(cast(KernelBase, mock_kernel), None)
    assert res() == None


@patch(f"{PKG}.make_path")
def test_get_diagnostic_fn_returns_callback(mock_make_path: Mock) -> None:
    mock_kernel = Mock()
    mock_basedir = "MOCK_BASEDIR"
    mock_outpath = "MOCK_PATH"
    mock_make_path.return_value = mock_outpath

    mock_diag_config = DiagnosticDataConfig(
        DiagnosticLevel.EXTREME, mock_basedir, False
    )

    def mock_callback(
        *, diagnostic_level: DiagnosticLevel, out_dir: str
    ) -> Tuple[DiagnosticLevel, str]:
        return (diagnostic_level, out_dir)

    mock_kernel.per_iteration_diagnostic = mock_callback

    callback = get_diagnostic_fn(cast(KernelBase, mock_kernel), mock_diag_config)
    mock_make_path.assert_called_once_with(mock_basedir, use_exact_path=False)
    res = callback()
    assert res == (DiagnosticLevel.EXTREME, mock_outpath)
