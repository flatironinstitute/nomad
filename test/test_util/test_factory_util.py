from typing import Tuple, cast
from unittest.mock import Mock, patch
from pytest import raises

from fi_nomad.types.enums import KernelStrategy, DiagnosticLevel
from fi_nomad.types.types import DiagnosticDataConfig
from fi_nomad.kernels.kernel_base import KernelBase


from fi_nomad.util.factory_util import (
    instantiate_kernel,
    do_diagnostic_configuration,
    empty_fn,
)

PKG = "fi_nomad.util.factory_util"


def test_instantiate_kernel_throws_on_unsupported_kernel() -> None:
    mock_data_in = Mock()
    with raises(ValueError, match="Unsupported"):
        _ = instantiate_kernel(KernelStrategy.TEST, mock_data_in)


def test_instantiate_kernel_throws_on_non_matching_kernel_params() -> None:
    mock_data_in = Mock()
    with raises(TypeError, match="kernel_params"):
        _ = instantiate_kernel(KernelStrategy.MOMENTUM_3_BLOCK_MODEL_FREE, mock_data_in)


@patch(f"{PKG}.BaseModelFree", autospec=True)
def test_insantiate_kernel_throws_on_instantiation_failure(mock_base: Mock) -> None:
    mock_data_in = Mock()
    mock_base.return_value = None
    with raises(ValueError, match="not initialized"):
        _ = instantiate_kernel(KernelStrategy.BASE_MODEL_FREE, mock_data_in)


@patch(f"{PKG}.RowwiseVarianceGaussianModelKernel", autospec=True)
@patch(f"{PKG}.SingleVarianceGaussianModelKernel", autospec=True)
@patch(f"{PKG}.BaseModelFree", autospec=True)
def test_instantiate_kernel_honors_strategy_selection(
    mock_base: Mock, mock_svgauss: Mock, mock_rwgauss: Mock
) -> None:
    mock_data_in = Mock()
    base_response = instantiate_kernel(KernelStrategy.BASE_MODEL_FREE, mock_data_in)
    # Confirms that we are actually calling do_diagnostic_configuration
    assert base_response.per_iteration_diagnostic == empty_fn
    mock_base.assert_called_once()
    _ = instantiate_kernel(KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE, mock_data_in)
    mock_svgauss.assert_called_once()
    _ = instantiate_kernel(KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE, mock_data_in)
    mock_rwgauss.assert_called_once()


def test_do_diagnostic_configuration_clears_fn_on_no_config() -> None:
    mock_kernel = Mock()
    res = do_diagnostic_configuration(cast(KernelBase, mock_kernel), None)
    assert res.per_iteration_diagnostic == empty_fn


def test_do_diagnostic_configuration_clears_fn_on_none_level() -> None:
    mock_kernel = Mock()
    mock_diag_config = DiagnosticDataConfig(DiagnosticLevel.NONE, "unused", False)
    res = do_diagnostic_configuration(cast(KernelBase, mock_kernel), mock_diag_config)
    assert res.per_iteration_diagnostic == empty_fn


@patch(f"{PKG}.make_path")
def test_get_diagnostic_fn_configures_kernel(mock_make_path: Mock) -> None:
    mock_kernel = Mock()
    mock_basedir = "MOCK_BASEDIR"
    mock_outpath = "MOCK_PATH"
    mock_make_path.return_value = mock_outpath

    mock_diag_config = DiagnosticDataConfig(
        DiagnosticLevel.EXTREME, mock_basedir, False
    )
    res = do_diagnostic_configuration(cast(KernelBase, mock_kernel), mock_diag_config)
    mock_make_path.assert_called_once_with(mock_basedir, use_exact_path=False)
    assert res.diagnostic_level == mock_diag_config.diagnostic_level
    assert str(res.out_dir) == mock_outpath
