"""Utility functions for instantiating kernels.

Functions:
    instantiate_kernel: Factory function for low-rank-estimator kernels.
    do_diagnostic_configuration: Configure kernel for per-iteration diagnostic output.

"""

from typing import Optional, cast

from fi_nomad.kernels import (
    KernelBase,
    BaseModelFree,
    RowwiseVarianceGaussianModelKernel,
    SingleVarianceGaussianModelKernel,
    Momentum3BlockModelFreeKernel,
)

from fi_nomad.types.enums import DiagnosticLevel

from fi_nomad.util.path_util import make_path

from fi_nomad.types import (
    KernelInputType,
    KernelSpecificParameters,
    KernelStrategy,
    DiagnosticDataConfig,
)


def empty_fn() -> None:  # pragma: no cover
    """No-op function, used to replace per-iteration diagnostic data output
    when caller requests no diagnostic output.
    """


def instantiate_kernel(
    s: KernelStrategy,
    data_in: KernelInputType,
    *,
    kernel_params: Optional[KernelSpecificParameters] = None,
    diagnostic_config: Optional[DiagnosticDataConfig] = None,
) -> KernelBase:
    """Factory function to instantiate and configure a decomposition kernel.

    Args:
        s: The defined KernelStrategy to instantiate
        data_in: Input data for the kernel
        kernel_params: Optional kernel-specific parameters. Defaults to None.
        diagnostic_config: Optional configuration for per-iteration diagnostic
            data output. Defaults to None (turning the feature off).

    Raises:
        ValueError: Raised if an unrecognized kernel type is requested.

    Returns:
        The instantiated decomposition kernel, conforming to the standard interface.
    """
    kernel: Optional[KernelBase] = None
    if s == KernelStrategy.BASE_MODEL_FREE:
        kernel = BaseModelFree(data_in)
    elif s == KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE:
        kernel = SingleVarianceGaussianModelKernel(data_in)
    elif s == KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE:
        kernel = RowwiseVarianceGaussianModelKernel(data_in)
    elif s == KernelStrategy.MOMENTUM_3_BLOCK_MODEL_FREE:
        kernel = Momentum3BlockModelFreeKernel(data_in, kernel_params)
    else:
        raise ValueError(f"Unsupported kernel strategy {s}")
    if kernel is None:
        raise ValueError("Error in kernel configuration: kernel not initialized.")
    kernel = do_diagnostic_configuration(kernel, diagnostic_config)
    return kernel


def do_diagnostic_configuration(
    kernel: KernelBase, config: Optional[DiagnosticDataConfig]
) -> KernelBase:
    """Configure kernel for per-iteration diagnostic output.

    Note that if the "config" object is None/omitted, or if the requested level
    is NONE, the kernel's diagnostic function will be replaced with a no-op.

    Args:
        kernel: Instantiated kernel capable of reporting
        config: Configuration object for diagnostic data setup. Defaults to None.

    Returns:
        The kernel, with stored config, and per-iteration method replaced if
        turned off by the caller.
    """
    if config is None or config.diagnostic_level == DiagnosticLevel.NONE:
        # rather than execute a test on every iteration, replace with no-op
        kernel.per_iteration_diagnostic = empty_fn  # type: ignore[method-assign]
        return kernel
    config = cast(DiagnosticDataConfig, config)
    kernel.diagnostic_level = config.diagnostic_level
    req_basedir = config.diagnostic_output_basedir
    req_exact = config.use_exact_diagnostic_basepath
    kernel.out_dir = make_path(req_basedir, use_exact_path=req_exact)
    return kernel
