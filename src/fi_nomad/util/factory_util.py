"""Utility functions for instantiating kernels.

Functions:
    instantiate_kernel: Factory function for low-rank-estimator kernels.
    get_diagnostic_fn: Given a kernel and diagnostic config, generate a per-iteration callback.

"""
from typing import Callable, Optional, cast

from fi_nomad.kernels import (
    KernelBase,
    BaseModelFree,
    RowwiseVarianceGaussianModelKernel,
    SingleVarianceGaussianModelKernel,
)

from fi_nomad.util.path_util import make_path

from fi_nomad.types import (
    KernelInputType,
    KernelSpecificParameters,
    KernelStrategy,
    DiagnosticDataConfig,
)


def instantiate_kernel(
    s: KernelStrategy,
    data_in: KernelInputType,
    kernel_params: Optional[KernelSpecificParameters] = None,
) -> KernelBase:
    """Factory function to instantiate and configure a decomposition kernel.

    Args:
        s: The defined KernelStrategy to instantiate
        data_in: Input data for the kernel
        kernel_params: Optional kernel-specific parameters. Defaults to None.

    Raises:
        NotImplementedError: Raised if optional kernel parameters are passed.
        ValueError: Raised if an unrecognized kernel type is requested.

    Returns:
        The instantiated decomposition kernel, conforming to the standard interface.
    """
    if kernel_params is not None:
        raise NotImplementedError(
            "No kernel is using these yet. Remove this note when implemented."
        )
    if s == KernelStrategy.BASE_MODEL_FREE:
        return BaseModelFree(data_in)
    if s == KernelStrategy.GAUSSIAN_MODEL_SINGLE_VARIANCE:
        return SingleVarianceGaussianModelKernel(data_in)
    if s == KernelStrategy.GAUSSIAN_MODEL_ROWWISE_VARIANCE:
        return RowwiseVarianceGaussianModelKernel(data_in)
    raise ValueError(f"Unsupported kernel strategy {s}")


def get_diagnostic_fn(
    kernel: KernelBase, config: Optional[DiagnosticDataConfig]
) -> Callable[[], None]:
    """Get appropriate callback for this kernel's per-iteration diagnostic output.

    Note that if the "config" object is None/omitted, the returned value will be
    a no-op. Also, because we are returning a callback that is instantiated before
    the main loop begins, implementing kernels should not expect to be able to
    change configuration values during loop execution.

    Args:
        kernel: Instantiated kernel capable of reporting
        config: Configuration object for diagnostic data setup. Defaults to None.

    Returns:
        A parameterless callback which the main decompose loop can execute at
        every pass, to allow the kernel to do whatever kernel-specific detailed
        diagnostic logging is desired (e.g. writing data files, internal state).
    """
    if config is None:
        return lambda: None
    config = cast(DiagnosticDataConfig, config)

    req_level = config.diagnostic_level
    req_basedir = config.diagnostic_output_basedir
    req_exact = config.use_exact_diagnostic_basepath
    diag_outdir = make_path(req_basedir, use_exact_path=req_exact)
    return lambda: kernel.per_iteration_diagnostic(
        diagnostic_level=req_level, out_dir=diag_outdir
    )
