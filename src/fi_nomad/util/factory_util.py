"""Utility functions for instantiating kernels.

Functions:
    instantiate_kernel: Factory function for low-rank-estimator kernels.

"""
from typing import Optional

from fi_nomad.kernels import (
    KernelBase,
    BaseModelFree,
    RowwiseVarianceGaussianModelKernel,
    SingleVarianceGaussianModelKernel,
)

from fi_nomad.types import (
    KernelInputType,
    KernelSpecificParameters,
    KernelStrategy,
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
