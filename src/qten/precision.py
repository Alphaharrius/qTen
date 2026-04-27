"""
Global numeric precision configuration for QTen.

This module stores the default floating and complex dtypes used by QTen helper
functions when constructing NumPy arrays and PyTorch tensors. The configuration
keeps NumPy and PyTorch precision choices paired so real and complex values use
consistent 32-bit or 64-bit families throughout geometry, tensor, and physics
workflows.

The default precision is 64-bit: `torch.float64`, `torch.complex128`,
`np.float64`, and `np.complex128`.
"""

from typing import Literal, Union
from collections import namedtuple
import numpy as np
import torch

_torch_float_dtype = torch.float64
_torch_complex_dtype = torch.complex128
_np_float_dtype = np.dtype(np.float64)
_np_complex_dtype = np.dtype(np.complex128)

PrecisionInput = Union[Literal["32", "64"], int, torch.dtype, np.dtype]

PrecisionInfo = namedtuple(
    "PrecisionInfo", ["torch_float", "torch_complex", "np_float", "np_complex"]
)
"""
Container describing the active real and complex dtype family.

Attributes
----------
torch_float : torch.dtype
    PyTorch real floating dtype.
torch_complex : torch.dtype
    PyTorch complex dtype paired with `torch_float`.
np_float : np.dtype
    NumPy real floating dtype.
np_complex : np.dtype
    NumPy complex dtype paired with `np_float`.
"""


def _normalize_precision(
    precision: PrecisionInput,
) -> PrecisionInfo:
    """
    Normalize a precision selector into paired NumPy and PyTorch dtypes.

    Parameters
    ----------
    precision : PrecisionInput
        Precision selector. Accepted values are `"32"`, `32`, `"64"`, `64`,
        `torch.float32`, `torch.complex64`, `torch.float64`,
        `torch.complex128`, `np.float32`, `np.complex64`, `np.float64`,
        `np.complex128`, or equivalent `np.dtype` instances.

    Returns
    -------
    PrecisionInfo
        Paired PyTorch and NumPy real/complex dtypes.

    Raises
    ------
    ValueError
        If `precision` does not describe a supported 32-bit or 64-bit dtype
        family.
    """
    if precision in ("32", 32):
        return PrecisionInfo(
            torch.float32,
            torch.complex64,
            np.dtype(np.float32),
            np.dtype(np.complex64),
        )
    if precision in ("64", 64):
        return PrecisionInfo(
            torch.float64,
            torch.complex128,
            np.dtype(np.float64),
            np.dtype(np.complex128),
        )

    if isinstance(precision, np.dtype) or (
        isinstance(precision, type) and issubclass(precision, np.generic)
    ):
        np_dt = np.dtype(precision)
        if np_dt in (np.dtype(np.float32), np.dtype(np.complex64)):
            return PrecisionInfo(
                torch.float32,
                torch.complex64,
                np.dtype(np.float32),
                np.dtype(np.complex64),
            )
        if np_dt in (np.dtype(np.float64), np.dtype(np.complex128)):
            return PrecisionInfo(
                torch.float64,
                torch.complex128,
                np.dtype(np.float64),
                np.dtype(np.complex128),
            )

    if isinstance(precision, torch.dtype):
        if precision in (torch.float32, torch.complex64):
            return PrecisionInfo(
                torch.float32,
                torch.complex64,
                np.dtype(np.float32),
                np.dtype(np.complex64),
            )
        if precision in (torch.float64, torch.complex128):
            return PrecisionInfo(
                torch.float64,
                torch.complex128,
                np.dtype(np.float64),
                np.dtype(np.complex128),
            )

    raise ValueError("Precision must be 32/64, torch.dtype, or np.dtype")


def set_precision(
    precision: PrecisionInput,
    set_torch_default: bool = True,
) -> None:
    """
    Set QTen's global real and complex dtype family.

    The selected precision updates the dtype values returned by
    [`get_precision_config()`][qten.precision.get_precision_config]. When
    `set_torch_default=True`, it also calls `torch.set_default_dtype()` with the
    selected real PyTorch dtype.

    Supported values
    ----------------
    - `"32"` or `32`: use `torch.float32`, `torch.complex64`, `np.float32`, and
      `np.complex64`.
    - `"64"` or `64`: use `torch.float64`, `torch.complex128`, `np.float64`,
      and `np.complex128`.
    - A supported real or complex `torch.dtype` selects its matching dtype
      family.
    - A supported real or complex `np.dtype` or NumPy scalar dtype selects its
      matching dtype family.

    Parameters
    ----------
    precision : PrecisionInput
        Precision selector describing the desired 32-bit or 64-bit dtype family.
    set_torch_default : bool
        If `True`, update PyTorch's process-wide default floating dtype to the
        selected real dtype. If `False`, only QTen's stored precision config is
        updated.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `precision` does not describe a supported 32-bit or 64-bit dtype
        family.
    """
    global _torch_float_dtype, _torch_complex_dtype
    global _np_float_dtype, _np_complex_dtype

    precision_info = _normalize_precision(precision)

    if set_torch_default:
        torch.set_default_dtype(precision_info.torch_float)

    _torch_float_dtype = precision_info.torch_float
    _torch_complex_dtype = precision_info.torch_complex
    _np_float_dtype = precision_info.np_float
    _np_complex_dtype = precision_info.np_complex


def get_precision_config() -> PrecisionInfo:
    """
    Return QTen's current global precision configuration.

    Returns
    -------
    PrecisionInfo
        Named tuple with fields `torch_float`, `torch_complex`, `np_float`, and
        `np_complex`.

    See Also
    --------
    [`set_precision(precision, set_torch_default)`][qten.precision.set_precision]
        Update the stored precision configuration.
    """
    return PrecisionInfo(
        _torch_float_dtype,
        _torch_complex_dtype,
        _np_float_dtype,
        _np_complex_dtype,
    )
