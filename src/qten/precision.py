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


def _normalize_precision(
    precision: PrecisionInput,
) -> PrecisionInfo:
    """
    Normalize user precision input to (torch_float, torch_complex, np_float, np_complex).
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
    Sets default precision and updates global dtype references.
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
    Returns current global precision config (torch_float, torch_complex, np_float, np_complex).
    """
    return PrecisionInfo(
        _torch_float_dtype,
        _torch_complex_dtype,
        _np_float_dtype,
        _np_complex_dtype,
    )
