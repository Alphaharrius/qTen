"""
Top-level public API for QTen.

This package re-exports the most commonly used tensor, device, and precision
entry points so users can work from `qten` directly without importing deep
submodules in day-to-day code.

Main exports
------------
- [`Tensor`][qten.linalg.tensors.Tensor]
  StateSpace-aware tensor wrapper over `torch.Tensor`.
- [`Device`][qten.utils.devices.Device]
  Logical device descriptor used by QTen objects.
- [`set_precision`][qten.precision.set_precision]
  Configure numeric precision defaults for the library.

Tensor construction and manipulation
------------------------------------
- [`zeros`][qten.linalg.tensors.zeros], [`ones`][qten.linalg.tensors.ones], [`eye`][qten.linalg.tensors.eye], [`one_hot`][qten.linalg.tensors.one_hot]
- [`permute`][qten.linalg.tensors.permute], [`transpose`][qten.linalg.tensors.transpose], [`squeeze`][qten.linalg.tensors.squeeze], [`unsqueeze`][qten.linalg.tensors.unsqueeze]
- [`replace_dim`][qten.linalg.tensors.replace_dim], [`factorize_dim`][qten.linalg.tensors.factorize_dim], [`product_dims`][qten.linalg.tensors.product_dims], [`promote_rank`][qten.linalg.tensors.promote_rank]
- [`align`][qten.linalg.tensors.align], [`align_all`][qten.linalg.tensors.align_all], [`expand_to_union`][qten.linalg.tensors.expand_to_union], [`union_dims`][qten.linalg.tensors.union_dims]

Tensor algebra and queries
--------------------------
- [`matmul`][qten.linalg.tensors.matmul], [`norm`][qten.linalg.tensors.norm], [`mean`][qten.linalg.tensors.mean]
- [`all`][qten.linalg.tensors.all], [`allclose`][qten.linalg.tensors.allclose], [`equal`][qten.linalg.tensors.equal], [`isclose`][qten.linalg.tensors.isclose]
- [`argmax`][qten.linalg.tensors.argmax], [`argmin`][qten.linalg.tensors.argmin], [`nonzero`][qten.linalg.tensors.nonzero], [`where`][qten.linalg.tensors.where]
- [`real`][qten.linalg.tensors.real], [`imag`][qten.linalg.tensors.imag], [`conj`][qten.linalg.tensors.conj], [`astype`][qten.linalg.tensors.astype]

Linear-algebra helpers
----------------------
- [`mapping_matrix`][qten.linalg.tensors.mapping_matrix], [`kernel_tensor`][qten.linalg.tensors.kernel_tensor], [`cat`][qten.linalg.tensors.cat]

Devices and I/O
---------------
- [`at_device`][qten.linalg.tensors.at_device]
  Context manager forcing newly created tensors onto a chosen logical device.
- [`io`][qten.utils.io]
  Save/load helpers exposed as a convenience namespace.

Subpackages
-----------
For broader domain APIs, see:
- [`qten.geometries`][qten.geometries]
- [`qten.linalg`][qten.linalg]
- [`qten.phys`][qten.phys]
- [`qten.pointgroups`][qten.pointgroups]
- [`qten.symbolics`][qten.symbolics]
- [`qten.utils`][qten.utils]
"""

from .precision import set_precision as set_precision

from .linalg.tensors import (
    Tensor as Tensor,
    align as align,
    align_all as align_all,
    all as all,
    allclose as allclose,
    at_device as at_device,
    argmax as argmax,
    argmin as argmin,
    abs as abs,
    astype as astype,
    cat as cat,
    conj as conj,
    equal as equal,
    einsum as einsum,
    expand_to_union as expand_to_union,
    eye as eye,
    factorize_dim as factorize_dim,
    kernel_tensor as kernel_tensor,
    mapping_matrix as mapping_matrix,
    matmul as matmul,
    mean as mean,
    norm as norm,
    nonzero as nonzero,
    one_hot as one_hot,
    imag as imag,
    isclose as isclose,
    ones as ones,
    permute as permute,
    product_dims as product_dims,
    promote_rank as promote_rank,
    real as real,
    rank as rank,
    replace_dim as replace_dim,
    squeeze as squeeze,
    transpose as transpose,
    union_dims as union_dims,
    unsqueeze as unsqueeze,
    where as where,
    zeros as zeros,
)

from .utils.devices import Device as Device
from .utils import io as io
