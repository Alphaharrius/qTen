"""
StateSpace-aware tensor primitives and helpers for QTen.

This module defines [`Tensor`][qten.linalg.tensors.Tensor], QTen's wrapper
around `torch.Tensor`, together with the dimension-aware operations that
preserve symbolic axis metadata during tensor algebra.

Core API
--------
- [`Tensor`][qten.linalg.tensors.Tensor]
  Tensor wrapper pairing raw torch data with symbolic
  [`StateSpace`][qten.symbolics.state_space.StateSpace] dims.
- [`at_device`][qten.linalg.tensors.at_device]
  Context manager for forcing newly created QTen tensors onto a device.
- [`matmul`][qten.linalg.tensors.matmul]
  StateSpace-aware matrix multiplication.
- [`einsum`][qten.linalg.tensors.einsum]
  Einstein-summation helper with StateSpace-aware axis alignment.
- [`kron`][qten.linalg.tensors.kron]
  StateSpace-aware Kronecker product.
- [`align`][qten.linalg.tensors.align]
  Reorder or expand a tensor axis so it matches a target symbolic dimension.
- [`union_dims`][qten.linalg.tensors.union_dims]
  Broadcast-compatible dimension merge used by arithmetic and comparison ops.

Construction helpers
--------------------
- [`zeros`][qten.linalg.tensors.zeros]
- [`ones`][qten.linalg.tensors.ones]
- [`eye`][qten.linalg.tensors.eye]
- [`kernel_tensor`][qten.linalg.tensors.kernel_tensor]
- [`mapping_matrix`][qten.linalg.tensors.mapping_matrix]

Shape and metadata transforms
-----------------------------
- [`permute`][qten.linalg.tensors.permute]
- [`transpose`][qten.linalg.tensors.transpose]
- [`replace_dim`][qten.linalg.tensors.replace_dim]
- [`update_dim`][qten.linalg.tensors.update_dim]
- [`factorize_dim`][qten.linalg.tensors.factorize_dim]
- [`product_dims`][qten.linalg.tensors.product_dims]
- [`promote_rank`][qten.linalg.tensors.promote_rank]

Indexing and selection
----------------------
- [`where`][qten.linalg.tensors.where]
- [`nonzero`][qten.linalg.tensors.nonzero]
- [`index_add`][qten.linalg.tensors.index_add]
- [`index_add_`][qten.linalg.tensors.index_add_]

Conventions
-----------
Every public helper in this module treats `tensor.dims` as part of the tensor's
meaning, not just its shape. Functions therefore describe not only how data is
transformed, but also how the symbolic
[`StateSpace`][qten.symbolics.state_space.StateSpace] metadata is preserved,
reordered, merged, or reduced.
"""

from typing import (
    Self,
    NamedTuple,
    Tuple,
    Type,
    TypeVar,
    Generic,
    Union,
    Sequence,
    cast,
    Dict,
    Any,
    Optional,
    Callable,
    TypeAlias,
    get_origin,
)
from typing_extensions import override
from contextlib import ContextDecorator
from functools import wraps, reduce
from itertools import product
from numbers import Number
from dataclasses import dataclass, replace
from threading import local
from types import EllipsisType
from collections import OrderedDict
import builtins

from multimethod import DispatchError, multimethod
import torch

from ..abstracts import Convertible, HasKroneckerProduct, Operable
from ..plottings import Plottable
from ..precision import get_precision_config
from ..utils.devices import Device, DeviceBounded
from ..validations import need_validation
from ..symbolics import (
    StateSpace,
    BroadcastSpace,
    IndexSpace,
    StateSpaceFactorization,
    embedding_order,
    permutation_order,
    same_rays,
)


T = TypeVar("T", bound=torch.Tensor)
TensorType = TypeVar("TensorType", bound="Tensor[Any]")
"""
The `torch.Tensor` types to be used in `Tensor`. 
This is a type variable that can be any subclass of `torch.Tensor`, 
such as `torch.FloatTensor`, `torch.DoubleTensor`, etc.
"""


_TENSOR_DEVICE_STATE = local()


def _tensor_device_stack() -> list[Device]:
    stack = getattr(_TENSOR_DEVICE_STATE, "stack", None)
    if stack is None:
        stack = []
        _TENSOR_DEVICE_STATE.stack = stack
    return cast(list[Device], stack)


def _forced_tensor_device() -> Optional[Device]:
    stack = cast(tuple[Device, ...], tuple(getattr(_TENSOR_DEVICE_STATE, "stack", ())))
    if not stack:
        return None
    return stack[-1]


class at_device(ContextDecorator):
    """
    Temporarily force newly created QTen tensors onto a specific device.

    This applies to [`Tensor(...)`][qten.linalg.tensors.Tensor] construction within the current thread,
    including tensors created indirectly by helper functions in this module.
    Nested scopes are supported; the innermost device takes precedence.
    """

    def __init__(self, device: Device | str):
        self.device = Device.new(device) if isinstance(device, str) else device

    def __enter__(self) -> "at_device":
        """
        Activate the context for the current thread.

        The target device is validated eagerly by resolving its underlying
        `torch.device`. The resolved device is then pushed onto the thread-local
        device stack used by `Tensor.__post_init__`.

        Returns
        -------
        at_device
            This context manager instance.
        """
        self.device.torch_device()
        _tensor_device_stack().append(self.device)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """
        Deactivate the current device-forcing scope.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type raised inside the context, if any.
        exc : BaseException | None
            Exception instance raised inside the context, if any.
        tb : Any
            Traceback object associated with `exc`, if any.

        Notes
        -----
        The exception information is ignored; this method only restores the
        previous thread-local device stack. Any active exception continues to
        propagate normally.
        """
        stack = _tensor_device_stack()
        stack.pop()
        if not stack:
            delattr(_TENSOR_DEVICE_STATE, "stack")


def _check_data_compatible_with_dims(tensor: "Tensor") -> None:
    """
    Validator function to check that a tensor's data shape matches its dims.

    This is a standalone function version of `ValidateDataDimsCompatibility.validate`
    for use in `@need_validation` without needing to define a separate class.
    """
    shape = tuple(d.dim for d in tensor.dims)
    if tensor.data.shape != shape:
        raise ValueError(
            f"Tensor data shape {tensor.data.shape} does not match expected shape {shape}."
        )


@need_validation(_check_data_compatible_with_dims)
@dataclass(frozen=True, eq=False)
class Tensor(
    Generic[T], Operable, Plottable, Convertible, DeviceBounded, HasKroneckerProduct
):
    r"""
    StateSpace-aware tensor wrapper over `torch.Tensor`.

    A [`Tensor`][qten.linalg.tensors.Tensor] pairs raw tensor data with symbolic axis metadata in `dims`.
    Each entry of `dims` is a [`StateSpace`][qten.symbolics.state_space.StateSpace] whose size must match the
    corresponding axis of `data`. This lets tensor operations preserve not only
    shapes, but also the semantic identity and ordering of axes.

    Core model
    ----------
    - `data` stores the underlying numeric values as a `torch.Tensor`.
    - `dims` stores one [`StateSpace`][qten.symbolics.state_space.StateSpace] per axis.
    - `tensor.data.shape` must equal `tuple(dim.dim for dim in tensor.dims)`.
    - Axes are aligned by [`StateSpace`][qten.symbolics.state_space.StateSpace] compatibility rather than by position
      alone. If two axes represent the same rays in different orders, QTen can
      permute one operand to match the other.
    - Singleton broadcast axes are represented symbolically by
      [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace].

    Attributes
    ----------
    data : T
        Underlying `torch.Tensor` storing the numeric values.
    dims : Tuple[StateSpace, ...]
        Symbolic dimension metadata, with one
        [`StateSpace`][qten.symbolics.state_space.StateSpace] per axis of
        `data`.

    Construction
    ------------
    Create tensors directly from a torch tensor and a matching dims tuple:

    [`Tensor(data=torch.randn(2, 3), dims=(left_space, right_space))`][qten.linalg.tensors.Tensor]

    Use `Tensor.scalar(number)` to construct a rank-0 tensor.

    Registered operations
    ---------------------
    The public arithmetic and comparison operators on [`Tensor`][qten.linalg.tensors.Tensor] are implemented by
    multimethod registrations on [`Operable`][qten.abstracts.Operable]. Those inherited
    `__xxx__` members are hidden from the generated API page, so this section is
    the canonical reference for Tensor-specific operator behavior.

    Matrix multiplication
    ---------------------
    `a @ b` contracts two tensors with StateSpace-aware matrix multiplication.
    The actual logic is implemented by [`matmul`][qten.linalg.tensors.matmul], which:

    - aligns shared contraction axes by [`StateSpace`][qten.symbolics.state_space.StateSpace],
    - supports batch broadcasting over leading axes,
    - preserves output metadata on the surviving axes.

    For ordinary rank-2 matrix axes this is the contraction

    \((A B)_{ik} = \sum_j A_{ij} B_{jk}\), with the contracted index matched through symbolic
    [`StateSpace`][qten.symbolics.state_space.StateSpace] metadata.

    Addition and subtraction
    ------------------------
    `a + b` and `a - b` operate on two tensors using StateSpace-aware alignment.

    - If ranks differ, the lower-rank operand is promoted with leading
      [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axes.
    - Output metadata is computed from [`union_dims(..., allow_merge=True)`][qten.linalg.tensors.union_dims].
    - Broadcast axes are materialized with [`expand_to_union`][qten.linalg.tensors.expand_to_union].
    - If two compatible axes represent the same rays in different orders, the
      right-hand operand is embedded into the merged output ordering before the
      data is accumulated.

    Scalar addition and subtraction are not element-wise broadcast operations.

    - `a + c` and `c + a` treat `a` as a matrix or batch of matrices over its
      last two axes and compute \(A + cI\).
    - `a - c` computes \(A - cI\).
    - `c - a` computes \(cI - A\).
    - These operations therefore require metadata that can construct
      [`eye`][qten.linalg.tensors.eye] on the tensor dims.

    Equivalently, scalar shifts act as \(A \mapsto A + cI\) on the final two
    axes, not as element-wise broadcasting over every entry.

    Negation and scaling
    --------------------
    - `-a` negates the tensor element-wise while preserving dims.
    - `a * c` and `c * a` perform element-wise scalar multiplication.
    - `a / c` performs element-wise scalar division.

    Equality and ordered comparisons
    --------------------------------
    `a == b`, `a < b`, `a <= b`, `a > b`, and `a >= b` use symmetric
    StateSpace-aware comparison semantics.

    - operands are rank-promoted to a common rank,
    - dims are merged with [`union_dims(..., allow_merge=False)`][qten.linalg.tensors.union_dims],
    - both operands are aligned to the merged dims,
    - torch broadcasting is validated against the merged symbolic shape,
    - the result is a bool [`Tensor`][qten.linalg.tensors.Tensor] on the merged dims.

    Scalar comparisons promote the scalar through `Tensor.scalar(...)`, so:

    - `a < c`, `a <= c`, `a > c`, `a >= c` follow the same tensor-tensor
      comparison pipeline,
    - `c < a`, `c <= a`, `c > a`, `c >= a` use the reflected comparison with
      the scalar converted to a rank-0 tensor first.

    Comparison semantics
    --------------------
    Comparisons use symmetric StateSpace-aware alignment:
    - operands are rank-promoted with leading [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace] axes when
      needed,
    - dims are merged with [`union_dims(..., allow_merge=False)`][qten.linalg.tensors.union_dims],
    - operands are aligned to the merged dims,
    - runtime broadcast compatibility is validated,
    - the result is a bool [`Tensor`][qten.linalg.tensors.Tensor] with those merged dims.

    Ordered comparisons on complex tensors are not supported and defer to
    PyTorch's runtime error behavior.

    Shape and axis transforms
    -------------------------
    - [`permute(*order)`][qten.linalg.tensors.Tensor.permute]: reorder axes.
    - [`transpose(dim0, dim1)`][qten.linalg.tensors.Tensor.transpose]: swap two axes.
    - [`h(dim0, dim1)`][qten.linalg.tensors.Tensor.h]: conjugate transpose across two axes.
    - [`unsqueeze(dim)`][qten.linalg.tensors.Tensor.unsqueeze]: insert a singleton [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axis.
    - [`squeeze(dim)`][qten.linalg.tensors.Tensor.squeeze]: remove a [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axis if present.
    - [`replace_dim(dim, new_dim)`][qten.linalg.tensors.Tensor.replace_dim]: replace metadata for one axis with size
      validation.
    - [`update_dim(dim, func)`][qten.linalg.tensors.Tensor.update_dim]: transform one axis metadata with a callback
      before validation.
    - [`factorize_dim(dim, rule)`][qten.linalg.tensors.Tensor.factorize_dim]: split one axis into multiple factor spaces.
    - [`product_dims(*groups)`][qten.linalg.tensors.Tensor.product_dims]: combine groups of axes into tensor-product axes.
    - [`promote_rank(tensor, target_rank)`][qten.linalg.tensors.promote_rank]: prepend broadcast axes.

    Alignment and metadata utilities
    --------------------------------
    - [`align(dim, target_dim)`][qten.linalg.tensors.Tensor.align]: align one axis to a compatible [`StateSpace`][qten.symbolics.state_space.StateSpace].
    - [`align_all(dims)`][qten.linalg.tensors.Tensor.align_all]: align all axes to a target dims tuple.
    - [`expand_to_union(union_dims)`][qten.linalg.tensors.Tensor.expand_to_union]: materialize data expansion for
      [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axes.
    - [`dim_types()`][qten.linalg.tensors.Tensor.dim_types]: return the runtime types of all dims.
    - [`rank()`][qten.linalg.tensors.Tensor.rank]: return the tensor rank.

    Reductions and element queries
    ------------------------------
    - [`all(dim=None, keepdim=False)`][qten.linalg.tensors.Tensor.all]: logical AND reduction.
    - [`mean(dim=None)`][qten.linalg.tensors.Tensor.mean]: arithmetic mean reduction.
    - [`argmax(dim)`][qten.linalg.tensors.Tensor.argmax], [`argmin(dim)`][qten.linalg.tensors.Tensor.argmin]: index reductions.
    - [`item()`][qten.linalg.tensors.Tensor.item]: extract the Python scalar from a rank-0 tensor.
    - [`equal(other)`][qten.linalg.tensors.Tensor.equal]: exact equality after metadata alignment, returning
      Python `bool`.
    - [`allclose(other, ...)`][qten.linalg.tensors.Tensor.allclose]: approximate equality after metadata alignment,
      returning Python `bool`.

    Boolean-mask helpers
    --------------------
    - [`where(input, other)`][qten.linalg.tensors.Tensor.where]: use this tensor as a bool mask and select between
      two tensors.
    - [`where()`][qten.linalg.tensors.Tensor.where]: return index tensors for `True` entries.
    - [`nonzero(as_tuple=True)`][qten.linalg.tensors.Tensor.nonzero]: return index tensors for nonzero / `True`
      entries.

    Indexing
    --------
    `__getitem__` supports:
    - Python integers, slices, `None`, and `...`,
    - [`StateSpace`][qten.symbolics.state_space.StateSpace] / [`Convertible`][qten.abstracts.Convertible] axis selection and reindexing,
    - [`Tensor`][qten.linalg.tensors.Tensor] advanced indices.

    Tensor advanced indexing is metadata-aware and can align tensor index dims
    before dispatching to torch indexing. Boolean tensor indices are not
    supported in `__getitem__`; use comparison masks together with `where` or
    `nonzero` instead.

    Autograd and devices
    --------------------
    - [`attach()`][qten.linalg.tensors.Tensor.attach]: return a leaf tensor with `requires_grad=True`.
    - [`detach()`][qten.linalg.tensors.Tensor.detach]: detach from autograd without cloning storage.
    - [`clone()`][qten.linalg.tensors.Tensor.clone]: deep-copy tensor data.
    - [`grad`][qten.linalg.tensors.Tensor.grad]: wrapped gradient tensor, if available.
    - [`requires_grad`][qten.linalg.tensors.Tensor.requires_grad]: autograd flag from underlying data.
    - [`backward(...)`][qten.linalg.tensors.Tensor.backward]: autograd backward pass.
    - `device`: logical QTen device view of the underlying torch device.
    - [`to_device(device)`][qten.linalg.tensors.Tensor.to_device]: move data to another logical device.

    Factory and module-level companion functions
    --------------------------------------------
    The module also exposes helpers that create or operate on [`Tensor`][qten.linalg.tensors.Tensor]:
    [`matmul`][qten.linalg.tensors.matmul], `permute`, `transpose`, `conj`, `unsqueeze`, `squeeze`,
    `align`, `align_all`, `all`, `mean`, `norm`, `argmax`, `argmin`, `astype`,
    [`one_hot`][qten.linalg.tensors.one_hot], `equal`, `allclose`, `expand_to_union`, [`union_dims`][qten.linalg.tensors.union_dims],
    [`mapping_matrix`][qten.linalg.tensors.mapping_matrix], [`eye`][qten.linalg.tensors.eye], [`zeros`][qten.linalg.tensors.zeros], [`ones`][qten.linalg.tensors.ones], [`kernel_tensor`][qten.linalg.tensors.kernel_tensor], [`cat`][qten.linalg.tensors.cat],
    `replace_dim`, `update_dim`, `factorize_dim`, `product_dims`, [`promote_rank`][qten.linalg.tensors.promote_rank],
    `where`, and `nonzero`.
    """

    data: T
    """
    Underlying `torch.Tensor` storing the numeric values. Its shape must match
    the dimensions implied by `dims`.
    """
    dims: Tuple[StateSpace, ...]
    """
    Symbolic dimension metadata, with one
    [`StateSpace`][qten.symbolics.state_space.StateSpace] per axis of `data`.
    These dimensions preserve axis identity and ordering beyond raw shape.
    """

    def __post_init__(self) -> None:
        """
        Finalize construction after dataclass initialization.

        If an [`at_device`][qten.linalg.tensors.at_device] context is active for the current thread, this method
        moves `data` onto that forced device before the frozen dataclass
        instance escapes to user code. When no device-forcing context is
        active, construction is left unchanged.

        Notes
        -----
        Because [`Tensor`][qten.linalg.tensors.Tensor] is a frozen dataclass, the post-init device update uses
        `object.__setattr__` to replace `data` in-place during initialization.
        """
        forced_device = _forced_tensor_device()
        if forced_device is None:
            return

        target = forced_device.torch_device()
        if self.data.device != target:
            object.__setattr__(self, "data", cast(T, self.data.to(target)))

    @staticmethod
    def scalar(number: Number, *, device: Optional[Device] = None) -> "Tensor":
        """
        Create a 0-dimensional [`Tensor`][qten.linalg.tensors.Tensor] from a scalar number.

        Parameters
        ----------
        number : Number
            The scalar value to convert into a tensor.
        device : Optional[Device], optional
            Device to place the scalar on, by default None (CPU).

        Returns
        -------
        Tensor
            A 0-dimensional tensor containing the given number.
        """
        precision = get_precision_config()
        dtype = (
            precision.torch_complex
            if isinstance(number, complex)
            else precision.torch_float
        )
        torch_device = device.torch_device() if device is not None else None
        data = torch.tensor(number, dtype=dtype, device=torch_device)
        return Tensor(data=data, dims=())

    def astype(self, dtype: torch.dtype) -> Self:
        """
        Return a new tensor with the same dims and converted data dtype.

        This method delegates to [`astype(tensor, dtype)`][qten.linalg.tensors.astype],
        which applies `tensor.data.to(dtype=...)` to the underlying torch data.

        See Also
        --------
        [`astype(tensor, dtype)`][qten.linalg.tensors.astype]
            Functional form with the same behavior.

        Parameters
        ----------
        dtype : torch.dtype
            Target PyTorch dtype.

            This must be a [`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype)
            object such as `torch.float32`, `torch.float64`,
            `torch.complex64`, `torch.complex128`, `torch.int64`, or
            `torch.bool`. Any dtype accepted by
            `torch.Tensor.to(dtype=...)` is valid here.

        Returns
        -------
        Self
            A new tensor of the same wrapper type whose data has dtype `dtype`.
        """
        return astype(self, dtype)

    def equal(self, other: "Tensor") -> bool:
        """
        Compare this tensor to another tensor for exact equality.

        See Also
        --------
        [`equal(a, b)`][qten.linalg.tensors.equal]
            Functional form with the full comparison semantics.

        Behavior
        --------
        - Attempts to align `other.dims` to `self.dims` using `align_all`.
        - If dimension alignment is not possible, returns `False`.
        - If alignment succeeds, compares aligned data via `torch.equal`.

        Parameters
        ----------
        other : Tensor
            The tensor to compare against this tensor.

        Returns
        -------
        bool
            True if tensors are exactly equal after alignment; otherwise `False`.
        """
        return equal(self, other)

    def allclose(
        self,
        other: "Tensor",
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> bool:
        """
        Compare this tensor to another tensor for approximate equality.

        See Also
        --------
        [`allclose(a, b, ...)`][qten.linalg.tensors.allclose]
            Functional form with the full comparison semantics.

        Behavior
        --------
        - Attempts to align `other.dims` to `self.dims` using `align_all`.
        - If dimension alignment is not possible, returns `False`.
        - If alignment succeeds, compares aligned data via `torch.allclose`.

        Parameters
        ----------
        other : Tensor
            The tensor to compare against this tensor.
        rtol : float, optional
            Relative tolerance used by `torch.allclose`.
        atol : float, optional
            Absolute tolerance used by `torch.allclose`.
        equal_nan : bool, optional
            Whether `NaN` values are considered equal.

        Returns
        -------
        bool
            True if tensors are close after alignment; otherwise `False`.
        """
        return allclose(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def isclose(
        self,
        other: Union["Tensor", Number],
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> Self:
        """
        Perform element-wise approximate equality comparison.

        This is the mask-producing counterpart to `allclose`: it returns a bool
        [`Tensor`][qten.linalg.tensors.Tensor] instead of a Python bool.

        Supported forms
        ---------------
        [`tensor.isclose(other)`][qten.linalg.tensors.Tensor.isclose]
            Method form.

        [`isclose(tensor, other)`][qten.linalg.tensors.isclose]
            Functional equivalent.

        Parameter forms
        ---------------
        `other : Tensor`
            Compared after symbolic alignment and broadcast handling.

        `other : Number`
            Promoted through `Tensor.scalar(other)` before applying the same
            comparison rules.

        Parameters
        ----------
        other : Tensor | Number
            Comparison target. If `other` is a
            [`Tensor`][qten.linalg.tensors.Tensor], it is aligned and
            broadcast against `self`. If `other` is a scalar number, it is
            promoted through `Tensor.scalar(other)` before comparison.
        rtol : float, optional
            Relative tolerance passed to `torch.isclose`.
        atol : float, optional
            Absolute tolerance passed to `torch.isclose`.
        equal_nan : bool, optional
            Whether `NaN` values are considered equal.

        Returns
        -------
        Self
            Boolean tensor mask with the merged symbolic output dims.

        See Also
        --------
        [`isclose(a, b, ...)`][qten.linalg.tensors.isclose]
            Functional form with the full behavior description.
        """
        return isclose(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def conj(self) -> Self:
        """
        Compute the complex conjugate of the given tensor.

        See Also
        --------
        [`conj(tensor)`][qten.linalg.tensors.conj]
            Functional form with the same behavior.

        Returns
        -------
        Self
            The complex conjugate of the tensor.
        """
        return conj(self)

    def real(self) -> Self:
        """
        Return the real part of the tensor, preserving dims.

        For complex-valued tensors this drops the imaginary component. For
        real-valued tensors this returns a tensor with the same values.

        See Also
        --------
        [`real(tensor)`][qten.linalg.tensors.real]
            Functional form with the same behavior.
        """
        return real(self)

    def imag(self) -> Self:
        """
        Return the imaginary part of the tensor, preserving dims.

        For complex-valued tensors this returns the imaginary component. For
        real-valued tensors this returns zeros with the corresponding real dtype.

        See Also
        --------
        [`imag(tensor)`][qten.linalg.tensors.imag]
            Functional form with the same behavior.
        """
        return imag(self)

    def abs(self) -> Self:
        """
        Return the element-wise absolute value / magnitude of the tensor.

        For complex-valued tensors this returns the magnitude.

        See Also
        --------
        [`abs(tensor)`][qten.linalg.tensors.abs]
            Functional form with the same behavior.
        """
        return abs(self)

    def permute(self, *order: Union[int, Sequence[int]]) -> Self:
        """
        Permute the dimensions according to the specified order.

        See Also
        --------
        [`permute(tensor, *order)`][qten.linalg.tensors.permute]
            Functional form with the full permutation semantics.

        Parameters
        ----------
        order : Union[int, Sequence[int]]
            The desired order of dimensions.

        Returns
        -------
        Self
            The permuted tensor.
        """
        return permute(self, *order)

    def transpose(self, dim0: int, dim1: int) -> Self:
        """
        Transpose the specified dimensions.

        See Also
        --------
        [`transpose(tensor, dim0, dim1)`][qten.linalg.tensors.transpose]
            Functional form with the full transpose semantics.

        Parameters
        ----------
        dim0 : int
            The first dimension to transpose.
        dim1 : int
            The second dimension to transpose.

        Returns
        -------
        Self
            The transposed tensor.
        """
        return transpose(self, dim0, dim1)

    def h(self, dim0: int, dim1: int) -> Self:
        """
        Hermitian transpose (conjugate transpose) of the specified dimensions.

        See Also
        --------
        [`conj(tensor)`][qten.linalg.tensors.conj]
        [`transpose(tensor, dim0, dim1)`][qten.linalg.tensors.transpose]

        Parameters
        ----------
        dim0 : int
            The first dimension to transpose.
        dim1 : int
            The second dimension to transpose.

        Returns
        -------
        Self
            The Hermitian transposed tensor.
        """
        return self.conj().transpose(dim0, dim1)

    def align(self, dim: int, target_dim: StateSpace) -> Self:
        """
        Align the specified dimension to the target StateSpace.

        Behavior
        --------
        Delegates to [`align`][qten.linalg.tensors.align]. This may:

        - leave the tensor unchanged if the axis already matches,
        - expand a broadcast axis to `target_dim`,
        - permute the axis if the same rays appear in a different order,
        - raise if the axis is not symbolically compatible with `target_dim`.

        Use cases
        ---------
        - reorder one axis to match another tensor before contraction,
        - materialize a broadcast axis as a concrete symbolic space.

        Parameters
        ----------
        dim : int
            The dimension index to align.
        target_dim : StateSpace
            The target StateSpace to align to.

        Returns
        -------
        Self
            The aligned tensor.

        See Also
        --------
        [`align(tensor, dim, target_dim)`][qten.linalg.tensors.align]
            Functional form with the full behavior description.
        """
        return align(self, dim, target_dim)

    def align_all(self, dims: Tuple[StateSpace, ...]) -> Self:
        """
        Align all tensor dimensions to `dims`.

        Behavior
        --------
        Delegates to [`align_all`][qten.linalg.tensors.align_all], aligning the
        tensor axis-by-axis to the requested symbolic layout.

        Use cases
        ---------
        - normalize one tensor to the metadata layout of another before
          comparison or arithmetic,
        - prepare a tensor for an API that expects a specific symbolic axis
          ordering.

        Parameters
        ----------
        dims : Tuple[StateSpace, ...]
            The target dimensions to align to.

        Returns
        -------
        Self
            The aligned tensor.

        Raises
        ------
        ValueError
            If the provided `dims` are not compatible with the tensor's current dimensions.

        See Also
        --------
        [`align_all(tensor, dims)`][qten.linalg.tensors.align_all]
            Functional form with the full behavior description.
        """
        return align_all(self, dims)

    def all(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> Self:
        """
        Return whether all elements evaluate to `True`.

        Parameters
        ----------
        dim : Optional[Union[int, Tuple[int, ...]]], optional
            Reduction axis (or axes). If `None`, reduce over all dimensions.
        keepdim : bool, optional
            If `True`, retains the reduced axis as [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace].

        Returns
        -------
        Self
            Boolean tensor after reduction.

        See Also
        --------
        [`all(tensor, dim=None, keepdim=False)`][qten.linalg.tensors.all]
            Functional form with the full reduction semantics.
        """
        return all(self, dim=dim, keepdim=keepdim)

    def where(
        self,
        input: Optional["Tensor"] = None,
        other: Optional["Tensor"] = None,
        index_type: Type[Any] = None,
    ) -> Union[
        "Tensor",
        Tuple["Tensor", ...],
        Tuple[Tuple[int, ...], ...],
        StateSpace,
    ]:
        """
        Apply `where` using this tensor as the boolean condition mask.

        Supported call forms
        --------------------
        - `condition.where(input, other)`:
          elementwise selection between `input` and `other`.
        - `condition.where()`:
          returns index tensors of `True` entries.
        - `condition.where(index_type=...)`:
          returns the requested index representation for `True` entries.

        Semantics
        ---------
        This method requires `condition.data.dtype == torch.bool`.

        For `condition.where(input, other)`:
        - `condition`, `input`, and `other` are promoted to a common rank by
          prepending leading [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axes as needed.
        - metadata is merged with
          [`union_dims(condition.dims, input.dims, other.dims, allow_merge=False)`][qten.linalg.tensors.union_dims].
        - all three operands are aligned to those merged dims and broadcast-expanded.
        - selection is then applied elementwise with `torch.where`.

        For `condition.where()` and `condition.where(index_type=...)`:
        - behavior follows `torch.where(condition)` /
          `torch.nonzero(condition, as_tuple=True)`.
        - for a rank-`R` mask, the Tensor form returns `R` one-dimensional
          index tensors, one per axis.
        - each returned Tensor index uses `IndexSpace.linear(nnz)`, where
          `nnz` is the number of `True` entries.

        Parameters
        ----------
        input : Optional[Tensor], optional
            Tensor selected where `condition` is `True` in the selection form
            `condition.where(input, other)`.
        other : Optional[Tensor], optional
            Tensor selected where `condition` is `False` in the selection form
            `condition.where(input, other)`.
        index_type : Type[Any], optional
            Only used for the condition-only form `condition.where(...)`.

            Supported values are
            [`Tensor`][qten.linalg.tensors.Tensor] to return one rank-1
            integer tensor per axis, `tuple` / `Tuple` to return Python
            coordinate tuples, and
            [`StateSpace`][qten.symbolics.state_space.StateSpace] to return the
            selected symbolic subspace for rank-1 conditions.

        Returns
        -------
        Union[Tensor, Tuple[Tensor, ...], Tuple[Tuple[int, ...], ...], StateSpace]
            Return value depends on call form. `condition.where(input, other)`
            returns a single [`Tensor`][qten.linalg.tensors.Tensor] with
            `dims == union_dims(condition.dims, input.dims, other.dims,
            allow_merge=False)`, after all operands are aligned and broadcast.
            `condition.where()` and `condition.where(index_type=Tensor)` return
            `Tuple[Tensor, ...]` with one rank-1 index tensor per condition
            axis, each with dims `(IndexSpace.linear(nnz),)`. Using
            `index_type=tuple` or `Tuple` returns one Python coordinate tuple
            per `True` entry. Using `index_type=StateSpace` returns the
            selected subspace for rank-1 conditions only.

        Raises
        ------
        TypeError
            If `condition.data` is not boolean, if only one of `input`/`other`
            is provided, or if `index_type` is passed together with
            input/`other`.
        ValueError
            If operands cannot be aligned/broadcast to shared union dims, or if
            index_type=StateSpace is requested for a non-rank-1 mask.

        See Also
        --------
        [`where(...)`][qten.linalg.tensors.where]
            Functional form with the full supported-form documentation.
        """
        if index_type is None:
            index_type = Tensor
        if input is None and other is None:
            return where(self, index_type=index_type)
        if input is None or other is None:
            raise TypeError(
                "Tensor.where supports either where() or where(input, other)"
            )
        if index_type is not Tensor:
            raise TypeError("index_type is only supported for where()")
        return where(self, input, other)

    def nonzero(
        self, as_tuple: bool = True, index_type: Type[Any] = None
    ) -> Union[
        Tuple["Tensor", ...],
        Tuple[Tuple[int, ...], ...],
        StateSpace,
    ]:
        """
        Return indices of non-zero / `True` entries for this tensor.

        Currently supports only `as_tuple=True`, matching
        `torch.nonzero(..., as_tuple=True)`.

        Parameters
        ----------
        as_tuple : bool, optional
            Must be `True`.
        index_type : Type[Any], optional
            Requested index representation. Supported values are
            [`Tensor`][qten.linalg.tensors.Tensor] to return one rank-1
            integer tensor per axis, `tuple` / `Tuple` to return Python
            coordinate tuples, and
            [`StateSpace`][qten.symbolics.state_space.StateSpace] to return the
            selected symbolic subspace for rank-1 conditions.

        Returns
        -------
        Union[Tuple[Tensor, ...], Tuple[Tuple[int, ...], ...], StateSpace]
            Index representation selected by `index_type`.

        Raises
        ------
        NotImplementedError
            If `as_tuple` is `False`.

        See Also
        --------
        [`nonzero(condition, ...)`][qten.linalg.tensors.nonzero]
            Functional form with the full supported-form documentation.
        """
        if index_type is None:
            index_type = Tensor
        return nonzero(self, as_tuple=as_tuple, index_type=index_type)

    def unsqueeze(self, dim: int) -> Self:
        """
        Unsqueeze the specified dimension.

        See Also
        --------
        [`unsqueeze(tensor, dim)`][qten.linalg.tensors.unsqueeze]
            Functional form with the full behavior description.

        Parameters
        ----------
        dim : int
            The dimension to unsqueeze.

        Returns
        -------
        Self
            The unsqueezed tensor.
        """
        return unsqueeze(self, dim)

    def squeeze(self, dim: int) -> Self:
        """
        Squeeze the specified dimension.

        See Also
        --------
        [`squeeze(tensor, dim)`][qten.linalg.tensors.squeeze]
            Functional form with the full behavior description.

        Parameters
        ----------
        dim : int
            The dimension to squeeze.

        Returns
        -------
        Self
            The squeezed tensor.
        """
        return squeeze(self, dim)

    def index_add(
        self,
        dim: int,
        index: "Tensor[Any]",
        source: "Tensor",
        alpha: Union[int, float, complex] = 1,
    ) -> Self:
        """
        Return a copy of this tensor with indexed additions applied along `dim`.

        Parameters
        ----------
        dim : int
            Dimension along which to accumulate updates.
        index : Tensor
            Rank-1 integer tensor of destination indices. Its single
            [`StateSpace`][qten.symbolics.state_space.StateSpace] defines the symbolic order of the updates.
        source : Tensor
            Tensor of update values. It must have the same rank as `self`.
        alpha : Union[int, float, complex], optional
            Scalar multiplier applied to `source` before accumulation.

        Alignment rules
        ---------------
        - `index` must be a rank-1 integer tensor.
        - `source` must have the same rank as `self`.
        - On non-indexed axes, `source` is aligned to `self`.
        - On the indexed axis, `source` is aligned to `index.dims[0]`, so the
          source rows or blocks follow the same symbolic order as the index
          list.

        Returns
        -------
        Self
            A new tensor with the same dimensions as `self`.

        See Also
        --------
        [`index_add(tensor, dim, index, source, alpha=1)`][qten.linalg.tensors.index_add]
            Functional form with the full behavior description.
        """
        return index_add(self, dim=dim, index=index, source=source, alpha=alpha)

    def index_add_(
        self,
        dim: int,
        index: "Tensor[Any]",
        source: "Tensor",
        alpha: Union[int, float, complex] = 1,
    ) -> Self:
        """
        In-place variant of `index_add`.

        Parameters
        ----------
        dim : int
            Dimension along which to accumulate updates.
        index : Tensor
            Rank-1 integer tensor of destination indices.
        source : Tensor
            Tensor of update values. It must have the same rank as `self`.
        alpha : Union[int, float, complex], optional
            Scalar multiplier applied to `source` before accumulation.

        Returns
        -------
        Self
            This tensor after in-place accumulation.

        See Also
        --------
        [`index_add_(tensor, dim, index, source, alpha=1)`][qten.linalg.tensors.index_add_]
            Functional in-place form with the full behavior description.
        """
        return index_add_(self, dim=dim, index=index, source=source, alpha=alpha)

    def rank(self) -> int:
        """
        Get the rank (number of dimensions) of the tensor.

        See Also
        --------
        [`rank(tensor)`][qten.linalg.tensors.rank]
            Functional form.

        Returns
        -------
        int
            The rank of the tensor.
        """
        return rank(self)

    def mean(self, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> Self:
        """
        Compute the mean over specified dimension(s).

        See Also
        --------
        [`mean(tensor, dim=None)`][qten.linalg.tensors.mean]
            Functional form with the full reduction semantics.

        Parameters
        ----------
        dim : Optional[Union[int, Tuple[int, ...]]], optional
            Reduction axis (or axes). If `None`, reduce over all dimensions.

        Returns
        -------
        Self
            A new tensor with the specified dimensions reduced.
        """
        return mean(self, dim)

    def norm(
        self,
        ord: Optional[Union[int, float, str]] = None,
        dim: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Self:
        """
        Compute a vector or matrix norm over the specified dimension(s).

        This method delegates to [`norm(tensor, ord=None, dim=None)`][qten.linalg.tensors.norm],
        which in turn forwards the numeric computation to
        [`torch.linalg.norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.norm.html).

        Supported `ord` values
        ----------------------
        The accepted values depend on whether `dim` selects a vector norm or a
        matrix norm:

        - `dim` is an `int`: vector norm.
          Supported `ord` values include `None`, `0`, any finite `int` or
          `float`, `float("inf")`, and `-float("inf")`.
        - `dim` is a 2-tuple: matrix norm.
          Supported `ord` values include `None`, `"fro"`, `"nuc"`, `1`, `-1`,
          `2`, `-2`, `float("inf")`, and `-float("inf")`.
        - `dim is None`:
          PyTorch decides whether to interpret the input as a flattened vector
          or as a 1D/2D tensor norm depending on `ord`. See the linked
          `torch.linalg.norm` reference for the exact dispatch rules.

        Parameters
        ----------
        ord : Optional[Union[int, float, str]], optional
            Norm order forwarded to `torch.linalg.norm`.

            Common choices are `None` for the default norm chosen by PyTorch,
            `2` for the Euclidean vector norm or spectral matrix norm, `1` for
            an L1 vector norm or induced 1 matrix norm, `float("inf")` for
            max-based norms, `"fro"` for the Frobenius matrix norm, and
            `"nuc"` for the nuclear matrix norm.
        dim : Optional[Union[int, Tuple[int, int]]], optional
            Reduction axis or axes. Use an `int` to compute a vector norm along
            one axis, a `Tuple[int, int]` to compute a matrix norm over two
            axes, or `None` to let PyTorch use its default
            `torch.linalg.norm` semantics.

        Returns
        -------
        Self
            A new tensor with the specified dimensions reduced.

        See Also
        --------
        [`norm(tensor, ord=None, dim=None)`][qten.linalg.tensors.norm]
            Functional form with the full reduction semantics.
        """
        return norm(self, ord=ord, dim=dim)

    def argmax(self, dim: int) -> Self:
        """
        Compute the indices of the maximum values over a specified dimension.

        See Also
        --------
        [`argmax(tensor, dim)`][qten.linalg.tensors.argmax]
            Functional form with the full reduction semantics.

        Parameters
        ----------
        dim : int
            The dimension to reduce.

        Returns
        -------
        Self
            A new tensor with the specified dimension reduced.
        """
        return argmax(self, dim)

    def argmin(self, dim: int) -> Self:
        """
        Compute the indices of the minimum values over a specified dimension.

        See Also
        --------
        [`argmin(tensor, dim)`][qten.linalg.tensors.argmin]
            Functional form with the full reduction semantics.

        Parameters
        ----------
        dim : int
            The dimension to reduce.

        Returns
        -------
        Self
            A new tensor with the specified dimension reduced.
        """
        return argmin(self, dim)

    def expand_to_union(self, union_dims: list[StateSpace]) -> Self:
        """
        Expand the tensor to the union of the specified dimensions.

        See Also
        --------
        [`expand_to_union(tensor, union_dims)`][qten.linalg.tensors.expand_to_union]
            Functional form with the full broadcast-expansion semantics.

        Parameters
        ----------
        union_dims : list[StateSpace]
            The dimensions to expand to the union of.

        Returns
        -------
        Self
            The expanded tensor.
        """
        return expand_to_union(self, union_dims)

    def item(self) -> Union[Number, int, float]:
        """
        Return the value of a 0-dimensional tensor as a standard Python number.

        Returns
        -------
        number
            The value of the tensor.

        Raises
        ------
        ValueError
            If the tensor is not 0-dimensional.
        """
        if self.rank() != 0:
            raise ValueError(
                f"Tensor.item() only works for rank-0 tensors, got rank {self.rank()}"
            )
        return self.data.item()

    @override
    @property
    def device(self) -> Device:
        """
        Return the logical device associated with the tensor data.

        Raises
        ------
        ValueError
            If the underlying torch device type is not one of the supported
            QTen device mappings.
        """
        torch_device = self.data.device
        if torch_device.type == "cpu":
            return Device("cpu")
        if torch_device.type == "cuda":
            return Device("gpu", torch_device.index)
        if torch_device.type == "mps":
            return Device("gpu")
        raise ValueError(f"Unsupported tensor device type: {torch_device.type}")

    @override
    def to_device(self, device: Device) -> Self:
        """
        Copy the tensor data to the specified logical device and return a new tensor.
        """
        target = device.torch_device()
        if self.data.device == target:
            return self
        return replace(self, data=cast(T, self.data.to(target)))

    @property
    def requires_grad(self) -> bool:
        """
        Check if the tensor data requires gradient tracking.

        Returns
        -------
        bool
            True if the tensor data requires gradient tracking, False otherwise.
        """
        return self.data.requires_grad

    @property
    def grad(self) -> Optional[Self]:
        """
        Return the accumulated gradient wrapped as a QTen tensor.

        Returns
        -------
        Optional[Self]
            The current gradient with the same dims, or `None` if no gradient
            has been accumulated.
        """
        grad = self.data.grad
        if grad is None:
            return None
        return replace(self, data=cast(T, grad))

    def backward(
        self,
        gradient: Optional[Self] = None,
        retain_graph: Optional[bool] = None,
        create_graph: bool = False,
        inputs: Optional[Sequence[Self]] = None,
    ) -> None:
        """
        Run autograd backward from this tensor.

        Parameters
        ----------
        gradient : Optional[Self], optional
            Upstream gradient for non-scalar tensors.
        retain_graph : Optional[bool], optional
            Whether to retain the autograd graph after backward.
        create_graph : bool, optional
            Whether to construct the derivative graph.
        inputs : Optional[Sequence[Self]], optional
            Restrict gradient accumulation to the specified leaf inputs.
        """
        grad_data = gradient.align_all(self.dims).data if gradient is not None else None
        input_data = [tensor.data for tensor in inputs] if inputs is not None else None
        self.data.backward(
            gradient=grad_data,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=input_data,
        )

    def attach(self) -> Self:
        """
        Enable gradient tracking for the tensor data and return the attached [`Tensor`][qten.linalg.tensors.Tensor] instance.

        Behavior
        --------
        - If [`requires_grad`][qten.linalg.tensors.Tensor.requires_grad] is already `True`, this returns `self` unchanged.
        - Otherwise, this detaches the underlying data from any existing autograd graph,
          clones it to ensure a fresh leaf tensor, and sets [`requires_grad`][qten.linalg.tensors.Tensor.requires_grad] to `True`.
        - The returned tensor preserves the original `dims`, device, and dtype.

        Returns
        -------
        Self
            The new tensor of the same wrapper type with gradient tracking enabled.
        """
        if self.data.requires_grad:
            return self
        return replace(
            self,
            data=cast(T, self.data.detach().clone().requires_grad_(True)),
        )

    def detach(self) -> Self:
        """
        Disable gradient tracking for the tensor data and create a new [`Tensor`][qten.linalg.tensors.Tensor] instance.

        Behavior
        --------
        - Always returns a new [`Tensor`][qten.linalg.tensors.Tensor] whose data is a detached view of the
          original tensor (no clone), so it shares storage with the original.
        - The returned tensor preserves the original `dims`, device, and dtype.

        Returns
        -------
        Self
            The new tensor of the same wrapper type with gradient tracking disabled.
        """
        return replace(self, data=cast(T, self.data.detach()))

    def clone(self) -> Self:
        """
        Create a deep copy of the tensor.

        Returns
        -------
        Self
            The cloned tensor.
        """
        return replace(self, data=cast(T, self.data.clone()))

    def replace_dim(self, dim: int, new_dim: StateSpace) -> Self:
        """
        Replace the StateSpace at the specified dimension with a new StateSpace.

        See Also
        --------
        [`replace_dim(tensor, dim, new_dim)`][qten.linalg.tensors.replace_dim]
            Functional form with the full metadata replacement semantics.

        Parameters
        ----------
        dim : int
            The index of the dimension to replace.
        new_dim : StateSpace
            The new StateSpace to assign to the dimension.

        Returns
        -------
        Self
            A new tensor of the same wrapper type with the updated dimension.
        """
        return replace_dim(self, dim, new_dim)

    def update_dim(self, dim: int, func: Callable[[StateSpace], StateSpace]) -> Self:
        """
        Transform the StateSpace at the specified dimension with a callback.

        The callback receives the current dimension metadata and must return
        the replacement [`StateSpace`][qten.symbolics.state_space.StateSpace].
        Size validation and index normalization follow the same rules as
        [`replace_dim`][qten.linalg.tensors.Tensor.replace_dim].

        See Also
        --------
        [`update_dim(tensor, dim, func)`][qten.linalg.tensors.update_dim]
            Functional form with the full metadata update semantics.

        Parameters
        ----------
        dim : int
            The index of the dimension to update.
        func : Callable[[StateSpace], StateSpace]
            Callback that maps the current StateSpace to a replacement.

        Returns
        -------
        Self
            A new tensor of the same wrapper type with the updated dimension.
        """
        return update_dim(self, dim, func)

    def __getitem__(self, key):
        """
        Index tensor data with [`TensorIndexing`][qten.linalg.tensors.TensorIndexing] and return a new [`Tensor`][qten.linalg.tensors.Tensor].

        Parameters
        ----------
        key : Any
            Index expression accepted by QTen tensor indexing. Supported tokens
            include Python integers, slices, `None`, `...`, [`StateSpace`][qten.symbolics.state_space.StateSpace]
            objects, [`Convertible`][qten.abstracts.Convertible] objects that can be converted to
            [`StateSpace`][qten.symbolics.state_space.StateSpace], and QTen [`Tensor`][qten.linalg.tensors.Tensor] index tensors.

        Returns
        -------
        Tensor
            A new tensor with indexed data and output metadata compiled from the
            provided key.

        Index normalization
        -------------------
        - A non-tuple key is treated as a 1-tuple.
        - At most one ellipsis (`...`) is allowed.
        - `...` expands to the required number of full slices (`:`) based on
          source-axis-consuming tokens (all non-`None` entries).
        - If fewer source-axis-consuming indices are provided than rank, missing
          trailing full slices (`:`) are appended.

        Per-token semantics
        -------------------
        - `int`: selects one element along the current source axis and removes
          that output dimension.
        - `slice`:
          - full slice `:` preserves the current [`StateSpace`][qten.symbolics.state_space.StateSpace],
          - non-full slice uses `self.dims[axis][slice]`, except
            [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axes where metadata follows sliced size:
            size `1` keeps [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace], size `0` becomes
            `IndexSpace.linear(0)`.
        - `None`: inserts a new output axis with [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] and does not
          consume a source axis.
        - [`StateSpace`][qten.symbolics.state_space.StateSpace] / [`Convertible`][qten.abstracts.Convertible]:
          - indexing a [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axis with any non-[`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace]
            [`StateSpace`][qten.symbolics.state_space.StateSpace] is rejected (`IndexError`),
          - if equal to current axis space: behaves like full slice,
          - if same span: uses permutation indexing and output dim is the index
            [`StateSpace`][qten.symbolics.state_space.StateSpace],
          - if contained subspace: uses embedding indexing and output dim is the
            subspace,
          - otherwise raises `IndexError`.
        - [`Tensor`][qten.linalg.tensors.Tensor] index:
          - `bool` dtype is not supported (`NotImplementedError`),
          - lower-rank tensor indices are left-padded with singleton/broadcast
            axes to match the maximum tensor-index rank before metadata
            union/alignment,
          - index metadata is aligned to the union of all tensor-index dims.

        Mode rules
        ----------
        - Mixing [`Tensor`][qten.linalg.tensors.Tensor] indices with [`StateSpace`][qten.symbolics.state_space.StateSpace]/[`Convertible`][qten.abstracts.Convertible] indices is
          rejected (`ValueError`).
        - If at least one [`Tensor`][qten.linalg.tensors.Tensor] index is present, data indexing is executed
          in one torch advanced-indexing call.
        - Otherwise indexing is applied step-by-step per axis (including tuple
          index_select steps), so per-axis [`StateSpace`][qten.symbolics.state_space.StateSpace] tuple indices are not
          jointly broadcast by torch.

        Output dim ordering for tensor advanced indexing
        -----------------------------------------------
        Let `tensor_union_dims` be the broadcast/union dims of all tensor index
        tensors.
        - If tensor index tokens form one contiguous block in the normalized key,
          `tensor_union_dims` is inserted at that block position.
        - If tensor index tokens are separated by non-tensor tokens,
          `tensor_union_dims` is moved to the front of output dims.

        Raises
        ------
        IndexError
            If the key contains too many indices, incompatible [`StateSpace`][qten.symbolics.state_space.StateSpace]
            metadata, or more than one ellipsis.
        ValueError
            If tensor indices are mixed with [`StateSpace`][qten.symbolics.state_space.StateSpace] / [`Convertible`][qten.abstracts.Convertible]
            indices in one indexing operation.
        NotImplementedError
            If boolean tensor indices are used in a mode that QTen does not
            support through `__getitem__`.

        Notes
        -----
        This method delegates key analysis to [`TensorIndexing`][qten.linalg.tensors.TensorIndexing]. When advanced
        tensor indexing is present, indexing is executed in one torch call. In
        all other cases, indexing is applied step-by-step so QTen can preserve
        per-axis [`StateSpace`][qten.symbolics.state_space.StateSpace] semantics.
        """
        if not isinstance(key, tuple):
            key = (key,)
        compiled = TensorIndexing(self.dims, key).compile()
        if compiled.has_tensor_index:
            indices = tuple(
                idx.to(self.data.device) if isinstance(idx, torch.Tensor) else idx
                for idx in compiled.indices
            )
            data = self.data[indices]
            return Tensor(data=data, dims=compiled.dims)

        data = self.data
        axis = 0
        for step in compiled.indices_steps:
            if step is None:
                data = data.unsqueeze(axis)
                axis += 1
                continue
            if isinstance(step, tuple):
                index_data = torch.tensor(step, dtype=torch.long, device=data.device)
                data = data.index_select(axis, index_data)
                axis += 1
                continue
            key_step = (slice(None),) * axis + (step,)
            data = data[key_step]
            if not isinstance(step, int):
                axis += 1
        return Tensor(data=data, dims=compiled.dims)

    def factorize_dim(self, dim: int, rule: StateSpaceFactorization) -> Self:
        """
        Factorize one [`StateSpace`][qten.symbolics.state_space.StateSpace]-like dimension into multiple subspaces.

        For a tensor with shape `(A, B)`, factorizing the `0`th dimension with
        `factorized=(A1, A2)` and a compatible `align_dim` produces shape
        `(A1, A2, B)`.

        Parameters
        ----------
        dim : int
            The index of the dimension to factorize.
        rule : StateSpaceFactorization
            The factorization rule. Its `factorized` field gives the output
            factor spaces that replace the selected axis, and its `align_dim`
            field gives a compatible reordering of the original axis used
            before reshape.

        Returns
        -------
        Self
            A new tensor of the same wrapper type with the specified dimension factorized.

        See Also
        --------
        [`factorize_dim(tensor, dim, rule)`][qten.linalg.tensors.factorize_dim]
            Functional form with the full factorization semantics.
        """
        return factorize_dim(self, dim, rule)

    def product_dims(self, *indices_group: Tuple[int, ...]) -> Self:
        """
        Combine selected tensor dimensions into product dimensions.

        Each entry in `indices_group` defines one output product dimension.
        For a group `(i0, i1, ..., ik)`, the returned tensor contains a single
        axis whose size is the product of the grouped axis sizes and whose
        [`StateSpace`][qten.symbolics.state_space.StateSpace] is `self.dims[i0] @ self.dims[i1] @ ... @ self.dims[ik]`.
        Dimensions not listed in any group are preserved as-is.

        Negative indices are supported and follow Python indexing rules.
        Grouped dimensions do not need to be contiguous in the input tensor; the
        method reorders axes internally, performs one reshape, and returns the
        result in the canonical output order.

        Use cases
        ---------
        - flatten several symbolic axes into one composite axis before a
          decomposition,
        - build tensor-product state spaces explicitly in the metadata.

        Parameters
        ----------
        indices_group : Tuple[int, ...]
            One or more non-empty groups of dimension indices to combine.
            Indices must be unique across all groups (a dimension can belong to
            at most one group).

        Returns
        -------
        Self
            A new tensor of the same wrapper type where each requested group is
            replaced by one product dimension and all non-grouped dimensions
            are retained.

        See Also
        --------
        [`product_dims(tensor, *indices_group)`][qten.linalg.tensors.product_dims]
            Functional form with the full product-dimension semantics.

        Raises
        ------
        IndexError
            If any provided index is out of range for the tensor rank.
        ValueError
            If any group is empty, if a group contains duplicate indices, or if
            the same index appears in more than one group.
        """
        return product_dims(self, *indices_group)

    @override
    def kron(self, other: "Tensor") -> "Tensor":
        """
        Compute the StateSpace-aware Kronecker product with another tensor.

        This method delegates to [`kron(left, right)`][qten.linalg.tensors.kron].
        The Kronecker product is position-wise across axes: each output axis is
        built from the tensor product of the corresponding symbolic dimensions.

        Parameters
        ----------
        other : Tensor
            Right operand of the Kronecker product.

        Returns
        -------
        Tensor
            Tensor with data `torch.kron(self.data, other.data)` and
            position-wise tensor-product dimensions.

        See Also
        --------
        [`kron(left, right)`][qten.linalg.tensors.kron]
            Functional form with full semantics.
        """
        return kron(self, other)

    def dim_types(self) -> Tuple[type, ...]:
        """
        Return a tuple of the types of the dimensions in the tensor.

        Returns
        -------
        Tuple[type, ...]
            A tuple containing the types of each dimension in the tensor.
        """
        return tuple(type(dim) for dim in self.dims)

    def __repr__(self) -> str:
        """
        Return a compact developer-facing representation of the tensor.

        The representation summarizes:

        - the execution device class (`CPU` or `GPU`),
        - whether gradients are tracked,
        - and one `TypeName:size` entry per axis in `dims`.

        Returns
        -------
        str
            A one-line summary suitable for debugging and interactive use.
        """
        device_type = self.data.device.type
        device = "GPU" if device_type in {"cuda", "mps"} else "CPU"
        if self.dims:
            shape = ", ".join(f"{type(dim).__name__}:{dim.dim}" for dim in self.dims)
            shape_repr = f"({shape})"
        else:
            shape_repr = "()"
        return f"<{device} Tensor grad={self.data.requires_grad} shape={shape_repr}>"

    __str__ = __repr__  # Override str to use the same representation


def auto_promote_dtype(func):
    """Decorator to automatically promote input Tensors to a common dtype."""

    @wraps(func)
    def wrapper(left, right, *args, **kwargs):
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            common_dtype = torch.promote_types(left.data.dtype, right.data.dtype)
            if left.data.dtype != common_dtype:
                left = replace(left, data=left.data.to(common_dtype))
            if right.data.dtype != common_dtype:
                right = replace(right, data=right.data.to(common_dtype))
        return func(left, right, *args, **kwargs)

    return wrapper


def _common_device(*tensors: Tensor) -> Device:
    """Return the common execution device for a group of tensors."""
    for tensor in tensors:
        if tensor.device.name == "gpu":
            return tensor.device
    return tensors[0].device


def _promote_to_device(*tensors: Tensor) -> Tuple[Tensor, ...]:
    """Move tensors to a common execution device when needed."""
    target_device = _common_device(*tensors)
    return tuple(
        tensor if tensor.device == target_device else tensor.to_device(target_device)
        for tensor in tensors
    )


def auto_promote_device(func):
    """Decorator to automatically promote input Tensors to a common device."""

    @wraps(func)
    def wrapper(left, right, *args, **kwargs):
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            left, right = _promote_to_device(left, right)
        return func(left, right, *args, **kwargs)

    return wrapper


def auto_promote(func):
    """Decorator to automatically promote input Tensors to a common dtype/device."""

    return auto_promote_device(auto_promote_dtype(func))


def _match_dims_for_matmul(left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
    if left.rank() == 1:
        left = left.unsqueeze(0)
    if right.rank() == 1:
        right = right.unsqueeze(-1)

    if left.rank() > right.rank():
        right = promote_rank(right, left.rank())
    elif right.rank() > left.rank():
        left = promote_rank(left, right.rank())
    return left, right


def _parse_einsum_term(term: str) -> tuple[list[str], int | None]:
    labels: list[str] = []
    ellipsis_pos: int | None = None
    i = 0
    while i < len(term):
        ch = term[i]
        if ch.isspace():
            i += 1
            continue
        if ch == ".":
            if term[i : i + 3] != "...":
                raise ValueError(f"Invalid einsum term {term!r}: malformed ellipsis")
            if ellipsis_pos is not None:
                raise ValueError(
                    f"Invalid einsum term {term!r}: multiple ellipses are not allowed"
                )
            ellipsis_pos = len(labels)
            i += 3
            continue
        if ch.isalpha():
            labels.append(ch)
            i += 1
            continue
        raise ValueError(
            f"Invalid einsum term {term!r}: expected letters or ellipsis, got {ch!r}"
        )
    return labels, ellipsis_pos


def _merge_einsum_dim(
    current: StateSpace | None, candidate: StateSpace, label: str
) -> StateSpace:
    if current is None:
        return candidate
    if isinstance(current, BroadcastSpace):
        return candidate
    if isinstance(candidate, BroadcastSpace):
        return current
    if same_rays(current, candidate):
        return current
    raise ValueError(
        f"einsum label {label!r} has incompatible dimensions: "
        f"{type(current).__name__}:{current.dim} vs "
        f"{type(candidate).__name__}:{candidate.dim}"
    )


def _normalize_einsum_operands(
    equation: str, operands: tuple[Tensor, ...]
) -> tuple[str, list[list[str]], list[str], tuple[Tensor, ...]]:
    if not operands:
        raise ValueError("einsum expects at least one operand")

    if "->" in equation:
        lhs, rhs = equation.split("->", 1)
        output_term = rhs.strip()
    else:
        lhs = equation
        output_term = None

    input_terms = [term.strip() for term in lhs.split(",")]
    if len(input_terms) != len(operands):
        raise ValueError(
            f"einsum equation expects {len(input_terms)} operands, got {len(operands)}"
        )

    parsed_terms = [_parse_einsum_term(term) for term in input_terms]
    any_ellipsis = any(ellipsis_pos is not None for _, ellipsis_pos in parsed_terms)
    ellipsis_rank = 0
    for (labels, ellipsis_pos), operand, term in zip(
        parsed_terms, operands, input_terms
    ):
        named_count = len(labels)
        if ellipsis_pos is None:
            if operand.rank() != named_count:
                raise ValueError(
                    f"einsum term {term!r} expects rank {named_count}, got {operand.rank()}"
                )
            continue
        current_ellipsis_rank = operand.rank() - named_count
        if current_ellipsis_rank < 0:
            raise ValueError(
                f"einsum term {term!r} has too many labels for rank {operand.rank()}"
            )
        ellipsis_rank = max(ellipsis_rank, current_ellipsis_rank)

    ellipsis_labels = [f"@ELLIPSIS{idx}" for idx in range(ellipsis_rank)]
    expanded_terms: list[list[str]] = []
    normalized_operands: list[Tensor] = []
    normalized_input_terms = (
        [
            term if ellipsis_pos is not None else f"...{term}"
            for term, (_, ellipsis_pos) in zip(input_terms, parsed_terms)
        ]
        if any_ellipsis
        else input_terms.copy()
    )
    normalized_equation = ",".join(normalized_input_terms)
    if output_term is not None:
        normalized_equation = f"{normalized_equation}->{output_term}"

    for (labels, ellipsis_pos), operand in zip(parsed_terms, operands):
        current = operand
        if ellipsis_pos is None:
            if ellipsis_rank:
                for _ in range(ellipsis_rank):
                    current = current.unsqueeze(0)
                expanded_terms.append([*ellipsis_labels, *labels])
            else:
                expanded_terms.append(labels.copy())
            normalized_operands.append(current)
            continue

        current_ellipsis_rank = current.rank() - len(labels)
        missing = ellipsis_rank - current_ellipsis_rank
        for _ in range(missing):
            current = current.unsqueeze(ellipsis_pos)
        expanded_terms.append(
            [
                *labels[:ellipsis_pos],
                *ellipsis_labels,
                *labels[ellipsis_pos:],
            ]
        )
        normalized_operands.append(current)

    if output_term is None:
        label_counts: Dict[str, int] = OrderedDict()
        for term in expanded_terms:
            for label in term:
                if label in ellipsis_labels:
                    continue
                label_counts[label] = label_counts.get(label, 0) + 1
        output_labels = [
            *ellipsis_labels,
            *sorted(label for label, count in label_counts.items() if count == 1),
        ]
        return (
            normalized_equation,
            expanded_terms,
            output_labels,
            tuple(normalized_operands),
        )

    output_labels_raw, output_ellipsis_pos = _parse_einsum_term(output_term)
    if output_ellipsis_pos is None:
        output_labels = output_labels_raw
    else:
        output_labels = [
            *output_labels_raw[:output_ellipsis_pos],
            *ellipsis_labels,
            *output_labels_raw[output_ellipsis_pos:],
        ]

    available_labels = {label for term in expanded_terms for label in term}
    for label in output_labels:
        if label not in available_labels:
            raise ValueError(
                f"einsum output label {label!r} does not appear in any input operand"
            )

    return (
        normalized_equation,
        expanded_terms,
        output_labels,
        tuple(normalized_operands),
    )


def _promote_einsum_operands(operands: Sequence[Tensor]) -> tuple[Tensor, ...]:
    if not operands:
        return ()
    target_device = _common_device(*operands)
    common_dtype = reduce(
        torch.promote_types,
        (operand.data.dtype for operand in operands[1:]),
        operands[0].data.dtype,
    )
    promoted: list[Tensor] = []
    for operand in operands:
        current = operand
        if current.device != target_device:
            current = current.to_device(target_device)
        if current.data.dtype != common_dtype:
            current = replace(current, data=current.data.to(common_dtype))
        promoted.append(current)
    return tuple(promoted)


def einsum(equation: str, *operands: Tensor) -> Tensor:
    r"""
    Contract tensors with Einstein summation while aligning symbolic dims.

    This mirrors `torch.einsum` at the numeric level, but first reconciles
    every labeled axis through QTen's [`StateSpace`][qten.symbolics.state_space.StateSpace]
    semantics:

    - same labeled axes must be symbolically compatible,
    - [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace] may
      expand to a concrete labeled space,
    - axes with the same rays but different ordering are permuted into a
      common canonical ordering before contraction.

    Equation guide
    --------------
    The equation uses the same label syntax as `torch.einsum`.

    - Each input operand is described by one comma-separated label term.
    - Labels that appear in multiple operands identify axes that should be
      multiplied together.
    - Labels omitted from the output are summed out.
    - Labels kept in the output determine the order of the output `dims`.
    - `...` is supported and follows torch's broadcast convention for unnamed
      axes. It may appear at the start, middle, or end of a term.
    - `...ij` means "match the last two labeled axes and let `...` absorb the
      preceding axes"; `i...j` means "match the first labeled axis, the last
      labeled axis, and let `...` absorb the axes in between".
    - If any input term uses `...`, QTen normalizes any input term that omits
      it by inserting an ellipsis in the corresponding position before calling
      `torch.einsum`. This lets mixed equations such as `"...ij,ij->...ij"`
      behave like torch-style broadcasted contractions.
    - QTen only normalizes input terms. The explicit output term, if present,
      is preserved as written.

    For example:

    - `"ij,ij->ij"` means elementwise multiplication.
    - `"ij,jk->ik"` means matrix multiplication over the shared `j` axis.
    - `"abc,dbe->acde"` means multiply over the shared `b` axis and keep the
      surviving axes in the explicit output order `(a, c, d, e)`.
    - `"...ij,ij->...ij"` means "broadcast the `ij` operand across the leading
      unnamed axes of the other operand, then keep those axes in the output".
    - `"i...j,ij->i...j"` means "broadcast the `ij` operand across the unnamed
      middle axes between `i` and `j`".

    Symbolic alignment rules
    ------------------------
    Before dispatching to `torch.einsum`, QTen aligns axes label-by-label:

    - if two axes with the same label already share the same
      [`StateSpace`][qten.symbolics.state_space.StateSpace], nothing changes,
    - if they have the same rays in a different order, the operand is
      permuted to a common ordering,
    - if one side is [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace],
      it is expanded to the concrete shared space,
    - if two same-labeled concrete axes are not symbolically compatible, the
      call raises `ValueError`.

    This means einsum compatibility is stricter than raw shape-only tensor
    math: matching sizes alone are not enough when a label represents a
    symbolic axis.

    Examples
    --------
    Elementwise product with broadcast:

    ```python
    left = Tensor(data=torch.randn(2, 1), dims=(row_space, BroadcastSpace()))
    right = Tensor(data=torch.randn(1, 3), dims=(BroadcastSpace(), col_space))

    out = einsum("ij,ij->ij", left, right)
    # out.dims == (row_space, col_space)
    ```

    Matrix multiplication:

    ```python
    left = Tensor(data=torch.randn(m.dim, k.dim), dims=(m, k))
    right = Tensor(data=torch.randn(k.dim, n.dim), dims=(k, n))

    out = einsum("ij,jk->ik", left, right)
    # out.dims == (m, n)
    ```

    Mixed leading ellipsis:

    ```python
    batch = IndexSpace.linear(5)
    i = IndexSpace.linear(2)
    j = IndexSpace.linear(3)

    left = Tensor(data=torch.randn(batch.dim, i.dim, j.dim), dims=(batch, i, j))
    right = Tensor(data=torch.randn(i.dim, j.dim), dims=(i, j))

    out = einsum("...ij,ij->...ij", left, right)
    # Equivalent numeric contraction: torch.einsum("...ij,...ij->...ij", ...)
    # out.dims == (batch, i, j)
    ```

    Mixed middle ellipsis:

    ```python
    i = IndexSpace.linear(2)
    middle = IndexSpace.linear(4)
    j = IndexSpace.linear(3)

    left = Tensor(data=torch.randn(i.dim, middle.dim, j.dim), dims=(i, middle, j))
    right = Tensor(data=torch.randn(i.dim, j.dim), dims=(i, j))

    out = einsum("i...j,ij->i...j", left, right)
    # Equivalent numeric contraction: torch.einsum("i...j,i...j->i...j", ...)
    # out.dims == (i, middle, j)
    ```

    Mixed trailing ellipsis:

    ```python
    i = IndexSpace.linear(2)
    j = IndexSpace.linear(3)
    tail = IndexSpace.linear(4)

    left = Tensor(data=torch.randn(i.dim, j.dim, tail.dim), dims=(i, j, tail))
    right = Tensor(data=torch.randn(i.dim, j.dim), dims=(i, j))

    out = einsum("ij...,ij->ij...", left, right)
    # Equivalent numeric contraction: torch.einsum("ij...,ij...->ij...", ...)
    # out.dims == (i, j, tail)
    ```

    Higher-rank contraction over one shared index:

    ```python
    out = einsum("abc,dbe->acde", x, y)
    ```

    This computes
    \(out[a, c, d, e] = \sum_b x[a, b, c] \; y[d, b, e]\).

    Higher-rank contraction over multiple shared indices:

    ```python
    out = einsum("abcd,bcde->ae", x, y)
    ```

    This sums over the shared `b`, `c`, and `d` labels and keeps only the
    surviving `a` and `e` axes in the output.

    Parameters
    ----------
    equation : str
        Einstein summation equation in the same format accepted by
        `torch.einsum`.
    *operands : Tensor
        Input tensors whose ranks must match the equation terms after any
        `...` expansion.

    Returns
    -------
    Tensor
        Tensor whose data is computed by `torch.einsum` on the aligned operand
        data and whose `dims` follow the einsum output labels.

    Raises
    ------
    ValueError
        If the equation does not match the operand ranks or if any shared label
        maps to incompatible symbolic dimensions.

    See Also
    --------
    [`matmul`][qten.linalg.tensors.matmul]
        Specialized contraction helper for standard matrix multiplication.
    [`align`][qten.linalg.tensors.align]
        Axis-alignment primitive used to reconcile same-labeled symbolic dims.
    """
    (
        normalized_equation,
        expanded_terms,
        output_labels,
        normalized_operands,
    ) = _normalize_einsum_operands(equation, operands)

    label_dims: Dict[str, StateSpace] = {}
    for term, operand in zip(expanded_terms, normalized_operands):
        for axis, label in enumerate(term):
            label_dims[label] = _merge_einsum_dim(
                label_dims.get(label), operand.dims[axis], label
            )

    aligned_operands: list[Tensor] = []
    for term, operand in zip(expanded_terms, normalized_operands):
        current = operand
        for axis, label in enumerate(term):
            current = current.align(axis, label_dims[label])
        aligned_operands.append(current)

    promoted_operands = _promote_einsum_operands(aligned_operands)
    data = torch.einsum(
        normalized_equation, *(operand.data for operand in promoted_operands)
    )
    dims = tuple(label_dims[label] for label in output_labels)
    return Tensor(data=data, dims=dims)


def _kron_dim(left_dim: StateSpace, right_dim: StateSpace) -> StateSpace:
    if isinstance(left_dim, BroadcastSpace):
        return right_dim
    if isinstance(right_dim, BroadcastSpace):
        return left_dim
    if isinstance(left_dim, HasKroneckerProduct) and isinstance(
        right_dim, HasKroneckerProduct
    ):
        return cast(StateSpace, left_dim.kron(right_dim))
    raise TypeError(
        "kron only supports dimensions implementing HasKroneckerProduct "
        "(BroadcastSpace allowed as neutral axis), got "
        f"{type(left_dim).__name__} and {type(right_dim).__name__}."
    )


@auto_promote
def kron(left: Tensor, right: Tensor) -> Tensor:
    r"""
    Compute the StateSpace-aware Kronecker product between two tensors.

    Semantics
    ---------
    This function applies `torch.kron(left.data, right.data)` and propagates
    symbolic metadata axis-by-axis:

    - The operands must have the same rank.
    - For each axis `i`, output dim `i` is `left.dims[i] @ right.dims[i]`.
    - Axis pairs are matched by position, not by name or irrep content. The
      `i`-th dim of `left` is combined only with the `i`-th dim of `right`.
    - Non-broadcast axes must implement
      [`HasKroneckerProduct`][qten.abstracts.HasKroneckerProduct].
    - For Hilbert-space-like dims, the basis-level tensor product requires
      disjoint concrete irrep types between the paired dims. For example, a
      dim whose basis states contain irrep types `(int, str)` can be combined
      with one containing `(float,)`, but combining it with one containing
      `(int,)` is rejected because `int` appears on both sides.
    - [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] is treated
      as a neutral singleton axis, so `BroadcastSpace @ X` and
      `X @ BroadcastSpace` both map to `X`.

    Allowed dim products
    --------------------
    Let \(L_i = \mathrm{left.dims}[i]\) and
    \(R_i = \mathrm{right.dims}[i]\). The metadata part of
    [`kron(left, right)`][qten.linalg.tensors.kron] is allowed only when every
    paired dim product

    \[
        L_i \otimes R_i
    \]

    is defined. For Hilbert-space-like dims, write
    \(\operatorname{types}(D)\) for the concrete irrep types present in dim
    \(D\). Then the paired product requires

    \[
        \operatorname{types}(L_i) \cap \operatorname{types}(R_i) = \varnothing.
    \]

    For example, a dim with irrep types \((\mathrm{int}, \mathrm{str})\) can
    be paired with one whose irrep types are \((\mathrm{float},)\), but not
    with one whose irrep types are \((\mathrm{int},)\), because the shared
    \(\mathrm{int}\) type would give multiplicity greater than one in the
    tensor-product basis.

    Parameters
    ----------
    left : Tensor
        Left operand.
    right : Tensor
        Right operand.

    Returns
    -------
    Tensor
        Tensor with Kronecker-product data and axis-wise tensor-product dims.

    Raises
    ------
    ValueError
        If `left` and `right` do not have the same rank, or if a dim-level
        Kronecker product rejects overlapping irrep types.
    TypeError
        If any axis pair is not compatible with Kronecker-product semantics
        (except neutral broadcast axes).
    """
    if left.rank() != right.rank():
        raise ValueError(
            "kron expects tensors with the same rank: "
            f"got {left.rank()} and {right.rank()}."
        )

    new_dims = tuple(
        _kron_dim(left_dim, right_dim)
        for left_dim, right_dim in zip(left.dims, right.dims)
    )
    return Tensor(data=torch.kron(left.data, right.data), dims=new_dims)


@auto_promote
def matmul(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform matrix multiplication between two Tensors with StateSpace-aware
    alignment and torch-style rank handling.

    Supported forms
    ---------------
    [`matmul(left, right)`][qten.linalg.tensors.matmul]
        Functional form.

    [`left @ right`][qten.linalg.tensors.matmul]
        Operator form provided by [`Operable`][qten.abstracts.Operable] and
        dispatched to this function.

    Both operands must be at least 1D. If either operand is 1D, this follows
    `torch.matmul` behavior by temporarily unsqueezing it to 2D, performing the
    matmul, then squeezing out the added dimension(s).

    The function first makes the tensors have the same number of dimensions by
    unsqueezing leading dimensions with [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace]. It then aligns any
    leading (batch) dimensions so that [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] can expand to concrete
    StateSpaces and any non-broadcast StateSpaces are reordered to match. Finally,
    the right tensor's second-to-last dimension is aligned to the left tensor's
    last dimension, and `torch.matmul` is applied.

    The contraction always happens between `left.dims[-1]` and `right.dims[-2]`.
    Leading dimensions behave like batch dimensions and follow the broadcast and
    alignment rules described above. The output keeps all aligned leading
    dimensions (including any [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] that remain), drops the contracted
    dimension, and appends the right-most dimension from `right`.

    Use cases
    ---------
    - contract two symbolic matrix/tensor operators while preserving axis
      labels,
    - multiply batched operators whose batch axes need symbolic alignment,
    - handle vector-matrix, matrix-vector, and vector-vector products with the
      same API used for higher-rank tensors.

    Parameters
    ----------
    left : Tensor
        The left tensor operand.
    right : Tensor
        The right tensor operand.

    Returns
    -------
    Tensor
        A tensor with data `torch.matmul(left.data, right.data)` and dimensions
        left.dims[:-1] + right.dims[-1:], after the alignment and any
        1D squeeze handling.

    Raises
    ------
    ValueError
        If either operand is 0D or any StateSpace alignment fails during the
        broadcast or contraction alignment steps.
    """
    left_rank = left.rank()
    right_rank = right.rank()

    if left_rank < 1:
        raise ValueError("Left tensor must have rank at least 1 for matmul!")
    if right_rank < 1:
        raise ValueError("Right tensor must have rank at least 1 for matmul!")

    left, right = _match_dims_for_matmul(left, right)

    # Reconcile batch dimensions using strict union semantics.
    batch_target = union_dims(left.dims[:-2], right.dims[:-2], allow_merge=False)

    # Align only batch dimensions to the shared batch target.
    left_target_dims = batch_target + left.dims[-2:]
    right_target_dims = batch_target + right.dims[-2:]
    left = left.align_all(left_target_dims)
    right = right.align_all(right_target_dims)

    # Materialize data broadcast where any batch axis is still broadcast-backed.
    left = left.expand_to_union(list(left_target_dims))
    right = right.expand_to_union(list(right_target_dims))

    right = right.align(-2, left.dims[-1])
    data = torch.matmul(left.data, right.data)
    new_dims = left.dims[:-1] + right.dims[-1:]

    prod = Tensor(data=data, dims=new_dims)

    if left_rank == 1 and right_rank == 1:
        prod = prod.squeeze(0).squeeze(-1)
    elif right_rank == 1:
        prod = prod.squeeze(-1)
    elif left_rank == 1:
        prod = prod.squeeze(-2)

    return prod


@Operable.__matmul__.register
def _(left: Tensor, right: Tensor) -> Tensor:
    """
    Contract two tensors with `@` using StateSpace-aware matrix multiplication.

    Parameters
    ----------
    left : Tensor
        Left operand.
    right : Tensor
        Right operand.

    Returns
    -------
    Tensor
        Product tensor returned by `matmul(left, right)`.

    Notes
    -----
    The actual contraction logic, batch broadcasting, and metadata alignment
    are implemented by [`matmul`][qten.linalg.tensors.matmul].
    """
    return matmul(left, right)


def _match_dims_for_tensoradd(left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
    if left.rank() > right.rank():
        right = promote_rank(right, left.rank())
    elif right.rank() > left.rank():
        left = promote_rank(left, right.rank())
    return left, right


@Operable.__add__.register
@auto_promote
def _(left: Tensor, right: Tensor) -> Tensor:
    """
    Add two tensors with the same order of dimensions.
    If the intra-ordering within the [`StateSpace`][qten.symbolics.state_space.StateSpace]s differ,
    the `right` tensor is permuted to match the ordering
    of the `left` tensor before addition.

    Parameters
    ----------
    left : Tensor
        The left tensor to add.
    right : Tensor
        The right tensor to add.

    Returns
    -------
    Tensor
        The resulting tensor on the union of StateSpaces.
    """
    left, right = _match_dims_for_tensoradd(left, right)

    # Calculate merged dimensions once using the shared union helper.
    merged_dims = union_dims(left.dims, right.dims, allow_merge=True)

    # Expand BroadcastSpace to the union StateSpace to ensure data expansion
    left = left.expand_to_union(list(merged_dims))
    right = right.expand_to_union(list(merged_dims))

    # calculate the new shape
    new_shape = tuple(u.dim for u in merged_dims)
    new_data = torch.zeros(new_shape, dtype=left.data.dtype, device=left.data.device)
    # fill the left tensor into the new data
    left_slices = tuple(slice(0, d.dim) for d in left.dims)
    new_data[left_slices] = left.data
    # fill the right tensor into the new data
    right_embedding_order = (
        torch.tensor(embedding_order(r, u), dtype=torch.long, device=left.data.device)
        for r, u in zip(right.dims, merged_dims)
    )
    new_data.index_put_(
        torch.meshgrid(*right_embedding_order, indexing="ij"),
        right.data,
        accumulate=True,
    )

    return Tensor(data=new_data, dims=merged_dims)


@Operable.__eq__.register
def _(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform element-wise equality comparison between two tensors.

    Parameters
    ----------
    left : Tensor
        Left operand.
    right : Tensor
        Right operand.

    Returns
    -------
    Tensor
        Boolean tensor with merged [`StateSpace`][qten.symbolics.state_space.StateSpace] metadata.

    Notes
    -----
    Comparison follows symmetric broadcast semantics:

    - compute strict shared union dims with
      `union_dims(..., allow_merge=False)`,
    - align both operands to those dims,
    - rely on torch runtime broadcasting for any singleton axes,
    - return a boolean tensor with `dims == union_dims`.
    """
    return _tensor_comparison_op(left, right, torch.eq)


def _tensor_comparison_op(
    left: TensorType,
    right: Tensor,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> TensorType:
    """
    Perform an element-wise comparison between two tensors using symmetric
    StateSpace-aware alignment and broadcasting.
    """
    promoted_left, promoted_right = _promote_to_device(left, right)
    left = cast(TensorType, promoted_left)
    right = promoted_right

    target_rank = max(left.rank(), right.rank())
    left = cast(TensorType, promote_rank(left, target_rank))
    right = promote_rank(right, target_rank)
    merged_dims = union_dims(left.dims, right.dims, allow_merge=False)
    aligned_left = left.align_all(merged_dims)
    aligned_right = right.align_all(merged_dims)
    expected_shape = tuple(dim.dim for dim in merged_dims)
    try:
        runtime_shape = torch.broadcast_shapes(
            aligned_left.data.shape, aligned_right.data.shape
        )
    except RuntimeError as e:
        raise ValueError(
            "operands are not broadcastable after StateSpace alignment: "
            f"left_shape={tuple(aligned_left.data.shape)}, "
            f"right_shape={tuple(aligned_right.data.shape)}, "
            f"merged_dims={_format_dims(merged_dims)}"
        ) from e
    if runtime_shape != expected_shape:
        raise ValueError(
            "StateSpace dims do not match runtime broadcast shape: "
            f"merged_dims={_format_dims(merged_dims)}, "
            f"expected_shape={expected_shape}, runtime_shape={tuple(runtime_shape)}"
        )
    return replace(
        left, data=op(aligned_left.data, aligned_right.data), dims=merged_dims
    )


def _binary_elementwise_mask_op(
    left: TensorType,
    right: Tensor,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> TensorType:
    """
    Apply a boolean-mask-producing binary tensor op with symmetric StateSpace
    alignment and broadcasting.
    """
    promoted_left, promoted_right = _promote_to_device(left, right)
    left = cast(TensorType, promoted_left)
    right = promoted_right

    target_rank = max(left.rank(), right.rank())
    left = cast(TensorType, promote_rank(left, target_rank))
    right = promote_rank(right, target_rank)
    merged_dims = union_dims(left.dims, right.dims, allow_merge=False)
    aligned_left = left.align_all(merged_dims)
    aligned_right = right.align_all(merged_dims)
    expected_shape = tuple(dim.dim for dim in merged_dims)
    try:
        runtime_shape = torch.broadcast_shapes(
            aligned_left.data.shape, aligned_right.data.shape
        )
    except RuntimeError as e:
        raise ValueError(
            "operands are not broadcastable after StateSpace alignment: "
            f"left_shape={tuple(aligned_left.data.shape)}, "
            f"right_shape={tuple(aligned_right.data.shape)}, "
            f"merged_dims={_format_dims(merged_dims)}"
        ) from e
    if runtime_shape != expected_shape:
        raise ValueError(
            "StateSpace dims do not match runtime broadcast shape: "
            f"merged_dims={_format_dims(merged_dims)}, "
            f"expected_shape={expected_shape}, runtime_shape={tuple(runtime_shape)}"
        )
    return replace(
        left, data=op(aligned_left.data, aligned_right.data), dims=merged_dims
    )


@Operable.__lt__.register
def _(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform element-wise less-than comparison between two tensors.

    Parameters
    ----------
    left : Tensor
        Left operand.
    right : Tensor
        Right operand.

    Returns
    -------
    Tensor
        Boolean tensor on the merged output metadata.
    """
    return _tensor_comparison_op(left, right, torch.lt)


@Operable.__le__.register
def _(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform element-wise less-than-or-equal comparison between two tensors.

    Parameters
    ----------
    left : Tensor
        Left operand.
    right : Tensor
        Right operand.

    Returns
    -------
    Tensor
        Boolean tensor on the merged output metadata.
    """
    return _tensor_comparison_op(left, right, torch.le)


@Operable.__gt__.register
def _(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform element-wise greater-than comparison between two tensors.

    Parameters
    ----------
    left : Tensor
        Left operand.
    right : Tensor
        Right operand.

    Returns
    -------
    Tensor
        Boolean tensor on the merged output metadata.
    """
    return _tensor_comparison_op(left, right, torch.gt)


@Operable.__ge__.register
def _(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform element-wise greater-than-or-equal comparison between two tensors.

    Parameters
    ----------
    left : Tensor
        Left operand.
    right : Tensor
        Right operand.

    Returns
    -------
    Tensor
        Boolean tensor on the merged output metadata.
    """
    return _tensor_comparison_op(left, right, torch.ge)


@Operable.__lt__.register
def _(left: Tensor, right: Number) -> Tensor:
    """
    Compare a tensor to a scalar with element-wise `<`.

    Parameters
    ----------
    left : Tensor
        Tensor operand.
    right : Number
        Scalar operand promoted to `Tensor.scalar(right)`.

    Returns
    -------
    Tensor
        Boolean tensor with the same comparison semantics as tensor-tensor
        comparison.
    """
    return left < Tensor.scalar(right)


@Operable.__lt__.register
def _(left: Number, right: Tensor) -> Tensor:
    """
    Compare a scalar to a tensor with element-wise `<`.

    Parameters
    ----------
    left : Number
        Scalar operand promoted to `Tensor.scalar(left)`.
    right : Tensor
        Tensor operand.

    Returns
    -------
    Tensor
        Boolean tensor with the same comparison semantics as tensor-tensor
        comparison.
    """
    return Tensor.scalar(left) < right


@Operable.__le__.register
def _(left: Tensor, right: Number) -> Tensor:
    """
    Compare a tensor to a scalar with element-wise `<=`.

    Parameters
    ----------
    left : Tensor
        Tensor operand.
    right : Number
        Scalar operand promoted to `Tensor.scalar(right)`.

    Returns
    -------
    Tensor
        Boolean tensor with merged comparison metadata.
    """
    return left <= Tensor.scalar(right)


@Operable.__le__.register
def _(left: Number, right: Tensor) -> Tensor:
    """
    Compare a scalar to a tensor with element-wise `<=`.

    Parameters
    ----------
    left : Number
        Scalar operand promoted to `Tensor.scalar(left)`.
    right : Tensor
        Tensor operand.

    Returns
    -------
    Tensor
        Boolean tensor with merged comparison metadata.
    """
    return Tensor.scalar(left) <= right


@Operable.__gt__.register
def _(left: Tensor, right: Number) -> Tensor:
    """
    Compare a tensor to a scalar with element-wise `>`.

    Parameters
    ----------
    left : Tensor
        Tensor operand.
    right : Number
        Scalar operand promoted to `Tensor.scalar(right)`.

    Returns
    -------
    Tensor
        Boolean tensor with merged comparison metadata.
    """
    return left > Tensor.scalar(right)


@Operable.__gt__.register
def _(left: Number, right: Tensor) -> Tensor:
    """
    Compare a scalar to a tensor with element-wise `>`.

    Parameters
    ----------
    left : Number
        Scalar operand promoted to `Tensor.scalar(left)`.
    right : Tensor
        Tensor operand.

    Returns
    -------
    Tensor
        Boolean tensor with merged comparison metadata.
    """
    return Tensor.scalar(left) > right


@Operable.__ge__.register
def _(left: Tensor, right: Number) -> Tensor:
    """
    Compare a tensor to a scalar with element-wise `>=`.

    Parameters
    ----------
    left : Tensor
        Tensor operand.
    right : Number
        Scalar operand promoted to `Tensor.scalar(right)`.

    Returns
    -------
    Tensor
        Boolean tensor with merged comparison metadata.
    """
    return left >= Tensor.scalar(right)


@Operable.__ge__.register
def _(left: Number, right: Tensor) -> Tensor:
    """
    Compare a scalar to a tensor with element-wise `>=`.

    Parameters
    ----------
    left : Number
        Scalar operand promoted to `Tensor.scalar(left)`.
    right : Tensor
        Tensor operand.

    Returns
    -------
    Tensor
        Boolean tensor with merged comparison metadata.
    """
    return Tensor.scalar(left) >= right


@Operable.__neg__.register
def _(tensor: Tensor) -> Tensor:
    """
    Perform negation on the given tensor.

    Parameters
    ----------
    tensor : Tensor
        The tensor to negate.

    Returns
    -------
    TensorType
        The negated tensor, preserving the input wrapper type.
    """
    return replace(tensor, data=-tensor.data)


@Operable.__sub__.register
def _(left: Tensor, right: Tensor) -> Tensor:
    """
    Subtract the right tensor from the left tensor with the same order of dimensions.
    If the intra-ordering within the [`StateSpace`][qten.symbolics.state_space.StateSpace]s differ, the `right` tensor is
    permuted to match the ordering of the `left` tensor before addition.

    Parameters
    ----------
    left : Tensor
        The tensor from which to subtract.
    right : Tensor
        The tensor to subtract.

    Returns
    -------
    Tensor
        The resulting tensor after subtraction.
    """
    return left + (-right)


@Operable.__mul__.register
def _(left: Number, right: Tensor) -> Tensor:
    """
    Perform element-wise multiplication of a number and a tensor.

    Parameters
    ----------
    left : Number
        The scalar value.
    right : Tensor
        The tensor.

    Returns
    -------
    Tensor
        A new tensor with each element multiplied by the scalar.
    """
    return Tensor(data=left * right.data, dims=right.dims)


@Operable.__mul__.register
def _(left: Tensor, right: Number) -> Tensor:
    """
    Perform element-wise multiplication of a tensor and a number.

    Parameters
    ----------
    left : Tensor
        The tensor.
    right : Number
        The scalar value.

    Returns
    -------
    Tensor
        A new tensor with each element multiplied by the scalar.
    """
    return Tensor(data=left.data * right, dims=left.dims)


@Operable.__add__.register
def _(left: Number, right: Tensor) -> Tensor:
    r"""
    Add a number to the diagonal of the tensor (broadcasting over batch dimensions).

    This treats the tensor as a batch of matrices defined by the last two
    dimensions. The scalar is added to the diagonal elements of each matrix.
    For rank-2 tensors this is equivalent to \(cI + M\).

    Parameters
    ----------
    left : Number
        The scalar value to add to the diagonal.
    right : Tensor
        The target tensor (must be at least rank 2).

    Returns
    -------
    Tensor
        The result of adding the scalar to the diagonal.
    """
    iden = eye(right.dims)
    return left * iden + right


@Operable.__add__.register
def _(left: Tensor, right: Number) -> Tensor:
    r"""
    Add a number to the diagonal of the tensor (broadcasting over batch dimensions).

    This treats the tensor as a batch of matrices defined by the last two
    dimensions. The scalar is added to the diagonal elements of each matrix.
    For rank-2 tensors this is equivalent to \(M + cI\).

    Parameters
    ----------
    left : Tensor
        The target tensor (must be at least rank 2).
    right : Number
        The scalar value to add to the diagonal.

    Returns
    -------
    Tensor
        The result of adding the scalar to the diagonal.
    """
    iden = eye(left.dims)
    return left + right * iden


@Operable.__sub__.register
def _(left: Number, right: Tensor) -> Tensor:
    r"""
    Subtract a tensor from a number (broadcasted on diagonal).

    This treats the tensor as a batch of matrices defined by the last two
    dimensions. The operation is performed as \(cI - T\), where \(I\) is the
    identity matrix broadcast over batch dimensions.

    Parameters
    ----------
    left : Number
        The scalar value.
    right : Tensor
        The tensor to subtract.

    Returns
    -------
    Tensor
        The result of the subtraction.
    """
    iden = eye(right.dims)
    return left * iden + (-right)


@Operable.__sub__.register
def _(left: Tensor, right: Number) -> Tensor:
    r"""
    Subtract a number from a tensor (broadcasted on diagonal).

    This treats the tensor as a batch of matrices defined by the last two
    dimensions. The operation is performed as \(T - cI\), where \(I\) is the
    identity matrix broadcast over batch dimensions.

    Parameters
    ----------
    left : Tensor
        The tensor.
    right : Number
        The scalar value to subtract from the diagonal.

    Returns
    -------
    Tensor
        The result of the subtraction.
    """
    iden = eye(left.dims)
    return left + (-right) * iden


@Operable.__truediv__.register
def _(left: Tensor, right: Number) -> Tensor:
    """
    Perform element-wise division of a tensor by a number.

    Parameters
    ----------
    left : Tensor
        The tensor.
    right : Number
        The scalar divisor.

    Returns
    -------
    Tensor
        A new tensor with each element divided by the scalar.
    """
    return left * (1.0 / right)  # type: ignore[operator]


def permute(tensor: TensorType, *order: Union[int, Sequence[int]]) -> TensorType:
    """
    Reorder tensor axes and their symbolic dimensions.

    This is the metadata-aware analogue of `torch.Tensor.permute`. It applies
    the same permutation to `tensor.data` and `tensor.dims`, so the returned
    tensor keeps its symbolic axes in sync with the raw data layout.

    Parameters
    ----------
    tensor : Tensor
        Tensor whose axes will be reordered.
    order : Union[int, Sequence[int]]
        Desired axis order. This may be passed either as variadic integers or
        as a single tuple/list.

    Returns
    -------
    TensorType
        Tensor with permuted data and correspondingly permuted `dims`,
        preserving the input wrapper type.

    Raises
    ------
    ValueError
        If the permutation length does not match `tensor.rank()`.
    """
    _order: Tuple[int, ...]
    if len(order) == 1 and isinstance(order[0], (tuple, list)):
        _order = tuple(order[0])
    else:
        # We assume that if it's not a single list/tuple, it's a sequence of ints
        _order = cast(Tuple[int, ...], tuple(order))

    if len(_order) != tensor.rank():
        raise ValueError(
            f"Permutation order length {len(_order)} does not match tensor dimensions {tensor.rank()}!"
        )

    new_data = tensor.data.permute(_order)
    new_dims = tuple(tensor.dims[i] for i in _order)

    return replace(tensor, data=new_data, dims=new_dims)


def transpose(tensor: TensorType, dim0: int, dim1: int) -> TensorType:
    """
    Swap two tensor axes and their symbolic dimensions.

    This is a two-axis specialization of
    [`permute`][qten.linalg.tensors.permute]. Both the raw data and the
    corresponding [`StateSpace`][qten.symbolics.state_space.StateSpace]
    metadata are transposed together.

    Parameters
    ----------
    tensor : Tensor
        Tensor whose axes will be swapped.
    dim0 : int
        The first dimension to transpose.
    dim1 : int
        The second dimension to transpose.

    Returns
    -------
    TensorType
        Tensor with the selected axes exchanged, preserving the input wrapper
        type.
    """
    new_data = tensor.data.transpose(dim0, dim1)

    # Convert tuple to list to modify
    new_dims_list = list(tensor.dims)
    # Swap elements
    new_dims_list[dim0], new_dims_list[dim1] = new_dims_list[dim1], new_dims_list[dim0]

    return replace(tensor, data=new_data, dims=tuple(new_dims_list))


def conj(tensor: TensorType) -> TensorType:
    """
    Return the element-wise complex conjugate of a tensor.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.

    Returns
    -------
    TensorType
        Tensor with conjugated data and unchanged dims, preserving the input
        wrapper type.
    """
    return replace(tensor, data=tensor.data.conj())


def real(tensor: TensorType) -> TensorType:
    """
    Return the real part of a tensor.

    The symbolic dimensions are unchanged. The returned tensor uses the real
    dtype associated with the input data.
    """
    return replace(tensor, data=cast(T, tensor.data.real))


def imag(tensor: TensorType) -> TensorType:
    """
    Return the imaginary part of a tensor.

    The symbolic dimensions are unchanged. For real-valued tensors this returns
    a zero tensor with the corresponding real dtype.
    """
    if tensor.data.is_complex():
        return replace(tensor, data=cast(T, tensor.data.imag))
    return replace(tensor, data=cast(T, torch.zeros_like(tensor.data)))


def abs(tensor: TensorType) -> TensorType:
    """
    Return the element-wise absolute value or magnitude of a tensor.

    The symbolic dimensions are unchanged. Complex inputs produce a real-valued
    magnitude tensor following PyTorch semantics.
    """
    return replace(tensor, data=cast(T, tensor.data.abs()))


def unsqueeze(tensor: TensorType, dim: int) -> TensorType:
    """
    Insert a singleton broadcast axis into a tensor.

    This is the metadata-aware analogue of `torch.unsqueeze`. The new axis is
    labeled with [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace]
    so later alignment and broadcasting operations can treat it as a symbolic
    singleton axis.

    Parameters
    ----------
    tensor : Tensor
        The tensor to unsqueeze.
    dim : int
        The dimension to unsqueeze.

    Returns
    -------
    TensorType
        Tensor with one extra singleton axis and a matching inserted
        [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] dim.
    """
    if dim < 0:
        dim = dim + len(tensor.dims) + 1
    new_data = tensor.data.unsqueeze(dim)
    new_dims = tensor.dims[:dim] + (BroadcastSpace(),) + tensor.dims[dim:]

    return replace(tensor, data=new_data, dims=new_dims)


def squeeze(tensor: TensorType, dim: int) -> TensorType:
    """
    Remove a singleton broadcast axis from a tensor.

    Only axes labeled by
    [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace] are
    removed. If the specified axis is not a broadcast axis, the input tensor is
    returned unchanged.

    Parameters
    ----------
    tensor : Tensor
        The tensor to squeeze.
    dim : int
        The dimension to squeeze.

    Returns
    -------
    TensorType
        Tensor with the requested broadcast axis removed, preserving the input
        wrapper type.
    """
    if dim < 0:
        dim = dim + len(tensor.dims)
    if not isinstance(tensor.dims[dim], BroadcastSpace):
        return tensor  # No squeezing needed if not BroadcastSpace

    new_data = tensor.data.squeeze(dim)
    new_dims = tensor.dims[:dim] + tensor.dims[dim + 1 :]

    return replace(tensor, data=new_data, dims=new_dims)


def _normalize_dim(dim: int, rank: int) -> int:
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise IndexError(f"Dimension index {dim} out of range for rank {rank}")
    return dim


def _prepare_index_add_operands(
    tensor: TensorType,
    dim: int,
    index: Tensor[Any],
    source: Tensor,
) -> Tuple[int, TensorType, Tensor[Any], Tensor]:
    dim = _normalize_dim(dim, tensor.rank())

    if index.rank() != 1:
        raise ValueError(
            f"index_add requires rank-1 index tensor, got rank {index.rank()}"
        )
    if index.data.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "index_add requires index tensor with dtype torch.int32 or torch.int64"
        )
    if source.rank() != tensor.rank():
        raise ValueError(
            f"index_add requires source rank {tensor.rank()}, got rank {source.rank()}"
        )

    index_dim = index.dims[0]
    target_dims = tensor.dims[:dim] + (index_dim,) + tensor.dims[dim + 1 :]

    promoted_tensor, promoted_index, promoted_source = cast(
        Tuple[TensorType, Tensor[Any], Tensor],
        _promote_to_device(tensor, index, source),
    )
    try:
        aligned_source = promoted_source.align_all(target_dims)
    except (IndexError, TypeError, ValueError) as e:
        raise ValueError(
            "index_add could not align source to tensor/index dimensions on "
            "non-indexed axes: "
            f"source={_format_dims(promoted_source.dims)}, "
            f"target={_format_dims(target_dims)}"
        ) from e

    if aligned_source.data.shape[dim] != promoted_index.data.numel():
        raise ValueError(
            "index_add requires source size along dim to match index length: "
            f"source={aligned_source.data.shape[dim]}, index={promoted_index.data.numel()}"
        )

    expected_shape = list(promoted_tensor.data.shape)
    expected_shape[dim] = promoted_index.data.numel()
    if tuple(aligned_source.data.shape) != tuple(expected_shape):
        raise ValueError(
            "index_add requires source shape to match tensor shape on non-indexed axes: "
            f"source={tuple(aligned_source.data.shape)}, expected={tuple(expected_shape)}"
        )

    return dim, promoted_tensor, promoted_index, aligned_source


def index_add(
    tensor: TensorType,
    dim: int,
    index: Tensor[Any],
    source: Tensor,
    alpha: Union[int, float, complex] = 1,
) -> TensorType:
    """
    Return a copy of `tensor` with `source` accumulated into positions `index`.

    This is a metadata-aware wrapper over `torch.index_add`. It preserves the
    dimensions of `tensor`, requires a rank-1 integer `index`, and aligns
    `source` to `tensor` on all non-indexed axes before dispatch.

    Behavior
    --------
    - `index` describes destination positions along axis `dim`.
    - `index.dims[0]` is treated as the symbolic ordering of the updates.
    - `source` is aligned to `tensor` on every non-indexed axis.
    - on the indexed axis, `source` must already match the ordering and length
      implied by `index`.

    Use cases
    ---------
    - scatter-add updates coming from a symbolic subspace or sampled index set,
    - accumulate block contributions into a larger tensor while preserving the
      destination tensor metadata.
    """
    dim, promoted_tensor, promoted_index, aligned_source = _prepare_index_add_operands(
        tensor, dim, index, source
    )
    out = torch.index_add(
        promoted_tensor.data,
        dim,
        promoted_index.data,
        aligned_source.data,
        alpha=alpha,
    )
    return replace(promoted_tensor, data=out)


def index_add_(
    tensor: TensorType,
    dim: int,
    index: Tensor[Any],
    source: Tensor,
    alpha: Union[int, float, complex] = 1,
) -> TensorType:
    """
    In-place metadata-aware wrapper over `torch.Tensor.index_add_`.

    Supported forms
    ---------------
    [`index_add_(tensor, dim, index, source, alpha=1)`][qten.linalg.tensors.index_add_]
        Functional in-place form.

    [`tensor.index_add_(dim, index, source, alpha=1)`][qten.linalg.tensors.Tensor.index_add_]
        Method form.

    The returned object is `tensor` itself. This is useful when you want the
    side effect of accumulation without allocating a new wrapper object.
    """
    dim = _normalize_dim(dim, tensor.rank())

    if tensor.device != index.device:
        index = index.to_device(tensor.device)
    if tensor.device != source.device:
        source = source.to_device(tensor.device)

    _, _, prepared_index, aligned_source = _prepare_index_add_operands(
        tensor, dim, index, source
    )
    tensor.data.index_add_(dim, prepared_index.data, aligned_source.data, alpha=alpha)
    return tensor


def align(tensor: TensorType, dim: int, target_dim: StateSpace) -> TensorType:
    """
    Align one tensor axis to a target symbolic dimension.

    Alignment is QTen's core metadata-aware axis reordering operation:

    - if the source axis already matches `target_dim`, the tensor is returned
      unchanged,
    - if the source axis is
      [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace], it is
      expanded to the size of `target_dim`,
    - if the source and target axes span the same rays in a different order,
      the data is permuted along that axis to match `target_dim`,
    - otherwise alignment fails.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    dim : int
        The dimension index to align.
    target_dim : StateSpace
        The target StateSpace to align to.

    Returns
    -------
    TensorType
        Tensor whose selected axis matches `target_dim`, preserving the input
        wrapper type.

    Raises
    ------
    IndexError
        If `dim` is out of range for the tensor rank.
    ValueError
        If the source and target dimensions are not symbolically compatible for
        alignment.
    """
    if dim < 0:
        dim = dim + len(tensor.dims)
    if dim < 0 or dim >= len(tensor.dims):
        raise IndexError(
            f"Dimension index {dim} out of range for rank {len(tensor.dims)}"
        )

    current_dim = tensor.dims[dim]
    if isinstance(target_dim, BroadcastSpace):
        return tensor  # No alignment needed for BroadcastSpace
    if current_dim is target_dim or current_dim == target_dim:
        return tensor

    if isinstance(current_dim, BroadcastSpace):
        # Expand broadcast dimension to match the target StateSpace size.
        expanded_shape = list(tensor.data.shape)
        expanded_shape[dim] = target_dim.dim
        aligned_data = tensor.data.expand(*expanded_shape)
        return replace(
            tensor,
            data=aligned_data,
            dims=tensor.dims[:dim] + (target_dim,) + tensor.dims[dim + 1 :],
        )

    if type(current_dim) is not type(target_dim):
        raise ValueError(
            f"Cannot align dimensions with different StateSpace types: "
            f"current={type(current_dim).__name__}:{current_dim.dim} vs "
            f"target={type(target_dim).__name__}:{target_dim.dim}"
        )
    if not same_rays(current_dim, target_dim):
        raise ValueError(
            f"StateSpace at axis {dim} cannot be aligned to target StateSpace: "
            f"current={type(current_dim).__name__}:{current_dim.dim}, "
            f"target={type(target_dim).__name__}:{target_dim.dim}"
        )

    try:
        target_order = permutation_order(current_dim, target_dim)
    except ValueError as e:
        raise ValueError(
            f"StateSpace at axis {dim} cannot be aligned to target StateSpace: "
            f"current={type(current_dim).__name__}:{current_dim.dim}, "
            f"target={type(target_dim).__name__}:{target_dim.dim}"
        ) from e
    if target_order == tuple(range(current_dim.dim)):
        return tensor
    aligned_data = torch.index_select(
        tensor.data,
        dim,
        torch.tensor(target_order, dtype=torch.long, device=tensor.data.device),
    )

    aligned_tensor = replace(
        tensor,
        data=aligned_data,
        dims=tensor.dims[:dim] + (target_dim,) + tensor.dims[dim + 1 :],
    )

    return aligned_tensor


def align_all(tensor: TensorType, dims: Tuple[StateSpace, ...]) -> TensorType:
    """
    Align every tensor axis to a target symbolic layout.

    This applies [`align`][qten.linalg.tensors.align] axis-by-axis and is the
    standard utility used before value comparisons and metadata-aware
    broadcasting.

    Parameters
    ----------
    tensor : Tensor
        The tensor to align.
    dims : Tuple[StateSpace, ...]
        Target dimensions for each axis.

    Returns
    -------
    TensorType
        Tensor whose `dims` match the requested target layout, preserving the
        input wrapper type.

    Raises
    ------
    ValueError
        If rank does not match or any dimension cannot be aligned.
    """
    if len(dims) != tensor.rank():
        raise ValueError(
            f"Cannot align rank-{tensor.rank()} tensor to rank-{len(dims)} dims: "
            f"current={_format_dims(tensor.dims)}, target={_format_dims(dims)}"
        )

    aligned = tensor
    for idx, target_dim in enumerate(dims):
        aligned = aligned.align(idx, target_dim)
    return aligned


def all(
    tensor: TensorType,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
) -> TensorType:
    """
    Reduce a tensor with logical AND, matching `torch.all` semantics.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    dim : Optional[Union[int, Tuple[int, ...]]], optional
        Reduction axis (or axes). If `None`, reduce over all dimensions.
    keepdim : bool, optional
        If `True`, retains the reduced axis as [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace].

    Returns
    -------
    TensorType
        Boolean tensor with reduced dimensions, preserving the input wrapper
        type.

    Notes
    -----
    When `keepdim=True`, reduced axes are retained as
    [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace] dimensions
    rather than their original concrete spaces.

    Raises
    ------
    IndexError
        If any requested reduction axis is out of range for the tensor rank.
    """
    if dim is None:
        return replace(tensor, data=torch.all(tensor.data), dims=())

    rank_ = tensor.rank()
    if isinstance(dim, int):
        dims_tuple: Tuple[int, ...] = (dim,)
    else:
        dims_tuple = dim

    normalized_dims: list[int] = []
    for d in dims_tuple:
        nd = d
        if nd < 0:
            nd += rank_
        if nd < 0 or nd >= rank_:
            raise IndexError(f"Dimension index {d} out of range for rank {rank_}")
        normalized_dims.append(nd)

    reduced_dims = tuple(normalized_dims)
    reduced_dims_set = set(reduced_dims)

    reduced = torch.all(tensor.data, dim=dim, keepdim=keepdim)
    if keepdim:
        new_dims = tuple(
            BroadcastSpace() if idx in reduced_dims_set else current_dim
            for idx, current_dim in enumerate(tensor.dims)
        )
    else:
        new_dims = tuple(
            current_dim
            for idx, current_dim in enumerate(tensor.dims)
            if idx not in reduced_dims_set
        )
    return replace(tensor, data=reduced, dims=new_dims)


def rank(tensor: Tensor) -> int:
    """
    Get the rank (number of dimensions) of the tensor.

    Parameters
    ----------
    tensor : Tensor
        The tensor whose rank is to be determined.

    Returns
    -------
    int
        The rank of the tensor.
    """
    return len(tensor.dims)


def mean(
    tensor: TensorType, dim: Optional[Union[int, Tuple[int, ...]]] = None
) -> TensorType:
    """
    Compute the mean over one or more tensor axes.

    This mirrors `torch.mean` for the supported `dim` forms while updating the
    symbolic dimension metadata to remove the reduced axes.

    Parameters
    ----------
    tensor : Tensor
        The tensor to reduce.
    dim : Optional[Union[int, Tuple[int, ...]]], optional
        Reduction axis (or axes). If `None`, reduce over all dimensions.

    Returns
    -------
    TensorType
        Tensor with the requested axes reduced and removed from `dims`,
        preserving the input wrapper type.

    Raises
    ------
    IndexError
        If any requested reduction axis is out of range for the tensor rank.
    """
    if dim is None:
        return replace(tensor, data=tensor.data.mean(), dims=())

    rank_ = tensor.rank()
    if isinstance(dim, int):
        dims_tuple: Tuple[int, ...] = (dim,)
    else:
        dims_tuple = dim

    normalized_dims: list[int] = []
    for d in dims_tuple:
        nd = d
        if nd < 0:
            nd += rank_
        if nd < 0 or nd >= rank_:
            raise IndexError(f"Dimension index {d} out of range for rank {rank_}")
        normalized_dims.append(nd)

    reduced_dims_set = set(normalized_dims)
    reduced = tensor.data.mean(dim=dim)
    new_dims = tuple(
        current_dim
        for idx, current_dim in enumerate(tensor.dims)
        if idx not in reduced_dims_set
    )
    return replace(tensor, data=reduced, dims=new_dims)


def norm(
    tensor: TensorType,
    ord: Optional[Union[int, float, str]] = None,
    dim: Optional[Union[int, Tuple[int, int]]] = None,
) -> TensorType:
    """
    Compute a vector or matrix norm with metadata-aware dimension reduction.

    This forwards to `torch.linalg.norm` for the numeric computation, then
    removes the reduced axes from the symbolic output dims.

    See Also
    --------
    [`torch.linalg.norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.norm.html)
        Official PyTorch reference for the underlying numeric operation.
    [`torch.linalg.vector_norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html)
        Clearer vector-only norm API in PyTorch.
    [`torch.linalg.matrix_norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html)
        Clearer matrix-only norm API in PyTorch.

    Behavior
    --------
    The interpretation of `ord` depends on `dim`:

    - `dim` is an `int`: compute a vector norm along that axis.
    - `dim` is a 2-tuple: compute a matrix norm over those two axes.
    - `dim is None`: follow PyTorch's `torch.linalg.norm` rules. In
      particular, `ord=None` flattens the tensor and computes a vector 2-norm,
      while `ord != None` expects PyTorch's documented 1D/2D behavior.

    Supported `ord` values
    ----------------------
    Vector-norm forms (`dim` is an `int`)
    - `None`
    - `0`
    - any finite `int` or `float`
    - `float("inf")`
    - `-float("inf")`

    Matrix-norm forms (`dim` is a 2-tuple)
    - `None`
    - `"fro"`
    - `"nuc"`
    - `1`, `-1`
    - `2`, `-2`
    - `float("inf")`
    - `-float("inf")`

    Parameters
    ----------
    tensor : Tensor
        The tensor to reduce.
    ord : Optional[Union[int, float, str]], optional
        Order of the norm forwarded to `torch.linalg.norm`.

        Common examples:
        - `ord=2` for the Euclidean vector norm or spectral matrix norm
        - `ord=1` for an L1 vector norm or induced 1 matrix norm
        - `ord=float("inf")` for max-based norms
        - `ord="fro"` for the Frobenius matrix norm
        - `ord="nuc"` for the nuclear matrix norm
    dim : Optional[Union[int, Tuple[int, int]]], optional
        Reduction axis or axes.

        - `int`: vector norm
        - `Tuple[int, int]`: matrix norm
        - `None`: use PyTorch's default `torch.linalg.norm` behavior

    Returns
    -------
    TensorType
        Tensor containing the requested norm values with reduced axes removed
        from `dims`.

    Raises
    ------
    IndexError
        If any requested reduction axis is out of range for the tensor rank.
    ValueError
        If `dim` contains duplicate axes.
    """
    reduced = torch.linalg.norm(tensor.data, ord=ord, dim=dim)
    if dim is None:
        return replace(tensor, data=reduced, dims=())

    rank_ = tensor.rank()
    dims_tuple: Tuple[int, ...]
    if isinstance(dim, int):
        dims_tuple = (dim,)
    else:
        dims_tuple = dim

    normalized_dims: list[int] = []
    for d in dims_tuple:
        nd = d
        if nd < 0:
            nd += rank_
        if nd < 0 or nd >= rank_:
            raise IndexError(f"Dimension index {d} out of range for rank {rank_}")
        if nd in normalized_dims:
            raise ValueError("norm dim entries must be unique")
        normalized_dims.append(nd)

    reduced_dims_set = set(normalized_dims)
    new_dims = tuple(
        current_dim
        for idx, current_dim in enumerate(tensor.dims)
        if idx not in reduced_dims_set
    )
    return replace(tensor, data=reduced, dims=new_dims)


def argmax(tensor: TensorType, dim: int) -> TensorType:
    """
    Return indices of maximum values along one tensor axis.

    Parameters
    ----------
    tensor : Tensor
        The tensor to reduce.
    dim : int
        The dimension to reduce.

    Returns
    -------
    TensorType
        Integer-valued tensor with the reduced axis removed from `dims`,
        preserving the input wrapper type.

    Raises
    ------
    IndexError
        If `dim` is out of range for the tensor rank.
    """
    if dim < 0:
        dim += tensor.rank()
    if dim < 0 or dim >= tensor.rank():
        raise IndexError(f"Dimension index {dim} out of range for rank {tensor.rank()}")

    return replace(
        tensor,
        data=tensor.data.argmax(dim=dim),
        dims=tensor.dims[:dim] + tensor.dims[dim + 1 :],
    )


def argmin(tensor: TensorType, dim: int) -> TensorType:
    """
    Return indices of minimum values along one tensor axis.

    Parameters
    ----------
    tensor : Tensor
        The tensor to reduce.
    dim : int
        The dimension to reduce.

    Returns
    -------
    TensorType
        Integer-valued tensor with the reduced axis removed from `dims`,
        preserving the input wrapper type.

    Raises
    ------
    IndexError
        If `dim` is out of range for the tensor rank.
    """
    if dim < 0:
        dim += tensor.rank()
    if dim < 0 or dim >= tensor.rank():
        raise IndexError(f"Dimension index {dim} out of range for rank {tensor.rank()}")

    return replace(
        tensor,
        data=tensor.data.argmin(dim=dim),
        dims=tensor.dims[:dim] + tensor.dims[dim + 1 :],
    )


def one_hot(
    tensor: Tensor[torch.LongTensor], dim: StateSpace
) -> Tensor[torch.LongTensor]:
    """
    One-hot encode an integer-valued tensor using a provided class StateSpace.

    The output appends `dim` as the last axis, and uses `dim.dim` as
    `num_classes`.

    Parameters
    ----------
    tensor : Tensor
        Input tensor containing class indices.
    dim : StateSpace
        Output class dimension. Class indices are assumed to be ordered as
        `[0, 1, ..., dim.dim - 1]`.

    Returns
    -------
    Tensor
        A new tensor with one extra trailing dimension for class channels.

    Raises
    ------
    TypeError
        If `tensor.data` is not integer-valued.
    ValueError
        If any class index is outside the range `0 <= index < dim.dim`.
    """
    if tensor.data.is_floating_point() or tensor.data.is_complex():
        raise TypeError("one_hot expects integer-valued tensor data")

    indices = tensor.data.to(dtype=torch.long)
    if indices.numel() > 0:
        if torch.any(indices < 0) or torch.any(indices >= dim.dim):
            raise ValueError(f"one_hot index out of range for num_classes={dim.dim}")

    return Tensor(
        data=cast(
            torch.LongTensor, torch.nn.functional.one_hot(indices, num_classes=dim.dim)
        ),
        dims=tensor.dims + (dim,),
    )


def astype(tensor: TensorType, dtype: torch.dtype) -> TensorType:
    """
    Return a new tensor with data converted to `dtype`.

    This is the metadata-preserving dtype-conversion helper for QTen tensors.
    Only the underlying torch data dtype changes; `tensor.dims` is left
    unchanged.

    See Also
    --------
    [`torch.Tensor.to`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)
        Underlying PyTorch dtype/device conversion API.

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    dtype : torch.dtype
        Target PyTorch dtype.

        This must be a [`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype)
        object such as `torch.float32`, `torch.float64`, `torch.complex64`,
        `torch.complex128`, `torch.int64`, or `torch.bool`. Any dtype
        accepted by `torch.Tensor.to(dtype=...)` is valid here.

    Returns
    -------
    TensorType
        A new tensor with converted data and unchanged dims, preserving the input wrapper type.
    """
    return replace(tensor, data=tensor.data.to(dtype=dtype))


def allclose(
    a: Tensor,
    b: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Compare two tensors for approximate equality with dimension-aware alignment.

    Supported forms
    ---------------
    [`allclose(a, b, ...)`][qten.linalg.tensors.allclose]
        Functional form.

    [`a.allclose(b, ...)`][qten.linalg.tensors.Tensor.allclose]
        Method form.

    This function first aligns `b` to `a` by calling `b.align_all(a.dims)`.
    If alignment fails (for example, mismatched rank or non-alignable
    [`StateSpace`][qten.symbolics.state_space.StateSpace]s), this function returns `False` instead of raising.
    When alignment succeeds, the function compares data values using
    `torch.allclose`.

    After alignment, comparison is delegated directly to `torch.allclose`,
    preserving native PyTorch behavior for dtype/device handling.

    Parameters
    ----------
    a : Tensor
        Reference tensor defining the target dimension layout.
    b : Tensor
        Tensor that will be aligned to `a` before comparison.
    rtol : float, optional
        Relative tolerance used by `torch.allclose`.
    atol : float, optional
        Absolute tolerance used by `torch.allclose`.
    equal_nan : bool, optional
        Whether `NaN` values are considered equal.

    Returns
    -------
    bool
        True if values are close after successful alignment; `False` if
        alignment fails or values are not close.

    Use cases
    ---------
    - compare tensors that may differ only by symbolic axis ordering,
    - validate numerical results in tests without manually aligning metadata
      first.
    """
    a, b = _promote_to_device(a, b)

    try:
        aligned_b = b.align_all(a.dims)
    except (IndexError, TypeError, ValueError, RuntimeError):
        return False

    return torch.allclose(
        a.data, aligned_b.data, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def isclose(
    a: TensorType,
    b: Union[Tensor, Number],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> TensorType:
    """
    Perform element-wise approximate equality comparison with dimension-aware
    alignment and broadcasting.

    This returns a bool [`Tensor`][qten.linalg.tensors.Tensor] mask, unlike `allclose`, which reduces to a
    Python bool.

    Supported forms
    ---------------
    [`isclose(a, b, ...)`][qten.linalg.tensors.isclose]
        Functional form.

    [`a.isclose(b, ...)`][qten.linalg.tensors.Tensor.isclose]
        Method form.

    Parameter forms
    ---------------
    `b : Tensor`
        Compare two tensors after symbolic alignment and broadcast handling.

    `b : Number`
        The scalar is promoted through `Tensor.scalar(b)` before applying the
        same tensor-tensor comparison logic.

    Parameters
    ----------
    a : TensorType
        Left-hand tensor operand.
    b : Tensor | Number
        Right-hand comparison target. If `b` is a
        [`Tensor`][qten.linalg.tensors.Tensor], it is aligned and broadcast
        against `a`. If `b` is a scalar number, it is promoted through
        `Tensor.scalar(b)` before comparison.
    rtol : float, optional
        Relative tolerance passed to `torch.isclose`.
    atol : float, optional
        Absolute tolerance passed to `torch.isclose`.
    equal_nan : bool, optional
        Whether `NaN` values are considered equal.

    Returns
    -------
    TensorType
        Boolean tensor mask on the merged symbolic output dims.

    See Also
    --------
    [`torch.isclose`](https://pytorch.org/docs/stable/generated/torch.isclose.html)
        Underlying PyTorch element-wise closeness check.

    Use cases
    ---------
    - build boolean masks for thresholding numerical errors,
    - compare a tensor against another tensor or a scalar tolerance target
      while preserving symbolic output dims.
    """
    if not isinstance(b, Tensor):
        b = Tensor.scalar(b)
    return _binary_elementwise_mask_op(
        a,
        b,
        lambda left, right: torch.isclose(
            left, right, rtol=rtol, atol=atol, equal_nan=equal_nan
        ),
    )


def equal(a: Tensor, b: Tensor) -> bool:
    """
    Compare two tensors for exact equality with dimension-aware alignment.

    Supported forms
    ---------------
    [`equal(a, b)`][qten.linalg.tensors.equal]
        Functional form.

    [`a.equal(b)`][qten.linalg.tensors.Tensor.equal]
        Method form.

    This function first aligns `b` to `a` by calling `b.align_all(a.dims)`.
    If alignment fails (for example, mismatched rank or non-alignable
    [`StateSpace`][qten.symbolics.state_space.StateSpace]s), this function returns `False` instead of raising.
    When alignment succeeds, the function compares data values using
    `torch.equal`.

    After alignment, equality is delegated directly to `torch.equal`, preserving
    native PyTorch equality behavior for dtype/device handling.

    Parameters
    ----------
    a : Tensor
        Reference tensor defining the target dimension layout.
    b : Tensor
        Tensor that will be aligned to `a` before comparison.

    Returns
    -------
    bool
        True if values are exactly equal after successful alignment; `False`
        if alignment fails or values are not equal.
    """
    a, b = _promote_to_device(a, b)

    try:
        aligned_b = b.align_all(a.dims)
    except (IndexError, TypeError, ValueError, RuntimeError):
        return False

    return torch.equal(a.data, aligned_b.data)


def expand_to_union(tensor: TensorType, union_dims: list[StateSpace]) -> TensorType:
    """
    Expand broadcast axes to match a merged symbolic dimension layout.

    This helper is typically used after
    [`union_dims`][qten.linalg.tensors.union_dims] determines the target dims
    for a broadcasted operation. Only axes labeled by
    [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace] are
    expanded; concrete non-broadcast axes are preserved as-is.

    Parameters
    ----------
    tensor : TensorType
        Input tensor whose broadcast axes may need to be materialized.
    union_dims : list[StateSpace]
        Target symbolic dimensions, typically produced by
        [`union_dims(...)`][qten.linalg.tensors.union_dims]. This must have the
        same length as `tensor.dims`.

    Returns
    -------
    TensorType
        Tensor whose broadcast axes have been expanded to the corresponding
        concrete sizes in `union_dims`. If no expansion is needed, returns the
        input tensor unchanged.

    Raises
    ------
    ValueError
        If `union_dims` does not match the tensor rank or does not describe a
        layout compatible with the underlying data shape.
    """

    if not any(isinstance(d, BroadcastSpace) for d in tensor.dims):
        return tensor
    target_shape = []
    new_dims = []
    needs_expansion = False

    for dim, u_dim, size in zip(tensor.dims, union_dims, tensor.data.shape):
        if isinstance(dim, BroadcastSpace) and not isinstance(u_dim, BroadcastSpace):
            target_shape.append(u_dim.dim)
            new_dims.append(u_dim)
            needs_expansion = True
        else:
            target_shape.append(size)
            new_dims.append(dim)

    if not needs_expansion:
        return tensor

    return replace(tensor, data=tensor.data.expand(target_shape), dims=tuple(new_dims))


def union_dims(
    *dims: Tuple[StateSpace, ...], allow_merge: bool = False
) -> Tuple[StateSpace, ...]:
    """
    Compute a broadcast-compatible union of multiple dimension tuples.

    Use cases
    ---------
    - determine the target metadata layout for broadcasted arithmetic,
    - validate whether two or more symbolic tensor layouts can participate in a
      common operation,
    - build the merged dims later passed to
      [`align_all`][qten.linalg.tensors.align_all] and
      [`expand_to_union`][qten.linalg.tensors.expand_to_union].

    This function merges dimension metadata axis-by-axis across one or more
    tuples of [`StateSpace`][qten.symbolics.state_space.StateSpace]s. All input tuples must have the same rank.

    Merge rule per axis (`allow_merge=False`)
    -------------------
    - [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] + concrete [`StateSpace`][qten.symbolics.state_space.StateSpace] -> concrete [`StateSpace`][qten.symbolics.state_space.StateSpace]
    - [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] + [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] -> [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace]
    - concrete + concrete:
      - if `same_rays(...)` is `True`, keeps the first (left-most) one
      - otherwise raises `ValueError`

    Merge rule per axis (`allow_merge=True`)
    -----------------------------------------------
    - Uses StateSpace union semantics directly (`left_dim + right_dim`) after
      rank checks. This supports disjoint-axis union behavior used by tensor add.

    Parameters
    ----------
    dims : Tuple[StateSpace, ...]
        One or more dimension tuples to merge.
    allow_merge : bool, optional
        If `False`, enforces strict compatibility for concrete dimensions.
        If `True`, merges concrete dimensions via `+`.

    Returns
    -------
    Tuple[StateSpace, ...]
        Merged dimensions with the same rank as each input tuple.

    Raises
    ------
    ValueError
        If no tuples are provided, ranks differ, or any axis is incompatible
        in strict mode.
    """
    if not dims:
        raise ValueError("union_dims expects at least one dims tuple")

    rank = len(dims[0])
    if any(len(current) != rank for current in dims):
        ranks = ", ".join(str(len(current)) for current in dims)
        raise ValueError(
            f"union_dims requires all dims tuples to have the same rank: got ranks=[{ranks}]"
        )

    merged = list(dims[0])
    for current in dims[1:]:
        for axis, (left_dim, right_dim) in enumerate(zip(merged, current)):
            if not allow_merge:
                if isinstance(left_dim, BroadcastSpace):
                    merged[axis] = right_dim
                    continue
                if isinstance(right_dim, BroadcastSpace):
                    continue
                if same_rays(left_dim, right_dim):
                    continue
                raise ValueError(
                    f"union_dims incompatible at axis {axis}: "
                    f"{type(left_dim).__name__}:{left_dim.dim} vs "
                    f"{type(right_dim).__name__}:{right_dim.dim}; "
                    f"left={_format_dims(tuple(merged))}, right={_format_dims(current)}"
                )
            merged[axis] = cast(StateSpace, left_dim + right_dim)

    return tuple(merged)


def _cat_dim(dims: Sequence[StateSpace]) -> StateSpace:
    if not dims:
        raise ValueError("cat requires at least one dimension to concatenate")

    first = dims[0]
    if isinstance(first, BroadcastSpace):
        raise ValueError("cat does not support concatenation along BroadcastSpace")

    if builtins.all(isinstance(dim, IndexSpace) for dim in dims):
        return IndexSpace.linear(sum(dim.dim for dim in dims))

    if not builtins.all(type(dim) is type(first) for dim in dims):
        raise ValueError("cat dimension types must match across all tensors")

    elements: list[Any] = []
    seen: set[Any] = set()
    for dim in dims:
        for element in dim.elements():
            if element in seen:
                raise ValueError(
                    "cat requires concatenated structured dimensions to be disjoint"
                )
            seen.add(element)
            elements.append(element)

    return type(first)(
        structure=OrderedDict((element, i) for i, element in enumerate(elements))
    )


def cat(tensors: Sequence[TensorType], dim: int = 0) -> TensorType:
    """
    Concatenate tensors along an existing dimension with metadata-aware alignment.

    Non-concatenated dimensions must represent the same rays and are aligned to
    the ordering of the first tensor before concatenation. The concatenated
    output dimension is rebuilt by ordered append. [`IndexSpace`][qten.symbolics.state_space.IndexSpace] dimensions are
    resized linearly; structured dimensions require fully disjoint labels.
    Any overlap at all, even partial overlap, raises `ValueError` rather than
    being merged.

    Parameters
    ----------
    tensors : Sequence[TensorType]
        Tensors to concatenate. All tensors must have the same rank.
    dim : int, default `0`
        Axis along which to concatenate.

    Returns
    -------
    TensorType
        Concatenated tensor whose non-concatenated dims match the first input
        and whose concatenated dim is rebuilt to describe the appended axis.

    Raises
    ------
    ValueError
        If `tensors` is empty, if the input tensors do not all have the same
        rank, if any non-concatenated axes do not span the same rays, or if
        the concatenated symbolic dimension cannot be rebuilt consistently
        (for example because dimension types differ, concatenation is attempted
        along [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace], or
        structured labels overlap in any amount).
    IndexError
        If `dim` is out of range for the input tensor rank.
    """
    if not tensors:
        raise ValueError("cat expects at least one tensor")

    first = tensors[0]
    rank_ = first.rank()
    if dim < 0:
        dim += rank_
    if dim < 0 or dim >= rank_:
        raise IndexError(f"Dimension index {dim} out of range for rank {rank_}")

    for tensor in tensors[1:]:
        if tensor.rank() != rank_:
            raise ValueError("All tensors passed to cat must have the same rank")

    common_dtype = first.data.dtype
    for tensor in tensors[1:]:
        common_dtype = torch.promote_types(common_dtype, tensor.data.dtype)

    promoted = tuple(
        tensor
        if tensor.data.dtype == common_dtype
        else replace(tensor, data=tensor.data.to(common_dtype))
        for tensor in _promote_to_device(*tensors)
    )

    aligned: list[TensorType] = []
    for tensor in promoted:
        current = tensor
        for axis in range(rank_):
            if axis == dim:
                continue
            target_dim = first.dims[axis]
            if not same_rays(current.dims[axis], target_dim):
                raise ValueError(
                    f"All non-concatenated dims must have the same rays; "
                    f"axis {axis} differs between {current.dims[axis]} and {target_dim}"
                )
            current = current.align(axis, target_dim)
        aligned.append(current)

    out_dim = _cat_dim([tensor.dims[dim] for tensor in aligned])
    out_dims = first.dims[:dim] + (out_dim,) + first.dims[dim + 1 :]
    out_data = torch.cat([tensor.data for tensor in aligned], dim=dim)
    return replace(first, data=out_data, dims=out_dims)


def mapping_matrix(
    from_space: StateSpace,
    to_space: StateSpace,
    mapping: Dict[Any, Any],
    factors: Optional[Dict[Tuple[Any, Any], int | float | complex]] = None,
    *,
    device: Optional[Device] = None,
) -> Tensor:
    """
    Create a sector-wise mapping matrix between two state spaces.

    Use cases
    ---------
    - build explicit basis-change or selection matrices between symbolic
      spaces,
    - embed one set of labeled states into another using a sparse symbolic
      correspondence,
    - construct structured linear maps for use in tensor contractions.

    For each `(from_marker, to_marker)` pair in `mapping`, this function inserts
    an identity block from the corresponding sector in `from_space` to the
    corresponding sector in `to_space`. Each block can be scaled by
    `factors[(from_marker, to_marker)]`; if omitted, a factor of `1` is used.

    Parameters
    ----------
    from_space : StateSpace
        Source state space defining the row dimension and source sectors.
    to_space : StateSpace
        Target state space defining the column dimension and target sectors.
    mapping : Dict[Any, Any]
        Dictionary mapping sector markers in `from_space` to sector markers in
        `to_space`. For each entry, a block is written between the slices
        implied by the integer indices in `from_space.structure` and
        `to_space.structure`.
    factors : Optional[Dict[Tuple[Any, Any], int | float | complex]], optional
        Optional per-entry factors. Keys are `(from_marker, to_marker)` tuples.
        Scalar values scale entries. Missing keys default to `1`.

    Returns
    -------
    Tensor
        Rank-2 tensor with dimensions `(from_space, to_space)` containing the
        assembled mapping matrix in complex precision.

    Raises
    ------
    ValueError
        If any mapped source and target sectors have different sizes.
    """
    if factors is None:
        factors = {}

    precision = get_precision_config()
    torch_device = device.torch_device() if device is not None else None
    mat = torch.zeros(
        (from_space.dim, to_space.dim),
        dtype=precision.torch_complex,
        device=torch_device,
    )
    for fm, tm in mapping.items():
        findex = from_space.structure[fm]
        tindex = to_space.structure[tm]
        factor = factors.get((fm, tm), 1)
        mat[findex, tindex] = cast(Any, factor)

    return Tensor(data=mat, dims=(from_space, to_space))


def eye(dims: Tuple[StateSpace, ...], *, device: Optional[Device] = None) -> Tensor:
    """
    Create an identity matrix tensor from the last two symbolic dimensions.

    This helper interprets `dims[-2:]` as the matrix axes and returns a rank-2
    identity tensor on those spaces. Any leading dimensions in `dims` are
    ignored; this function does not build a batched identity tensor.

    Parameters
    ----------
    dims : Tuple[StateSpace, ...]
        Dimension tuple whose last two entries define the row and column
        spaces of the identity matrix.
    device : Optional[Device], optional
        Device on which to place the returned tensor.

    Returns
    -------
    Tensor
        Rank-2 identity tensor with dims `(dims[-2], dims[-1])`.

    Raises
    ------
    ValueError
        If `dims` has rank less than 2.
    """
    if len(dims) < 2:
        raise ValueError(
            f"Identity tensor creation requires at least rank 2, got rank {len(dims)}!"
        )
    matrix_dims = dims[-2:]
    rows = matrix_dims[0].dim
    cols = matrix_dims[1].dim
    torch_device = device.torch_device() if device is not None else None
    return Tensor(data=torch.eye(rows, cols, device=torch_device), dims=matrix_dims)


def zeros(dims: Tuple[StateSpace, ...], *, device: Optional[Device] = None) -> Tensor:
    """
    Create a zero-filled tensor on the requested symbolic dimensions.

    Parameters
    ----------
    dims : Tuple[StateSpace, ...]
        StateSpace dimensions defining the tensor shape.
    device : Optional[Device], optional
        Device to place the tensor on, by default None (CPU).

    Returns
    -------
    Tensor
        Tensor of zeros with `shape == tuple(dim.dim for dim in dims)` and
        dims equal to `dims`.
    """
    shape = tuple(dim.dim for dim in dims)
    torch_device = device.torch_device() if device is not None else None
    return Tensor(data=torch.zeros(shape, device=torch_device), dims=dims)


def ones(dims: Tuple[StateSpace, ...], *, device: Optional[Device] = None) -> Tensor:
    """
    Create a one-filled tensor on the requested symbolic dimensions.

    Parameters
    ----------
    dims : Tuple[StateSpace, ...]
        StateSpace dimensions defining the tensor shape.
    device : Optional[Device], optional
        Device to place the tensor on, by default None (CPU).

    Returns
    -------
    Tensor
        Tensor of ones with `shape == tuple(dim.dim for dim in dims)` and dims
        equal to `dims`.
    """
    shape = tuple(dim.dim for dim in dims)
    torch_device = device.torch_device() if device is not None else None
    return Tensor(data=torch.ones(shape, device=torch_device), dims=dims)


def kernel_tensor(
    ker: Callable[..., Number],
    dims: Tuple[StateSpace, ...],
    *,
    device: Optional[Device] = None,
) -> Tensor[torch.Tensor]:
    """
    Build a tensor by evaluating a scalar-valued kernel over StateSpace elements.

    For each multi-index `(i0, i1, ..., iN)` this evaluates:
    `ker(dims[0].elements()[i0], dims[1].elements()[i1], ..., dims[N].elements()[iN])`
    and stores the result at that tensor position.

    Parameters
    ----------
    ker : Callable[..., Number]
        Scalar-valued callable that accepts one element from each state space in
        `dims`.
    dims : Tuple[StateSpace, ...]
        Output tensor dimensions.

    Returns
    -------
    Tensor
        Tensor with `dims` and values produced by `ker`.

    Notes
    -----
    This is the most direct construction helper when tensor entries are defined
    by symbolic basis elements rather than by raw numeric arrays.

    Raises
    ------
    ValueError
        If any state space reports a number of elements different from its
        declared `dim`.
    """
    torch_device = device.torch_device() if device is not None else None
    if not dims:
        return Tensor(data=torch.as_tensor(ker(), device=torch_device), dims=dims)

    element_axes = tuple(dim.elements() for dim in dims)
    for axis, dim in zip(element_axes, dims):
        if len(axis) != dim.dim:
            raise ValueError(
                f"kernel_tensor expects one element per index for each StateSpace; "
                f"got len(elements)={len(axis)} and dim={dim.dim} for {type(dim).__name__}"
            )

    values = [ker(*args) for args in product(*element_axes)]
    data = torch.as_tensor(values, device=torch_device).reshape(
        *(len(axis) for axis in element_axes)
    )
    return Tensor(data=data, dims=dims)


def _normalize_dim_index(rank: int, dim: int) -> int:
    if dim < 0:
        dim += rank

    if dim < 0 or dim >= rank:
        raise IndexError(
            f"Dimension index {dim} out of range for tensor of rank {rank}"
        )

    return dim


def replace_dim(tensor: TensorType, dim: int, new_dim: StateSpace) -> TensorType:
    """
    Replace one symbolic dimension without changing tensor data values.

    This is a metadata-only operation. It is valid only when `new_dim` has the
    same size as the underlying data axis (or is
    [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace] for a
    singleton axis).

    Parameters
    ----------
    tensor : Tensor
        The tensor to modify.
    dim : int
        The index of the dimension to replace.
    new_dim : StateSpace
        The new StateSpace to assign to the dimension.

    Returns
    -------
    TensorType
        Tensor with unchanged data and the requested replacement dimension.

    Raises
    ------
    IndexError
        If `dim` is out of range for the tensor rank.
    ValueError
        If `new_dim` is not size-compatible with the selected data axis.
    """
    dim = _normalize_dim_index(len(tensor.dims), dim)

    current_size = tensor.data.shape[dim]

    # Check size compatibility.
    # BroadcastSpace represents a singleton axis and only matches size 1.
    if isinstance(new_dim, BroadcastSpace):
        if current_size != 1:
            raise ValueError(
                f"Cannot replace dimension of size {current_size} with BroadcastSpace (expects size 1)."
            )
    elif new_dim.dim != current_size:
        raise ValueError(
            f"New StateSpace size {new_dim.dim} does not match tensor data size {current_size} at dimension {dim}!"
        )

    new_dims = list(tensor.dims)
    new_dims[dim] = new_dim
    return replace(tensor, dims=tuple(new_dims))


def update_dim(
    tensor: TensorType,
    dim: int,
    func: Callable[[StateSpace], StateSpace],
) -> TensorType:
    """
    Update one symbolic dimension by applying a callback to the current metadata.

    This is a metadata-only operation. The callback is applied to
    `tensor.dims[dim]`, and the returned StateSpace is then validated through
    [`replace_dim`][qten.linalg.tensors.replace_dim].

    Parameters
    ----------
    tensor : Tensor
        The tensor to modify.
    dim : int
        The index of the dimension to update.
    func : Callable[[StateSpace], StateSpace]
        Callback that maps the current StateSpace to a replacement.

    Returns
    -------
    TensorType
        Tensor with unchanged data and the requested updated dimension.

    Raises
    ------
    IndexError
        If `dim` is out of range for the tensor rank.
    ValueError
        If the callback returns a size-incompatible StateSpace.
    """
    dim = _normalize_dim_index(len(tensor.dims), dim)
    return replace_dim(tensor, dim, func(tensor.dims[dim]))


def factorize_dim(
    tensor: TensorType, dim: int, rule: StateSpaceFactorization
) -> TensorType:
    """
    Factorize one symbolic dimension into multiple output axes.

    For a tensor with shape `(A, B)`, factorizing the `0`th dimension with
    `factorized=(A1, A2)` and a compatible `align_dim` produces shape
    `(A1, A2, B)`.

    The source axis is first aligned to `rule.align_dim`, then reshaped into
    the factor spaces listed in `rule.factorized`.

    How to construct `rule`
    -----------------------
    [`StateSpaceFactorization`][qten.symbolics.state_space.StateSpaceFactorization]
    has two fields:

    - `factorized`: the tuple of output spaces that should replace the selected
      axis after reshaping.
    - `align_dim`: a symbolic dimension with the same total size as the input
      axis, but with an ordering compatible with flattening/unflattening into
      `factorized`.

    The key requirement is:
    `prod(subspace.dim for subspace in rule.factorized) == rule.align_dim.dim`.

    In practice, you usually do not construct this object by hand. When the
    target axis is a
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace], the usual
    workflow is to derive the rule from
    [`HilbertSpace.factorize(...)`][qten.symbolics.hilbert_space.HilbertSpace.factorize].

    Use cases
    ---------
    - split a flattened composite axis into its constituent symbolic factors,
    - recover tensor-product structure after linear-algebra operations that
      temporarily flattened an axis.

    Example
    -------
    ```python
    space = tensor.dims[0]  # suppose this is a homogeneous HilbertSpace
    rule = space.factorize((int,), (str,))

    # `rule.factorized` is the tuple of output factor spaces.
    # `rule.align_dim` is the reordered version of `space` whose basis order is
    # compatible with reshaping into those factors.

    out = factorize_dim(tensor, 0, rule)
    ```

    If you want to see the equivalent low-level structure, the rule behaves
    like:

    ```python
    left_factor, right_factor = rule.factorized
    align_dim = rule.align_dim

    # Equivalent internal steps:
    aligned = tensor.align(0, align_dim)
    out = factorize_dim(tensor, 0, rule)
    # out.dims == (left_factor, right_factor, *tensor.dims[1:])
    ```

    Parameters
    ----------
    tensor : Tensor
        The tensor to modify.
    dim : int
        The index of the dimension to factorize.
    rule : StateSpaceFactorization
        The factorization rule.

    Returns
    -------
    TensorType
        Tensor with the requested axis replaced by multiple factor axes,
        preserving the input wrapper type.

    Raises
    ------
    IndexError
        If `dim` is out of range for the tensor rank.
    ValueError
        If the product of `rule.factorized` sizes does not match
        `rule.align_dim.dim`, or if the selected axis cannot be aligned to
        `rule.align_dim`.
    """
    rank = tensor.rank()
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise IndexError(f"Dimension index {dim} out of range for rank {rank}")

    align_dim = rule.align_dim
    factorized_sizes = tuple(d.dim for d in rule.factorized)
    expected_size = 1
    for s in factorized_sizes:
        expected_size *= s
    if expected_size != align_dim.dim:
        raise ValueError(
            f"Factorized dimensions product {expected_size} does not match "
            f"align_dim size {align_dim.dim}"
        )

    aligned = tensor.align(dim, align_dim)
    new_shape = (
        tuple(aligned.data.shape[:dim])
        + factorized_sizes
        + tuple(aligned.data.shape[dim + 1 :])
    )
    new_data = aligned.data.reshape(new_shape)
    new_dims = tensor.dims[:dim] + rule.factorized + tensor.dims[dim + 1 :]
    return replace(tensor, data=new_data, dims=new_dims)


def _product_dims_normalize_groups(
    rank: int, indices_group: Tuple[Tuple[int, ...], ...]
) -> Tuple[list[Tuple[int, ...]], set[int]]:
    normalized_groups: list[Tuple[int, ...]] = []
    grouped_indices: set[int] = set()

    for group_idx, group in enumerate(indices_group):
        if len(group) == 0:
            raise ValueError("Each indices_group entry must be non-empty")

        normalized: list[int] = []
        seen_in_group = set()
        for idx in group:
            if idx < 0:
                idx += rank
            if idx < 0 or idx >= rank:
                raise IndexError(f"Dimension index {idx} out of range for rank {rank}")
            if idx in seen_in_group:
                raise ValueError(f"Duplicate index {idx} in group {group_idx}")
            if idx in grouped_indices:
                raise ValueError(f"Dimension index {idx} appears in multiple groups")
            seen_in_group.add(idx)
            grouped_indices.add(idx)
            normalized.append(idx)
        normalized_groups.append(tuple(normalized))

    return normalized_groups, grouped_indices


def _product_dims_build_slots(
    rank: int, normalized_groups: list[Tuple[int, ...]], grouped_indices: set[int]
) -> list[Tuple[bool, Tuple[int, ...]]]:
    groups_by_anchor = {min(group): group for group in normalized_groups}
    slots: list[Tuple[bool, Tuple[int, ...]]] = []
    for idx in range(rank):
        group = groups_by_anchor.get(idx)
        if group is not None:
            slots.append((True, group))
        elif idx not in grouped_indices:
            slots.append((False, (idx,)))
    return slots


def _format_dims(dims: Tuple[StateSpace, ...]) -> str:
    if not dims:
        return "()"
    return "(" + ", ".join(f"{type(dim).__name__}:{dim.dim}" for dim in dims) + ")"


def product_dims(tensor: TensorType, *indices_group: Tuple[int, ...]) -> TensorType:
    """
    Combine selected tensor dimensions into product dimensions.

    Each entry in `indices_group` defines one output product dimension.
    For a group `(i0, i1, ..., ik)`, the returned tensor contains a single
    axis whose size is the product of the grouped axis sizes and whose
    [`StateSpace`][qten.symbolics.state_space.StateSpace] is `self.dims[i0] @ self.dims[i1] @ ... @ self.dims[ik]`.
    Dimensions not listed in any group are preserved as-is.

    Negative indices are supported and follow Python indexing rules.
    Grouped dimensions do not need to be contiguous in the input tensor; the
    method reorders axes internally, performs one reshape, and returns the
    result in the canonical output order.

    Use cases
    ---------
    - flatten several symbolic axes into one composite Hilbert or state space,
    - prepare tensors for matrix decompositions or contractions that expect a
      single product axis.

    Parameters
    ----------
    tensor : Tensor
        The tensor to modify.
    indices_group : Tuple[int, ...]
        One or more non-empty groups of dimension indices to combine.
        Indices must be unique across all groups (a dimension can belong to
        at most one group).

    Returns
    -------
    TensorType
        A new tensor where each requested group is replaced by one product
        dimension and all non-grouped dimensions are retained.

    Raises
    ------
    IndexError
        If any provided index is out of range for the tensor rank.
    ValueError
        If any group is empty, if a group contains duplicate indices, or if
        the same index appears in more than one group.
    """
    if not indices_group:
        return tensor

    rank = tensor.rank()
    normalized_groups, grouped_indices = _product_dims_normalize_groups(
        rank, indices_group
    )
    slots = _product_dims_build_slots(rank, normalized_groups, grouped_indices)

    permute_order = tuple(idx for _, group in slots for idx in group)
    permuted = tensor.permute(permute_order)

    new_shape: list[int] = []
    new_dims: list[StateSpace] = []
    cursor = 0

    def _accumulate_group_size(acc: int, offset: int) -> int:
        return acc * permuted.data.shape[cursor + offset]

    def _tensor_product_dims(acc: StateSpace, g_idx: int) -> StateSpace:
        return _kron_dim(acc, tensor.dims[g_idx])

    for is_grouped, group in slots:
        if is_grouped:
            combined_size = reduce(_accumulate_group_size, range(len(group)), 1)
            combined_dim = reduce(
                _tensor_product_dims, group[1:], tensor.dims[group[0]]
            )
            new_shape.append(combined_size)
            new_dims.append(cast(StateSpace, combined_dim))
        else:
            idx = group[0]
            new_shape.append(permuted.data.shape[cursor])
            new_dims.append(tensor.dims[idx])
        cursor += len(group)

    return replace(
        tensor, data=permuted.data.reshape(tuple(new_shape)), dims=tuple(new_dims)
    )


def promote_rank(tensor: Tensor, target_rank: int) -> Tensor:
    """
    Return `tensor` with leading broadcast axes prepended to reach `target_rank`.

    This function preserves the existing axis order and values while adding
    `target_rank - tensor.rank()` leading singleton axes in `tensor.data`.
    The corresponding leading entries in `dims` are [`BroadcastSpace()`][qten.symbolics.state_space.BroadcastSpace].

    Parameters
    ----------
    tensor : Tensor
        Input tensor.
    target_rank : int
        Desired output rank. Must satisfy `target_rank >= tensor.rank()`.

    Returns
    -------
    Tensor
        `tensor` if no promotion is needed; otherwise a tensor with prepended
        broadcast axes and matching prepended [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] dims.

    Use cases
    ---------
    - normalize ranks before symmetric binary operations,
    - make scalar/vector/tensor operands participate in one broadcast-aware
      code path without losing symbolic meaning.

    Raises
    ------
    ValueError
        If `target_rank < tensor.rank()`.
    """
    current_rank = tensor.rank()
    if target_rank < current_rank:
        raise ValueError(
            f"Cannot promote rank {current_rank} tensor to lower target rank {target_rank}"
        )
    if target_rank == current_rank:
        return tensor

    prepend_count = target_rank - current_rank
    new_dims = (BroadcastSpace(),) * prepend_count + tensor.dims
    new_shape = (1,) * prepend_count + tuple(tensor.data.shape)
    return Tensor(data=tensor.data.reshape(new_shape), dims=new_dims)


@multimethod
def _where(condition: Tensor, input: Tensor, other: Tensor) -> Tensor:
    """
    Implementation for `Tensor.where(input, other)`.
    """
    if condition.data.dtype != torch.bool:
        raise TypeError("where expects condition.data to have dtype torch.bool")

    condition, input, other = _promote_to_device(condition, input, other)

    target_rank = max(condition.rank(), input.rank(), other.rank())
    condition = promote_rank(condition, target_rank)
    input = promote_rank(input, target_rank)
    other = promote_rank(other, target_rank)

    merged_dims = union_dims(condition.dims, input.dims, other.dims, allow_merge=False)
    condition = condition.align_all(merged_dims).expand_to_union(list(merged_dims))
    input = input.align_all(merged_dims).expand_to_union(list(merged_dims))
    other = other.align_all(merged_dims).expand_to_union(list(merged_dims))
    return Tensor(
        data=torch.where(condition.data, input.data, other.data),
        dims=merged_dims,
    )


@_where.register
def _(
    condition: Tensor, index_type: Type[Any] = Tensor
) -> Union[
    Tuple[Tensor, ...],
    Tuple[Tuple[int, ...], ...],
    StateSpace,
]:
    """
    Return the nonzero locations of a boolean condition tensor.

    Parameters
    ----------
    condition : Tensor
        Boolean tensor whose nonzero entries are to be reported.
    index_type : Type[Any], optional
        Output representation. Supported values are:

        - [`Tensor`][qten.linalg.tensors.Tensor]: return one integer index tensor per axis,
        - `tuple` / `Tuple`: return Python index tuples,
        - [`StateSpace`][qten.symbolics.state_space.StateSpace]: for rank-1 conditions, return the selected subspace.

    Returns
    -------
    Tuple[Tensor, ...] | Tuple[Tuple[int, ...], ...] | StateSpace
        Nonzero indices encoded according to `index_type`.

    Raises
    ------
    TypeError
        If `condition` is not boolean or `index_type` is unsupported.
    ValueError
        If `index_type is StateSpace` but `condition` is not rank 1.
    """
    if condition.data.dtype != torch.bool:
        raise TypeError("where expects condition.data to have dtype torch.bool")

    rows = torch.nonzero(condition.data, as_tuple=False)
    indices = torch.where(condition.data)
    nnz = indices[0].numel() if len(indices) > 0 else 0
    index_dim = IndexSpace.linear(nnz)
    origin = get_origin(index_type)
    if index_type is Tensor:
        return tuple(Tensor(data=idx, dims=(index_dim,)) for idx in indices)
    if index_type is tuple or origin is tuple:
        return tuple(tuple(int(v) for v in row.tolist()) for row in rows)
    if index_type is StateSpace:
        if condition.rank() != 1:
            raise ValueError(
                "StateSpace index output is only supported for rank-1 conditions"
            )
        selected = [int(row[0].item()) for row in rows]
        return condition.dims[0][selected]
    raise TypeError("index_type must be one of Tensor, tuple/Tuple, or StateSpace")


def where(*args, **kwargs):
    """
    Dispatch to the overloaded public [`where`][qten.linalg.tensors.where]
    implementations.

    Supported forms
    ---------------
    [`where(condition, input, other)`][qten.linalg.tensors.where]
        Element-wise selection, analogous to `torch.where(condition, input, other)`.

    [`where(condition, index_type=Tensor)`][qten.linalg.tensors.where]
        Return nonzero locations as one integer [`Tensor`][qten.linalg.tensors.Tensor]
        per axis.

    [`where(condition, index_type=tuple)`][qten.linalg.tensors.where]
        Return nonzero locations as Python coordinate tuples.

    [`where(condition, index_type=StateSpace)`][qten.linalg.tensors.where]
        For rank-1 conditions only, return the selected
        [`StateSpace`][qten.symbolics.state_space.StateSpace].

    This wrapper exists so multimethod dispatch errors can be translated into
    user-facing `TypeError` exceptions when the underlying implementation
    rejects a particular call signature.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to the overloaded `where` variants.
    **kwargs : Any
        Keyword arguments forwarded to the overloaded `where` variants.

    Returns
    -------
    Any
        Result produced by the matching overloaded `where` implementation.

    Raises
    ------
    TypeError
        If multimethod dispatch fails because the matched implementation raised
        `TypeError`.
    DispatchError
        If dispatch fails for another reason.
    """
    try:
        return _where(*args, **kwargs)
    except DispatchError as ex:
        if isinstance(ex.__cause__, TypeError):
            raise ex.__cause__
        raise


def nonzero(
    condition: Tensor, as_tuple: bool = True, index_type: Type[Any] = Tensor
) -> Union[
    Tuple[Tensor, ...],
    Tuple[Tuple[int, ...], ...],
    StateSpace,
]:
    """
    Return indices of non-zero or `True` entries.

    This currently supports only `as_tuple=True` and follows
    `torch.nonzero(condition.data, as_tuple=True)` semantics.

    Supported forms
    ---------------
    [`nonzero(condition)`][qten.linalg.tensors.nonzero]
        Return one integer [`Tensor`][qten.linalg.tensors.Tensor] per axis.

    [`nonzero(condition, index_type=tuple)`][qten.linalg.tensors.nonzero]
        Return Python coordinate tuples.

    [`nonzero(condition, index_type=StateSpace)`][qten.linalg.tensors.nonzero]
        For rank-1 conditions only, return the selected symbolic subspace.

    Parameters
    ----------
    condition : Tensor
        Input tensor.
    as_tuple : bool, optional
        Must be `True`.
    index_type : Type[Any], optional
        Requested index representation. Supported values are [`Tensor`][qten.linalg.tensors.Tensor],
        `tuple` / `Tuple`, and [`StateSpace`][qten.symbolics.state_space.StateSpace].

    Returns
    -------
    Union[Tuple[Tensor, ...], Tuple[Tuple[int, ...], ...], StateSpace]
        Index representation selected by `index_type`.

    Notes
    -----
    This is the public nonzero helper. [`where(condition)`][qten.linalg.tensors.where]
    provides the same index extraction behavior through the overloaded `where`
    API.

    Raises
    ------
    NotImplementedError
        If `as_tuple` is `False`.
    """
    if not as_tuple:
        raise NotImplementedError("nonzero currently supports only as_tuple=True")

    rows = torch.nonzero(condition.data, as_tuple=False)
    indices = torch.nonzero(condition.data, as_tuple=True)
    nnz = indices[0].numel() if len(indices) > 0 else 0
    index_dim = IndexSpace.linear(nnz)
    origin = get_origin(index_type)
    if index_type is Tensor:
        return tuple(Tensor(data=idx, dims=(index_dim,)) for idx in indices)
    if index_type is tuple or origin is tuple:
        return tuple(tuple(int(v) for v in row.tolist()) for row in rows)
    if index_type is StateSpace:
        if condition.rank() != 1:
            raise ValueError(
                "StateSpace index output is only supported for rank-1 conditions"
            )
        selected = [int(row[0].item()) for row in rows]
        return condition.dims[0][selected]
    raise TypeError("index_type must be one of Tensor, tuple/Tuple, or StateSpace")


TensorIndexType: TypeAlias = Union[
    int, slice, EllipsisType, None, Convertible, StateSpace, Tensor
]
"""
Public key token types accepted by `TensorIndexing` input keys.

This includes qten-level index forms (`StateSpace`, `Convertible`,
`Tensor`) and Python indexing tokens (`int`, `slice`, `None`, `...`).
"""

TorchIndexType: TypeAlias = Union[
    int, slice, EllipsisType, None, torch.Tensor, Tuple[int, ...]
]
"""
Low-level compiled index token types that can be executed against torch tensors.

In addition to standard tokens, `Tuple[int, ...]` is used for
permutation/embedding style StateSpace indexing steps.
"""


class TensorIndexing:
    """
    Compile mixed indexing keys into torch indices plus [`StateSpace`][qten.symbolics.state_space.StateSpace] metadata.

    This class is the single indexing compiler used by `Tensor.__getitem__`.
    It accepts raw key tokens (including `...`) and produces:

    - `indices`: a tuple consumable by torch indexing for one-shot execution.
    - `dims`: output [`StateSpace`][qten.symbolics.state_space.StateSpace] metadata after indexing.
    - `indices_steps`: per-token executable steps for sequential indexing.
    - `has_tensor_index`: whether advanced tensor-index mode is active.

    Key token types
    ---------------
    Supported input tokens in `indices`:
    - `int`
    - `slice`
    - `None`
    - `Ellipsis`
    - [`StateSpace`][qten.symbolics.state_space.StateSpace]
    - [`Convertible`][qten.abstracts.Convertible] (converted to [`StateSpace`][qten.symbolics.state_space.StateSpace])
    - [`Tensor`][qten.linalg.tensors.Tensor] (qten tensor index)

    Normalization rules
    -------------------
    1. Expand at most one ellipsis (`...`) into the required number of full
       slices `:`, where only non-`None` tokens consume source axes.
    2. Append trailing full slices when fewer source-axis-consuming tokens than
       tensor rank are provided.
    3. Reject keys with too many source-axis-consuming tokens.
    4. In tensor-index mode, left-pad lower-rank tensor indices with leading
       [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axes (via unsqueeze) so all tensor indices share one
       rank before `tensor_union_dims` is computed.

    Per-token compile rules
    -----------------------
    - `int`: consumes one source axis; removes that axis from output metadata.
    - `slice`:
      - full `:` preserves the source axis [`StateSpace`][qten.symbolics.state_space.StateSpace],
      - non-full uses `dim[slice]`, except for [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] where
        output metadata follows sliced size ([`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] for size `1`,
        `IndexSpace.linear(0)` for size `0`).
    - `None`: inserts one [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] axis; consumes no source axis.
    - [`StateSpace`][qten.symbolics.state_space.StateSpace] / [`Convertible`][qten.abstracts.Convertible]:
      - on [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] source axes, only [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] is accepted;
        other [`StateSpace`][qten.symbolics.state_space.StateSpace] indices raise `IndexError`,
      - equal space -> full-slice behavior,
      - same span -> permutation index (`Tuple[int, ...]`),
      - contained subspace -> embedding index (`Tuple[int, ...]`),
      - otherwise raises `IndexError`.
    - [`Tensor`][qten.linalg.tensors.Tensor]:
      - bool dtype is unsupported (`NotImplementedError`),
      - lower-rank tensor indices are internally left-padded to the maximum
        tensor-index rank before union/alignment,
      - aligns to `tensor_union_dims` (union over all tensor-index dims),
      - consumes one source axis and contributes advanced metadata dims.

    Mode and compatibility rules
    ----------------------------
    - Mixed [`Tensor`][qten.linalg.tensors.Tensor] with [`StateSpace`][qten.symbolics.state_space.StateSpace]/[`Convertible`][qten.abstracts.Convertible] in one key is rejected
      (`ValueError`).
    - Tensor-index mode:
      - uses torch advanced indexing semantics for data access,
      - output metadata places `tensor_union_dims` at contiguous tensor block
        position, or at the front if tensor indices are separated.
    - Non-tensor mode:
      - preserves per-axis semantics using `indices_steps`, allowing caller-side
        sequential execution to avoid unintended cross-axis advanced broadcast.

    Notes
    -----
    This class compiles indexing intent; execution strategy (one-shot vs
    stepwise) is chosen by `Tensor.__getitem__` using `has_tensor_index`.

    Most users should not construct this class directly. It exists as the
    implementation detail behind QTen tensor indexing and is documented here so
    the generated API explains how symbolic index forms are interpreted.
    """

    def __init__(
        self, dims: Tuple[StateSpace, ...], indices: Tuple[TensorIndexType, ...]
    ):
        self.dims = dims
        self.rank = len(dims)

        promoted_indices: list[TensorIndexType] = self._promote_tensor_indices(indices)
        self.indices = tuple(promoted_indices)
        self.non_none_indices = tuple(idx for idx in self.indices if idx is not None)

        tensor_indices = tuple(
            cast(Tensor, idx)
            for idx in self.indices
            if isinstance(idx, Tensor) and idx.data.dtype != torch.bool
        )
        self.tensor_union_dims = (
            union_dims(*(idx.dims for idx in tensor_indices), allow_merge=False)
            if tensor_indices
            else ()
        )

    @staticmethod
    def _promote_tensor_indices(
        indices: Tuple[TensorIndexType, ...],
    ) -> list[TensorIndexType]:
        promoted: list[TensorIndexType] = list(indices)
        tensor_positions = [
            i
            for i, idx in enumerate(indices)
            if isinstance(idx, Tensor) and idx.data.dtype != torch.bool
        ]
        if tensor_positions:
            tensor_entries = tuple(cast(Tensor, indices[i]) for i in tensor_positions)
            max_tensor_rank = max(idx.rank() for idx in tensor_entries)
            for pos in tensor_positions:
                promoted[pos] = promote_rank(
                    cast(Tensor, indices[pos]),
                    max_tensor_rank,
                )
        return promoted

    def _normalize(self) -> Tuple[TensorIndexType, ...]:
        return self._pad_missing_slices(self._expand_ellipsis())

    @staticmethod
    def _consumed_axes(idx: TensorIndexType) -> int:
        if idx is None:
            return 0
        if isinstance(idx, Tensor) and idx.data.dtype == torch.bool:
            return idx.rank()
        return 1

    def _expand_ellipsis(self) -> Tuple[TensorIndexType, ...]:
        if not any(idx is Ellipsis for idx in self.indices):
            # No need to expand if ellipsis is not present
            return self.indices
        ellipsis_count = sum(idx is Ellipsis for idx in self.indices)
        if ellipsis_count > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        # Split the indices around the ellipsis
        ellipsis_pos = next(i for i, idx in enumerate(self.indices) if idx is Ellipsis)
        left_indices = self.indices[:ellipsis_pos]
        right_indices = self.indices[ellipsis_pos + 1 :]

        # Calculate how many dimensions the ellipsis should expand to
        consumed = sum(self._consumed_axes(idx) for idx in left_indices + right_indices)
        num_full_slices = self.rank - consumed
        return left_indices + (slice(None),) * num_full_slices + right_indices

    def _pad_missing_slices(
        self, indices: Tuple[TensorIndexType, ...]
    ) -> Tuple[TensorIndexType, ...]:
        consumed = sum(self._consumed_axes(idx) for idx in indices)
        if consumed > self.rank:
            raise IndexError("Too many indices for tensor")
        if consumed < self.rank:
            return indices + (slice(None),) * (self.rank - consumed)
        return indices

    @multimethod
    def _compile(self, idx: int, v: int) -> Tuple[int, Tuple[StateSpace, ...], int]:
        return idx + 1, tuple(), v

    @_compile.register
    def _(self, idx: int, v: slice) -> Tuple[int, Tuple[StateSpace, ...], slice]:
        dim = self.dims[idx]
        # Check if its a full slice `:`
        if v.start is None and v.stop is None and v.step is None:
            return idx + 1, (dim,), v

        if isinstance(dim, BroadcastSpace):
            # BroadcastSpace always reports dim==1, but non-full slicing can
            # produce an empty axis at runtime (e.g., unsqueeze(0)[1:]).
            out_size = len(range(dim.dim)[v])
            out_dim: StateSpace = dim if out_size == 1 else IndexSpace.linear(out_size)
            return idx + 1, (out_dim,), v

        return idx + 1, (dim[v],), v

    @_compile.register
    def _(self, idx: int, _: None) -> Tuple[int, Tuple[StateSpace, ...], None]:
        return idx, (BroadcastSpace(),), None

    @_compile.register
    def _(
        self, idx: int, v: StateSpace
    ) -> Tuple[int, Tuple[StateSpace, ...], Union[Tuple[int, ...], slice]]:
        dim = self.dims[idx]

        if isinstance(dim, BroadcastSpace):
            if isinstance(v, BroadcastSpace):
                return idx + 1, (dim,), slice(None)
            raise IndexError(
                "Cannot index a BroadcastSpace axis with StateSpace metadata. "
                f"BroadcastSpace represents a singleton broadcast axis without "
                f"concrete basis elements; received {type(v).__name__}(dim={v.dim}). "
                "Use slice/int/None indexing instead."
            )

        if dim == v:
            return idx + 1, (dim,), slice(None)
        if same_rays(dim, v):
            return idx + 1, (v,), permutation_order(dim, v)
        if v in dim:
            return idx + 1, (v,), embedding_order(v, dim)

        raise IndexError(
            f"Unable to index dimension with {v} that is not contained in {dim}"
        )

    @_compile.register
    def _(
        self, idx: int, v: Convertible
    ) -> Tuple[int, Tuple[StateSpace, ...], Union[Tuple[int, ...], slice]]:
        return self._compile(idx, v.convert(StateSpace))

    @_compile.register
    def _(
        self, idx: int, v: Tensor
    ) -> Tuple[int, Tuple[StateSpace, ...], torch.Tensor]:
        if v.data.dtype == torch.bool:
            target_dims = self.dims[idx : idx + v.rank()]
            v = v.align_all(target_dims)
            nnz = v.data.count_nonzero().item()
            if v.rank() == 1:
                out_dim = v.nonzero(index_type=StateSpace)
            else:
                out_dim = IndexSpace.linear(int(nnz))
            return idx + v.rank(), (out_dim,), v.data

        v = v.align_all(self.tensor_union_dims)
        return idx + 1, self.tensor_union_dims, v.data

    class CompiledIndices(NamedTuple):
        indices: Tuple[TorchIndexType, ...]
        dims: Tuple[StateSpace, ...]
        indices_steps: Tuple[Union[int, slice, None, Tuple[int, ...]], ...]
        has_tensor_index: bool

    def _compile_entries(
        self, normalized_indices: Tuple[TensorIndexType, ...]
    ) -> Tuple[
        list[Tuple[TensorIndexType, Tuple[StateSpace, ...], TorchIndexType]],
        Tuple[TorchIndexType, ...],
    ]:
        entries: list[
            Tuple[TensorIndexType, Tuple[StateSpace, ...], TorchIndexType]
        ] = []
        torch_indices: Tuple[TorchIndexType, ...] = tuple()
        n = 0
        for idx in normalized_indices:
            n, compiled_dim, compiled_idx = self._compile(n, idx)
            entries.append((idx, compiled_dim, compiled_idx))
            torch_indices += (compiled_idx,)
        return entries, torch_indices

    def _compose_compiled_dims(
        self,
        entries: list[Tuple[TensorIndexType, Tuple[StateSpace, ...], TorchIndexType]],
    ) -> Tuple[StateSpace, ...]:
        tensor_positions_all = [
            i for i, (idx, _, _) in enumerate(entries) if isinstance(idx, Tensor)
        ]
        if len(tensor_positions_all) == 0:
            return tuple(dim for _, dims, _ in entries for dim in dims)

        has_bool_tensor = any(
            isinstance(idx, Tensor) and idx.data.dtype == torch.bool
            for idx, _, _ in entries
        )
        if has_bool_tensor and len(tensor_positions_all) > 1:
            advanced_dims = self._mixed_bool_advanced_dims(entries)
            first_tensor_pos = tensor_positions_all[0]
            last_tensor_pos = tensor_positions_all[-1]

            if last_tensor_pos - first_tensor_pos + 1 != len(tensor_positions_all):
                non_tensor_dims = tuple(
                    dim
                    for idx, dims, _ in entries
                    if not isinstance(idx, Tensor)
                    for dim in dims
                )
                return advanced_dims + non_tensor_dims

            compiled_dims: Tuple[StateSpace, ...] = tuple()
            for i, (idx, dims, _) in enumerate(entries):
                if i == first_tensor_pos:
                    compiled_dims += advanced_dims
                if isinstance(idx, Tensor):
                    continue
                compiled_dims += dims
            return compiled_dims

        tensor_positions = [
            i
            for i, (idx, _, _) in enumerate(entries)
            if isinstance(idx, Tensor) and idx.data.dtype != torch.bool
        ]
        if len(tensor_positions) == 0:
            return tuple(dim for _, dims, _ in entries for dim in dims)

        first_tensor_pos = tensor_positions[0]
        last_tensor_pos = tensor_positions[-1]
        if last_tensor_pos - first_tensor_pos + 1 != len(tensor_positions):
            non_tensor_dims = tuple(
                dim
                for idx, dims, _ in entries
                if not (isinstance(idx, Tensor) and idx.data.dtype != torch.bool)
                for dim in dims
            )
            return self.tensor_union_dims + non_tensor_dims

        compiled_dims: Tuple[StateSpace, ...] = tuple()
        for i, (idx, dims, _) in enumerate(entries):
            if i == first_tensor_pos:
                compiled_dims += self.tensor_union_dims
            if isinstance(idx, Tensor) and idx.data.dtype != torch.bool:
                continue
            compiled_dims += dims
        return compiled_dims

    def _mixed_bool_advanced_dims(
        self,
        entries: list[Tuple[TensorIndexType, Tuple[StateSpace, ...], TorchIndexType]],
    ) -> Tuple[StateSpace, ...]:
        shapes: list[Tuple[int, ...]] = []
        if self.tensor_union_dims:
            shapes.append(tuple(dim.dim for dim in self.tensor_union_dims))

        for idx, _, _ in entries:
            if not isinstance(idx, Tensor) or idx.data.dtype != torch.bool:
                continue
            shapes.append((int(idx.data.count_nonzero().item()),))

        if not shapes:
            return tuple()

        broadcast_shape = torch.broadcast_shapes(*shapes)
        return tuple(IndexSpace.linear(size) for size in broadcast_shape)

    def compile(self) -> CompiledIndices:
        """
        Compile the raw key into executable indices and output metadata.

        Returns
        -------
        CompiledIndices
            A tuple-like record containing:
            - `indices`: torch-compatible index tuple for one-shot execution.
            - `dims`: output [`StateSpace`][qten.symbolics.state_space.StateSpace] tuple after indexing.
            - `indices_steps`: per-token steps for sequential execution.
            - `has_tensor_index`: whether tensor advanced-index mode is active.

        Raises
        ------
        ValueError
            If [`Tensor`][qten.linalg.tensors.Tensor] indices are mixed with [`StateSpace`][qten.symbolics.state_space.StateSpace]/[`Convertible`][qten.abstracts.Convertible]
            indices in the same key.
        IndexError
            If normalization fails (e.g. too many indices) or a [`StateSpace`][qten.symbolics.state_space.StateSpace]
            index is not compatible with the corresponding source dimension.
        NotImplementedError
            If boolean [`Tensor`][qten.linalg.tensors.Tensor] index masking is requested.
        """
        normalized_indices = self._normalize()
        has_tensor_index = any(isinstance(idx, Tensor) for idx in normalized_indices)
        has_statespace_index = any(
            isinstance(idx, StateSpace)
            or (isinstance(idx, Convertible) and not isinstance(idx, Tensor))
            for idx in normalized_indices
        )
        if has_tensor_index and has_statespace_index:
            raise ValueError(
                "Tensor advanced indexing cannot be mixed with StateSpace/Convertible indexing"
            )
        entries, torch_indices = self._compile_entries(normalized_indices)
        compiled_dims = self._compose_compiled_dims(entries)
        indices_steps = tuple(
            cast(Union[int, slice, None, Tuple[int, ...]], compiled_idx)
            for _, _, compiled_idx in entries
        )
        return self.CompiledIndices(
            indices=torch_indices,
            dims=compiled_dims,
            indices_steps=indices_steps,
            has_tensor_index=has_tensor_index,
        )
