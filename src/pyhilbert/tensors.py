from typing import (
    Tuple,
    TypeVar,
    Generic,
    Union,
    Sequence,
    cast,
    Dict,
    Any,
    Optional,
    Callable,
)
from numbers import Number
from dataclasses import dataclass, replace
from multipledispatch import dispatch  # type: ignore[import-untyped]
import torch
from .precision import get_precision_config
from functools import wraps, reduce
from itertools import product

from .abstracts import Convertible, Operable, Plottable
from .state_space import (
    StateSpace,
    BroadcastSpace,
    StateSpaceFactorization,
    embedding_order,
    same_span,
    flat_permutation_order,
    restructure,
)


T = TypeVar("T", bound=torch.Tensor)
"""
The `torch.Tensor` types to be used in `Tensor`. 
This is a type variable that can be any subclass of `torch.Tensor`, 
such as `torch.FloatTensor`, `torch.DoubleTensor`, etc.
"""


@dataclass(frozen=True, eq=False)
class Tensor(Generic[T], Operable, Plottable, Convertible):
    data: T
    dims: Tuple[StateSpace, ...]

    def astype(self, dtype: torch.dtype) -> "Tensor":
        """
        Return a new tensor with the same dims and converted data dtype.

        Parameters
        ----------
        `dtype` : `torch.dtype`
            Target data type.

        Returns
        -------
        `Tensor`
            A new tensor whose data has dtype `dtype`.
        """
        return astype(self, dtype)

    def equal(self, other: "Tensor") -> bool:
        """
        Compare this tensor to another tensor for exact equality.

        Behavior
        --------
        - Attempts to align `other.dims` to `self.dims` using `align_all`.
        - If dimension alignment is not possible, returns `False`.
        - If alignment succeeds, compares aligned data via `torch.equal`.

        Parameters
        ----------
        `other` : `Tensor`
            The tensor to compare against this tensor.

        Returns
        -------
        `bool`
            `True` if tensors are exactly equal after alignment; otherwise `False`.
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

        Behavior
        --------
        - Attempts to align `other.dims` to `self.dims` using `align_all`.
        - If dimension alignment is not possible, returns `False`.
        - If alignment succeeds, compares aligned data via `torch.allclose`.

        Parameters
        ----------
        `other` : `Tensor`
            The tensor to compare against this tensor.
        `rtol` : `float`, optional
            Relative tolerance used by `torch.allclose`.
        `atol` : `float`, optional
            Absolute tolerance used by `torch.allclose`.
        `equal_nan` : `bool`, optional
            Whether `NaN` values are considered equal.

        Returns
        -------
        `bool`
            `True` if tensors are close after alignment; otherwise `False`.
        """
        return allclose(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def conj(self) -> "Tensor":
        """
        Compute the complex conjugate of the given tensor.

        Returns
        -------
        `Tensor`
            The complex conjugate of the tensor.
        """
        return conj(self)

    def permute(self, *order: Union[int, Sequence[int]]) -> "Tensor":
        """
        Permute the dimensions according to the specified order.

        Parameters
        ----------
        order : `Union[int, Sequence[int]]`
            The desired order of dimensions.

        Returns
        -------
        `Tensor`
            The permuted tensor.
        """
        return permute(self, *order)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        """
        Transpose the specified dimensions.

        Parameters
        ----------
        dim0 : `int`
            The first dimension to transpose.
        dim1 : `int`
            The second dimension to transpose.

        Returns
        -------
        `Tensor`
            The transposed tensor.
        """
        return transpose(self, dim0, dim1)

    def h(self, dim0: int, dim1: int) -> "Tensor":
        """
        Hermitian transpose (conjugate transpose) of the specified dimensions.

        Parameters
        ----------
        dim0 : `int`
            The first dimension to transpose.
        dim1 : `int`
            The second dimension to transpose.

        Returns
        -------
        `Tensor`
            The Hermitian transposed tensor.
        """
        return self.conj().transpose(dim0, dim1)

    def align(self, dim: int, target_dim: StateSpace) -> "Tensor":
        """
        Align the specified dimension to the target StateSpace.

        Parameters
        ----------
        dim : `int`
            The dimension index to align.
        target_dim : `StateSpace`
            The target StateSpace to align to.

        Returns
        -------
        `Tensor`
            The aligned tensor.
        """
        return align(self, dim, target_dim)

    def align_all(self, dims: Tuple[StateSpace, ...]) -> "Tensor":
        """
        Align all tensor dimensions to `dims`.

        Parameters
        ----------
        dims : `Tuple[StateSpace, ...]`
            The target dimensions to align to.

        Returns
        -------
        `Tensor`
            The aligned tensor.

        Raises
        ------
        `ValueError`
            If the provided `dims` are not compatible with the tensor's current dimensions.
        """
        return align_all(self, dims)

    def all(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> "Tensor":
        """
        Return whether all elements evaluate to `True`.

        Parameters
        ----------
        `dim` : `Optional[Union[int, Tuple[int, ...]]]`, optional
            Reduction axis (or axes). If `None`, reduce over all dimensions.
        `keepdim` : `bool`, optional
            If `True`, retains the reduced axis as `BroadcastSpace`.

        Returns
        -------
        `Tensor`
            Boolean tensor after reduction.
        """
        return all(self, dim=dim, keepdim=keepdim)

    def unsqueeze(self, dim: int) -> "Tensor":
        """
        Unsqueeze the specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to unsqueeze.

        Returns
        -------
        `Tensor`
            The unsqueezed tensor.
        """
        return unsqueeze(self, dim)

    def squeeze(self, dim: int) -> "Tensor":
        """
        Squeeze the specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to squeeze.

        Returns
        -------
        `Tensor`
            The squeezed tensor.
        """
        return squeeze(self, dim)

    def rank(self) -> int:
        """
        Get the rank (number of dimensions) of the tensor.

        Returns
        -------
        `int`
            The rank of the tensor.
        """
        return rank(self)

    def mean(self, dim: int) -> "Tensor":
        """
        Compute the mean over a specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to reduce.

        Returns
        -------
        `Tensor`
            A new tensor with the specified dimension reduced.
        """
        return mean(self, dim)

    def argmax(self, dim: int) -> "Tensor":
        """
        Compute the indices of the maximum values over a specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to reduce.

        Returns
        -------
        `Tensor`
            A new tensor with the specified dimension reduced.
        """
        return argmax(self, dim)

    def argmin(self, dim: int) -> "Tensor":
        """
        Compute the indices of the minimum values over a specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to reduce.

        Returns
        -------
        `Tensor`
            A new tensor with the specified dimension reduced.
        """
        return argmin(self, dim)

    def expand_to_union(self, union_dims: list[StateSpace]) -> "Tensor":
        """
        Expand the tensor to the union of the specified dimensions.

        Parameters
        ----------
        union_dims : `list[StateSpace]`
            The dimensions to expand to the union of.

        Returns
        -------
        `Tensor`
            The expanded tensor.
        """
        return expand_to_union(self, union_dims)

    def item(self) -> Union[Number, int, float]:
        """
        Return the value of a 0-dimensional tensor as a standard Python number.

        Returns
        -------
        `number`
            The value of the tensor.

        Raises
        ------
        ValueError
            If the tensor is not 0-dimensional.
        """
        return self.data.item()

    def cpu(self) -> "Tensor":
        """
        Copy the tensor data to CPU memory and create a new `Tensor` instance.

        Returns
        -------
        `Tensor`
            The new `Tensor` instance with copied data on CPU.
        """
        return Tensor(data=self.data.cpu(), dims=self.dims)

    def gpu(self) -> "Tensor":
        """
        Copy the tensor data to GPU memory and create a new `Tensor` instance.

        Returns
        -------
        `Tensor`
            The new `Tensor` instance with copied data on GPU.

        Raises
        ------
        RuntimeError
            If GPU is not available on this system.
        """
        if torch.cuda.is_available():
            return Tensor(data=self.data.cuda(), dims=self.dims)
        elif torch.backends.mps.is_available():
            return Tensor(data=self.data.to("mps"), dims=self.dims)
        else:
            raise RuntimeError(
                "Only CUDA and MPS devices are supported for GPU operations!"
            )

    @property
    def requires_grad(self) -> bool:
        """
        Check if the tensor data requires gradient tracking.

        Returns
        -------
        `bool`
            True if the tensor data requires gradient tracking, False otherwise.
        """
        return self.data.requires_grad

    def attach(self) -> "Tensor":
        """
        Enable gradient tracking for the tensor data and return the attached `Tensor` instance.

        Behavior
        --------
        - If `requires_grad` is already `True`, this returns `self` unchanged.
        - Otherwise, this detaches the underlying data from any existing autograd graph,
          clones it to ensure a fresh leaf tensor, and sets `requires_grad` to `True`.
        - The returned tensor preserves the original `dims`, device, and dtype.

        Returns
        -------
        `Tensor`
            The new `Tensor` instance with gradient tracking enabled.
        """
        if self.data.requires_grad:
            return self
        return Tensor(
            data=self.data.detach().clone().requires_grad_(True), dims=self.dims
        )

    def detach(self) -> "Tensor":
        """
        Disable gradient tracking for the tensor data and create a new `Tensor` instance.

        Behavior
        --------
        - Always returns a new `Tensor` whose data is a detached view of the
          original tensor (no clone), so it shares storage with the original.
        - The returned tensor preserves the original `dims`, device, and dtype.

        Returns
        -------
        `Tensor`
            The new `Tensor` instance with gradient tracking disabled.
        """
        return Tensor(data=self.data.detach(), dims=self.dims)

    def clone(self) -> "Tensor":
        """
        Create a deep copy of the tensor.

        Returns
        -------
        `Tensor`
            The cloned tensor.
        """
        return Tensor(data=self.data.clone(), dims=self.dims)

    def replace_dim(self, dim: int, new_dim: StateSpace) -> "Tensor":
        """
        Replace the StateSpace at the specified dimension with a new StateSpace.

        Parameters
        ----------
        tensor : `Tensor`
            The tensor to modify.
        dim : `int`
            The index of the dimension to replace.
        new_dim : `StateSpace`
            The new StateSpace to assign to the dimension.

        Returns
        -------
        `Tensor`
            A new Tensor with the updated dimension.
        """
        return replace_dim(self, dim, new_dim)

    def __getitem__(self, key):
        """
        Index tensor data with either numeric slicing or StateSpace-aware slicing.

        Supported conventions
        ---------------------
        - Normal indexing: when no `StateSpace`/`Convertible` indices are present,
          this forwards directly to `self.data[key]` and returns a `torch.Tensor`.
        - StateSpace indexing: when any `StateSpace`/`Convertible` index is present,
          this returns a `Tensor` and applies StateSpace-aware selection rules.

        Convertible indices
        -------------------
        Any index object implementing `Convertible` is accepted, as long as
        `index.convert(StateSpace)` is defined and returns a `StateSpace`.
        This allows domain-specific index types (for example, basis/momentum/band
        objects) to be used directly in tensor indexing.

        Notes
        -----
        - In StateSpace-aware mode, integer/range indices cannot be mixed in.
        - Only full slices `:` are allowed alongside StateSpace-aware indices.
        """
        if key is Ellipsis:
            key = (slice(None),) * len(self.dims)
        elif not isinstance(key, tuple):
            key = (key,)

        if Ellipsis in key:
            ellipsis_at = key.index(Ellipsis)
            missing = len(self.dims) - (len(key) - 1)
            key = key[:ellipsis_at] + (slice(None),) * missing + key[ellipsis_at + 1 :]

        non_none = sum(1 for k in key if k is not None)
        if non_none < len(self.dims):
            key = key + (slice(None),) * (len(self.dims) - non_none)
        if non_none > len(self.dims):
            raise IndexError("Too many indices for tensor")

        # Decide indexing convention: StateSpace/Convertible vs normal (int/slice/range).
        has_hilbert_indices = any(isinstance(k, (StateSpace, Convertible)) for k in key)
        if has_hilbert_indices:
            if any(isinstance(k, (int, range)) for k in key if k is not None):
                raise ValueError(
                    "Hilbert indexing cannot be mixed with integer/range indexing"
                )
            if any(
                isinstance(k, slice) and k != slice(None, None, None)
                for k in key
                if k is not None
            ):
                raise ValueError(
                    "Hilbert indexing only allows full slices ':' when mixed with StateSpace/Convertible indices"
                )
        else:
            return self.data[key]

        return _tensor_getitem_hilbert(self, key)

    def factorize_dim(self, dim: int, rule: StateSpaceFactorization) -> "Tensor":
        """
        Factorize one `StateSpace`-like dimension into multiple subspaces.

        For a tensor with shape `(A, B)`, factorizing the `0`th dimension with
        `factorized=(A1, A2)` and a compatible `align_dim` produces shape
        `(A1, A2, B)`.

        Parameters
        ----------
        `dim` : `int`
            The index of the dimension to factorize.
        `rule` : `StateSpaceFactorization`
            The factorization rule.

        Returns
        -------
        `Tensor`
            A new tensor with the specified dimension factorized.
        """
        return factorize_dim(self, dim, rule)

    def product_dims(self, *indices_group: Tuple[int, ...]) -> "Tensor":
        """
        Combine selected tensor dimensions into product dimensions.

        Each entry in `indices_group` defines one output product dimension.
        For a group `(i0, i1, ..., ik)`, the returned tensor contains a single
        axis whose size is the product of the grouped axis sizes and whose
        `StateSpace` is `self.dims[i0] @ self.dims[i1] @ ... @ self.dims[ik]`.
        Dimensions not listed in any group are preserved as-is.

        Negative indices are supported and follow Python indexing rules.
        Grouped dimensions do not need to be contiguous in the input tensor; the
        method reorders axes internally, performs one reshape, and returns the
        result in the canonical output order.

        Parameters
        ----------
        `indices_group` : `Tuple[int, ...]`
            One or more non-empty groups of dimension indices to combine.
            Indices must be unique across all groups (a dimension can belong to
            at most one group).

        Returns
        -------
        `Tensor`
            A new tensor where each requested group is replaced by one product
            dimension and all non-grouped dimensions are retained.

        Raises
        ------
        `IndexError`
            If any provided index is out of range for the tensor rank.
        `ValueError`
            If any group is empty, if a group contains duplicate indices, or if
            the same index appears in more than one group.
        """
        return product_dims(self, *indices_group)

    def dim_types(self) -> Tuple[type, ...]:
        """
        Return a tuple of the types of the dimensions in the tensor.

        Returns
        -------
        `Tuple[type, ...]`
            A tuple containing the types of each dimension in the tensor.
        """
        return tuple(type(dim) for dim in self.dims)

    def __repr__(self) -> str:
        device_type = self.data.device.type
        device = "GPU" if device_type in {"cuda", "mps"} else "CPU"
        if self.dims:
            shape = ", ".join(f"{type(dim).__name__}:{dim.dim}" for dim in self.dims)
            shape_repr = f"({shape})"
        else:
            shape_repr = "()"
        return f"<{device} Tensor grad={self.data.requires_grad} shape={shape_repr}>"

    __str__ = __repr__  # Override str to use the same representation


def auto_promote(func):
    """Decorator to automatically promote input Tensors to a common dtype."""

    @wraps(func)
    def wrapper(left, right, *args, **kwargs):
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            common_dtype = torch.promote_types(left.data.dtype, right.data.dtype)
            if left.data.dtype != common_dtype:
                left = Tensor(data=left.data.to(common_dtype), dims=left.dims)
            if right.data.dtype != common_dtype:
                right = Tensor(data=right.data.to(common_dtype), dims=right.dims)
        return func(left, right, *args, **kwargs)

    return wrapper


def _tensor_getitem_hilbert(tensor: Tensor, key: Tuple[object, ...]) -> Tensor:
    data = tensor.data
    new_dims = list(tensor.dims)
    dim_index = 0
    dim_pos = 0
    for k in key:
        if k is None:
            data = data.unsqueeze(dim_index)
            new_dims.insert(dim_index, BroadcastSpace())
            dim_index += 1
            continue
        if dim_pos >= len(tensor.dims):
            raise IndexError("Too many indices for tensor")
        dim = tensor.dims[dim_pos]
        dim_pos += 1
        if not isinstance(k, (StateSpace, slice)) and isinstance(k, Convertible):
            try:
                converted = k.convert(StateSpace)
            except NotImplementedError as e:
                raise TypeError(
                    f"{type(k).__name__} cannot be converted to StateSpace for tensor indexing"
                ) from e
            assert isinstance(converted, StateSpace), (
                "Convertible.convert(StateSpace) must return a StateSpace"
            )
            k = converted
        if isinstance(k, StateSpace):
            if not set(k.structure.keys()).issubset(dim.structure.keys()):
                raise ValueError("StateSpace index is not a subspace of tensor dim")
            sub_space = replace(k, structure=restructure(k.structure))
            idx = torch.tensor(
                embedding_order(sub_space, dim),
                dtype=torch.long,
                device=data.device,
            )
            data = data.index_select(dim_index, idx)
            new_dims[dim_index] = sub_space
            dim_index += 1
            continue
        if isinstance(k, slice):
            if k != slice(None, None, None):
                raise TypeError(
                    "Only full slice ':' is supported with StateSpace indexing"
                )
            dim_index += 1
            continue
        raise TypeError(f"Unsupported index type for StateSpace slicing: {type(k)}")

    return Tensor(data=data, dims=tuple(new_dims))


def _match_dims_for_matmul(left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
    if left.rank() == 1:
        left = left.unsqueeze(0)
    if right.rank() == 1:
        right = right.unsqueeze(-1)

    if left.rank() > right.rank():
        # Unsqueeze right tensor
        for _ in range(left.rank() - right.rank()):
            right = right.unsqueeze(0)
    elif right.rank() > left.rank():
        # Unsqueeze left tensor
        for _ in range(right.rank() - left.rank()):
            left = left.unsqueeze(0)
    return left, right


def _align_dims_for_matmul(left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
    ignores = set()
    for n, ld in enumerate(left.dims[:-2]):
        if not isinstance(ld, BroadcastSpace):
            continue
        rd = right.dims[n]
        if isinstance(rd, BroadcastSpace):
            continue
        left = left.align(n, rd)
        ignores.add(n)

    for n, ld in enumerate(left.dims[:-2]):
        if n in ignores:
            continue
        right = right.align(n, ld)

    return left, right


@auto_promote
def matmul(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform matrix multiplication between two Tensors with StateSpace-aware
    alignment and torch-style rank handling.

    Both operands must be at least 1D. If either operand is 1D, this follows
    `torch.matmul` behavior by temporarily unsqueezing it to 2D, performing the
    matmul, then squeezing out the added dimension(s).

    The function first makes the tensors have the same number of dimensions by
    unsqueezing leading dimensions with `BroadcastSpace`. It then aligns any
    leading (batch) dimensions so that `BroadcastSpace` can expand to concrete
    StateSpaces and any non-broadcast StateSpaces are reordered to match. Finally,
    the right tensor's second-to-last dimension is aligned to the left tensor's
    last dimension, and `torch.matmul` is applied.

    The contraction always happens between `left.dims[-1]` and `right.dims[-2]`.
    Leading dimensions behave like batch dimensions and follow the broadcast and
    alignment rules described above. The output keeps all aligned leading
    dimensions (including any `BroadcastSpace` that remain), drops the contracted
    dimension, and appends the right-most dimension from `right`.

    Parameters
    ----------
    left : `Tensor`
        The left tensor operand.
    right : `Tensor`
        The right tensor operand.

    Returns
    -------
    `Tensor`
        A tensor with data `torch.matmul(left.data, right.data)` and dimensions
        `left.dims[:-1] + right.dims[-1:]`, after the alignment and any
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
    left, right = _align_dims_for_matmul(left, right)

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


@dispatch(Tensor, Tensor)
def operator_matmul(left: Tensor, right: Tensor) -> Tensor:
    """Perform matrix multiplication (contraction) between two `Tensor`."""
    return matmul(left, right)


def _match_dims_for_tensoradd(left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:
    if left.rank() > right.rank():
        # Unsqueeze right tensor
        for _ in range(left.rank() - right.rank()):
            right = right.unsqueeze(0)
    elif right.rank() > left.rank():
        # Unsqueeze left tensor
        for _ in range(right.rank() - left.rank()):
            left = left.unsqueeze(0)
    return left, right


@dispatch(Tensor, Tensor)
@auto_promote
def operator_add(left: Tensor, right: Tensor) -> Tensor:
    """
    Add two tensors with the same order of dimensions.
    If the intra-ordering within the `StateSpace`s differ,
    the `right` tensor is permuted to match the ordering
    of the `left` tensor before addition.

    Parameters
    ----------
    left : `Tensor`
        The left tensor to add.
    right : `Tensor`
        The right tensor to add.

    Returns
    -------
    `Tensor`
        The resulting tensor on the union of StateSpaces.
    """
    left, right = _match_dims_for_tensoradd(left, right)

    # calculate the union of the StateSpaces
    union_dims = []
    for l_dim, r_dim in zip(left.dims, right.dims):
        union_dims.append(l_dim + r_dim)

    # Expand BroadcastSpace to the union StateSpace to ensure data expansion
    left = left.expand_to_union(union_dims)
    right = right.expand_to_union(union_dims)

    # calculate the new shape
    new_shape = tuple(u.dim for u in union_dims)
    new_data = torch.zeros(new_shape, dtype=left.data.dtype, device=left.data.device)
    # fill the left tensor into the new data
    left_slices = tuple(slice(0, d.dim) for d in left.dims)
    new_data[left_slices] = left.data
    # fill the right tensor into the new data
    right_embedding_order = (
        torch.tensor(embedding_order(r, u), dtype=torch.long, device=left.data.device)
        for r, u in zip(right.dims, union_dims)
    )
    new_data.index_put_(
        torch.meshgrid(*right_embedding_order, indexing="ij"),
        right.data,
        accumulate=True,
    )

    return Tensor(data=new_data, dims=tuple(union_dims))


@dispatch(Tensor, Tensor)
def operator_eq(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform element-wise equality comparison between two tensors.

    The `right` tensor is aligned to `left.dims` before comparison.
    """
    aligned_right = right.align_all(left.dims)
    return Tensor(data=left.data == aligned_right.data, dims=left.dims)


@dispatch(Tensor)
def operator_neg(tensor: Tensor) -> Tensor:
    """
    Perform negation on the given tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to negate.

    Returns
    -------
    `Tensor`
        The negated tensor.
    """
    return Tensor(data=-tensor.data, dims=tensor.dims)


@dispatch(Tensor, Tensor)
def operator_sub(left: Tensor, right: Tensor) -> Tensor:
    """
    Subtract the right tensor from the left tensor with the same order of dimensions.
    If the intra-ordering within the `StateSpace`s differ, the `right` tensor is
    permuted to match the ordering of the `left` tensor before addition.

    Parameters
    ----------
    left : `Tensor`
        The tensor from which to subtract.
    right : `Tensor`
        The tensor to subtract.

    Returns
    -------
    `Tensor`
        The resulting tensor after subtraction.
    """
    return left + (-right)


@dispatch(Number, Tensor)
def operator_mul(left: Number, right: Tensor) -> Tensor:
    """
    Perform element-wise multiplication of a number and a tensor.

    Parameters
    ----------
    left : `Number`
        The scalar value.
    right : `Tensor`
        The tensor.
    Returns
    -------
    `Tensor`
        A new tensor with each element multiplied by the scalar.
    """
    return Tensor(data=left * right.data, dims=right.dims)


@dispatch(Tensor, Number)  # type: ignore[no-redef]
def operator_mul(left: Tensor, right: Number) -> Tensor:
    """
    Perform element-wise multiplication of a tensor and a number.

    Parameters
    ----------
    left : `Tensor`
        The tensor.
    right : `Number`
        The scalar value.
    Returns
    -------
    `Tensor`
        A new tensor with each element multiplied by the scalar.
    """
    return Tensor(data=left.data * right, dims=left.dims)


@dispatch(Number, Tensor)  # type: ignore[no-redef]
def operator_add(left: Number, right: Tensor) -> Tensor:
    """
    Add a number to the diagonal of the tensor (broadcasting over batch dimensions).

    This treats the tensor as a batch of matrices (defined by the last two dimensions).
    The scalar is added to the diagonal elements of these matrices.
    For rank-2 tensors, this is equivalent to M + c*I.
    Parameters
    ----------
    left : `Number`
        The scalar value to add to the diagonal.
    right : `Tensor`
        The target tensor (must be at least rank 2).
    Returns
    -------
    `Tensor`
        The result of adding the scalar to the diagonal.
    """
    eye = identity(right.dims)
    return left * eye + right


@dispatch(Tensor, Number)  # type: ignore[no-redef]
def operator_add(left: Tensor, right: Number) -> Tensor:
    """
    Add a number to the diagonal of the tensor (broadcasting over batch dimensions).

    This treats the tensor as a batch of matrices (defined by the last two dimensions).
    The scalar is added to the diagonal elements of these matrices.
    For rank-2 tensors, this is equivalent to M + c*I.
    Parameters
    ----------
    left : `Tensor`
        The target tensor (must be at least rank 2).
    right : `Number`
        The scalar value to add to the diagonal.
    Returns
    -------
    `Tensor`
        The result of adding the scalar to the diagonal.
    """
    eye = identity(left.dims)
    return left + right * eye


@dispatch(Number, Tensor)  # type: ignore[no-redef]
def operator_sub(left: Number, right: Tensor) -> Tensor:
    """
    Subtract a tensor from a number (broadcasted on diagonal).

    This treats the tensor as a batch of matrices (defined by the last two dimensions).
    The operation is performed as (c*I - T), where I is the identity matrix broadcasted
    over the batch dimensions.
    Parameters
    ----------
    left : `Number`
        The scalar value.
    right : `Tensor`
        The tensor to subtract.
    Returns
    -------
    `Tensor`
        The result of the subtraction.
    """
    eye = identity(right.dims)
    return left * eye + (-right)


@dispatch(Tensor, Number)  # type: ignore[no-redef]
def operator_sub(left: Tensor, right: Number) -> Tensor:
    """
    Subtract a number from a tensor (broadcasted on diagonal).

    This treats the tensor as a batch of matrices (defined by the last two dimensions).
    The operation is performed as (T - c*I), where I is the identity matrix broadcasted
    over the batch dimensions.
    Parameters
    ----------
    left : `Tensor`
        The tensor.
    right : `Number`
        The scalar value to subtract from the diagonal.
    Returns
    -------
    `Tensor`
        The result of the subtraction.
    """
    eye = identity(left.dims)
    return left + (-right) * eye


@dispatch(Tensor, Number)
def operator_truediv(left: Tensor, right: Number) -> Tensor:
    """
    Perform element-wise division of a tensor by a number.
    Parameters
    ----------
    left : `Tensor`
        The tensor.
    right : `Number`
        The scalar divisor.
    Returns
    -------
    `Tensor`
        A new tensor with each element divided by the scalar.
    """
    return left * (1.0 / right)  # type: ignore[operator]


def permute(tensor: Tensor, *order: Union[int, Sequence[int]]) -> Tensor:
    """
    Permute the dimensions of the tensor according to the specified order.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to permute.
    order : `Union[int, Sequence[int]]`
        The desired order of dimensions.

    Returns
    -------
    `Tensor`
        The permuted tensor.
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

    return Tensor(data=new_data, dims=new_dims)


def transpose(tensor: Tensor, dim0: int, dim1: int) -> Tensor:
    """
    Transpose the specified dimensions of the tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to transpose.
    dim0 : `int`
        The first dimension to transpose.
    dim1 : `int`
        The second dimension to transpose.

    Returns
    -------
    `Tensor`
        The transposed tensor.
    """
    new_data = tensor.data.transpose(dim0, dim1)

    # Convert tuple to list to modify
    new_dims_list = list(tensor.dims)
    # Swap elements
    new_dims_list[dim0], new_dims_list[dim1] = new_dims_list[dim1], new_dims_list[dim0]

    return Tensor(data=new_data, dims=tuple(new_dims_list))


def conj(tensor: Tensor) -> Tensor:
    """
    Compute the complex conjugate of the given tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to conjugate.

    Returns
    -------
    `Tensor`
        The complex conjugate of the tensor.
    """
    return Tensor(data=tensor.data.conj(), dims=tensor.dims)


def unsqueeze(tensor: Tensor, dim: int) -> Tensor:
    """
    Unsqueeze the specified dimension of the tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to unsqueeze.
    dim : `int`
        The dimension to unsqueeze.

    Returns
    -------
    `Tensor`
        The unsqueezed tensor.
    """
    if dim < 0:
        dim = dim + len(tensor.dims) + 1
    new_data = tensor.data.unsqueeze(dim)
    new_dims = tensor.dims[:dim] + (BroadcastSpace(),) + tensor.dims[dim:]

    return Tensor(data=new_data, dims=new_dims)


def squeeze(tensor: Tensor, dim: int) -> Tensor:
    """
    Squeeze the specified dimension of the tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to squeeze.
    dim : `int`
        The dimension to squeeze.

    Returns
    -------
    `Tensor`
        The squeezed tensor.
    """
    if dim < 0:
        dim = dim + len(tensor.dims)
    if not isinstance(tensor.dims[dim], BroadcastSpace):
        return tensor  # No squeezing needed if not BroadcastSpace

    new_data = tensor.data.squeeze(dim)
    new_dims = tensor.dims[:dim] + tensor.dims[dim + 1 :]

    return Tensor(data=new_data, dims=new_dims)


def align(tensor: Tensor, dim: int, target_dim: StateSpace) -> Tensor:
    """
    Align the specified dimension of the tensor to the target StateSpace.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to align.
    dim : `int`
        The dimension index to align.
    target_dim : `StateSpace`
        The target StateSpace to align to.

    Returns
    -------
    `Tensor`
        The aligned tensor.
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

    if isinstance(current_dim, BroadcastSpace):
        # Expand broadcast dimension to match the target StateSpace size.
        expanded_shape = list(tensor.data.shape)
        expanded_shape[dim] = target_dim.dim
        aligned_data = tensor.data.expand(*expanded_shape)
        return Tensor(
            data=aligned_data,
            dims=tensor.dims[:dim] + (target_dim,) + tensor.dims[dim + 1 :],
        )

    if type(current_dim) is not type(target_dim):
        raise ValueError(
            f"Cannot align dimensions with different StateSpace types: "
            f"current dim={type(current_dim)} vs target dim={type(target_dim)}!"
        )
    if not same_span(current_dim, target_dim):
        raise ValueError(f"StateSpace at {dim} cannot be aligned to target StateSpace!")

    try:
        target_order = flat_permutation_order(current_dim, target_dim)
    except ValueError as e:
        raise ValueError(
            f"StateSpace at {dim} cannot be aligned to target StateSpace!"
        ) from e
    aligned_data = torch.index_select(
        tensor.data,
        dim,
        torch.tensor(target_order, dtype=torch.long, device=tensor.data.device),
    )

    aligned_tensor = Tensor(
        data=aligned_data,
        dims=tensor.dims[:dim] + (target_dim,) + tensor.dims[dim + 1 :],
    )

    return aligned_tensor


def align_all(tensor: Tensor, dims: Tuple[StateSpace, ...]) -> Tensor:
    """
    Align all dimensions of `tensor` to `dims`.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to align.
    dims : `Tuple[StateSpace, ...]`
        Target dimensions for each axis.

    Returns
    -------
    `Tensor`
        The aligned tensor.

    Raises
    ------
    `ValueError`
        If rank does not match or any dimension cannot be aligned.
    """
    if len(dims) != tensor.rank():
        raise ValueError(
            f"Cannot align rank-{tensor.rank()} tensor to rank-{len(dims)} dims"
        )

    aligned = tensor
    for idx, target_dim in enumerate(dims):
        aligned = aligned.align(idx, target_dim)
    return aligned


def all(
    tensor: Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
) -> Tensor:
    """
    Reduce a tensor with logical AND, matching `torch.all` semantics.

    Parameters
    ----------
    `tensor` : `Tensor`
        Input tensor.
    `dim` : `Optional[Union[int, Tuple[int, ...]]]`, optional
        Reduction axis (or axes). If `None`, reduce over all dimensions.
    `keepdim` : `bool`, optional
        If `True`, retains the reduced axis as `BroadcastSpace`.

    Returns
    -------
    `Tensor`
        Boolean tensor with reduced dimensions.
    """
    if dim is None:
        return Tensor(data=torch.all(tensor.data), dims=())

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
    return Tensor(data=reduced, dims=new_dims)


def rank(tensor: Tensor) -> int:
    """
    Get the rank (number of dimensions) of the tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor whose rank is to be determined.

    Returns
    -------
    `int`
        The rank of the tensor.
    """
    return len(tensor.dims)


def mean(tensor: Tensor, dim: int) -> Tensor:
    """
    Compute the mean over a specified dimension.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to reduce.
    dim : `int`
        The dimension to reduce.

    Returns
    -------
    `Tensor`
        A new tensor with the specified dimension reduced.
    """
    if dim < 0:
        dim += tensor.rank()
    if dim < 0 or dim >= tensor.rank():
        raise IndexError(f"Dimension index {dim} out of range for rank {tensor.rank()}")

    return Tensor(
        data=tensor.data.mean(dim=dim),
        dims=tensor.dims[:dim] + tensor.dims[dim + 1 :],
    )


def argmax(tensor: Tensor, dim: int) -> Tensor:
    """
    Compute the indices of the maximum values over a specified dimension.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to reduce.
    dim : `int`
        The dimension to reduce.

    Returns
    -------
    `Tensor`
        A new tensor with the specified dimension reduced.
    """
    if dim < 0:
        dim += tensor.rank()
    if dim < 0 or dim >= tensor.rank():
        raise IndexError(f"Dimension index {dim} out of range for rank {tensor.rank()}")

    return Tensor(
        data=tensor.data.argmax(dim=dim),
        dims=tensor.dims[:dim] + tensor.dims[dim + 1 :],
    )


def argmin(tensor: Tensor, dim: int) -> Tensor:
    """
    Compute the indices of the minimum values over a specified dimension.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to reduce.
    dim : `int`
        The dimension to reduce.

    Returns
    -------
    `Tensor`
        A new tensor with the specified dimension reduced.
    """
    if dim < 0:
        dim += tensor.rank()
    if dim < 0 or dim >= tensor.rank():
        raise IndexError(f"Dimension index {dim} out of range for rank {tensor.rank()}")

    return Tensor(
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
    tensor : `Tensor`
        Input tensor containing class indices.
    dim : `StateSpace`
        Output class dimension. Class indices are assumed to be ordered as
        `[0, 1, ..., dim.dim - 1]`.

    Returns
    -------
    `Tensor`
        A new tensor with one extra trailing dimension for class channels.
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


def astype(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    """
    Return a new tensor with data converted to `dtype`.

    Parameters
    ----------
    `tensor` : `Tensor`
        Input tensor.
    `dtype` : `torch.dtype`
        Target data type.

    Returns
    -------
    `Tensor`
        A new tensor with converted data and unchanged dims.
    """
    return Tensor(data=tensor.data.to(dtype=dtype), dims=tensor.dims)


def allclose(
    a: Tensor,
    b: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """
    Compare two tensors for approximate equality with dimension-aware alignment.

    This function first aligns `b` to `a` by calling `b.align_all(a.dims)`.
    If alignment fails (for example, mismatched rank or non-alignable
    `StateSpace`s), this function returns `False` instead of raising.
    When alignment succeeds, the function compares data values using
    `torch.allclose`.

    After alignment, comparison is delegated directly to `torch.allclose`,
    preserving native PyTorch behavior for dtype/device handling.

    Parameters
    ----------
    `a` : `Tensor`
        Reference tensor defining the target dimension layout.
    `b` : `Tensor`
        Tensor that will be aligned to `a` before comparison.
    `rtol` : `float`, optional
        Relative tolerance used by `torch.allclose`.
    `atol` : `float`, optional
        Absolute tolerance used by `torch.allclose`.
    `equal_nan` : `bool`, optional
        Whether `NaN` values are considered equal.

    Returns
    -------
    `bool`
        `True` if values are close after successful alignment; `False` if
        alignment fails or values are not close.
    """
    try:
        aligned_b = b.align_all(a.dims)
    except (IndexError, TypeError, ValueError, RuntimeError):
        return False

    return torch.allclose(
        a.data, aligned_b.data, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def equal(a: Tensor, b: Tensor) -> bool:
    """
    Compare two tensors for exact equality with dimension-aware alignment.

    This function first aligns `b` to `a` by calling `b.align_all(a.dims)`.
    If alignment fails (for example, mismatched rank or non-alignable
    `StateSpace`s), this function returns `False` instead of raising.
    When alignment succeeds, the function compares data values using
    `torch.equal`.

    After alignment, equality is delegated directly to `torch.equal`, preserving
    native PyTorch equality behavior for dtype/device handling.

    Parameters
    ----------
    a : `Tensor`
        Reference tensor defining the target dimension layout.
    b : `Tensor`
        Tensor that will be aligned to `a` before comparison.

    Returns
    -------
    `bool`
        `True` if values are exactly equal after successful alignment; `False`
        if alignment fails or values are not equal.
    """
    try:
        aligned_b = b.align_all(a.dims)
    except (IndexError, TypeError, ValueError, RuntimeError):
        return False

    return torch.equal(a.data, aligned_b.data)


def expand_to_union(tensor: Tensor, union_dims: list[StateSpace]) -> Tensor:
    """
    Expand BroadcastSpace dimensions in the tensor to match union_dims sizes.
    Performs expansion in a single pass to avoid intermediate Tensor creation.
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

    return Tensor(data=tensor.data.expand(target_shape), dims=tuple(new_dims))


def mapping_matrix(
    from_space: StateSpace,
    to_space: StateSpace,
    mapping: Dict[Any, Any],
    factors: Optional[
        Dict[Tuple[Any, Any], int | float | complex | torch.Tensor]
    ] = None,
) -> Tensor:
    """
    Create a sector-wise mapping matrix between two state spaces.

    For each `(from_marker, to_marker)` pair in `mapping`, this function inserts
    an identity block from the corresponding sector in `from_space` to the
    corresponding sector in `to_space`. Each block can be scaled by
    `factors[(from_marker, to_marker)]`; if omitted, a factor of `1` is used.

    Parameters
    ----------
    `from_space` : `StateSpace`
        Source state space defining the row dimension and source sectors.
    `to_space` : `StateSpace`
        Target state space defining the column dimension and target sectors.
    `mapping` : `Dict[Any, Any]`
        Dictionary mapping sector markers in `from_space` to sector markers in
        `to_space`. For each entry, a block is written between the slices
        returned by `from_space.get_slice(...)` and `to_space.get_slice(...)`.
    `factors` : `Optional[Dict[Tuple[Any, Any], int | float | complex | torch.Tensor]]`, optional
        Optional per-entry factors. Keys are `(from_marker, to_marker)` tuples.
        Scalar values scale an identity block; tensor values are inserted
        directly as the block matrix. Missing keys default to `1`.

    Returns
    -------
    `Tensor`
        Rank-2 tensor with dimensions `(from_space, to_space)` containing the
        assembled mapping matrix in complex precision.

    Raises
    ------
    `ValueError`
        If any mapped source and target sectors have different sizes.
    """
    if factors is None:
        factors = {}

    precision = get_precision_config()
    mat = torch.zeros((from_space.dim, to_space.dim), dtype=precision.torch_complex)
    for fm, tm in mapping.items():
        fslice = from_space.get_slice(fm)
        tslice = to_space.get_slice(tm)

        flen = fslice.stop - fslice.start
        tlen = tslice.stop - tslice.start
        factor = factors.get((fm, tm), 1)
        if torch.is_tensor(factor):
            if tuple(factor.shape) != (flen, tlen):
                raise ValueError(
                    f"Cannot insert factor block with shape {tuple(factor.shape)} for sector shape {(flen, tlen)}"
                )
            mat[fslice, tslice] = factor.to(dtype=mat.dtype, device=mat.device)
            continue

        if flen != tlen:
            raise ValueError(
                f"Cannot create mapping matrix between sectors of different sizes: {flen} != {tlen}"
            )

        mat[fslice, tslice] = (
            torch.eye(flen, dtype=mat.dtype, device=mat.device) * factor
        )

    return Tensor(data=mat, dims=(from_space, to_space))


# TODO: Add hilbert_mapping(a: HilbertSpace, b: HilbertSpace) that map between two HilbertSpace, supports internal spans by flattening them.


def identity(dims: Tuple[StateSpace, ...]) -> Tensor:
    """
    Create an identity tensor based on the last two dimensions.
    Returns a rank-2 Tensor corresponding to the identity of the matrix part.
    """
    if len(dims) < 2:
        raise ValueError(
            f"Identity tensor creation requires at least rank 2, got rank {len(dims)}!"
        )
    matrix_dims = dims[-2:]
    rows = matrix_dims[0].dim
    cols = matrix_dims[1].dim
    return Tensor(data=torch.eye(rows, cols), dims=matrix_dims)


def zeros(dims: Tuple[StateSpace, ...]) -> Tensor:
    """
    Create a zero-filled tensor with shape defined by `dims`.

    Parameters
    ----------
    dims : `Tuple[StateSpace, ...]`
        StateSpace dimensions defining the tensor shape.

    Returns
    -------
    `Tensor`
        A tensor of zeros with `shape == tuple(dim.dim for dim in dims)`.
    """
    shape = tuple(dim.dim for dim in dims)
    return Tensor(data=torch.zeros(shape), dims=dims)


def ones(dims: Tuple[StateSpace, ...]) -> Tensor:
    """
    Create a one-filled tensor with shape defined by `dims`.

    Parameters
    ----------
    dims : `Tuple[StateSpace, ...]`
        StateSpace dimensions defining the tensor shape.

    Returns
    -------
    `Tensor`
        A tensor of ones with `shape == tuple(dim.dim for dim in dims)`.
    """
    shape = tuple(dim.dim for dim in dims)
    return Tensor(data=torch.ones(shape), dims=dims)


def kernel_tensor(
    ker: Callable[..., Number], dims: Tuple[StateSpace, ...]
) -> Tensor[torch.Tensor]:
    """
    Build a tensor by evaluating a scalar-valued kernel over StateSpace elements.

    For each multi-index `(i0, i1, ..., iN)` this evaluates:
    `ker(dims[0].elements()[i0], dims[1].elements()[i1], ..., dims[N].elements()[iN])`
    and stores the result at that tensor position.

    Parameters
    ----------
    `ker` : `Callable[..., Number]`
        Scalar-valued callable that accepts one element from each state space in
        `dims`.
    `dims` : `Tuple[StateSpace, ...]`
        Output tensor dimensions.

    Returns
    -------
    `Tensor`
        Tensor with `dims` and values produced by `ker`.
    """
    if not dims:
        return Tensor(data=torch.as_tensor(ker()), dims=dims)

    element_axes = tuple(dim.elements() for dim in dims)
    for axis, dim in zip(element_axes, dims):
        if len(axis) != dim.dim:
            raise ValueError(
                f"kernel_tensor expects one element per index for each StateSpace; "
                f"got len(elements)={len(axis)} and dim={dim.dim} for {type(dim).__name__}"
            )

    values = [ker(*args) for args in product(*element_axes)]
    data = torch.as_tensor(values).reshape(*(len(axis) for axis in element_axes))
    return Tensor(data=data, dims=dims)


def replace_dim(tensor: Tensor, dim: int, new_dim: StateSpace) -> Tensor:
    """
    Replace the StateSpace at the specified dimension with a new StateSpace.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to modify.
    dim : `int`
        The index of the dimension to replace.
    new_dim : `StateSpace`
        The new StateSpace to assign to the dimension.

    Returns
    -------
    `Tensor`
        A new Tensor with the updated dimension.
    """
    if dim < 0:
        dim += len(tensor.dims)

    if dim < 0 or dim >= len(tensor.dims):
        raise IndexError(
            f"Dimension index {dim} out of range for tensor of rank {len(tensor.dims)}"
        )

    current_size = tensor.data.shape[dim]

    # Check size compatibility
    # If new_dim is BroadcastSpace with no structure (size 0), it matches data dimension of size 1.
    if isinstance(new_dim, BroadcastSpace) and new_dim.dim == 0:
        if current_size != 1:
            raise ValueError(
                f"Cannot replace dimension of size {current_size} with empty BroadcastSpace (expects size 1)."
            )
    elif new_dim.dim != current_size:
        raise ValueError(
            f"New StateSpace size {new_dim.dim} does not match tensor data size {current_size} at dimension {dim}!"
        )

    new_dims = list(tensor.dims)
    new_dims[dim] = new_dim
    return Tensor(data=tensor.data, dims=tuple(new_dims))


def factorize_dim(tensor: Tensor, dim: int, rule: StateSpaceFactorization) -> Tensor:
    """
    Factorize one `StateSpace`-like dimension into multiple subspaces.

    For a tensor with shape `(A, B)`, factorizing the `0`th dimension with
    `factorized=(A1, A2)` and a compatible `align_dim` produces shape
    `(A1, A2, B)`.

    Parameters
    ----------
    `tensor` : `Tensor`
        The tensor to modify.
    `dim` : `int`
        The index of the dimension to factorize.
    `rule` : `StateSpaceFactorization`
        The factorization rule.

    Returns
    -------
    `Tensor`
        A new tensor with the specified dimension factorized.
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
    return Tensor(data=new_data, dims=new_dims)


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


def product_dims(tensor: Tensor, *indices_group: Tuple[int, ...]) -> Tensor:
    """
    Combine selected tensor dimensions into product dimensions.

    Each entry in `indices_group` defines one output product dimension.
    For a group `(i0, i1, ..., ik)`, the returned tensor contains a single
    axis whose size is the product of the grouped axis sizes and whose
    `StateSpace` is `self.dims[i0] @ self.dims[i1] @ ... @ self.dims[ik]`.
    Dimensions not listed in any group are preserved as-is.

    Negative indices are supported and follow Python indexing rules.
    Grouped dimensions do not need to be contiguous in the input tensor; the
    method reorders axes internally, performs one reshape, and returns the
    result in the canonical output order.

    Parameters
    ----------
    `tensor` : `Tensor`
        The tensor to modify.
    `indices_group` : `Tuple[int, ...]`
        One or more non-empty groups of dimension indices to combine.
        Indices must be unique across all groups (a dimension can belong to
        at most one group).

    Returns
    -------
    `Tensor`
        A new tensor where each requested group is replaced by one product
        dimension and all non-grouped dimensions are retained.

    Raises
    ------
    `IndexError`
        If any provided index is out of range for the tensor rank.
    `ValueError`
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
        return cast(StateSpace, acc @ tensor.dims[g_idx])

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

    return Tensor(data=permuted.data.reshape(tuple(new_shape)), dims=tuple(new_dims))
