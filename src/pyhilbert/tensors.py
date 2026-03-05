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
    Literal,
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
    IndexSpace,
    StateSpaceFactorization,
    embedding_order,
    permutation_order,
    same_span,
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

    @dispatch(Number)
    @staticmethod
    def scalar(number: Number) -> "Tensor":
        """
        Create a 0-dimensional `Tensor` from a scalar number.

        Parameters
        ----------
        number : `Number`
            The scalar value to convert into a tensor.

        Returns
        -------
        `Tensor`
            A 0-dimensional tensor containing the given number.
        """
        precision = get_precision_config()
        dtype = (
            precision.torch_complex
            if isinstance(number, complex)
            else precision.torch_float
        )
        data = torch.tensor(number, dtype=dtype)
        return Tensor(data=data, dims=())

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

    def where(
        self,
        input: Optional["Tensor"] = None,
        other: Optional["Tensor"] = None,
    ) -> Union["Tensor", Tuple["Tensor", ...]]:
        """
        Apply `where` using this tensor as the boolean condition mask.

        Supported call forms
        --------------------
        - `condition.where(input, other)`:
          elementwise selection between `input` and `other`.
        - `condition.where()`:
          returns index tensors of `True` entries.

        Parameters
        ----------
        `input` : `Optional[Tensor]`, optional
            Tensor selected where `condition` is `True`.
        `other` : `Optional[Tensor]`, optional
            Tensor selected where `condition` is `False`.

        Returns
        -------
        `Union[Tensor, Tuple[Tensor, ...]]`
            Return value depends on call form:
            - For `condition.where(input, other)`, returns a single `Tensor`
              with `dims == condition.dims`. Data is selected elementwise:
              `input` where condition is `True`, and `other` where condition is
              `False` (after both are aligned to `condition.dims`).
            - For `condition.where()`, returns `Tuple[Tensor, ...]` with one
              1D index tensor per condition axis (same ordering as
              `torch.where(condition)` / `torch.nonzero(as_tuple=True)`).
              Each returned tensor has shape `(nnz,)` and
              `dims == (IndexSpace.linear(nnz),)`, where `nnz` is the number of
              `True` entries in `condition`.

        Raises
        ------
        `TypeError`
            If only one of `input`/`other` is provided.
        """
        if input is None and other is None:
            return where(self)
        if input is None or other is None:
            raise TypeError(
                "Tensor.where supports either where() or where(input, other)"
            )
        return where(self, input, other)

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

    def mean(self, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> "Tensor":
        """
        Compute the mean over specified dimension(s).

        Parameters
        ----------
        dim : `Optional[Union[int, Tuple[int, ...]]]`, optional
            Reduction axis (or axes). If `None`, reduce over all dimensions.

        Returns
        -------
        `Tensor`
            A new tensor with the specified dimensions reduced.
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
        if self.rank() != 0:
            raise ValueError(
                f"Tensor.item() only works for rank-0 tensors, got rank {self.rank()}"
            )
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

    def device(
        self, device: Optional[Literal["cpu", "gpu"]] = None
    ) -> "Tensor" | Literal["cpu", "gpu"]:
        """
        ### Provided `device`
        Copy the tensor data to the specified device and create a new `Tensor` instance.
        See ``Tensor.cpu()`` and ``Tensor.gpu()`` for device-specific behavior and requirements.

        ### No `device` argument
        If `device` is `None`, this returns a string indicating the current device type of the tensor data:
        - Returns `"gpu"` if the data is on a CUDA or MPS device.
        - Returns `"cpu"` if the data is on a CPU device.
        """
        if device is None:
            device_type = self.data.device.type
            return "gpu" if device_type in {"cuda", "mps"} else "cpu"
        elif device == "cpu":
            return self.cpu()
        elif device == "gpu":
            return self.gpu()

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
        Index tensor data with one of three indexing conventions.

        Supported conventions
        ---------------------
        - Normal indexing: when no `StateSpace`/`Convertible`/`Tensor` indices are present,
          this forwards directly to `self.data[key]` and returns a `torch.Tensor`.
        - StateSpace indexing: when any `StateSpace`/`Convertible` index is present,
          this returns a `Tensor` and applies StateSpace-aware selection rules.
        - Advanced Tensor indexing: when any `Tensor` index is present, this
          returns a `Tensor` and applies torch advanced-index semantics with
          `Tensor` indices plus `:`, `...`, and `None`.

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
        - In advanced Tensor mode, only `Tensor`, full slices `:`, `...`, and
          `None` are supported.
        """
        if key is Ellipsis:
            key = (slice(None),) * len(self.dims)
        elif not isinstance(key, tuple):
            key = (key,)

        ellipsis_at = next((i for i, k in enumerate(key) if k is Ellipsis), -1)
        if ellipsis_at != -1:
            missing = len(self.dims) - (len(key) - 1)
            key = key[:ellipsis_at] + (slice(None),) * missing + key[ellipsis_at + 1 :]

        non_none = sum(1 for k in key if k is not None)
        if non_none < len(self.dims):
            key = key + (slice(None),) * (len(self.dims) - non_none)
        if non_none > len(self.dims):
            raise IndexError("Too many indices for tensor")

        has_tensor_indices = any(isinstance(k, Tensor) for k in key)
        has_hilbert_indices = any(
            isinstance(k, StateSpace)
            or (isinstance(k, Convertible) and not isinstance(k, Tensor))
            for k in key
        )
        if has_tensor_indices:
            if has_hilbert_indices:
                raise ValueError(
                    "Tensor advanced indexing cannot be mixed with StateSpace/Convertible indexing"
                )
            for k in key:
                if k is None:
                    continue
                if isinstance(k, Tensor):
                    continue
                if isinstance(k, slice) and k == slice(None, None, None):
                    continue
                raise TypeError(
                    "Advanced Tensor indexing only supports Tensor indices, full ':' slices, and None"
                )
            return _tensor_getitem_advanced(self, key)

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
        if not isinstance(k, slice) and isinstance(k, Convertible):
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


def _tensor_getitem_advanced(tensor: Tensor, key: Tuple[object, ...]) -> Tensor:
    # Drop `None` entries first; they insert output axes but do not consume source axes.
    core_key = tuple(k for k in key if k is not None)
    # Record where advanced (pyhilbert Tensor) indices appear in the core key.
    tensor_positions = [i for i, k in enumerate(core_key) if isinstance(k, Tensor)]
    # Guard: this helper must only run when at least one advanced index is present.
    if not tensor_positions:
        raise ValueError("_tensor_getitem_advanced requires at least one Tensor index")

    # Collect advanced index tensors in key order.
    tensor_indices = [cast(Tensor, k) for k in core_key if isinstance(k, Tensor)]
    # Broadcast/align all advanced index tensors for both data and metadata dims.
    aligned_indices, advanced_dims = _broadcast_advanced_index_tensors(tensor_indices)

    # Replace advanced indices in the core key with aligned raw torch tensors.
    aligned_iter = iter(aligned_indices)
    # Build a torch-compatible key without `None` entries.
    aligned_core_key: list[object] = []
    # Walk each token and substitute `.data` for advanced index tensors.
    for k in core_key:
        # Advanced index token -> consume next aligned tensor index.
        if isinstance(k, Tensor):
            aligned_core_key.append(next(aligned_iter).data)
        # Basic token (currently `:`) is kept unchanged.
        else:
            aligned_core_key.append(k)
    # Final core key used for shape/rank inference.
    torch_core_key = tuple(aligned_core_key)
    # Run core indexing once (without `None`) to infer advanced-axis placement.
    core_data = tensor.data[torch_core_key]

    # Rebuild full key including `None` entries for the actual indexing output.
    aligned_iter = iter(aligned_indices)
    # `None` stays as `None`; advanced Tensor tokens become aligned `.data`.
    torch_key_full = tuple(
        next(aligned_iter).data if isinstance(k, Tensor) else k for k in key
    )
    # Execute final torch indexing result data.
    data = tensor.data[torch_key_full]

    # Set form for contiguous advanced-block detection.
    tensor_pos_set = set(tensor_positions)
    # True if all advanced index positions form one contiguous block in `core_key`.
    contiguous_advanced = max(tensor_pos_set) - min(tensor_pos_set) + 1 == len(
        tensor_pos_set
    )

    # Count basic preserved axes (`:`) in core key.
    num_basic = sum(
        1 for k in core_key if isinstance(k, slice) and k == slice(None, None, None)
    )
    # Contiguous advanced-index case: advanced axes stay at block position.
    if contiguous_advanced:
        # First and last positions of the advanced block.
        first = min(tensor_pos_set)
        last = max(tensor_pos_set)
        # Number of preserved basic axes before the advanced block.
        basic_before = sum(
            1
            for idx, k in enumerate(core_key)
            if idx < first and isinstance(k, slice) and k == slice(None, None, None)
        )
        # Number of preserved basic axes after the advanced block.
        basic_after = sum(
            1
            for idx, k in enumerate(core_key)
            if idx > last and isinstance(k, slice) and k == slice(None, None, None)
        )
        # Infer advanced rank in the core output.
        advanced_rank = core_data.ndim - basic_before - basic_after
        # Sanity guard against invalid inference.
        if advanced_rank < 0:
            raise ValueError(
                "Invalid advanced indexing rank produced by torch indexing"
            )
        # Extract expected advanced output shape segment from core output.
        expected_shape = core_data.shape[basic_before : basic_before + advanced_rank]
    # Separated advanced-index case: advanced axes are moved to front by torch.
    else:
        # Infer advanced rank in separated layout.
        advanced_rank = core_data.ndim - num_basic
        # Sanity guard against invalid inference.
        if advanced_rank < 0:
            raise ValueError(
                "Invalid advanced indexing rank produced by torch indexing"
            )
        # In separated case, advanced shape is the leading segment.
        expected_shape = core_data.shape[:advanced_rank]

    # Safety fallback: if metadata dims and actual torch shape diverge, use linear dims.
    if tuple(dim.dim for dim in advanced_dims) != tuple(int(s) for s in expected_shape):
        advanced_dims = tuple(IndexSpace.linear(int(size)) for size in expected_shape)

    # Build per-core-axis dim contributions before re-inserting `None` axes.
    segments: list[Tuple[StateSpace, ...]] = []
    # Anchor where advanced dims should appear for contiguous case.
    first_tensor_pos = min(tensor_pos_set)
    # Map each core key token to output dims contribution.
    for axis, axis_key in enumerate(core_key):
        # `:` preserves the corresponding source axis dim.
        if isinstance(axis_key, slice):
            segments.append((tensor.dims[axis],))
            continue
        # In contiguous mode, only the first advanced token emits advanced dims.
        if contiguous_advanced and axis == first_tensor_pos:
            segments.append(advanced_dims)
        # Other advanced tokens do not contribute extra output axes.
        else:
            segments.append(tuple())

    # In separated mode, advanced dims appear at the very front.
    result_dims: list[StateSpace] = (
        list(advanced_dims) if not contiguous_advanced else []
    )
    # Pointer through core segments while replaying original key (with `None`).
    non_none_axis = 0
    # Reconstruct final output dims in original key order.
    for axis_key in key:
        # `None` inserts a size-1 broadcast axis.
        if axis_key is None:
            result_dims.append(BroadcastSpace())
            continue
        # Non-None token consumes one core-axis segment.
        result_dims.extend(segments[non_none_axis])
        non_none_axis += 1

    # Return wrapped Tensor with torch-indexed data and computed output dims.
    return Tensor(data=data, dims=tuple(result_dims))


def _broadcast_advanced_index_tensors(
    indices: list[Tensor],
) -> Tuple[list[Tensor], Tuple[StateSpace, ...]]:
    # Guard: advanced indexing requires at least one tensor index.
    if not indices:
        raise ValueError("At least one advanced index tensor is required")

    # Determine broadcast working rank (left-pad all indices to this rank).
    max_rank = max(idx.rank() for idx in indices)

    # Left-pad each index tensor with leading BroadcastSpace axes.
    padded_indices: list[Tensor] = []
    # Process each raw advanced index tensor.
    for idx in indices:
        # Start from original index tensor.
        padded = idx
        # Insert leading singleton axes until ranks match.
        for _ in range(max_rank - idx.rank()):
            padded = padded.unsqueeze(0)
        # Save rank-normalized tensor.
        padded_indices.append(padded)

    # Use torch broadcast rules on raw data shapes to validate index broadcastability.
    try:
        target_shape = tuple(
            int(s)
            for s in torch.broadcast_shapes(*(idx.data.shape for idx in padded_indices))
        )
    # Convert torch runtime broadcast failure into package-level ValueError.
    except RuntimeError as e:
        raw_shapes = ", ".join(str(tuple(idx.data.shape)) for idx in padded_indices)
        raise ValueError(
            f"Advanced index tensors are not broadcastable in shape: shapes=[{raw_shapes}]"
        ) from e

    # Merge StateSpace metadata axis-wise across padded index tensors.
    try:
        base_union = union_dims(*(idx.dims for idx in padded_indices))
    # Convert metadata merge failure into advanced-index specific ValueError.
    except ValueError as e:
        dim_sigs = ", ".join(_format_dims(idx.dims) for idx in padded_indices)
        raise ValueError(
            f"Advanced index tensor dims are incompatible for broadcast: dims=[{dim_sigs}]"
        ) from e

    # If union metadata is still BroadcastSpace on an axis but torch's data
    # broadcast shape is concrete (>1), materialize that axis as IndexSpace.
    target_dims = tuple(
        IndexSpace.linear(size)
        if isinstance(dim, BroadcastSpace) and size != 1
        else dim
        for dim, size in zip(base_union, target_shape)
    )

    # Reorder/align each index tensor metadata to merged target dims.
    aligned = [idx.align_all(target_dims) for idx in padded_indices]
    # Realize data broadcasting to the merged target shape where needed.
    expanded = [expand_to_union(idx, list(target_dims)) for idx in aligned]
    # Return fully broadcasted advanced indices and their merged dims.
    return expanded, target_dims


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


@dispatch(Tensor, Tensor)
def operator_eq(left: Tensor, right: Tensor) -> Tensor:
    """
    Perform element-wise equality comparison between two tensors.

    Behavior follows symmetric broadcast comparison:
    - computes strict shared union dims with `union_dims(..., allow_merge=False)`
    - aligns both operands to the union dims
    - relies on torch runtime broadcasting for singleton/broadcast-backed axes
    - returns output with `dims == union_dims`
    """
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
    return Tensor(data=aligned_left.data == aligned_right.data, dims=merged_dims)


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
    iden = eye(right.dims)
    return left * iden + right


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
    iden = eye(left.dims)
    return left + right * iden


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
    iden = eye(right.dims)
    return left * iden + (-right)


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
    iden = eye(left.dims)
    return left + (-right) * iden


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
    if current_dim is target_dim or current_dim == target_dim:
        return tensor

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
            f"current={type(current_dim).__name__}:{current_dim.dim} vs "
            f"target={type(target_dim).__name__}:{target_dim.dim}"
        )
    if not same_span(current_dim, target_dim):
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
            f"Cannot align rank-{tensor.rank()} tensor to rank-{len(dims)} dims: "
            f"current={_format_dims(tensor.dims)}, target={_format_dims(dims)}"
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


def mean(tensor: Tensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
    """
    Compute the mean over specified dimension(s), matching `torch.mean` dim forms.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to reduce.
    dim : `Optional[Union[int, Tuple[int, ...]]]`, optional
        Reduction axis (or axes). If `None`, reduce over all dimensions.

    Returns
    -------
    `Tensor`
        A new tensor with the specified dimensions reduced.
    """
    if dim is None:
        return Tensor(data=tensor.data.mean(), dims=())

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
    return Tensor(data=reduced, dims=new_dims)


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


def union_dims(
    *dims: Tuple[StateSpace, ...], allow_merge: bool = False
) -> Tuple[StateSpace, ...]:
    """
    Compute a broadcast-compatible union of multiple dimension tuples.

    This function merges dimension metadata axis-by-axis across one or more
    tuples of `StateSpace`s. All input tuples must have the same rank.

    Merge rule per axis (`allow_merge=False`)
    -------------------
    - `BroadcastSpace` + concrete `StateSpace` -> concrete `StateSpace`
    - `BroadcastSpace` + `BroadcastSpace` -> `BroadcastSpace`
    - concrete + concrete:
      - if `same_span(...)` is `True`, keeps the first (left-most) one
      - otherwise raises `ValueError`

    Merge rule per axis (`allow_merge=True`)
    -----------------------------------------------
    - Uses StateSpace union semantics directly (`left_dim + right_dim`) after
      rank checks. This supports disjoint-axis union behavior used by tensor add.

    Parameters
    ----------
    `dims` : `Tuple[StateSpace, ...]`
        One or more dimension tuples to merge.
    `allow_merge` : `bool`, optional
        If `False`, enforces strict compatibility for concrete dimensions.
        If `True`, merges concrete dimensions via `+`.

    Returns
    -------
    `Tuple[StateSpace, ...]`
        Merged dimensions with the same rank as each input tuple.

    Raises
    ------
    `ValueError`
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
                if same_span(left_dim, right_dim):
                    continue
                raise ValueError(
                    f"union_dims incompatible at axis {axis}: "
                    f"{type(left_dim).__name__}:{left_dim.dim} vs "
                    f"{type(right_dim).__name__}:{right_dim.dim}; "
                    f"left={_format_dims(tuple(merged))}, right={_format_dims(current)}"
                )
            merged[axis] = cast(StateSpace, left_dim + right_dim)

    return tuple(merged)


def mapping_matrix(
    from_space: StateSpace,
    to_space: StateSpace,
    mapping: Dict[Any, Any],
    factors: Optional[Dict[Tuple[Any, Any], int | float | complex]] = None,
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
        implied by the integer indices in `from_space.structure` and
        `to_space.structure`.
    `factors` : `Optional[Dict[Tuple[Any, Any], int | float | complex]]`, optional
        Optional per-entry factors. Keys are `(from_marker, to_marker)` tuples.
        Scalar values scale entries. Missing keys default to `1`.

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
        findex = from_space.structure[fm]
        tindex = to_space.structure[tm]
        factor = factors.get((fm, tm), 1)
        mat[findex, tindex] = cast(Any, factor)

    return Tensor(data=mat, dims=(from_space, to_space))


def eye(dims: Tuple[StateSpace, ...]) -> Tensor:
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


def _format_dims(dims: Tuple[StateSpace, ...]) -> str:
    if not dims:
        return "()"
    return "(" + ", ".join(f"{type(dim).__name__}:{dim.dim}" for dim in dims) + ")"


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


@dispatch(Tensor, Tensor, Tensor)
def where(condition: Tensor[torch.BoolTensor], input: Tensor, other: Tensor) -> Tensor:
    """
    Select values from `input` and `other` using a boolean mask.

    This is the StateSpace-aware wrapper of `torch.where(condition, input, other)`.
    The three tensors are first symmetrically aligned/broadcast to shared union
    dims, then selection is applied elementwise:

    - if `condition[i]` is `True`, output uses `input[i]`
    - otherwise, output uses `other[i]`

    Parameters
    ----------
    `condition` : `Tensor[torch.BoolTensor]`
        Boolean mask tensor that defines the output dimensions.
    `input` : `Tensor`
        Values chosen where the mask is `True`. Must be alignable to
        `condition.dims`.
    `other` : `Tensor`
        Values chosen where the mask is `False`. Must be alignable to
        `condition.dims`.

    Returns
    -------
    `Tensor`
        Tensor with `dims == condition.dims`.

    Raises
    ------
    `TypeError`
        If `condition.data` is not boolean.
    `ValueError`
        If `input` or `other` cannot be aligned to `condition.dims`.
    """
    if condition.data.dtype != torch.bool:
        raise TypeError("where expects condition.data to have dtype torch.bool")

    merged_dims = union_dims(condition.dims, input.dims, other.dims, allow_merge=False)
    condition = condition.align_all(merged_dims).expand_to_union(list(merged_dims))
    input = input.align_all(merged_dims).expand_to_union(list(merged_dims))
    other = other.align_all(merged_dims).expand_to_union(list(merged_dims))
    return Tensor(
        data=torch.where(condition.data, input.data, other.data),
        dims=merged_dims,
    )


@dispatch(Tensor)  # type: ignore[no-redef]
def where(condition: Tensor[torch.BoolTensor]) -> Tuple[Tensor, ...]:
    """
    Return coordinate tensors of `True` entries in a boolean mask.

    This is the StateSpace-aware wrapper of `torch.where(condition)` and follows
    `torch.nonzero(condition, as_tuple=True)` semantics.

    For a mask of rank `R`, this returns `R` tensors. The `k`-th tensor stores
    the indices along axis `k` for each `True` element. All returned tensors are
    1D and share the same length, equal to the number of `True` entries.

    The 1D dimension on each returned tensor is `IndexSpace.linear(nnz)`, where
    `nnz` is the number of selected positions.

    Parameters
    ----------
    `condition` : `Tensor[torch.BoolTensor]`
        Boolean mask tensor.

    Returns
    -------
    `Tuple[Tensor, ...]`
        Tuple of index tensors with length matching `torch.where(condition)`.

    Raises
    ------
    `TypeError`
        If `condition.data` is not boolean.
    """
    if condition.data.dtype != torch.bool:
        raise TypeError("where expects condition.data to have dtype torch.bool")

    indices = torch.where(condition.data)
    nnz = indices[0].numel() if len(indices) > 0 else 0
    index_dim = IndexSpace.linear(nnz)
    return tuple(Tensor(data=idx, dims=(index_dim,)) for idx in indices)
