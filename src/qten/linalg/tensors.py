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
from functools import wraps, reduce
from itertools import product
from numbers import Number
from dataclasses import dataclass, replace
from types import EllipsisType

from multipledispatch import dispatch  # type: ignore[import-untyped]
import torch

from ..abstracts import Convertible, Operable
from ..plottings import Plottable
from ..precision import get_precision_config
from ..utils.devices import Device, DeviceBounded
from ..validations import need_validation
from ..symbolics.state_space import (
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
class Tensor(Generic[T], Operable, Plottable, Convertible, DeviceBounded):
    """
    StateSpace-aware tensor wrapper over `torch.Tensor`.

    A `Tensor` pairs raw tensor data with symbolic axis metadata in `dims`.
    Each entry of `dims` is a `StateSpace` whose size must match the
    corresponding axis of `data`. This lets tensor operations preserve not only
    shapes, but also the semantic identity and ordering of axes.

    Core model
    ----------
    - `data` stores the underlying numeric values as a `torch.Tensor`.
    - `dims` stores one `StateSpace` per axis.
    - `tensor.data.shape` must equal `tuple(dim.dim for dim in tensor.dims)`.
    - Axes are aligned by `StateSpace` compatibility rather than by position
      alone. If two axes represent the same rays in different orders, QTen can
      permute one operand to match the other.
    - Singleton broadcast axes are represented symbolically by
      `BroadcastSpace()`.

    Construction
    ------------
    Create tensors directly from a torch tensor and a matching dims tuple:

    `Tensor(data=torch.randn(2, 3), dims=(left_space, right_space))`

    Use `Tensor.scalar(number)` to construct a rank-0 tensor.

    Supported operators
    -------------------
    Tensor-Tensor:
    - `a @ b`: StateSpace-aware matrix multiplication / contraction.
    - `a + b`: StateSpace union add with metadata-aware alignment.
    - `a - b`: subtraction.
    - `a == b`: element-wise equality, returns a bool `Tensor`.
    - `a < b`, `a <= b`, `a > b`, `a >= b`: element-wise ordered comparisons,
      return bool `Tensor`s.

    Tensor-Number / Number-Tensor:
    - `a * c`, `c * a`: element-wise scalar multiplication.
    - `a / c`: element-wise scalar division.
    - `a < c`, `a <= c`, `a > c`, `a >= c`: scalar broadcast comparisons.
    - `c < a`, `c <= a`, `c > a`, `c >= a`: reflected scalar broadcast
      comparisons.

    Scalar add/sub semantics:
    - `a + c`, `c + a`, `a - c`, `c - a` treat `a` as a matrix or batch of
      matrices over its last two axes and add/subtract `c * I` on that matrix
      part.
    - These operations therefore require rank at least 2 through `eye(dims)`.
    - Scalar addition is not element-wise broadcasting.

    Comparison semantics
    --------------------
    Comparisons use symmetric StateSpace-aware alignment:
    - operands are rank-promoted with leading `BroadcastSpace()` axes when
      needed,
    - dims are merged with `union_dims(..., allow_merge=False)`,
    - operands are aligned to the merged dims,
    - runtime broadcast compatibility is validated,
    - the result is a bool `Tensor` with those merged dims.

    Ordered comparisons on complex tensors are not supported and defer to
    PyTorch's runtime error behavior.

    Shape and axis transforms
    -------------------------
    - `permute(*order)`: reorder axes.
    - `transpose(dim0, dim1)`: swap two axes.
    - `h(dim0, dim1)`: conjugate transpose across two axes.
    - `unsqueeze(dim)`: insert a singleton `BroadcastSpace` axis.
    - `squeeze(dim)`: remove a `BroadcastSpace` axis if present.
    - `replace_dim(dim, new_dim)`: replace metadata for one axis with size
      validation.
    - `factorize_dim(dim, rule)`: split one axis into multiple factor spaces.
    - `product_dims(*groups)`: combine groups of axes into tensor-product axes.
    - `promote_rank(tensor, target_rank)`: prepend broadcast axes.

    Alignment and metadata utilities
    --------------------------------
    - `align(dim, target_dim)`: align one axis to a compatible `StateSpace`.
    - `align_all(dims)`: align all axes to a target dims tuple.
    - `expand_to_union(union_dims)`: materialize data expansion for
      `BroadcastSpace` axes.
    - `dim_types()`: return the runtime types of all dims.
    - `rank()`: return the tensor rank.

    Reductions and element queries
    ------------------------------
    - `all(dim=None, keepdim=False)`: logical AND reduction.
    - `mean(dim=None)`: arithmetic mean reduction.
    - `argmax(dim)`, `argmin(dim)`: index reductions.
    - `item()`: extract the Python scalar from a rank-0 tensor.
    - `equal(other)`: exact equality after metadata alignment, returning
      Python `bool`.
    - `allclose(other, ...)`: approximate equality after metadata alignment,
      returning Python `bool`.

    Boolean-mask helpers
    --------------------
    - `where(input, other)`: use this tensor as a bool mask and select between
      two tensors.
    - `where()`: return index tensors for `True` entries.
    - `nonzero(as_tuple=True)`: return index tensors for nonzero / `True`
      entries.

    Indexing
    --------
    `__getitem__` supports:
    - Python integers, slices, `None`, and `...`,
    - `StateSpace` / `Convertible` axis selection and reindexing,
    - `Tensor` advanced indices.

    Tensor advanced indexing is metadata-aware and can align tensor index dims
    before dispatching to torch indexing. Boolean tensor indices are not
    supported in `__getitem__`; use comparison masks together with `where` or
    `nonzero` instead.

    Autograd and devices
    --------------------
    - `attach()`: return a leaf tensor with `requires_grad=True`.
    - `detach()`: detach from autograd without cloning storage.
    - `clone()`: deep-copy tensor data.
    - `grad`: wrapped gradient tensor, if available.
    - `requires_grad`: autograd flag from underlying data.
    - `backward(...)`: autograd backward pass.
    - `device`: logical QTen device view of the underlying torch device.
    - `to_device(device)`: move data to another logical device.

    Factory and module-level companion functions
    --------------------------------------------
    The module also exposes helpers that create or operate on `Tensor`:
    `matmul`, `permute`, `transpose`, `conj`, `unsqueeze`, `squeeze`,
    `align`, `align_all`, `all`, `mean`, `argmax`, `argmin`, `astype`,
    `one_hot`, `equal`, `allclose`, `expand_to_union`, `union_dims`,
    `mapping_matrix`, `eye`, `zeros`, `ones`, `kernel_tensor`,
    `replace_dim`, `factorize_dim`, `product_dims`, `promote_rank`,
    `where`, and `nonzero`.
    """

    data: T
    dims: Tuple[StateSpace, ...]

    @staticmethod
    def scalar(number: Number, *, device: Optional[Device] = None) -> "Tensor":
        """
        Create a 0-dimensional `Tensor` from a scalar number.

        Parameters
        ----------
        number : `Number`
            The scalar value to convert into a tensor.
        device : `Optional[Device]`, optional
            Device to place the scalar on, by default None (CPU).

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
        torch_device = device.torch_device() if device is not None else None
        data = torch.tensor(number, dtype=dtype, device=torch_device)
        return Tensor(data=data, dims=())

    def astype(self, dtype: torch.dtype) -> Self:
        """
        Return a new tensor with the same dims and converted data dtype.

        Parameters
        ----------
        `dtype` : `torch.dtype`
            Target data type.

        Returns
        -------
        `Self`
            A new tensor of the same wrapper type whose data has dtype `dtype`.
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
        `Tensor` instead of a Python bool.
        """
        return isclose(self, other, rtol=rtol, atol=atol, equal_nan=equal_nan)

    def conj(self) -> Self:
        """
        Compute the complex conjugate of the given tensor.

        Returns
        -------
        `Self`
            The complex conjugate of the tensor.
        """
        return conj(self)

    def real(self) -> Self:
        """
        Return the real part of the tensor, preserving dims.

        For complex-valued tensors this drops the imaginary component. For
        real-valued tensors this returns a tensor with the same values.
        """
        return real(self)

    def imag(self) -> Self:
        """
        Return the imaginary part of the tensor, preserving dims.

        For complex-valued tensors this returns the imaginary component. For
        real-valued tensors this returns zeros with the corresponding real dtype.
        """
        return imag(self)

    def abs(self) -> Self:
        """
        Return the element-wise absolute value / magnitude of the tensor.

        For complex-valued tensors this returns the magnitude.
        """
        return abs(self)

    def permute(self, *order: Union[int, Sequence[int]]) -> Self:
        """
        Permute the dimensions according to the specified order.

        Parameters
        ----------
        order : `Union[int, Sequence[int]]`
            The desired order of dimensions.

        Returns
        -------
        `Self`
            The permuted tensor.
        """
        return permute(self, *order)

    def transpose(self, dim0: int, dim1: int) -> Self:
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
        `Self`
            The transposed tensor.
        """
        return transpose(self, dim0, dim1)

    def h(self, dim0: int, dim1: int) -> Self:
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
        `Self`
            The Hermitian transposed tensor.
        """
        return self.conj().transpose(dim0, dim1)

    def align(self, dim: int, target_dim: StateSpace) -> Self:
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
        `Self`
            The aligned tensor.
        """
        return align(self, dim, target_dim)

    def align_all(self, dims: Tuple[StateSpace, ...]) -> Self:
        """
        Align all tensor dimensions to `dims`.

        Parameters
        ----------
        dims : `Tuple[StateSpace, ...]`
            The target dimensions to align to.

        Returns
        -------
        `Self`
            The aligned tensor.

        Raises
        ------
        `ValueError`
            If the provided `dims` are not compatible with the tensor's current dimensions.
        """
        return align_all(self, dims)

    def all(
        self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False
    ) -> Self:
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
        `Self`
            Boolean tensor after reduction.
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

        Parameters
        ----------
        `input` : `Optional[Tensor]`, optional
            Tensor selected where `condition` is `True`.
        `other` : `Optional[Tensor]`, optional
            Tensor selected where `condition` is `False`.
        `index_type` : `Type[Any]`, optional
            Only used for the condition-only form. Supported values are
            `Tensor`, `tuple` / `Tuple`, and `StateSpace`.

        Returns
        -------
        `Union[Tensor, Tuple[Tensor, ...], Tuple[Tuple[int, ...], ...], StateSpace]`
            Return value depends on call form:
            - For `condition.where(input, other)`, returns a single `Tensor`
              with `dims == union_dims(condition.dims, input.dims, other.dims,
              allow_merge=False)`. Data is selected elementwise after all
              operands are aligned/broadcast to these merged dims.
            - For `condition.where()`, returns `Tuple[Tensor, ...]` with one
              1D index tensor per condition axis (same ordering as
              `torch.where(condition)` / `torch.nonzero(as_tuple=True)`).
              Each returned tensor has shape `(nnz,)` and
              `dims == (IndexSpace.linear(nnz),)`, where `nnz` is the number of
              `True` entries in `condition`.
            - For `condition.where(index_type=Tensor)`, returns
              `Tuple[Tensor, ...]`.
            - For `condition.where(index_type=tuple)` or `Tuple`, returns one
              Python coordinate tuple per `True` entry.
            - For `condition.where(index_type=StateSpace)`, returns the
              selected subspace for rank-1 conditions only.

        Raises
        ------
        `TypeError`
            If only one of `input`/`other` is provided.
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
        `as_tuple` : `bool`, optional
            Must be `True`.
        `index_type` : `Type[Any]`, optional
            Requested index representation. Supported values are `Tensor`,
            `tuple` / `Tuple`, and `StateSpace`.

        Returns
        -------
        `Union[Tuple[Tensor, ...], Tuple[Tuple[int, ...], ...], StateSpace]`
            Index representation selected by `index_type`.

        Raises
        ------
        `NotImplementedError`
            If `as_tuple` is `False`.
        """
        if index_type is None:
            index_type = Tensor
        return nonzero(self, as_tuple=as_tuple, index_type=index_type)

    def unsqueeze(self, dim: int) -> Self:
        """
        Unsqueeze the specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to unsqueeze.

        Returns
        -------
        `Self`
            The unsqueezed tensor.
        """
        return unsqueeze(self, dim)

    def squeeze(self, dim: int) -> Self:
        """
        Squeeze the specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to squeeze.

        Returns
        -------
        `Self`
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

    def mean(self, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> Self:
        """
        Compute the mean over specified dimension(s).

        Parameters
        ----------
        dim : `Optional[Union[int, Tuple[int, ...]]]`, optional
            Reduction axis (or axes). If `None`, reduce over all dimensions.

        Returns
        -------
        `Self`
            A new tensor with the specified dimensions reduced.
        """
        return mean(self, dim)

    def argmax(self, dim: int) -> Self:
        """
        Compute the indices of the maximum values over a specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to reduce.

        Returns
        -------
        `Self`
            A new tensor with the specified dimension reduced.
        """
        return argmax(self, dim)

    def argmin(self, dim: int) -> Self:
        """
        Compute the indices of the minimum values over a specified dimension.

        Parameters
        ----------
        dim : `int`
            The dimension to reduce.

        Returns
        -------
        `Self`
            A new tensor with the specified dimension reduced.
        """
        return argmin(self, dim)

    def expand_to_union(self, union_dims: list[StateSpace]) -> Self:
        """
        Expand the tensor to the union of the specified dimensions.

        Parameters
        ----------
        union_dims : `list[StateSpace]`
            The dimensions to expand to the union of.

        Returns
        -------
        `Self`
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

    @override
    @property
    def device(self) -> Device:
        """
        Return the logical device associated with the tensor data.
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
        `bool`
            True if the tensor data requires gradient tracking, False otherwise.
        """
        return self.data.requires_grad

    @property
    def grad(self) -> Optional[Self]:
        """
        Return the accumulated gradient wrapped as a QTen tensor.

        Returns
        -------
        `Optional[Self]`
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
        `gradient` : `Optional[Self]`, optional
            Upstream gradient for non-scalar tensors.
        `retain_graph` : `Optional[bool]`, optional
            Whether to retain the autograd graph after backward.
        `create_graph` : `bool`, optional
            Whether to construct the derivative graph.
        `inputs` : `Optional[Sequence[Self]]`, optional
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
        Enable gradient tracking for the tensor data and return the attached `Tensor` instance.

        Behavior
        --------
        - If `requires_grad` is already `True`, this returns `self` unchanged.
        - Otherwise, this detaches the underlying data from any existing autograd graph,
          clones it to ensure a fresh leaf tensor, and sets `requires_grad` to `True`.
        - The returned tensor preserves the original `dims`, device, and dtype.

        Returns
        -------
        `Self`
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
        Disable gradient tracking for the tensor data and create a new `Tensor` instance.

        Behavior
        --------
        - Always returns a new `Tensor` whose data is a detached view of the
          original tensor (no clone), so it shares storage with the original.
        - The returned tensor preserves the original `dims`, device, and dtype.

        Returns
        -------
        `Self`
            The new tensor of the same wrapper type with gradient tracking disabled.
        """
        return replace(self, data=cast(T, self.data.detach()))

    def clone(self) -> Self:
        """
        Create a deep copy of the tensor.

        Returns
        -------
        `Self`
            The cloned tensor.
        """
        return replace(self, data=cast(T, self.data.clone()))

    def replace_dim(self, dim: int, new_dim: StateSpace) -> Self:
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
        `Self`
            A new tensor of the same wrapper type with the updated dimension.
        """
        return replace_dim(self, dim, new_dim)

    def __getitem__(self, key):
        """
        Index tensor data with `TensorIndexing` and return a new `Tensor`.

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
          - full slice `:` preserves the current `StateSpace`,
          - non-full slice uses `self.dims[axis][slice]`, except
            `BroadcastSpace` axes where metadata follows sliced size:
            size `1` keeps `BroadcastSpace`, size `0` becomes
            `IndexSpace.linear(0)`.
        - `None`: inserts a new output axis with `BroadcastSpace` and does not
          consume a source axis.
        - `StateSpace` / `Convertible`:
          - indexing a `BroadcastSpace` axis with any non-`BroadcastSpace`
            `StateSpace` is rejected (`IndexError`),
          - if equal to current axis space: behaves like full slice,
          - if same span: uses permutation indexing and output dim is the index
            `StateSpace`,
          - if contained subspace: uses embedding indexing and output dim is the
            subspace,
          - otherwise raises `IndexError`.
        - `Tensor` index:
          - `bool` dtype is not supported (`NotImplementedError`),
          - lower-rank tensor indices are left-padded with singleton/broadcast
            axes to match the maximum tensor-index rank before metadata
            union/alignment,
          - index metadata is aligned to the union of all tensor-index dims.

        Mode rules
        ----------
        - Mixing `Tensor` indices with `StateSpace`/`Convertible` indices is
          rejected (`ValueError`).
        - If at least one `Tensor` index is present, data indexing is executed
          in one torch advanced-indexing call.
        - Otherwise indexing is applied step-by-step per axis (including tuple
          index_select steps), so per-axis `StateSpace` tuple indices are not
          jointly broadcast by torch.

        Output dim ordering for tensor advanced indexing
        -----------------------------------------------
        Let `tensor_union_dims` be the broadcast/union dims of all tensor index
        tensors.
        - If tensor index tokens form one contiguous block in the normalized key,
          `tensor_union_dims` is inserted at that block position.
        - If tensor index tokens are separated by non-tensor tokens,
          `tensor_union_dims` is moved to the front of output dims.
        """
        if not isinstance(key, tuple):
            key = (key,)
        compiled = TensorIndexing(self.dims, key).compile()
        if compiled.has_tensor_index:
            data = self.data[compiled.indices]
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
        `Self`
            A new tensor of the same wrapper type with the specified dimension factorized.
        """
        return factorize_dim(self, dim, rule)

    def product_dims(self, *indices_group: Tuple[int, ...]) -> Self:
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
        `Self`
            A new tensor of the same wrapper type where each requested group is replaced by one product
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
                left = replace(left, data=left.data.to(common_dtype))
            if right.data.dtype != common_dtype:
                right = replace(right, data=right.data.to(common_dtype))
        return func(left, right, *args, **kwargs)

    return wrapper


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
        right = promote_rank(right, left.rank())
    elif right.rank() > left.rank():
        left = promote_rank(left, right.rank())
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
def operator_eq(left: TensorType, right: Tensor) -> TensorType:
    """
    Perform element-wise equality comparison between two tensors.

    Behavior follows symmetric broadcast comparison:
    - computes strict shared union dims with `union_dims(..., allow_merge=False)`
    - aligns both operands to the union dims
    - relies on torch runtime broadcasting for singleton/broadcast-backed axes
    - returns output with `dims == union_dims`
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


@dispatch(Tensor, Tensor)
def operator_lt(left: TensorType, right: Tensor) -> TensorType:
    """Perform element-wise less-than comparison between two tensors."""
    return _tensor_comparison_op(left, right, torch.lt)


@dispatch(Tensor, Tensor)
def operator_le(left: TensorType, right: Tensor) -> TensorType:
    """Perform element-wise less-than-or-equal comparison between two tensors."""
    return _tensor_comparison_op(left, right, torch.le)


@dispatch(Tensor, Tensor)
def operator_gt(left: TensorType, right: Tensor) -> TensorType:
    """Perform element-wise greater-than comparison between two tensors."""
    return _tensor_comparison_op(left, right, torch.gt)


@dispatch(Tensor, Tensor)
def operator_ge(left: TensorType, right: Tensor) -> TensorType:
    """Perform element-wise greater-than-or-equal comparison between two tensors."""
    return _tensor_comparison_op(left, right, torch.ge)


@dispatch(Tensor, Number)
def operator_lt(left: TensorType, right: Number) -> TensorType:
    """Perform element-wise less-than comparison between a tensor and a scalar."""
    return operator_lt(left, Tensor.scalar(right))


@dispatch(Number, Tensor)
def operator_lt(left: Number, right: TensorType) -> TensorType:  # type: ignore[no-redef]
    """Perform element-wise less-than comparison between a scalar and a tensor."""
    return operator_lt(Tensor.scalar(left), right)


@dispatch(Tensor, Number)
def operator_le(left: TensorType, right: Number) -> TensorType:
    """Perform element-wise less-than-or-equal comparison between a tensor and a scalar."""
    return operator_le(left, Tensor.scalar(right))


@dispatch(Number, Tensor)
def operator_le(left: Number, right: TensorType) -> TensorType:  # type: ignore[no-redef]
    """Perform element-wise less-than-or-equal comparison between a scalar and a tensor."""
    return operator_le(Tensor.scalar(left), right)


@dispatch(Tensor, Number)
def operator_gt(left: TensorType, right: Number) -> TensorType:
    """Perform element-wise greater-than comparison between a tensor and a scalar."""
    return operator_gt(left, Tensor.scalar(right))


@dispatch(Number, Tensor)
def operator_gt(left: Number, right: TensorType) -> TensorType:  # type: ignore[no-redef]
    """Perform element-wise greater-than comparison between a scalar and a tensor."""
    return operator_gt(Tensor.scalar(left), right)


@dispatch(Tensor, Number)
def operator_ge(left: TensorType, right: Number) -> TensorType:
    """Perform element-wise greater-than-or-equal comparison between a tensor and a scalar."""
    return operator_ge(left, Tensor.scalar(right))


@dispatch(Number, Tensor)
def operator_ge(left: Number, right: TensorType) -> TensorType:  # type: ignore[no-redef]
    """Perform element-wise greater-than-or-equal comparison between a scalar and a tensor."""
    return operator_ge(Tensor.scalar(left), right)


@dispatch(Tensor)
def operator_neg(tensor: TensorType) -> TensorType:
    """
    Perform negation on the given tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to negate.

    Returns
    -------
    `TensorType`
        The negated tensor, preserving the input wrapper type.
    """
    return replace(tensor, data=-tensor.data)


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


def permute(tensor: TensorType, *order: Union[int, Sequence[int]]) -> TensorType:
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
    `TensorType`
        The permuted tensor, preserving the input wrapper type.
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
    `TensorType`
        The transposed tensor, preserving the input wrapper type.
    """
    new_data = tensor.data.transpose(dim0, dim1)

    # Convert tuple to list to modify
    new_dims_list = list(tensor.dims)
    # Swap elements
    new_dims_list[dim0], new_dims_list[dim1] = new_dims_list[dim1], new_dims_list[dim0]

    return replace(tensor, data=new_data, dims=tuple(new_dims_list))


def conj(tensor: TensorType) -> TensorType:
    """
    Compute the complex conjugate of the given tensor.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to conjugate.

    Returns
    -------
    `TensorType`
        The complex conjugate of the tensor, preserving the input wrapper type.
    """
    return replace(tensor, data=tensor.data.conj())


def real(tensor: TensorType) -> TensorType:
    """
    Return the real part of a tensor, preserving dims and wrapper type.
    """
    return replace(tensor, data=cast(T, tensor.data.real))


def imag(tensor: TensorType) -> TensorType:
    """
    Return the imaginary part of a tensor, preserving dims and wrapper type.

    For real-valued tensors this is a zero tensor with the corresponding real dtype.
    """
    if tensor.data.is_complex():
        return replace(tensor, data=cast(T, tensor.data.imag))
    return replace(tensor, data=cast(T, torch.zeros_like(tensor.data)))


def abs(tensor: TensorType) -> TensorType:
    """
    Return the element-wise absolute value / magnitude of a tensor, preserving
    dims and wrapper type.
    """
    return replace(tensor, data=cast(T, tensor.data.abs()))


def unsqueeze(tensor: TensorType, dim: int) -> TensorType:
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
    `TensorType`
        The unsqueezed tensor, preserving the input wrapper type.
    """
    if dim < 0:
        dim = dim + len(tensor.dims) + 1
    new_data = tensor.data.unsqueeze(dim)
    new_dims = tensor.dims[:dim] + (BroadcastSpace(),) + tensor.dims[dim:]

    return replace(tensor, data=new_data, dims=new_dims)


def squeeze(tensor: TensorType, dim: int) -> TensorType:
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
    `TensorType`
        The squeezed tensor, preserving the input wrapper type.
    """
    if dim < 0:
        dim = dim + len(tensor.dims)
    if not isinstance(tensor.dims[dim], BroadcastSpace):
        return tensor  # No squeezing needed if not BroadcastSpace

    new_data = tensor.data.squeeze(dim)
    new_dims = tensor.dims[:dim] + tensor.dims[dim + 1 :]

    return replace(tensor, data=new_data, dims=new_dims)


def align(tensor: TensorType, dim: int, target_dim: StateSpace) -> TensorType:
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
    `TensorType`
        The aligned tensor, preserving the input wrapper type.
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
    Align all dimensions of `tensor` to `dims`.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to align.
    dims : `Tuple[StateSpace, ...]`
        Target dimensions for each axis.

    Returns
    -------
    `TensorType`
        The aligned tensor, preserving the input wrapper type.

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
    tensor: TensorType,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
) -> TensorType:
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
    `TensorType`
        Boolean tensor with reduced dimensions, preserving the input wrapper type.
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
    tensor : `Tensor`
        The tensor whose rank is to be determined.

    Returns
    -------
    `int`
        The rank of the tensor.
    """
    return len(tensor.dims)


def mean(
    tensor: TensorType, dim: Optional[Union[int, Tuple[int, ...]]] = None
) -> TensorType:
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
    `TensorType`
        A new tensor with the specified dimensions reduced, preserving the input wrapper type.
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


def argmax(tensor: TensorType, dim: int) -> TensorType:
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
    `TensorType`
        A new tensor with the specified dimension reduced, preserving the input wrapper type.
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
    Compute the indices of the minimum values over a specified dimension.

    Parameters
    ----------
    tensor : `Tensor`
        The tensor to reduce.
    dim : `int`
        The dimension to reduce.

    Returns
    -------
    `TensorType`
        A new tensor with the specified dimension reduced, preserving the input wrapper type.
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


def astype(tensor: TensorType, dtype: torch.dtype) -> TensorType:
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
    `TensorType`
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

    This returns a bool `Tensor` mask, unlike `allclose`, which reduces to a
    Python bool.
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


def expand_to_union(tensor: TensorType, union_dims: list[StateSpace]) -> TensorType:
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

    return replace(tensor, data=tensor.data.expand(target_shape), dims=tuple(new_dims))


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
      - if `same_rays(...)` is `True`, keeps the first (left-most) one
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
    torch_device = device.torch_device() if device is not None else None
    return Tensor(data=torch.eye(rows, cols, device=torch_device), dims=matrix_dims)


def zeros(dims: Tuple[StateSpace, ...], *, device: Optional[Device] = None) -> Tensor:
    """
    Create a zero-filled tensor with shape defined by `dims`.

    Parameters
    ----------
    dims : `Tuple[StateSpace, ...]`
        StateSpace dimensions defining the tensor shape.
    device : `Optional[Device]`, optional
        Device to place the tensor on, by default None (CPU).

    Returns
    -------
    `Tensor`
        A tensor of zeros with `shape == tuple(dim.dim for dim in dims)`.
    """
    shape = tuple(dim.dim for dim in dims)
    torch_device = device.torch_device() if device is not None else None
    return Tensor(data=torch.zeros(shape, device=torch_device), dims=dims)


def ones(dims: Tuple[StateSpace, ...], *, device: Optional[Device] = None) -> Tensor:
    """
    Create a one-filled tensor with shape defined by `dims`.

    Parameters
    ----------
    dims : `Tuple[StateSpace, ...]`
        StateSpace dimensions defining the tensor shape.
    device : `Optional[Device]`, optional
        Device to place the tensor on, by default None (CPU).

    Returns
    -------
    `Tensor`
        A tensor of ones with `shape == tuple(dim.dim for dim in dims)`.
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


def replace_dim(tensor: TensorType, dim: int, new_dim: StateSpace) -> TensorType:
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
    `TensorType`
        A new tensor with the updated dimension, preserving the input wrapper type.
    """
    if dim < 0:
        dim += len(tensor.dims)

    if dim < 0 or dim >= len(tensor.dims):
        raise IndexError(
            f"Dimension index {dim} out of range for tensor of rank {len(tensor.dims)}"
        )

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


def factorize_dim(
    tensor: TensorType, dim: int, rule: StateSpaceFactorization
) -> TensorType:
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
    `TensorType`
        A new tensor with the specified dimension factorized, preserving the input wrapper type.
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
    `TensorType`
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

    return replace(
        tensor, data=permuted.data.reshape(tuple(new_shape)), dims=tuple(new_dims)
    )


def promote_rank(tensor: Tensor, target_rank: int) -> Tensor:
    """
    Return `tensor` with leading broadcast axes prepended to reach `target_rank`.

    This function preserves the existing axis order and values while adding
    `target_rank - tensor.rank()` leading singleton axes in `tensor.data`.
    The corresponding leading entries in `dims` are `BroadcastSpace()`.

    Parameters
    ----------
    `tensor` : `Tensor`
        Input tensor.
    `target_rank` : `int`
        Desired output rank. Must satisfy `target_rank >= tensor.rank()`.

    Returns
    -------
    `Tensor`
        `tensor` if no promotion is needed; otherwise a tensor with prepended
        broadcast axes and matching prepended `BroadcastSpace` dims.

    Raises
    ------
    `ValueError`
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


@dispatch(Tensor, Tensor, Tensor)
def where(condition: Tensor[torch.BoolTensor], input: Tensor, other: Tensor) -> Tensor:
    """
    Select values from `input` and `other` using a boolean mask.

    This is the StateSpace-aware wrapper of `torch.where(condition, input, other)`.
    The three tensors are first promoted to a common rank by prepending leading
    `BroadcastSpace` axes (torch-style rank broadcasting), then symmetrically
    aligned/broadcast to shared union dims, and finally selection is applied
    elementwise:

    - if `condition[i]` is `True`, output uses `input[i]`
    - otherwise, output uses `other[i]`

    Parameters
    ----------
    `condition` : `Tensor[torch.BoolTensor]`
        Boolean mask tensor participating in symmetric union alignment.
    `input` : `Tensor`
        Values chosen where the mask is `True`. Must be compatible with
        `condition` and `other` under `union_dims(..., allow_merge=False)`.
    `other` : `Tensor`
        Values chosen where the mask is `False`. Must be compatible with
        `condition` and `input` under `union_dims(..., allow_merge=False)`.

    Returns
    -------
    `Tensor`
        Tensor with `dims == union_dims(condition.dims, input.dims, other.dims,
        allow_merge=False)`.

    Raises
    ------
    `TypeError`
        If `condition.data` is not boolean.
    `ValueError`
        If operands cannot be aligned/broadcast to shared union dims.
    """
    if condition.data.dtype != torch.bool:
        raise TypeError("where expects condition.data to have dtype torch.bool")

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


@dispatch(Tensor)  # type: ignore[no-redef]
def where(
    condition: Tensor[torch.BoolTensor], index_type: Type[Any] = Tensor
) -> Union[
    Tuple[Tensor, ...],
    Tuple[Tuple[int, ...], ...],
    StateSpace,
]:
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
    `Union[Tuple[Tensor, ...], Tuple[Tuple[int, ...], ...], StateSpace]`
        Index representation selected by `index_type`.

    Raises
    ------
    `TypeError`
        If `condition.data` is not boolean.
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


def nonzero(
    condition: Tensor, as_tuple: bool = True, index_type: Type[Any] = Tensor
) -> Union[
    Tuple[Tensor, ...],
    Tuple[Tuple[int, ...], ...],
    StateSpace,
]:
    """
    Return indices of non-zero / `True` entries.

    This currently supports only `as_tuple=True` and follows
    `torch.nonzero(condition.data, as_tuple=True)` semantics.

    Parameters
    ----------
    `condition` : `Tensor`
        Input tensor.
    `as_tuple` : `bool`, optional
        Must be `True`.
    `index_type` : `Type[Any]`, optional
        Requested index representation. Supported values are `Tensor`,
        `tuple` / `Tuple`, and `StateSpace`.

    Returns
    -------
    `Union[Tuple[Tensor, ...], Tuple[Tuple[int, ...], ...], StateSpace]`
        Index representation selected by `index_type`.

    Raises
    ------
    `NotImplementedError`
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
    Compile mixed indexing keys into torch indices plus `StateSpace` metadata.

    This class is the single indexing compiler used by `Tensor.__getitem__`.
    It accepts raw key tokens (including `...`) and produces:

    - `indices`: a tuple consumable by torch indexing for one-shot execution.
    - `dims`: output `StateSpace` metadata after indexing.
    - `indices_steps`: per-token executable steps for sequential indexing.
    - `has_tensor_index`: whether advanced tensor-index mode is active.

    Key token types
    ---------------
    Supported input tokens in `indices`:
    - `int`
    - `slice`
    - `None`
    - `Ellipsis`
    - `StateSpace`
    - `Convertible` (converted to `StateSpace`)
    - `Tensor` (qten tensor index)

    Normalization rules
    -------------------
    1. Expand at most one ellipsis (`...`) into the required number of full
       slices `:`, where only non-`None` tokens consume source axes.
    2. Append trailing full slices when fewer source-axis-consuming tokens than
       tensor rank are provided.
    3. Reject keys with too many source-axis-consuming tokens.
    4. In tensor-index mode, left-pad lower-rank tensor indices with leading
       `BroadcastSpace` axes (via unsqueeze) so all tensor indices share one
       rank before `tensor_union_dims` is computed.

    Per-token compile rules
    -----------------------
    - `int`: consumes one source axis; removes that axis from output metadata.
    - `slice`:
      - full `:` preserves the source axis `StateSpace`,
      - non-full uses `dim[slice]`, except for `BroadcastSpace` where
        output metadata follows sliced size (`BroadcastSpace` for size `1`,
        `IndexSpace.linear(0)` for size `0`).
    - `None`: inserts one `BroadcastSpace` axis; consumes no source axis.
    - `StateSpace` / `Convertible`:
      - on `BroadcastSpace` source axes, only `BroadcastSpace` is accepted;
        other `StateSpace` indices raise `IndexError`,
      - equal space -> full-slice behavior,
      - same span -> permutation index (`Tuple[int, ...]`),
      - contained subspace -> embedding index (`Tuple[int, ...]`),
      - otherwise raises `IndexError`.
    - `Tensor`:
      - bool dtype is unsupported (`NotImplementedError`),
      - lower-rank tensor indices are internally left-padded to the maximum
        tensor-index rank before union/alignment,
      - aligns to `tensor_union_dims` (union over all tensor-index dims),
      - consumes one source axis and contributes advanced metadata dims.

    Mode and compatibility rules
    ----------------------------
    - Mixed `Tensor` with `StateSpace`/`Convertible` in one key is rejected
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
            cast(Tensor, idx) for idx in self.indices if isinstance(idx, Tensor)
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
            i for i, idx in enumerate(indices) if isinstance(idx, Tensor)
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

        # None inserts output axes and does not consume source axes.
        consumed = sum(idx is not None for idx in left_indices + right_indices)
        # Calculate how many dimensions the ellipsis should expand to
        num_full_slices = self.rank - consumed
        return left_indices + (slice(None),) * num_full_slices + right_indices

    def _pad_missing_slices(
        self, indices: Tuple[TensorIndexType, ...]
    ) -> Tuple[TensorIndexType, ...]:
        non_none = sum(idx is not None for idx in indices)
        if non_none > self.rank:
            raise IndexError("Too many indices for tensor")
        if non_none < self.rank:
            return indices + (slice(None),) * (self.rank - non_none)
        return indices

    @dispatch(int, int)  # type: ignore[no-redef]
    def _compile(self, idx: int, v: int) -> Tuple[int, Tuple[StateSpace, ...], int]:
        return idx + 1, tuple(), v

    @dispatch(int, slice)  # type: ignore[no-redef]
    def _compile(self, idx: int, v: slice) -> Tuple[int, Tuple[StateSpace, ...], slice]:
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

    @dispatch(int, type(None))  # type: ignore[no-redef]
    def _compile(self, idx: int, _: None) -> Tuple[int, Tuple[StateSpace, ...], None]:
        return idx, (BroadcastSpace(),), None

    @dispatch(int, (Convertible, StateSpace))  # type: ignore[no-redef]
    def _compile(
        self, idx: int, v: Union[Convertible, StateSpace]
    ) -> Tuple[int, Tuple[StateSpace, ...], Union[Tuple[int, ...], slice]]:
        if isinstance(v, Convertible) and not isinstance(v, StateSpace):
            v = v.convert(StateSpace)
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

    @dispatch(int, Tensor)  # type: ignore[no-redef]
    def _compile(
        self, idx: int, v: Tensor
    ) -> Tuple[int, Tuple[StateSpace, ...], torch.Tensor]:
        if v.data.dtype == torch.bool:
            raise NotImplementedError("Boolean Tensor indexing is not supported yet")
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
        tensor_positions = [
            i for i, (idx, _, _) in enumerate(entries) if isinstance(idx, Tensor)
        ]
        if len(tensor_positions) == 0:
            return tuple(dim for _, dims, _ in entries for dim in dims)

        first_tensor_pos = tensor_positions[0]
        last_tensor_pos = tensor_positions[-1]
        if last_tensor_pos - first_tensor_pos + 1 != len(tensor_positions):
            non_tensor_dims = tuple(
                dim
                for idx, dims, _ in entries
                if not isinstance(idx, Tensor)
                for dim in dims
            )
            return self.tensor_union_dims + non_tensor_dims

        compiled_dims: Tuple[StateSpace, ...] = tuple()
        for i, (idx, dims, _) in enumerate(entries):
            if i == first_tensor_pos:
                compiled_dims += self.tensor_union_dims
            if isinstance(idx, Tensor):
                continue
            compiled_dims += dims
        return compiled_dims

    def compile(self) -> CompiledIndices:
        """
        Compile the raw key into executable indices and output metadata.

        Returns
        -------
        `CompiledIndices`
            A tuple-like record containing:
            - `indices`: torch-compatible index tuple for one-shot execution.
            - `dims`: output `StateSpace` tuple after indexing.
            - `indices_steps`: per-token steps for sequential execution.
            - `has_tensor_index`: whether tensor advanced-index mode is active.

        Raises
        ------
        `ValueError`
            If `Tensor` indices are mixed with `StateSpace`/`Convertible`
            indices in the same key.
        `IndexError`
            If normalization fails (e.g. too many indices) or a `StateSpace`
            index is not compatible with the corresponding source dimension.
        `NotImplementedError`
            If boolean `Tensor` index masking is requested.
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
