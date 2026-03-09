from collections.abc import Mapping
from abc import ABCMeta, ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Iterator,
    Any,
    Generic,
    List,
    Literal,
    Optional,
    Self,
    Tuple,
    Dict,
    TypeVar,
    Iterable,
    Callable,
    Type,
    cast,
)
import torch


_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(Mapping[_K, _V], Generic[_K, _V]):
    __slots__ = ("__items", "__hash")

    def __init__(self, *args: Any, **kwargs: Any):
        data = dict(*args, **kwargs)
        try:
            fitems = frozenset(data.items())  # ensures all keys/vals are hashable
        except TypeError as e:
            raise TypeError(
                "All keys and values must be hashable. "
                "Use deep_freeze() for nested mutables."
            ) from e
        object.__setattr__(
            self, "_FrozenDict__items", tuple(fitems)
        )  # hidden, immutable
        object.__setattr__(self, "_FrozenDict__hash", hash(fitems))

    # internal accessor that bypasses the guard
    def _items(self) -> Tuple[Tuple[_K, _V], ...]:
        return cast(
            Tuple[Tuple[_K, _V], ...],
            object.__getattribute__(self, "_FrozenDict__items"),
        )

    # --- Mapping interface ---
    def __len__(self) -> int:
        return len(self._items())

    def __iter__(self) -> Iterator[_K]:
        for k, _ in self._items():
            yield k

    def __getitem__(self, key: _K) -> _V:
        for k, v in self._items():
            if k == key:
                return v
        raise KeyError(key)

    # --- equality & hash ---
    def __hash__(self) -> int:
        return object.__getattribute__(self, "_FrozenDict__hash")

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Mapping):
            try:
                return frozenset(self._items()) == frozenset(other.items())
            except TypeError:
                return False
        return NotImplemented

    def __str__(self) -> str:
        return str(dict(self._items()))

    def __repr__(self) -> str:
        return repr(dict(self._items()))

    def __getattribute__(self, name: str):
        if name in {"_FrozenDict__items"}:
            raise AttributeError("Private storage is hidden")
        return super().__getattribute__(name)


# --- Plotting Helpers ---


def compute_bonds(
    coords: torch.Tensor, dim: int
) -> Tuple[
    List[Optional[float]], List[Optional[float]], Optional[List[Optional[float]]]
]:
    """
    Generate bond lines connecting nearest neighbors using PyTorch.
    Returns (x_lines, y_lines, z_lines) where lists contain coordinates separated by None.
    z_lines is None if dim != 3.
    """
    if coords.size(0) < 2:
        return [], [], None if dim != 3 else None

    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dists = torch.norm(diff, dim=-1)

    dists.fill_diagonal_(float("inf"))

    min_dist = torch.min(dists)
    if torch.isinf(min_dist):
        return [], [], None if dim != 3 else None

    tol = 1e-4
    pairs = torch.nonzero(dists <= min_dist + tol)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    if pairs.size(0) == 0:
        return [], [], None if dim != 3 else None

    p1 = coords[pairs[:, 0]]
    p2 = coords[pairs[:, 1]]

    p1_np = p1.numpy()
    p2_np = p2.numpy()

    x_lines: List[Optional[float]] = []
    y_lines: List[Optional[float]] = []
    z_lines: Optional[List[Optional[float]]] = [] if dim == 3 else None
    nan = None

    for i in range(len(p1_np)):
        x_lines.extend([p1_np[i, 0], p2_np[i, 0], nan])
        y_lines.extend([p1_np[i, 1], p2_np[i, 1], nan])
        if dim == 3 and z_lines is not None:
            z_lines.extend([p1_np[i, 2], p2_np[i, 2], nan])

    return x_lines, y_lines, z_lines


def matchby(
    source: Iterable[Any], dest: Iterable[Any], base_func: Callable[[Any], Any]
) -> Dict[Any, Any]:
    """
    Map elements from source to destination using a provided mapping function.
    Parameters
    ----------
    `source` : `Iterable[Any]`
        The source elements to be mapped.
    `dest` : `Iterable[Any]`
        The destination elements to map to.
    `base_func` : `Callable[[Any], Any]`
        A function that defines the comparison baseline.

    Returns
    -------
    `Dict[Any, Any]`
        A dictionary mapping each source element to its corresponding destination element `source -> dest`.
    """
    mapping: Dict[Any, Any] = {}

    source_base: Dict[Any, Any] = {m: base_func(m) for m in source}
    dest_base: Dict[Any, Any] = {base_func(m): m for m in dest}

    if len(dest_base) != len(tuple(dest)):
        raise ValueError("Destination elements have non-unique base values!")

    for sm, sb in source_base.items():
        if sb not in dest_base:
            raise ValueError(
                f"Source element {sm} with base {sb} has no match in destination!"
            )
        mapping[sm] = dest_base[sb]

    return mapping


def subtypes(cls: Type) -> Tuple[ABCMeta, ...]:
    """
    Return all transitive subclasses of a class.

    Parameters
    ----------
    `cls` : `Type`
        The class to inspect.

    Returns
    -------
    `Tuple[ABCMeta, ...]`
        A tuple containing all direct and indirect subclasses of `cls`.
    """
    out = set()
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub not in out:
            out.add(sub)
            stack.extend(sub.__subclasses__())
    return cast(Tuple[ABCMeta, ...], tuple(out))


def full_typename(cls: Type) -> str:
    """
    Get the full module and class name of a type.

    Parameters
    ----------
    `cls` : `Type`
        The class to get the name of.

    Returns
    -------
    `str`
        The full name of the class, including its module.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class Device:
    """
    Lightweight immutable device descriptor used by PyHilbert.

    The public device model is intentionally small:
    - `"cpu"` represents host execution.
    - `"gpu"` represents accelerated execution.

    For GPU devices, ``index`` optionally stores a CUDA device index. This index
    is only meaningful on CUDA-capable systems. On systems where CUDA is not
    available but PyTorch MPS support is available, GPU requests resolve to the
    single `"mps"` backend and ``index`` is ignored.

    Parameters
    ----------
    `name` : `Literal["cpu", "gpu"]`
        The logical device family.
    `index` : `Optional[int]`, default=`None`
        Optional CUDA device index. This should only be set when ``name`` is
        `"gpu"`.
    """

    name: Literal["cpu", "gpu"]
    index: Optional[int] = None

    @staticmethod
    def new(name: str) -> "Device":
        """
        Parse a user-facing device string into a ``Device`` instance.

        Supported inputs are:
        - `"cpu"`
        - `"gpu"`
        - `"gpu:<index>"`, where ``<index>`` is a non-negative integer

        Parameters
        ----------
        `name` : `str`
            Device string to parse.

        Returns
        -------
        `Device`
            Parsed immutable device descriptor.

        Raises
        ------
        `ValueError`
            If the input does not match one of the supported formats.
        """
        if name == "cpu":
            return Device(name="cpu")
        if name == "gpu":
            return Device(name="gpu")
        if name.startswith("gpu:"):
            index = name[4:]
            if index.isdigit():
                return Device(name="gpu", index=int(index))
        raise ValueError(f"Invalid device name: {name}")

    def torch_device(self) -> torch.device:
        """
        Resolve this logical device into the concrete PyTorch backend device.

        Resolution is runtime-dependent:
        - `"cpu"` always maps to ``torch.device("cpu")``.
        - `"gpu"` prefers CUDA when available.
        - If CUDA is unavailable and MPS is available, `"gpu"` maps to
          ``torch.device("mps")``.

        When CUDA is selected, an explicit ``index`` is used if present.
        Otherwise, the current CUDA device reported by PyTorch is used.

        Returns
        -------
        `torch.device`
            Concrete PyTorch device corresponding to this logical device.

        Raises
        ------
        `ValueError`
            If ``self.name`` is not a supported logical device value.
        `RuntimeError`
            If a GPU device is requested but neither CUDA nor MPS is available
            in the current environment.
        """
        if self.name == "cpu":
            return torch.device("cpu")
        if not self.name == "gpu":
            raise ValueError(f"Invalid device name: {self.name}")
        if torch.cuda.is_available():
            index = (
                self.index if self.index is not None else torch.cuda.current_device()
            )
            return torch.device("cuda", index)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("The current system does not have GPU devices!")

    def __str__(self) -> str:
        """
        Format the device using PyHilbert's logical device syntax.

        Returns
        -------
        `str`
            `"cpu"`, `"gpu"`, or `"gpu:<index>"`.

        Raises
        ------
        `ValueError`
            If ``self.name`` is not a supported logical device value.
        """
        match self.name:
            case "cpu":
                return "cpu"
            case "gpu":
                if self.index is not None:
                    return f"gpu:{self.index}"
                return "gpu"
            case _:
                raise ValueError(f"Invalid device name: {self.name}")

    __repr__ = __str__


class DeviceBounded(ABC):
    def cpu(self) -> Self:
        """
        Return a copy of this object residing on the CPU device.
        """
        return self.to_device(Device("cpu"))

    def gpu(self, index: Optional[int] = None) -> Self:
        """
        Return a copy of this object residing on a GPU device.

        Parameters
        ----------
        `index` : `Optional[int]`, default=`None`
            Optional CUDA device index. This should only be set when the target
            GPU backend supports multiple devices (e.g. CUDA). If not provided,
            the current device will be used.

        Returns
        -------
        `Self`
            A copy of this object on the specified GPU device.
        """
        device = Device("gpu", index)
        return self.to_device(device)

    @abstractmethod
    def to_device(self, device: Device) -> Self:
        """
        Return a copy of this object residing on the specified device.

        Parameters
        ----------
        `device` : `Device`
            The target device to move this object to.

        Returns
        -------
        `Self`
            A copy of this object on the specified device.
        """
        pass

    @property
    @abstractmethod
    def device(self) -> Device:
        """
        Return the current device this object resides on.

        Returns
        -------
        `Device`
            The current device of this object.
        """
        pass
