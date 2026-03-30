from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Literal,
    Optional,
    Self,
)
import torch


@dataclass(frozen=True)
class Device:
    """
    Lightweight immutable device descriptor used by QTen.

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
        if name.startswith("gpu"):
            parts = name.split(":")
            if len(parts) == 1:
                return Device(name="gpu")
            elif len(parts) == 2 and parts[1].isdigit():
                return Device(name="gpu", index=int(parts[1]))
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
        raise RuntimeError("The current system does not have GPU devices!")

    def __str__(self) -> str:
        """
        Format the device using QTen's logical device syntax.

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
