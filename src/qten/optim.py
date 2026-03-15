from dataclasses import replace
from typing import Callable, Self, TypeVar
from typing_extensions import override

import torch.nn as nn

from .utils.devices import Device, DeviceBounded
from .utils.collections_ext import FrozenDict
from .abstracts import Functional
from .linalg.tensors import Tensor


ModuleType = TypeVar("ModuleType", bound=type["Module"])
"""
Type variable representing a subclass object of :class:`Module`.

This is used to type the ``@nograd_tensors(...)`` class decorator so that the
decorated class preserves its original class type instead of being widened to a
plain ``type[Module]``.
"""


def nograd_tensors(*names: str) -> Callable[[ModuleType], ModuleType]:
    """
    Mark ``Tensor`` attributes as no-grad module state stored as buffers.

    The decorator stores the provided attribute names on the target class in the
    ``__nograd_tensors__`` class attribute. When an instance later assigns a
    :class:`~qten.tensors.Tensor` to one of those names, :meth:`Module.__setattr__`
    copies detached data into module-owned buffer storage rather than
    registering it as a PyTorch parameter.

    The annotation is inherited. Decorating a subclass extends the inherited set
    rather than replacing it.

    Parameters
    ----------
    `*names` : `str`
        Attribute names whose assigned ``Tensor`` values should be stored as
        module buffers rather than ``nn.Parameter`` instances.

    Returns
    -------
    `Callable[[ModuleType], ModuleType]`
        A class decorator that annotates the target ``Module`` subclass in place
        and returns that same class.

    Examples
    --------
    ```python
    @nograd_tensors("basis", "projector")
    class MyModule(Module):
        ...
    ```
    """

    def annotate(cls: ModuleType) -> ModuleType:
        inherited_names = getattr(cls, "__nograd_tensors__", ())
        cls.__nograd_tensors__ = frozenset((*inherited_names, *names))
        return cls

    return annotate


TENSOR_PARAM_PREFIX = "tensor:"
"""
Prefix used for the hidden ``nn.Parameter`` names registered for wrapped tensors.

When a public module attribute such as ``self.weight`` is assigned a
:class:`~qten.tensors.Tensor`, the wrapper object remains accessible under
the original attribute name while the actual PyTorch parameter is registered
under ``f"{TENSOR_PARAM_PREFIX}{name}"``. This keeps the wrapper and the
registered parameter distinct.
"""

TENSOR_BUFFER_PREFIX = "buffer:"
"""
Prefix used for the hidden buffers registered for no-grad wrapped tensors.

When a public module attribute such as ``self.basis`` is assigned a
:class:`~qten.tensors.Tensor` and its name is listed in
``__nograd_tensors__``, the underlying ``torch.Tensor`` is registered under
``f"{TENSOR_BUFFER_PREFIX}{name}"`` while the public attribute remains a
QTen ``Tensor`` wrapper.
"""


class Module(Functional, nn.Module, DeviceBounded):
    """
    QTen base module combining multiple dispatch, device tracking, and tensor wrapping.

    This class extends :class:`torch.nn.Module` with two QTen-specific
    behaviors:

    1. Public attributes may be assigned :class:`~qten.tensors.Tensor`
       objects directly. The underlying ``torch.Tensor`` data is automatically
       copied into module-owned storage. Trainable tensor attributes are
       registered as ``nn.Parameter`` values, while names listed in
       ``__nograd_tensors__`` are registered as buffers. In both cases the
       public attribute remains a QTen ``Tensor`` wrapper.
    2. The module is also a :class:`~qten.abstracts.Functional`, so calling
       the module dispatches through QTen's multiple-dispatch mechanism
       rather than PyTorch's usual ``forward`` path.

    The module additionally tracks its logical :class:`~qten.utils.Device`
    and automatically moves assigned :class:`~qten.utils.DeviceBounded`
    values onto the same device.

    This class is also useful as a structured container for optimizable tensors,
    even when the object is not meant to behave like a trainable "model" with a
    forward pass. In that usage, assign the tensors you want PyTorch to manage
    as public attributes, optimize them through the module's parameter
    interface, and call :meth:`export` when you need an independent
    non-differentiable :class:`~qten.tensors.Tensor` snapshot for
    downstream use outside the module.

    Ownership semantics are important:
    - Before assignment, a :class:`~qten.tensors.Tensor` is functionally
      defined by its own wrapper value.
    - After assignment to a ``Module`` attribute, the wrapper becomes
      module-bound: its ``data`` points at module-owned storage, either a
      registered ``nn.Parameter`` or registered buffer depending on
      whether the name is marked in ``__nograd_tensors__``.
    - Reading ``self.weight`` still returns a QTen ``Tensor`` wrapper, but
      its mutable/autograd state is now governed by the containing module and
      PyTorch parameter machinery.
    - No-grad tensor attributes remain part of the module's buffer state, so
      they follow standard PyTorch buffer behavior for device moves and
      serialization while remaining excluded from optimization.
    - Assignment is owning-by-value: the module copies the assigned tensor data
      into its own owned storage rather than aliasing the caller's tensor
      storage.
    - Use :meth:`export` or :meth:`export_all` if you need standalone tensor
      values whose subsequent state cannot be changed implicitly by optimizer
      steps, reassignment, or other module-side updates.
    """

    __nograd_tensors__: frozenset[str] = frozenset()
    """
    Names of public ``Tensor`` attributes stored as buffers.

    This is a class-level annotation populated by :func:`nograd_tensors`. The
    default value is an empty set, meaning assigned tensors are registered as
    parameters unless explicitly annotated otherwise.
    """

    def __init__(self):
        """
        Initialize the PyTorch module state and default the logical device to CPU.
        """
        nn.Module.__init__(self)
        self._device = Device("cpu")

    def export(self, name: str) -> Tensor:
        """
        Export a module-owned tensor as an independent, non-differentiable snapshot.

        This is intended for extracting a tensor attribute from the module for
        downstream use without preserving autograd linkage or shared storage
        with the original module-owned data. This is the recommended way to obtain
        a standalone tensor value from a module that is being used as a wrapper
        around optimizable tensors rather than purely as a callable model. This
        works uniformly for both parameter-backed tensors and buffer-backed
        no-grad tensors.

        Parameters
        ----------
        `name` : `str`
            Name of the public module attribute to export.

        Returns
        -------
        `Tensor`
            A detached cloned tensor with the same dims as the module attribute.

        Raises
        ------
        `AttributeError`
            If the module has no attribute named ``name``.
        `TypeError`
            If the named attribute is not a ``Tensor``.
        """
        tensor = getattr(self, name)
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Attribute {name!r} is not a Tensor.")
        return tensor.detach().clone()

    def export_all(self) -> FrozenDict[str, Tensor]:
        """
        Export all public tensor attributes from this module tree.

        This recursively traverses nested :class:`Module` instances and returns
        detached cloned tensor snapshots keyed by public dotted attribute names,
        for example ``"weight"`` or ``"inner.basis"``.

        Returns
        -------
        `FrozenDict[str, Tensor]`
            Mapping from public tensor names to independent exported tensors.
        """
        exported: dict[str, Tensor] = {}
        for prefix, module in self.named_modules():
            if not isinstance(module, Module):
                continue
            for name, tensor in module._iter_public_tensors():
                full_name = f"{prefix}.{name}" if prefix else name
                exported[full_name] = tensor.detach().clone()
        return FrozenDict(exported)

    def freeze(self) -> Self:
        """
        Disable gradient tracking for all module-owned parameters.

        This applies recursively through submodules using PyTorch's parameter
        traversal and returns ``self`` for fluent usage. Buffer-backed tensor
        state declared via ``@nograd_tensors(...)`` is not registered as
        parameters and therefore remains unaffected.

        Returns
        -------
        `Self`
            This module after all parameters have been frozen.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def unfreeze(self) -> Self:
        """
        Enable gradient tracking for all module-owned parameters.

        This applies recursively through submodules using PyTorch's parameter
        traversal and returns ``self`` for fluent usage. Buffer-backed tensor
        state declared via ``@nograd_tensors(...)`` is not registered as
        parameters and therefore remains unaffected.

        Returns
        -------
        `Self`
            This module after all parameters have been unfrozen.
        """
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self

    def _iter_public_tensors(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                yield name, value

    def _owned_tensor_data(self, name: str):
        parameter_name = self._tensor_parameter_name(name)
        if parameter_name in self._parameters:
            return self._parameters[parameter_name]
        buffer_name = self._tensor_buffer_name(name)
        if buffer_name in self._buffers:
            return self._buffers[buffer_name]
        return None

    @staticmethod
    def _tensor_parameter_name(name: str) -> str:
        """
        Build the hidden registered parameter name for a public tensor attribute.

        Parameters
        ----------
        `name` : `str`
            Public attribute name on the module, such as ``"weight"``.

        Returns
        -------
        `str`
            Internal parameter name used when registering the backing
            ``nn.Parameter`` with PyTorch.
        """
        return f"{TENSOR_PARAM_PREFIX}{name}"

    @staticmethod
    def _tensor_buffer_name(name: str) -> str:
        """
        Build the hidden registered buffer name for a public tensor attribute.

        Parameters
        ----------
        `name` : `str`
            Public attribute name on the module, such as ``"basis"``.

        Returns
        -------
        `str`
            Internal buffer name used when registering the backing tensor with
            PyTorch.
        """
        return f"{TENSOR_BUFFER_PREFIX}{name}"

    def _clear_tensor_parameter(self, name: str) -> None:
        """
        Remove the hidden registered parameter associated with a public attribute.

        This is used when a wrapped tensor attribute is deleted or overwritten by
        a non-``Tensor`` value so that stale parameters do not remain in
        ``named_parameters()`` or ``state_dict()``.

        Parameters
        ----------
        `name` : `str`
            Public attribute name whose hidden parameter registration should be
            removed if it exists.
        """
        parameter_name = self._tensor_parameter_name(name)
        if parameter_name in self._parameters:
            nn.Module.__delattr__(self, parameter_name)

    def _clear_tensor_buffer(self, name: str) -> None:
        """
        Remove the hidden registered buffer associated with a public attribute.

        Parameters
        ----------
        `name` : `str`
            Public attribute name whose hidden buffer registration should be
            removed if it exists.
        """
        buffer_name = self._tensor_buffer_name(name)
        if buffer_name in self._buffers:
            nn.Module.__delattr__(self, buffer_name)

    def _refresh_public_tensor_wrappers(self) -> None:
        for name, tensor in list(self._iter_public_tensors()):
            data = self._owned_tensor_data(name)
            if data is not None:
                nn.Module.__setattr__(self, name, replace(tensor, data=data))

    def __setattr__(self, name: str, value) -> None:
        """
        Assign an attribute, wrapping :class:`qten.tensors.Tensor` as module-owned state.

        Behavior
        --------
        - Any assigned :class:`~qten.utils.DeviceBounded` value is moved to
          ``self.device`` first.
        - Any assigned :class:`~qten.tensors.Tensor` is copied into
          module-owned storage.
        - For names listed in ``__nograd_tensors__``, the copied tensor is
          registered as a hidden buffer.
        - Otherwise the copied tensor is registered as a hidden
          :class:`torch.nn.Parameter`.
        - The public attribute itself remains a QTen ``Tensor`` wrapper
          whose ``data`` points at the owned buffer or registered
          parameter.
        - If a non-``Tensor`` value replaces a previously wrapped tensor
          attribute, the hidden parameter/buffer registration is removed.
        - Tensor assignment is owning-by-value: the module clones tensor data
          before storing it as a parameter or buffer, so later optimizer steps
          or in-place edits on the module do not alias the caller's original
          tensor storage.

        In other words, assigning a ``Tensor`` transfers ownership of its
        mutable/autograd state to the module. The attribute remains convenient
        to use as a QTen wrapper, but it should now be treated as
        module-bound state rather than an isolated functional value.

        Parameters
        ----------
        `name` : `str`
            Attribute name being assigned.
        `value` : `Any`
            Value to assign.
        """
        if isinstance(value, DeviceBounded):
            # Automatically move any assigned DeviceBounded objects to the same device as this module.
            value = value.to_device(self.device)
        if isinstance(value, Tensor):
            # Module assignment owns its parameter state by value rather than
            # aliasing the caller's tensor storage.
            source = value.data
            copied = source.detach().clone()
            if name in self.__nograd_tensors__:
                self._clear_tensor_parameter(name)
                self.register_buffer(self._tensor_buffer_name(name), copied)
                value = replace(
                    value, data=self._buffers[self._tensor_buffer_name(name)]
                )
            else:
                self._clear_tensor_buffer(name)
                data = nn.Parameter(copied, requires_grad=source.requires_grad)
                nn.Module.__setattr__(self, self._tensor_parameter_name(name), data)
                value = replace(value, data=data)
        else:
            self._clear_tensor_parameter(name)
            self._clear_tensor_buffer(name)
        nn.Module.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """
        Delete an attribute and remove any hidden tensor storage tied to it.

        Parameters
        ----------
        `name` : `str`
            Attribute name to delete.
        """
        self._clear_tensor_parameter(name)
        self._clear_tensor_buffer(name)
        nn.Module.__delattr__(self, name)

    __call__ = Functional.__call__

    @override
    def to_device(self, device: Device) -> Self:
        """
        Move the module and all owned tensor state to the specified device.

        This delegates to :meth:`torch.nn.Module.to`, which already applies the
        move recursively to parameters and buffers. Public tensor wrappers are
        then refreshed to point at the moved owned storage before the logical
        device record is updated.

        Parameters
        ----------
        `device` : `Device`
            Target logical device.

        Returns
        -------
        `Self`
            This module after the move.
        """
        nn.Module.to(self, device.torch_device())
        for module in self.modules():
            if not isinstance(module, Module):
                continue
            module._device = device
            module._refresh_public_tensor_wrappers()
        return self

    @override
    @property
    def device(self) -> Device:
        """
        Return the current logical device recorded for this module.

        Returns
        -------
        `Device`
            The logical device associated with the module.
        """
        return self._device
