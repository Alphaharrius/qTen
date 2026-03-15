import torch
import torch.nn as nn
import pytest

from qten.optim import (
    Module,
    TENSOR_BUFFER_PREFIX,
    TENSOR_PARAM_PREFIX,
    nograd_tensors,
)
from qten.symbolics.state_space import IndexSpace
from qten.linalg.tensors import Tensor
from qten.utils.devices import Device
from qten.utils.collections_ext import FrozenDict

VECTOR_DIM = (IndexSpace.linear(2),)


@nograd_tensors("basis")
class AffineBlock(Module):
    def __init__(self):
        super().__init__()
        self.weight = Tensor(data=torch.tensor([2.0, -1.0]), dims=VECTOR_DIM).attach()
        self.bias = Tensor(data=torch.tensor([0.5, 1.5]), dims=VECTOR_DIM).attach()
        self.basis = Tensor(data=torch.tensor([3.0, 4.0]), dims=VECTOR_DIM).attach()
        self.label = "affine"


class NestedBlock(Module):
    def __init__(self):
        super().__init__()
        self.inner = AffineBlock()
        self.scale = Tensor(data=torch.tensor([1.0, 2.0]), dims=VECTOR_DIM).attach()


class DispatchedAffineBlock(Module):
    def __init__(self):
        super().__init__()
        self.weight = Tensor(data=torch.tensor([2.0, -1.0]), dims=VECTOR_DIM).attach()
        self.bias = Tensor(data=torch.tensor([0.5, 1.5]), dims=VECTOR_DIM).attach()


@DispatchedAffineBlock.register(torch.Tensor)
def _apply_dispatched_affine(module: DispatchedAffineBlock, x: torch.Tensor) -> Tensor:
    return Tensor(data=module.weight.data * x + module.bias.data, dims=VECTOR_DIM)


@nograd_tensors("cache")
class ParentAnnotatedBlock(Module):
    def __init__(self):
        super().__init__()
        self.cache = Tensor(data=torch.tensor([5.0, 6.0]), dims=VECTOR_DIM).attach()


@nograd_tensors("projection")
class ChildAnnotatedBlock(ParentAnnotatedBlock):
    def __init__(self):
        super().__init__()
        self.projection = Tensor(
            data=torch.tensor([7.0, 8.0]), dims=VECTOR_DIM
        ).attach()
        self.weight = Tensor(data=torch.tensor([1.0, -3.0]), dims=VECTOR_DIM).attach()


class TestModules:
    @staticmethod
    def _assert_module_tensor_device_sync(module: Module) -> None:
        torch_device = module.device.torch_device()
        for name, tensor in module._iter_public_tensors():
            assert tensor.device == module.device
            assert tensor.data.device == torch_device
            parameter_name = module._tensor_parameter_name(name)
            if parameter_name in module._parameters:
                assert tensor.data is module._parameters[parameter_name]
                continue
            buffer_name = module._tensor_buffer_name(name)
            assert buffer_name in module._buffers
            assert tensor.data is module._buffers[buffer_name]

    def test_module_registers_grad_and_nograd_tensors(self):
        module = AffineBlock()

        assert isinstance(module, nn.Module)
        assert isinstance(module.weight, Tensor)
        assert isinstance(module.bias, Tensor)
        assert isinstance(module.basis, Tensor)

        weight_name = f"{TENSOR_PARAM_PREFIX}weight"
        bias_name = f"{TENSOR_PARAM_PREFIX}bias"
        basis_name = f"{TENSOR_BUFFER_PREFIX}basis"

        named_parameters = dict(module.named_parameters())
        assert set(named_parameters) == {weight_name, bias_name}
        assert named_parameters[weight_name] is module.weight.data
        assert named_parameters[bias_name] is module.bias.data

        named_buffers = dict(module.named_buffers())
        assert set(named_buffers) == {basis_name}
        assert named_buffers[basis_name] is module.basis.data

        assert module.weight.requires_grad
        assert module.bias.requires_grad
        assert not module.basis.requires_grad

        state_dict = module.state_dict()
        assert set(state_dict) == {weight_name, bias_name, basis_name}

    def test_module_tensor_assignment_copies_input_storage(self):
        source = Tensor(data=torch.tensor([9.0, -4.0]), dims=VECTOR_DIM).attach()

        module = Module()
        assert module.device == Device("cpu")
        module.weight = source

        assert module.weight.device == Device("cpu")
        assert module.weight.data is not source.data
        assert torch.equal(module.weight.data, source.data)

        with torch.no_grad():
            module.weight.data.add_(1.0)

        assert torch.equal(source.data, torch.tensor([9.0, -4.0]))
        assert torch.equal(module.weight.data.detach(), torch.tensor([10.0, -3.0]))

    def test_tensor_ownership_lifecycle_through_module_pipeline(self):
        source = Tensor(data=torch.tensor([1.0, 2.0]), dims=VECTOR_DIM).attach()
        source_before_assignment = source.clone()

        module = Module()
        assert module.device == Device("cpu")
        module.weight = source

        # Before assignment the source tensor is standalone; after assignment the
        # module owns an isolated parameter copy.
        assert source.equal(source_before_assignment)
        assert source.device == Device("cpu")
        assert module.device == Device("cpu")
        assert module.weight.device == Device("cpu")
        assert module.weight.data is not source.data
        assert module.weight.requires_grad == source.requires_grad
        assert torch.equal(module.weight.data.detach(), source.data.detach())

        loss = Tensor(data=(module.weight.data**2).sum(), dims=())
        loss.backward()
        assert module.weight.grad is not None
        assert module.weight.grad.equal(
            Tensor(data=torch.tensor([2.0, 4.0]), dims=VECTOR_DIM)
        )
        assert source.grad is None

        with torch.no_grad():
            module.weight.data.add_(3.0)

        # Module mutation does not change the original input after ownership transfer.
        assert torch.equal(source.data.detach(), torch.tensor([1.0, 2.0]))
        assert torch.equal(module.weight.data.detach(), torch.tensor([4.0, 5.0]))

        exported = module.export("weight")
        assert exported.data is not module.weight.data
        assert not exported.requires_grad
        assert exported.device == module.weight.device
        assert exported.equal(Tensor(data=torch.tensor([4.0, 5.0]), dims=VECTOR_DIM))

        with torch.no_grad():
            module.weight.data.add_(1.0)

        # Exported tensors are independent snapshots after leaving the module.
        assert exported.equal(Tensor(data=torch.tensor([4.0, 5.0]), dims=VECTOR_DIM))
        assert torch.equal(module.weight.data.detach(), torch.tensor([5.0, 6.0]))

    def test_module_backward_computes_grads_only_for_trainable_tensors(self):
        module = AffineBlock()
        x = torch.tensor([3.0, -2.0])
        target = torch.tensor([1.0, -1.0])

        output = module.weight.data * x + module.bias.data + module.basis.data
        loss = Tensor(data=((output - target) ** 2).sum(), dims=())
        loss.backward()

        expected_weight_grad = 2 * (output.detach() - target) * x
        expected_bias_grad = 2 * (output.detach() - target)

        assert module.weight.grad is not None
        assert module.bias.grad is not None
        assert module.weight.grad.equal(
            Tensor(data=expected_weight_grad, dims=VECTOR_DIM)
        )
        assert module.bias.grad.equal(Tensor(data=expected_bias_grad, dims=VECTOR_DIM))
        assert module.basis.grad is None

    def test_functional_dispatch_path_preserves_gradients(self):
        module = DispatchedAffineBlock()
        x = torch.tensor([3.0, -2.0])
        target = torch.tensor([1.0, -1.0])

        output = module(x)
        loss = Tensor(data=((output.data - target) ** 2).sum(), dims=())
        loss.backward()

        expected_weight_grad = 2 * (output.data.detach() - target) * x
        expected_bias_grad = 2 * (output.data.detach() - target)

        assert module.weight.grad is not None
        assert module.bias.grad is not None
        assert module.weight.grad.equal(
            Tensor(data=expected_weight_grad, dims=VECTOR_DIM)
        )
        assert module.bias.grad.equal(Tensor(data=expected_bias_grad, dims=VECTOR_DIM))

    def test_nograd_tensor_annotations_are_inherited_and_applied(self):
        module = ChildAnnotatedBlock()

        assert module.__nograd_tensors__ == frozenset({"cache", "projection"})
        assert not module.cache.requires_grad
        assert not module.projection.requires_grad
        assert module.weight.requires_grad

        output = module.weight.data + module.cache.data + module.projection.data
        Tensor(data=output.sum(), dims=()).backward()

        assert module.weight.grad is not None
        assert module.cache.grad is None
        assert module.projection.grad is None

    def test_nested_modules_are_registered_and_receive_gradients(self):
        module = NestedBlock()
        x = torch.tensor([1.5, -0.5])

        assert module.device == Device("cpu")
        assert module.inner.device == Device("cpu")
        assert module.scale.device == Device("cpu")
        assert module.inner.weight.device == Device("cpu")
        assert module.inner.bias.device == Device("cpu")
        assert module.inner.basis.device == Device("cpu")

        output = module.scale.data * (
            module.inner.weight.data * x
            + module.inner.bias.data
            + module.inner.basis.data
        )
        loss = Tensor(data=output.sum(), dims=())
        loss.backward()

        expected_inner_weight_grad = module.scale.data.detach() * x
        expected_inner_bias_grad = module.scale.data.detach()
        expected_scale_grad = (
            module.inner.weight.data.detach() * x
            + module.inner.bias.data.detach()
            + module.inner.basis.data.detach()
        )

        assert module.inner.weight.grad is not None
        assert module.inner.bias.grad is not None
        assert module.scale.grad is not None
        assert module.inner.basis.grad is None

        assert module.inner.weight.grad.equal(
            Tensor(data=expected_inner_weight_grad, dims=VECTOR_DIM)
        )
        assert module.inner.bias.grad.equal(
            Tensor(data=expected_inner_bias_grad, dims=VECTOR_DIM)
        )
        assert module.scale.grad.equal(
            Tensor(data=expected_scale_grad, dims=VECTOR_DIM)
        )

        named_parameters = dict(module.named_parameters())
        assert f"inner.{TENSOR_PARAM_PREFIX}weight" in named_parameters
        assert f"inner.{TENSOR_PARAM_PREFIX}bias" in named_parameters
        assert f"inner.{TENSOR_PARAM_PREFIX}basis" not in named_parameters
        assert f"{TENSOR_PARAM_PREFIX}scale" in named_parameters

        named_buffers = dict(module.named_buffers())
        assert f"inner.{TENSOR_BUFFER_PREFIX}basis" in named_buffers

    def test_nested_module_to_device_keeps_logical_device_in_sync(self):
        module = NestedBlock()

        moved = module.to_device(Device("cpu"))

        assert moved is module
        assert module.device == Device("cpu")
        assert module.inner.device == Device("cpu")
        assert module.scale.device == Device("cpu")
        assert module.inner.weight.device == Device("cpu")
        assert module.inner.bias.device == Device("cpu")
        assert module.inner.basis.device == Device("cpu")

    def test_module_device_and_owned_storage_stay_synchronized(self):
        module = NestedBlock()

        self._assert_module_tensor_device_sync(module)
        self._assert_module_tensor_device_sync(module.inner)

        module.freeze()
        self._assert_module_tensor_device_sync(module)
        self._assert_module_tensor_device_sync(module.inner)

        module.unfreeze()
        self._assert_module_tensor_device_sync(module)
        self._assert_module_tensor_device_sync(module.inner)

        module.to_device(Device("cpu"))
        self._assert_module_tensor_device_sync(module)
        self._assert_module_tensor_device_sync(module.inner)

    def test_module_export_and_parameter_cleanup_behavior(self):
        module = AffineBlock()

        exported = module.export("weight")
        assert isinstance(exported, Tensor)
        assert exported.equal(module.weight.detach().clone())
        assert exported.data is not module.weight.data
        assert not exported.requires_grad

        basis_export = module.export("basis")
        assert isinstance(basis_export, Tensor)
        assert basis_export.equal(module.basis.detach().clone())
        assert basis_export.data is not module.basis.data
        assert not basis_export.requires_grad

        with pytest.raises(TypeError, match="is not a Tensor"):
            module.export("label")

        module.weight = 3
        assert f"{TENSOR_PARAM_PREFIX}weight" not in dict(module.named_parameters())

        module.bias = Tensor(data=torch.tensor([7.0, 8.0]), dims=VECTOR_DIM).attach()
        assert (
            dict(module.named_parameters())[f"{TENSOR_PARAM_PREFIX}bias"]
            is module.bias.data
        )

        del module.basis
        assert not hasattr(module, "basis")
        assert f"{TENSOR_BUFFER_PREFIX}basis" not in dict(module.named_buffers())

    def test_export_all_recursively_exports_public_tensors(self):
        module = NestedBlock()

        exported = module.export_all()

        assert isinstance(exported, FrozenDict)
        assert set(exported) == {"scale", "inner.weight", "inner.bias", "inner.basis"}
        assert exported["scale"].equal(module.scale.detach().clone())
        assert exported["inner.weight"].equal(module.inner.weight.detach().clone())
        assert exported["inner.bias"].equal(module.inner.bias.detach().clone())
        assert exported["inner.basis"].equal(module.inner.basis.detach().clone())

        assert exported["scale"].data is not module.scale.data
        assert exported["inner.weight"].data is not module.inner.weight.data
        assert exported["inner.bias"].data is not module.inner.bias.data
        assert exported["inner.basis"].data is not module.inner.basis.data

    def test_module_respects_torch_no_grad_contexts(self):
        module = AffineBlock()
        x = torch.tensor([2.0, -1.0])

        with torch.no_grad():
            context_output = (
                module.weight.data * x + module.bias.data + module.basis.data
            )
            context_loss = Tensor(data=context_output.sum(), dims=())

        assert not context_output.requires_grad
        assert not context_loss.requires_grad
        with pytest.raises(RuntimeError, match="does not require grad"):
            context_loss.backward()

        @torch.no_grad()
        def build_loss() -> Tensor:
            decorated_output = (
                module.weight.data * x + module.bias.data + module.basis.data
            )
            return Tensor(data=decorated_output.sum(), dims=())

        decorated_loss = build_loss()
        assert not decorated_loss.requires_grad
        with pytest.raises(RuntimeError, match="does not require grad"):
            decorated_loss.backward()

    def test_module_freeze_and_unfreeze_toggle_parameter_grads(self):
        module = AffineBlock()

        assert module.freeze() is module
        assert not module.weight.requires_grad
        assert not module.bias.requires_grad
        assert not module.basis.requires_grad

        assert module.unfreeze() is module
        assert module.weight.requires_grad
        assert module.bias.requires_grad
        assert not module.basis.requires_grad

    def test_module_freeze_and_unfreeze_apply_recursively(self):
        module = NestedBlock()

        module.freeze()
        assert not module.scale.requires_grad
        assert not module.inner.weight.requires_grad
        assert not module.inner.bias.requires_grad
        assert not module.inner.basis.requires_grad

        module.unfreeze()
        assert module.scale.requires_grad
        assert module.inner.weight.requires_grad
        assert module.inner.bias.requires_grad
        assert not module.inner.basis.requires_grad
