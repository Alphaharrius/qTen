from contextlib import nullcontext

import pytest
import torch

from qten.linalg.tensors import (
    Tensor,
    at_device,
    allclose,
    cat,
    equal,
    isclose,
    matmul,
    where,
    zeros,
)
from qten.symbolics.state_space import IndexSpace
from qten.utils.devices import Device


def _has_gpu() -> bool:
    try:
        Device("gpu").torch_device()
        return True
    except RuntimeError:
        return False


HAS_GPU = _has_gpu()


VECTOR_DIM = (IndexSpace.linear(3),)
MATRIX_DIM = (IndexSpace.linear(2), IndexSpace.linear(2))
INDEX_DIM = (IndexSpace.linear(2),)


def _move(tensor: Tensor, device_name: str) -> Tensor:
    return tensor.to_device(Device(device_name))


def _device(name: str) -> Device:
    if name == "gpu" and HAS_GPU:
        try:
            td = Device("gpu").torch_device()
            if td.type == "cuda":
                return Device("gpu", td.index)
        except RuntimeError:
            pass
    return Device(name)


def _expected_device(*device_names: str) -> Device:
    return _device("gpu") if "gpu" in device_names else _device("cpu")


def _case(*device_names: str):
    needs_gpu = "gpu" in device_names
    marks = pytest.mark.skipif(not HAS_GPU and needs_gpu, reason="requires GPU support")
    return pytest.param(*device_names, marks=marks)


@pytest.mark.parametrize(
    ("left_device", "right_device"),
    [
        _case("cpu", "cpu"),
        _case("cpu", "gpu"),
        _case("gpu", "cpu"),
        _case("gpu", "gpu"),
    ],
)
class TestTensorCrossDeviceBinaryOps:
    def test_add_auto_promotes_device(self, left_device: str, right_device: str):
        left = _move(
            Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM), left_device
        )
        right = _move(
            Tensor(data=torch.tensor([4.0, 5.0, 6.0]), dims=VECTOR_DIM), right_device
        )

        result = left + right

        assert result.device == _expected_device(left_device, right_device)
        assert torch.equal(result.data.cpu(), torch.tensor([5.0, 7.0, 9.0]))

    def test_matmul_auto_promotes_device(self, left_device: str, right_device: str):
        left = _move(
            Tensor(
                data=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                dims=MATRIX_DIM,
            ),
            left_device,
        )
        right = _move(
            Tensor(
                data=torch.tensor([[2.0, 0.0], [1.0, 2.0]]),
                dims=MATRIX_DIM,
            ),
            right_device,
        )

        result = matmul(left, right)

        assert result.device == _expected_device(left_device, right_device)
        assert torch.equal(result.data.cpu(), torch.tensor([[4.0, 4.0], [10.0, 8.0]]))

    def test_eq_auto_promotes_device(self, left_device: str, right_device: str):
        left = _move(
            Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM), left_device
        )
        right = _move(
            Tensor(data=torch.tensor([1.0, 0.0, 3.0]), dims=VECTOR_DIM), right_device
        )

        result = left == right

        assert result.device == _expected_device(left_device, right_device)
        assert torch.equal(result.data.cpu(), torch.tensor([True, False, True]))

    def test_lt_auto_promotes_device(self, left_device: str, right_device: str):
        left = _move(
            Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM), left_device
        )
        right = _move(
            Tensor(data=torch.tensor([2.0, 1.0, 4.0]), dims=VECTOR_DIM), right_device
        )

        result = left < right

        assert result.device == _expected_device(left_device, right_device)
        assert torch.equal(result.data.cpu(), torch.tensor([True, False, True]))

    def test_isclose_auto_promotes_device(self, left_device: str, right_device: str):
        left = _move(
            Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM), left_device
        )
        right = _move(
            Tensor(
                data=torch.tensor([1.0 + 1e-6, 2.1, 3.0 - 1e-6]),
                dims=VECTOR_DIM,
            ),
            right_device,
        )

        result = isclose(left, right, atol=1e-5)

        assert result.device == _expected_device(left_device, right_device)
        assert torch.equal(result.data.cpu(), torch.tensor([True, False, True]))

    def test_allclose_auto_promotes_device(self, left_device: str, right_device: str):
        left = _move(
            Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM), left_device
        )
        right = _move(
            Tensor(
                data=torch.tensor([1.0 + 1e-6, 2.0 - 1e-6, 3.0]),
                dims=VECTOR_DIM,
            ),
            right_device,
        )

        assert allclose(left, right, atol=1e-5)

    def test_equal_auto_promotes_device(self, left_device: str, right_device: str):
        left = _move(
            Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM), left_device
        )
        right = _move(
            Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM), right_device
        )

        assert equal(left, right)

    def test_tensor_advanced_indexing_auto_promotes_index_device(
        self, left_device: str, right_device: str
    ):
        data = _move(
            Tensor(data=torch.tensor([10.0, 20.0, 30.0]), dims=VECTOR_DIM),
            left_device,
        )
        index = _move(
            Tensor(data=torch.tensor([2, 0], dtype=torch.long), dims=INDEX_DIM),
            right_device,
        )

        result = data[index]

        assert result.device == data.device
        assert result.dims == INDEX_DIM
        assert torch.equal(result.data.cpu(), torch.tensor([30.0, 10.0]))

    def test_cat_auto_promotes_device(self, left_device: str, right_device: str):
        left = _move(
            Tensor(data=torch.tensor([1.0, 2.0]), dims=(IndexSpace.linear(2),)),
            left_device,
        )
        right = _move(
            Tensor(data=torch.tensor([3.0, 4.0, 5.0]), dims=(IndexSpace.linear(3),)),
            right_device,
        )

        result = cat([left, right], dim=0)

        assert result.device == _expected_device(left_device, right_device)
        assert result.dims == (IndexSpace.linear(5),)
        assert torch.equal(result.data.cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


@pytest.mark.parametrize(
    ("condition_device", "input_device", "other_device"),
    [
        _case("cpu", "cpu", "cpu"),
        _case("cpu", "cpu", "gpu"),
        _case("cpu", "gpu", "cpu"),
        _case("cpu", "gpu", "gpu"),
        _case("gpu", "cpu", "cpu"),
        _case("gpu", "cpu", "gpu"),
        _case("gpu", "gpu", "cpu"),
        _case("gpu", "gpu", "gpu"),
    ],
)
class TestTensorCrossDeviceWhere:
    def test_where_auto_promotes_device(
        self, condition_device: str, input_device: str, other_device: str
    ):
        condition = _move(
            Tensor(data=torch.tensor([True, False, True]), dims=VECTOR_DIM),
            condition_device,
        )
        input = _move(
            Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM),
            input_device,
        )
        other = _move(
            Tensor(data=torch.tensor([10.0, 20.0, 30.0]), dims=VECTOR_DIM),
            other_device,
        )

        result = where(condition, input, other)

        assert result.device == _expected_device(
            condition_device, input_device, other_device
        )
        assert torch.equal(result.data.cpu(), torch.tensor([1.0, 20.0, 3.0]))


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(not HAS_GPU, reason="requires GPU support"),
        ),
    ],
)
class TestTensorDeviceContext:
    def test_tensor_constructor_forces_requested_device(self, device_name: str):
        with at_device(Device(device_name)):
            tensor = Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM)

        assert tensor.device == _device(device_name)

    def test_factories_inherit_requested_device(self, device_name: str):
        with at_device(device_name):
            tensor = zeros(VECTOR_DIM)

        assert tensor.device == _device(device_name)

    def test_nested_scope_restores_outer_device(self, device_name: str):
        outer = Device(device_name)
        inner = Device("cpu") if device_name == "gpu" else Device("gpu")
        use_inner_scope = inner.name != "gpu" or HAS_GPU
        inner_ctx = at_device(inner) if use_inner_scope else nullcontext()

        with at_device(outer):
            outer_tensor = Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=VECTOR_DIM)
            with inner_ctx:
                inner_tensor = Tensor(
                    data=torch.tensor([4.0, 5.0, 6.0]),
                    dims=VECTOR_DIM,
                )
            restored_tensor = Tensor(
                data=torch.tensor([7.0, 8.0, 9.0]),
                dims=VECTOR_DIM,
            )

        assert outer_tensor.device == _device(outer.name)
        assert restored_tensor.device == _device(outer.name)
        expected_inner = inner if use_inner_scope else outer
        assert inner_tensor.device == _device(expected_inner.name)
