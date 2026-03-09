import torch
import sympy as sy
from pyhilbert.tensors import Tensor
from pyhilbert.hilbert_space import U1Basis, hilbert
from pyhilbert.state_space import IndexSpace


def _state(tag: str, idx: int) -> U1Basis:
    return U1Basis(u1=sy.Integer(1), rep=((tag, idx),))


class TestTensorAutograd:
    def test_tensor_autograd_behavior(self):
        # Setup basic data
        dims = (IndexSpace.linear(3), IndexSpace.linear(3))
        data = torch.randn(3, 3)

        # --- 1. Test requires_grad property ---
        t_false = Tensor(data=data.clone(), dims=dims)
        assert not t_false.requires_grad, (
            "Should be False when data does not require grad"
        )

        t_true = Tensor(data=data.clone().requires_grad_(True), dims=dims)
        assert t_true.requires_grad, "Should be True when data requires grad"

        # --- 2. Test attach() ---
        # Case A: attach() on a tensor without grad
        t_attached = t_false.attach()
        assert t_attached is not t_false
        assert t_attached.requires_grad
        assert t_attached.data.is_leaf

        # Verify independence: modifying original shouldn't affect attached
        t_false.data[0, 0] += 100
        assert t_attached.data[0, 0] != t_false.data[0, 0], (
            "Attached tensor should be a deep copy/clone"
        )

        # Case B: attach() on a tensor ALREADY with grad
        t_reattached = t_true.attach()
        assert t_reattached is t_true, "Should return self if already attached"

        # --- 3. Test detach() ---
        # Should return NEW tensor, grad=False, SHARED storage
        t_detached = t_true.detach()
        assert t_detached is not t_true
        assert not t_detached.requires_grad

        # Verify shared storage: modifying detached data SHOULD affect original
        # (Note: we modify the detached one to avoid autograd errors on leaf variables)
        original_val = t_true.data[0, 0].item()
        t_detached.data[0, 0] += 50
        assert t_true.data[0, 0] == t_detached.data[0, 0], (
            "Detached tensor should share storage"
        )
        assert t_true.data[0, 0] != original_val

        # --- 4. Test clone() ---
        t_cloned = t_true.clone()
        assert t_cloned is not t_true
        assert t_cloned.requires_grad == t_true.requires_grad

        # Verify independence
        t_cloned.data[1, 1] += 200
        assert t_true.data[1, 1] != t_cloned.data[1, 1], (
            "Cloned tensor should have independent storage"
        )

    def test_autograd_application_flow(self):
        """
        Test a simulated application flow involving forward pass,
        gradient calculation, and detachment.
        """
        # 1. Simulate inputs and weights
        space = hilbert(_state("batch", i) for i in range(3))
        dims = (space,)

        x_data = torch.tensor([1.0, 2.0, 3.0])
        w_data = torch.tensor([0.5, 0.5, 0.5])

        # Create tensors
        x = Tensor(data=x_data, dims=dims)
        w = Tensor(data=w_data, dims=dims)

        # 2. Attach gradients
        x_input = x.attach()
        w_param = w.attach()

        # 3. Forward pass: y = x + w (using operator_add)
        y = x_input + w_param

        # 4. Compute loss (scalar)
        # Target: [2.0, 3.0, 4.0]
        target = torch.tensor([2.0, 3.0, 4.0])
        diff = y.data - target
        loss = Tensor(data=(diff**2).sum(), dims=())

        # 5. Backward pass
        loss.backward()

        # 6. Verify gradients
        # y = x + w
        # L = sum((y - t)^2) = sum((x + w - t)^2)
        # dL/dx = 2 * (x + w - t)
        # x=[1,2,3], w=[.5,.5,.5], t=[2,3,4]
        # x+w-t = [1.5, 2.5, 3.5] - [2, 3, 4] = [-0.5, -0.5, -0.5]
        # grad = 2 * (-0.5) = -1.0

        expected_grad = torch.tensor([-1.0, -1.0, -1.0])
        expected_grad_tensor = Tensor(data=expected_grad, dims=dims)

        assert w_param.grad is not None
        assert x_input.grad is not None
        assert w_param.grad.equal(expected_grad_tensor), (
            f"Expected grad {expected_grad_tensor}, got {w_param.grad}"
        )
        assert x_input.grad.equal(expected_grad_tensor)

        # 7. Test Detach
        w_detached = w_param.detach()
        y_val = x_input + w_detached
        loss_val = Tensor(data=y_val.data.sum(), dims=())

        # Zero gradients
        if x_input.data.grad is not None:
            x_input.data.grad.zero_()

        loss_val.backward()

        # x_input should have grad 1.0 (derivative of sum(x))
        assert x_input.grad is not None
        assert x_input.grad.equal(Tensor(data=torch.ones(3), dims=dims))

        # w_detached should not have grad
        assert not w_detached.requires_grad
        assert w_detached.grad is None

    def test_clone_autograd(self):
        """Test that clone() preserves autograd history."""
        dims = (IndexSpace.linear(1),)
        x = Tensor(data=torch.tensor([1.0]), dims=dims).attach()

        # Clone
        y = x.clone()

        # Operation
        z = y.data * 2
        loss = Tensor(data=z.sum(), dims=())
        loss.backward()

        # d(2*y)/dx = 2 * dy/dx = 2 * 1 = 2
        assert x.grad is not None
        assert x.grad.equal(Tensor(data=torch.tensor([2.0]), dims=dims))

    def test_backward_aligns_gradient_with_permuted_dimension_elements(self):
        left = hilbert(_state("left", i) for i in range(2))
        right = hilbert(_state("right", i) for i in range(3))
        right_permuted = hilbert(_state("right", i) for i in (2, 0, 1))

        x = Tensor(
            data=torch.arange(6.0, dtype=torch.float32).reshape(2, 3),
            dims=(left, right),
        ).attach()
        y = x.clone()
        upstream = Tensor(
            data=torch.tensor(
                [
                    [10.0, 20.0, 30.0],
                    [40.0, 50.0, 60.0],
                ]
            ),
            dims=(left, right_permuted),
        )

        y.backward(upstream)

        assert x.grad is not None
        assert x.grad.equal(
            Tensor(
                data=torch.tensor(
                    [
                        [20.0, 30.0, 10.0],
                        [50.0, 60.0, 40.0],
                    ]
                ),
                dims=(left, right),
            )
        )
