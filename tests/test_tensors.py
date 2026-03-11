import pytest
from typing import Tuple
import torch
import sympy as sy
from dataclasses import dataclass
from collections import OrderedDict
from pyhilbert import state_space
from pyhilbert.tensors import (
    Tensor,
    all as tensor_all,
    align_all,
    allclose,
    astype,
    equal,
    kernel_tensor,
    matmul,
    nonzero,
    one_hot,
    ones,
    union_dims,
    where,
    zeros,
)
from pyhilbert.hilbert_space import HilbertSpace, U1Basis, hilbert
from pyhilbert.state_space import (
    BroadcastSpace,
    IndexSpace,
    MomentumSpace,
)
from pyhilbert.utils import FrozenDict
from pyhilbert.tensors import unsqueeze


@dataclass(frozen=True)
class MockMode:
    attr: FrozenDict
    size: int

    def unit(self):
        return self


def make_mode(name: str, size: int) -> MockMode:
    return MockMode(attr=FrozenDict({"name": name}), size=size)


@dataclass(frozen=True)
class MockModeElement:
    mode: MockMode
    index: int

    @property
    def dim(self) -> int:
        return 1

    def unit(self):
        return self


def _mode_elements(mode: MockMode) -> Tuple[MockModeElement, ...]:
    return tuple(MockModeElement(mode=mode, index=i) for i in range(mode.size))


def _space_from_modes(*modes: MockMode) -> HilbertSpace:
    elements = tuple(el for mode in modes for el in _mode_elements(mode))
    return HilbertSpace(OrderedDict((el, i) for i, el in enumerate(elements)))


@pytest.fixture
def matmul_ctx():
    class Context:
        def __init__(self):
            # Define some dummy modes
            self.mode_a = make_mode("a", 2)
            self.mode_b = make_mode("b", 3)
            self.mode_c = make_mode("c", 4)
            self.space_a = _space_from_modes(self.mode_a)
            self.space_b = _space_from_modes(self.mode_b)
            self.space_c = _space_from_modes(self.mode_c)

            # Create StateSpaces
            # Space 1: A + B (dim 5)
            self.space1 = _space_from_modes(self.mode_a, self.mode_b)

            # Space 2: C (dim 4)
            self.space2 = self.space_c

            # Space 3: B + A (dim 5) - Same span as Space 1 but different order
            self.space3 = _space_from_modes(self.mode_b, self.mode_a)

    return Context()


class TestMatmul:
    def test_tensor_constructor_rejects_shape_dim_mismatch(self):
        space = _space_from_modes(make_mode("a", 2), make_mode("b", 3))

        with pytest.raises(ValueError, match="does not match expected shape"):
            Tensor(data=torch.randn(4), dims=(space,))

    def test_basic_matmul(self, matmul_ctx):
        # A (2, 5) x B (5, 4) -> C (2, 4)
        # Dimensions: (space2, space1) x (space1, space2)
        # Wait, matmul contracts left[-1] and right[-2]
        # left: (M, K) -> (space2, space1)
        # right: (K, N) -> (space1, space2)
        # result: (M, N) -> (space2, space2)

        data_left = torch.randn(matmul_ctx.space2.dim, matmul_ctx.space1.dim)
        tensor_left = Tensor(
            data=data_left, dims=(matmul_ctx.space2, matmul_ctx.space1)
        )

        data_right = torch.randn(matmul_ctx.space1.dim, matmul_ctx.space2.dim)
        tensor_right = Tensor(
            data=data_right, dims=(matmul_ctx.space1, matmul_ctx.space2)
        )

        result = matmul(tensor_left, tensor_right)

        expected_data = torch.matmul(data_left, data_right)
        assert torch.allclose(result.data, expected_data)
        assert result.dims == (matmul_ctx.space2, matmul_ctx.space2)

    def test_matmul_with_alignment(self, matmul_ctx):
        # Test where contraction dimensions have different internal order (space1 vs space3)
        # left: (space2, space1)
        # right: (space3, space2)
        # space1 and space3 cover {A, B} but in different order.

        data_left = torch.randn(matmul_ctx.space2.dim, matmul_ctx.space1.dim)
        tensor_left = Tensor(
            data=data_left, dims=(matmul_ctx.space2, matmul_ctx.space1)
        )

        # Create data for right tensor corresponding to space3 ordering
        # space3: B (0-3), A (3-5)
        # space1: A (0-2), B (2-5)
        # logical vector v in space3 order [b0, b1, b2, a0, a1]
        # logical vector v in space1 order [a0, a1, b0, b1, b2]

        data_right_s3 = torch.randn(matmul_ctx.space3.dim, matmul_ctx.space2.dim)
        tensor_right = Tensor(
            data=data_right_s3, dims=(matmul_ctx.space3, matmul_ctx.space2)
        )

        # When matmul(left, right) happens:
        # right is aligned to left.dims[-1] (space1).
        # alignment permutes right from space3 to space1.

        result = matmul(tensor_left, tensor_right)

        # Manually align right data to check result
        # Permutation from space3 to space1
        # space3 indices: 0,1,2 (B), 3,4 (A)
        # space1 indices: A first (so take 3,4 from space3), B second (take 0,1,2 from space3)
        # align indices: [3, 4, 0, 1, 2]
        indices = torch.tensor([3, 4, 0, 1, 2], dtype=torch.long)
        aligned_data_right = torch.index_select(data_right_s3, 0, indices)

        expected_data = torch.matmul(data_left, aligned_data_right)
        assert torch.allclose(result.data, expected_data), (
            "Data mismatch after alignment"
        )
        assert result.dims == (matmul_ctx.space2, matmul_ctx.space2)

    def test_matmul_rank3(self, matmul_ctx):
        # Test matmul between two 3-tensors to verify batch dimension handling
        # Left: (Batch, M, K) -> (Space2, Space2, Space1)
        # Right: (Batch, K, N) -> (Space2, Space1, Space2)
        # Result should be (Space2, Space2, Space2)

        dims_left = (matmul_ctx.space2, matmul_ctx.space2, matmul_ctx.space1)
        dims_right = (matmul_ctx.space2, matmul_ctx.space1, matmul_ctx.space2)

        data_left = torch.randn(
            matmul_ctx.space2.dim, matmul_ctx.space2.dim, matmul_ctx.space1.dim
        )
        data_right = torch.randn(
            matmul_ctx.space2.dim, matmul_ctx.space1.dim, matmul_ctx.space2.dim
        )

        t_left = Tensor(data_left, dims_left)
        t_right = Tensor(data_right, dims_right)

        result = matmul(t_left, t_right)

        # Check dimensions
        # Expected: Batch dim (Space2) + Left non-contracted (Space2) + Right non-contracted (Space2)
        expected_dims = (matmul_ctx.space2, matmul_ctx.space2, matmul_ctx.space2)
        assert result.dims == expected_dims

        # Check data
        # torch.matmul handles batch matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
        expected_data = torch.matmul(data_left, data_right)
        assert torch.allclose(result.data, expected_data)

    def test_broadcasting_missing_dims(self, matmul_ctx):
        # left: (space1) -> interpreted as (space1)
        # right: (space2, space1, space2)
        # unsqueezed left -> (Broadcast, Broadcast, space1)
        # This might fail because matmul expects left to be at least 1D?
        # Actually matmul aligns left[-1] with right[-2].
        # If left is 1D: dims=(space1,), left[-1] is space1.
        # If right is 3D: dims=(D1, space1, D2). right[-2] is space1.
        # Broadcast match:
        # left unsqueezed to match rank 3: (Broadcast, Broadcast, space1)
        # batch dims:
        # 0: left(Broadcast) vs right(D1) -> match
        # 1: left(Broadcast) vs right(space1) -> This is the contraction dim for right? No, right[-2] is contraction.
        # Wait, if right is (D1, space1, D2), rank is 3.
        # left is (space1), rank 1.
        # _match_dims makes left rank 3: (Broadcast, Broadcast, space1).
        # _align_dims:
        # n=0: left(Broadcast), right(D1) -> ok.
        # n=1: left(Broadcast), right(space1) -> ok.
        # Contraction: left[-1] (space1) vs right[-2] (space1). Match!

        # Let's try:
        # Left: (space1, )
        # Right: (space2, space1, space2)

        data_left = torch.randn(matmul_ctx.space1.dim)
        tensor_left = Tensor(data=data_left, dims=(matmul_ctx.space1,))

        data_right = torch.randn(
            matmul_ctx.space2.dim, matmul_ctx.space1.dim, matmul_ctx.space2.dim
        )
        tensor_right = Tensor(
            data=data_right,
            dims=(matmul_ctx.space2, matmul_ctx.space1, matmul_ctx.space2),
        )

        result = tensor_left @ tensor_right

        # Expected:
        # left broadcasted to (1, 1, space1.dim)
        # right is (space2.dim, space1.dim, space2.dim)
        # but the broadcasting logic in matmul implementation:
        # unsqueeze adds BroadcastSpace.
        # align expands BroadcastSpace if the other is not BroadcastSpace.
        # align(tensor, dim, target):
        # if current is Broadcast, expand data.

        # left becomes (space2, space1_broadcast?? No wait)
        # _align_dims_for_matopt iterates up to -2.
        # left dims: (Broadcast, Broadcast, space1)
        # right dims: (space2, space1, space2)
        # loop n=0: left[0] is Broadcast, right[0] is space2. left aligned to space2.
        # data_left expanded at dim 0.

        # loop n=1: left[1] is Broadcast, right[1] is space1.
        # But wait, contraction is left[-1] vs right[-2].
        # left[-1] is space1. right[-2] is space1.
        # So left is (space2, Broadcast, space1).
        # right is (space2, space1, space2).
        # This seems overlapping.
        # Contraction logic:
        # "The contraction always happens between left.dims[-1] and right.dims[-2]."

        # So for left=(space1), right=(space2, space1, space2)
        # left -> (Broadcast, Broadcast, space1)
        # align n=0: left aligned to space2 -> (space2, Broadcast, space1)
        # align n=1: left aligned to ? right[1] is space1.
        # left aligned to space1 -> (space2, space1, space1) (Broadcast expands to space1)
        # result dims: left[:-1] + right[-1:]
        # left[:-1] is (space2, space1). right[-1:] is (space2).
        # result: (space2, space1, space2).

        # But wait, torch.matmul logic:
        # left (B, N, K) x right (B, K, M)
        # here left is (space2, space1, space1). right is (space2, space1, space2).
        # torch.matmul(left, right) -> left shape (S2, S1, S1), right shape (S2, S1, S2).
        # batch dim S2 matches.
        # matrix mul: (S1, S1) x (S1, S2) -> (S1, S2).
        # final shape (S2, S1, S2).

        # Let's verify this behavior is what we expect.
        # Effectively batched vector-matrix mult?
        # A (K) x B (M, K, N)
        # -> A broadcasted to (M, K) -> (M, 1, K) or something?

        # Actually usually A(K) @ B(..., K, N) works in torch.
        # In torch, (K) @ (M, K, N) -> (M, N).
        # But here we force ranks to match.
        # left becomes (1, 1, K) -> expand to (M, 1, K)? No, align expands to target size.
        # left dim 0 aligned to right dim 0 (space2). size M.
        # left dim 1 aligned to right dim 1 (space1). size K.
        # so left becomes (M, K, K).
        # right is (M, K, N).
        # result (M, K, N).

        # Is this intended?
        # If I have a vector v and a batch of matrices M_i, I might want v @ M_i for all i.
        # v: (K). M: (Batch, K, N).
        # v -> (1, 1, K).
        # align dim 0: (Batch, 1, K). (Broadcast to Batch).
        # align dim 1: (Batch, K, K) ?? Why align dim 1 to K?
        # right dim 1 is K (from K, N).
        # So left becomes (Batch, K, K).
        # data is replicated K times along dim 1? That seems wrong for standard matmul.

        # Standard broadcast: (K) vs (B, K, N)
        # (K) treated as (1, K) if we follow numpy/torch broadcasting rules?
        # But here `_match_dims` unsqueezes at 0.
        # (K) -> (1, 1, K).
        # right is (B, K, N).
        # dim 0: 1 vs B -> expand to B. -> (B, 1, K).
        # dim 1: 1 vs K -> expand to K. -> (B, K, K).

        # This seems to imply PyHilbert's matmul might behave differently than torch.matmul if broadcasting aligns batch dims aggressively.
        # Or maybe I should not align dim 1 if it is the contraction dim for right?
        # `_align_dims_for_matopt` loops `left.dims[:-2]`.
        # So for left (B, B, K) it only aligns dim 0.
        # dim 1 is left.dims[-2]. It is skipped by loop.
        # So left remains (B, Broadcast, K).
        # right is (B, K, N).
        # torch.matmul((B, 1, K), (B, K, N)) -> (B, N).
        # Result dims before squeeze: left[:-1] (B, Broadcast) + right[-1:] (N) -> (B, Broadcast, N).
        # Since left is 1D, matmul squeezes the added dimension, yielding (B, N).

        expected_data_torch = torch.matmul(data_left, data_right)
        # torch result for (K) @ (M, K, N) is (M, N).

        result = matmul(tensor_left, tensor_right)

        # If the code works as analyzed:
        # left unsqueezed: (Broadcast, Broadcast, space1)
        # align loop over [:-2]: index 0 only.
        # index 0: left[0] (Broadcast) aligns to right[0] (space2).
        # left -> (space2, Broadcast, space1). Data shape (S2, 1, S1).
        # torch.matmul((S2, 1, S1), (S2, S1, S2))
        # -> (S2, 1, S2).
        # Result dims before squeeze: (space2, Broadcast, space2).
        # After squeeze (left is 1D): (space2, space2).

        # Check if result matches this expectation
        assert len(result.dims) == 2
        assert result.dims[0] == matmul_ctx.space2
        assert result.dims[1] == matmul_ctx.space2

        # data check
        # torch.matmul(v, M) -> (M, N) usually?
        # A = torch.randn(5)
        # B = torch.randn(4, 5, 4)
        # torch.matmul(A, B).shape -> (4, 4).
        # My result is (4, 4).
        assert torch.allclose(result.data, expected_data_torch)

    def test_broadcast_space_alignment(self, matmul_ctx):
        # Explicit broadcasting test using BroadcastSpace
        # left: (Broadcast, space1) -> effectively (1, 5)
        # right: (space2, space1, space2) -> (4, 5, 4)

        # Create a tensor with explicit BroadcastSpace
        # BroadcastSpace has size 1 implicitly for data generation usually?
        # But Tensor.data shape must match dims. BroadcastSpace doesn't have a fixed size property that returns 1?
        # StateSpace.dim relies on structure. BroadcastSpace structure is empty, size is 0?
        # Let's check StateSpace.dim implementation.
        # size returns structure[last].stop. If empty, 0.

        # If BroadcastSpace size is 0, we can't create a tensor with dim size 0 and expect it to broadcast to N?
        # Usually broadcasting dim has size 1.
        # But BroadcastSpace in PyHilbert seems to handle "unsqueezed" dims.
        # unsqueeze() creates data with dim size 1.

        # Let's use unsqueeze to create the tensor with BroadcastSpace
        data_orig = torch.randn(matmul_ctx.space1.dim)
        tensor_orig = Tensor(data=data_orig, dims=(matmul_ctx.space1,))

        tensor_left = unsqueeze(tensor_orig, 0)  # (Broadcast, space1)

        # right tensor
        data_right = torch.randn(
            matmul_ctx.space2.dim, matmul_ctx.space1.dim, matmul_ctx.space2.dim
        )
        tensor_right = Tensor(
            data=data_right,
            dims=(matmul_ctx.space2, matmul_ctx.space1, matmul_ctx.space2),
        )

        # This is effectively the same as missing dims test but we manually created the BroadcastSpace tensor
        result = matmul(tensor_left, tensor_right)

        assert len(result.dims) == 3
        assert result.dims[0] == matmul_ctx.space2
        # The result logic in matmul: left[:-1] + right[-1:]
        # left: (Broadcast, space1). left[:-1] -> (Broadcast,)
        # right: (space2, space1, space2). right[-1:] -> (space2,)
        # Wait, but during alignment:
        # left (Broadcast, space1), right (space2, space1, space2)
        # match dims -> left unsqueezed -> (Broadcast, Broadcast, space1)
        # So tensor_left which is (Broadcast, space1) is unsqueezed to (Broadcast, Broadcast, space1).
        # And proceeds as before.

        # What if we have (space2, Broadcast) x (space2, space1)?
        # left: (space2, Broadcast) (e.g. 4, 1)
        # right: (space2, space1) (e.g. 4, 5)
        # contraction: Broadcast vs space2.
        # Broadcast vs space2 -> This should fail or expand?
        # contraction requires alignment.
        # align(left, -1, space2). left[-1] is Broadcast.
        # align: if current is Broadcast, expand.
        # So left becomes (space2, space2). Data expanded.
        # Then matmul (S2, S2) x (S2, S1) -> (S2, S1).
        # This is outer product if S2 was 1, but here S2=4.
        # It's (4, 4) x (4, 5) -> (4, 5).
        # where (4, 4) is formed by repeating the column vector 4 times?

        # Test case where right tensor has BroadcastSpace at contraction dimension
        # left: (space2, space2) -> (4, 4)
        # right: (Broadcast, space1) -> (1, 5)
        # contraction: space2 (4) vs Broadcast (1).
        # matmul aligns right[-2] (Broadcast) to left[-1] (space2).
        # expected: right expands to (4, 5). matmul((4,4), (4,5)) -> (4, 5).

        data_left = torch.randn(matmul_ctx.space2.dim, matmul_ctx.space2.dim)
        tensor_left = Tensor(
            data=data_left, dims=(matmul_ctx.space2, matmul_ctx.space2)
        )

        data_right = torch.randn(1, matmul_ctx.space1.dim)
        tensor_right = Tensor(
            data=data_right, dims=(BroadcastSpace(), matmul_ctx.space1)
        )

        result = matmul(tensor_left, tensor_right)

        # Verify
        # right expanded
        expanded_right = data_right.expand(matmul_ctx.space2.dim, matmul_ctx.space1.dim)
        expected = torch.matmul(data_left, expanded_right)

        assert result.dims == (matmul_ctx.space2, matmul_ctx.space1)
        assert torch.allclose(result.data, expected)

    def test_incompatible_shapes(self, matmul_ctx):
        # left: (space2, space2) -> (4, 4)
        # right: (space1, space2) -> (5, 4)
        # contraction: space2 (4) vs space1 (5) -> Error

        t1 = Tensor(torch.randn(4, 4), (matmul_ctx.space2, matmul_ctx.space2))
        t2 = Tensor(torch.randn(5, 4), (matmul_ctx.space1, matmul_ctx.space2))

        with pytest.raises(ValueError):
            matmul(t1, t2)

    def test_matmul_rejects_scalar(self, matmul_ctx):
        t1 = Tensor(torch.tensor(1.0), ())
        t2 = Tensor(torch.randn(matmul_ctx.space2.dim), (matmul_ctx.space2,))

        with pytest.raises(ValueError):
            matmul(t1, t2)

    def test_singleton_vector_matmul(self):
        mode_one = make_mode("one", 1)
        structure = OrderedDict([(mode_one, 0)])
        space_one = HilbertSpace(structure=structure)

        data_left = torch.randn(space_one.dim)
        data_right = torch.randn(space_one.dim)
        tensor_left = Tensor(data=data_left, dims=(space_one,))
        tensor_right = Tensor(data=data_right, dims=(space_one,))

        result = matmul(tensor_left, tensor_right)
        expected_data = torch.matmul(data_left, data_right)

        assert result.dims == tuple()
        assert torch.allclose(result.data, expected_data)

    def test_matmul_dtype_promotion(self, matmul_ctx):
        # Test float @ complex -> complex

        # Left: float (space2, space1)
        data_left = torch.randn(
            matmul_ctx.space2.dim, matmul_ctx.space1.dim, dtype=torch.float32
        )
        t_left = Tensor(data=data_left, dims=(matmul_ctx.space2, matmul_ctx.space1))

        # Right: complex (space1, space2)
        data_right = torch.randn(
            matmul_ctx.space1.dim, matmul_ctx.space2.dim, dtype=torch.complex64
        )
        t_right = Tensor(data=data_right, dims=(matmul_ctx.space1, matmul_ctx.space2))

        result = matmul(t_left, t_right)

        assert result.data.dtype == torch.complex64
        expected = torch.matmul(data_left.to(torch.complex64), data_right)
        assert torch.allclose(result.data, expected)


@pytest.fixture
def tensor_add_ctx():
    class Context:
        def __init__(self):
            self.mode_a = make_mode("a", 2)
            self.mode_b = make_mode("b", 3)
            self.mode_c = make_mode("c", 2)
            self.mode_d = make_mode("d", 4)
            self.mode_m = make_mode("m", 1)

            self.space_a = _space_from_modes(self.mode_a)
            self.space_b = _space_from_modes(self.mode_b)
            self.space_c = _space_from_modes(self.mode_c)
            self.space_d = _space_from_modes(self.mode_d)
            self.space_m = _space_from_modes(self.mode_m)

            self.space_ab = _space_from_modes(self.mode_a, self.mode_b)
            self.space_ba = _space_from_modes(self.mode_b, self.mode_a)

    return Context()


class TestTensorAdd:
    def test_add_union_disjoint_axis(self, tensor_add_ctx):
        left_data = torch.randn(tensor_add_ctx.space_a.dim, tensor_add_ctx.space_b.dim)
        right_data = torch.randn(tensor_add_ctx.space_d.dim, tensor_add_ctx.space_b.dim)
        left = Tensor(
            data=left_data, dims=(tensor_add_ctx.space_a, tensor_add_ctx.space_b)
        )
        right = Tensor(
            data=right_data, dims=(tensor_add_ctx.space_d, tensor_add_ctx.space_b)
        )

        result = left + right
        expected_dims = (
            tensor_add_ctx.space_a + tensor_add_ctx.space_d,
            tensor_add_ctx.space_b,
        )

        assert result.dims == expected_dims
        assert torch.allclose(result.data[: tensor_add_ctx.space_a.dim, :], left_data)
        assert torch.allclose(
            result.data[
                tensor_add_ctx.space_a.dim : tensor_add_ctx.space_a.dim
                + tensor_add_ctx.space_d.dim,
                :,
            ],
            right_data,
        )

    def test_add_reorders_right_by_state_space(self, tensor_add_ctx):
        left_data = torch.zeros(tensor_add_ctx.space_ab.dim)
        right_data = torch.arange(tensor_add_ctx.space_ba.dim, dtype=left_data.dtype)
        left = Tensor(data=left_data, dims=(tensor_add_ctx.space_ab,))
        right = Tensor(data=right_data, dims=(tensor_add_ctx.space_ba,))

        result = left + right
        perm = torch.tensor(
            state_space.permutation_order(
                tensor_add_ctx.space_ba, tensor_add_ctx.space_ab
            ),
            dtype=torch.long,
        )
        expected = left_data + right_data.index_select(0, perm)

        assert result.dims == (tensor_add_ctx.space_ab,)
        assert torch.allclose(result.data, expected)

    def test_add_union_multiple_axes(self, tensor_add_ctx):
        left_data = torch.zeros(
            tensor_add_ctx.space_a.dim,
            tensor_add_ctx.space_b.dim,
            tensor_add_ctx.space_c.dim,
        )
        right_data = torch.randn(
            tensor_add_ctx.space_d.dim,
            tensor_add_ctx.space_b.dim,
            tensor_add_ctx.space_m.dim,
        )
        left = Tensor(
            data=left_data,
            dims=(
                tensor_add_ctx.space_a,
                tensor_add_ctx.space_b,
                tensor_add_ctx.space_c,
            ),
        )
        right = Tensor(
            data=right_data,
            dims=(
                tensor_add_ctx.space_d,
                tensor_add_ctx.space_b,
                tensor_add_ctx.space_m,
            ),
        )

        result = left + right
        expected_dims = (
            tensor_add_ctx.space_a + tensor_add_ctx.space_d,
            tensor_add_ctx.space_b,
            tensor_add_ctx.space_c + tensor_add_ctx.space_m,
        )

        assert result.dims == expected_dims
        a_slice = slice(
            tensor_add_ctx.space_a.dim,
            tensor_add_ctx.space_a.dim + tensor_add_ctx.space_d.dim,
        )
        c_slice = slice(
            tensor_add_ctx.space_c.dim,
            tensor_add_ctx.space_c.dim + tensor_add_ctx.space_m.dim,
        )
        assert torch.allclose(result.data[a_slice, :, c_slice], right_data)

    def test_add_accumulates_overlap(self, tensor_add_ctx):
        left_data = torch.ones(tensor_add_ctx.space_ab.dim)
        right_data = torch.ones(tensor_add_ctx.space_ab.dim)
        left = Tensor(data=left_data, dims=(tensor_add_ctx.space_ab,))
        right = Tensor(data=right_data, dims=(tensor_add_ctx.space_ab,))

        result = left + right

        assert result.dims == (tensor_add_ctx.space_ab,)
        assert torch.allclose(result.data, torch.full_like(left_data, 2.0))

    def test_add_rank_mismatch_broadcast(self, tensor_add_ctx):
        # Test adding tensors of different ranks (3-tensor + 4-tensor)
        # T3: (A, B, C)
        # T4: (D, A, B, C)
        # T3 should be broadcasted to (Broadcast, A, B, C) and then expanded to (D, A, B, C)

        t3_data = torch.ones(
            tensor_add_ctx.space_a.dim,
            tensor_add_ctx.space_b.dim,
            tensor_add_ctx.space_c.dim,
        )
        t3 = Tensor(
            data=t3_data,
            dims=(
                tensor_add_ctx.space_a,
                tensor_add_ctx.space_b,
                tensor_add_ctx.space_c,
            ),
        )

        t4_data = torch.ones(
            tensor_add_ctx.space_d.dim,
            tensor_add_ctx.space_a.dim,
            tensor_add_ctx.space_b.dim,
            tensor_add_ctx.space_c.dim,
        )
        t4 = Tensor(
            data=t4_data,
            dims=(
                tensor_add_ctx.space_d,
                tensor_add_ctx.space_a,
                tensor_add_ctx.space_b,
                tensor_add_ctx.space_c,
            ),
        )

        result = t3 + t4

        # Expected: T3 expanded to match T4 shape (dim 0 broadcasted) -> all 1s
        # T4 is all 1s
        # Result should be all 2s
        expected_dims = (
            tensor_add_ctx.space_d,
            tensor_add_ctx.space_a,
            tensor_add_ctx.space_b,
            tensor_add_ctx.space_c,
        )
        assert result.dims == expected_dims
        assert torch.allclose(result.data, torch.full_like(result.data, 2.0))

    def test_add_broadcastspace_unsqueeze_both_operands(self, tensor_add_ctx):
        # Regression: both operands carry the same BroadcastSpace axis via unsqueeze.
        # This path should still add elementwise on the singleton axis.
        left_base = Tensor(
            data=torch.arange(tensor_add_ctx.space_a.dim, dtype=torch.float32),
            dims=(tensor_add_ctx.space_a,),
        )
        right_base = Tensor(
            data=torch.full((tensor_add_ctx.space_a.dim,), 3.0, dtype=torch.float32),
            dims=(tensor_add_ctx.space_a,),
        )

        left = left_base.unsqueeze(0)
        right = right_base.unsqueeze(0)

        result = left + right

        assert len(result.dims) == 2
        assert isinstance(result.dims[0], BroadcastSpace)
        assert result.dims[1] == tensor_add_ctx.space_a
        assert result.data.shape == (1, tensor_add_ctx.space_a.dim)
        assert torch.allclose(result.data, left.data + right.data)

    def test_add_disjoint_matrices(self, tensor_add_ctx):
        # Test adding two matrices with disjoint spaces on both axes
        # M1: (A, B)
        # M2: (C, D)
        # Result: (A+C, B+D)

        m1_data = torch.ones(tensor_add_ctx.space_a.dim, tensor_add_ctx.space_b.dim)
        m1 = Tensor(data=m1_data, dims=(tensor_add_ctx.space_a, tensor_add_ctx.space_b))

        m2_data = torch.full(
            (tensor_add_ctx.space_c.dim, tensor_add_ctx.space_d.dim), 2.0
        )
        m2 = Tensor(data=m2_data, dims=(tensor_add_ctx.space_c, tensor_add_ctx.space_d))

        result = m1 + m2

        # Check dimensions
        expected_dims = (
            tensor_add_ctx.space_a + tensor_add_ctx.space_c,
            tensor_add_ctx.space_b + tensor_add_ctx.space_d,
        )
        assert result.dims == expected_dims

        # Check data structure (Block Diagonal)
        # Top-left (A x B) should be M1
        assert torch.allclose(
            result.data[: tensor_add_ctx.space_a.dim, : tensor_add_ctx.space_b.dim],
            m1_data,
        )

        # Bottom-right (C x D) should be M2
        # C is after A (indices [A.dim : A.dim+C.dim])
        # D is after B (indices [B.dim : B.dim+D.dim])
        assert torch.allclose(
            result.data[tensor_add_ctx.space_a.dim :, tensor_add_ctx.space_b.dim :],
            m2_data,
        )

        # Off-diagonal blocks should be zero
        # Top-right (A x D)
        assert torch.allclose(
            result.data[: tensor_add_ctx.space_a.dim, tensor_add_ctx.space_b.dim :],
            torch.zeros(tensor_add_ctx.space_a.dim, tensor_add_ctx.space_d.dim),
        )
        # Bottom-left (C x B)
        assert torch.allclose(
            result.data[tensor_add_ctx.space_a.dim :, : tensor_add_ctx.space_b.dim],
            torch.zeros(tensor_add_ctx.space_c.dim, tensor_add_ctx.space_b.dim),
        )

    def test_add_dtype_promotion(self, tensor_add_ctx):
        # Test float + complex -> complex promotion

        # Create float tensor on space_a
        float_data = torch.randn(tensor_add_ctx.space_a.dim, dtype=torch.float32)
        t_float = Tensor(data=float_data, dims=(tensor_add_ctx.space_a,))

        # Create complex tensor on space_a
        complex_data = torch.randn(tensor_add_ctx.space_a.dim, dtype=torch.complex64)
        t_complex = Tensor(data=complex_data, dims=(tensor_add_ctx.space_a,))

        # Add
        result = t_float + t_complex

        # Check dtype
        assert result.data.dtype == torch.complex64
        assert torch.allclose(result.data, float_data + complex_data)

        # Test reverse: complex + float
        result2 = t_complex + t_float
        assert result2.data.dtype == torch.complex64
        assert torch.allclose(result2.data, complex_data + float_data)

    def test_sub_dtype_promotion(self, tensor_add_ctx):
        # Test float - complex -> complex promotion

        # Create float tensor on space_a
        float_data = torch.randn(tensor_add_ctx.space_a.dim, dtype=torch.float32)
        t_float = Tensor(data=float_data, dims=(tensor_add_ctx.space_a,))

        # Create complex tensor on space_a
        complex_data = torch.randn(tensor_add_ctx.space_a.dim, dtype=torch.complex64)
        t_complex = Tensor(data=complex_data, dims=(tensor_add_ctx.space_a,))

        # Sub
        result = t_float - t_complex

        # Check dtype
        assert result.data.dtype == torch.complex64
        assert torch.allclose(result.data, float_data - complex_data)

        # Test reverse: complex - float
        result2 = t_complex - t_float
        assert result2.data.dtype == torch.complex64
        assert torch.allclose(result2.data, complex_data - float_data)


@pytest.fixture
def tensor_error_ctx():
    class Context:
        def __init__(self):
            self.mode_a = make_mode("a", 2)
            self.space_a = _space_from_modes(self.mode_a)

            # Create a MomentumSpace for type mismatch tests
            self.space_mom = MomentumSpace(
                structure=OrderedDict([("k1", 0), ("k2", 1)])
            )

    return Context()


class TestTensorErrorConditions:
    def test_add_incompatible_statespace_types(self, tensor_error_ctx):
        # HilbertSpace + MomentumSpace should fail
        t1 = Tensor(torch.randn(2), dims=(tensor_error_ctx.space_a,))
        t2 = Tensor(torch.randn(2), dims=(tensor_error_ctx.space_mom,))

        with pytest.raises(ValueError):
            t1 + t2

    def test_align_incompatible_types(self, tensor_error_ctx):
        t1 = Tensor(torch.randn(2), dims=(tensor_error_ctx.space_a,))
        with pytest.raises(ValueError):
            t1.align(0, tensor_error_ctx.space_mom)

    def test_align_different_span(self, tensor_error_ctx):
        mode_b = make_mode("b", 2)
        space_b = _space_from_modes(mode_b)

        t1 = Tensor(torch.randn(2), dims=(tensor_error_ctx.space_a,))
        # space_a and space_b have different keys -> different span
        with pytest.raises(ValueError):
            t1.align(0, space_b)

    def test_align_negative_dim(self, tensor_error_ctx):
        t1 = Tensor(
            torch.randn(2, 2),
            dims=(tensor_error_ctx.space_a, tensor_error_ctx.space_a),
        )
        aligned = t1.align(-1, tensor_error_ctx.space_a)
        assert aligned.dims == t1.dims
        assert aligned.data.shape == t1.data.shape

    def test_align_fast_path_returns_same_tensor_when_dim_matches(
        self, tensor_error_ctx
    ):
        t1 = Tensor(torch.randn(2), dims=(tensor_error_ctx.space_a,))
        aligned = t1.align(0, tensor_error_ctx.space_a)
        assert aligned is t1

    def test_permute_invalid_length(self, tensor_error_ctx):
        t1 = Tensor(
            torch.randn(2, 2), dims=(tensor_error_ctx.space_a, tensor_error_ctx.space_a)
        )
        with pytest.raises(ValueError):
            t1.permute(0)  # Need 2 indices

    def test_squeeze_invalid_dim(self, tensor_error_ctx):
        # Squeezing a non-broadcast dimension should return the tensor unchanged if we're not broadcasting
        # Implementation details: if not isinstance(tensor.dims[dim], hilbert.BroadcastSpace): return tensor
        t1 = Tensor(torch.randn(2), dims=(tensor_error_ctx.space_a,))
        t2 = t1.squeeze(0)
        assert t1 is t2

    def test_matmul_rank_zero(self, tensor_error_ctx):
        t1 = Tensor(torch.tensor(1.0), dims=())
        t2 = Tensor(torch.randn(2), dims=(tensor_error_ctx.space_a,))
        with pytest.raises(ValueError):
            matmul(t1, t2)


@pytest.fixture
def tensor_ops_ctx():
    class Context:
        def __init__(self):
            self.mode_a = make_mode("a", 2)
            self.space_a = _space_from_modes(self.mode_a)
            self.data = torch.randn(2)
            self.tensor = Tensor(self.data, (self.space_a,))

    return Context()


class TestTensorOperations:
    def test_zeros_helper(self, tensor_ops_ctx):
        dims = (tensor_ops_ctx.space_a, tensor_ops_ctx.space_a)
        out = zeros(dims)
        assert isinstance(out, Tensor)
        assert out.dims == dims
        assert out.data.shape == (2, 2)
        assert torch.equal(out.data, torch.zeros(2, 2))

    def test_ones_helper(self, tensor_ops_ctx):
        dims = (tensor_ops_ctx.space_a, tensor_ops_ctx.space_a)
        out = ones(dims)
        assert isinstance(out, Tensor)
        assert out.dims == dims
        assert out.data.shape == (2, 2)
        assert torch.equal(out.data, torch.ones(2, 2))

    def test_neg(self, tensor_ops_ctx):
        res = -tensor_ops_ctx.tensor
        assert torch.allclose(res.data, -tensor_ops_ctx.data)
        assert res.dims == tensor_ops_ctx.tensor.dims

    def test_sub(self, tensor_ops_ctx):
        t2 = Tensor(torch.randn(2), (tensor_ops_ctx.space_a,))
        res = tensor_ops_ctx.tensor - t2
        assert torch.allclose(res.data, tensor_ops_ctx.data - t2.data)

    def test_conj(self, tensor_ops_ctx):
        c_data = torch.randn(2, dtype=torch.complex64)
        c_tensor = Tensor(c_data, (tensor_ops_ctx.space_a,))
        res = c_tensor.conj()
        assert torch.allclose(res.data, c_data.conj())

    def test_transpose(self, tensor_ops_ctx):
        t = Tensor(torch.randn(2, 2), (tensor_ops_ctx.space_a, tensor_ops_ctx.space_a))
        res = t.transpose(0, 1)
        assert torch.allclose(res.data, t.data.transpose(0, 1))

    def test_unsupported_operators(self, tensor_ops_ctx):
        # Verify that unimplemented operators raise NotImplementedError
        t = tensor_ops_ctx.tensor

        with pytest.raises(NotImplementedError):
            _ = t * t

        with pytest.raises(NotImplementedError):
            _ = t / t

        with pytest.raises(NotImplementedError):
            _ = t**2

    def test_permute_variants(self, tensor_ops_ctx):
        # Setup a tensor with rank 3 to test permutation
        dims = (tensor_ops_ctx.space_a, tensor_ops_ctx.space_a, tensor_ops_ctx.space_a)
        data = torch.randn(2, 2, 2)
        t = Tensor(data, dims)

        # Test 1: Permute with unpacked arguments
        res1 = t.permute(2, 0, 1)
        assert torch.allclose(res1.data, data.permute(2, 0, 1))
        assert res1.dims == (dims[2], dims[0], dims[1])

        # Test 2: Permute with tuple argument
        res2 = t.permute((1, 2, 0))
        assert torch.allclose(res2.data, data.permute(1, 2, 0))
        assert res2.dims == (dims[1], dims[2], dims[0])

        # Test 3: Permute with list argument
        res3 = t.permute([0, 2, 1])
        assert torch.allclose(res3.data, data.permute(0, 2, 1))
        assert res3.dims == (dims[0], dims[2], dims[1])

    def test_expand_to_union_explicit(self, tensor_ops_ctx):
        # Test expand_to_union without going through add

        # Create a tensor with (BroadcastSpace, SpaceA)
        # Using unsqueeze to create a valid tensor with BroadcastSpace
        t_orig = tensor_ops_ctx.tensor  # (SpaceA,) size 2
        t_broad = t_orig.unsqueeze(0)  # (BroadcastSpace, SpaceA) - data shape (1, 2)

        # Target union: (SpaceA, SpaceA) - sizes (2, 2)
        union_dims = [tensor_ops_ctx.space_a, tensor_ops_ctx.space_a]

        t_expanded = t_broad.expand_to_union(union_dims)

        assert t_expanded.dims == tuple(union_dims)
        assert t_expanded.data.shape == (2, 2)

        # Verify data content (expanded along dim 0)
        expected_data = t_orig.data.expand(2, 2)
        assert torch.allclose(t_expanded.data, expected_data)

    def test_squeeze_unsqueeze_negative_indices(self, tensor_ops_ctx):
        # Tensor shape (2,)
        t = tensor_ops_ctx.tensor

        # Unsqueeze at -1 -> (2, 1)
        t_unsq = t.unsqueeze(-1)
        assert t_unsq.rank() == 2
        assert isinstance(t_unsq.dims[1], BroadcastSpace)
        assert t_unsq.data.shape == (2, 1)

        # Squeeze at -1 -> (2,)
        t_sq = t_unsq.squeeze(-1)
        assert t_sq.rank() == 1
        assert t_sq.dims == t.dims
        assert t_sq.data.shape == (2,)

    def test_clone_detach_semantics(self, tensor_ops_ctx):
        # Test that clone and detach preserve custom attributes (dims)
        t = tensor_ops_ctx.tensor

        # Clone
        t_clone = t.clone()
        assert t_clone.dims == t.dims
        assert t_clone is not t
        assert torch.allclose(t_clone.data, t.data)

        # Detach
        t_detach = t.detach()
        assert t_detach.dims == t.dims
        assert t_detach is not t
        assert not t_detach.requires_grad

        # Ensure underlying storage sharing for detach
        # (Modifying detached data should affect original if they share storage, usually)
        # Note: In PyTorch, detached tensor shares storage.
        # But here Tensor is a wrapper. t_detach.data should share storage with t.data
        t.data[0] += 1.0
        assert t_detach.data[0] == t.data[0]


class TestTensorScaler:
    @pytest.fixture
    def scaler_ctx(self):
        class Context:
            def __init__(self):
                self.mode_a = make_mode("a", 2)
                self.space_a = _space_from_modes(self.mode_a)
                # Rank-2 tensor (square matrix for identity ops)
                self.data_sq = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                self.tensor_sq = Tensor(
                    data=self.data_sq, dims=(self.space_a, self.space_a)
                )

                # Rank-1 tensor
                self.data_vec = torch.tensor([1.0, 2.0])
                self.tensor_vec = Tensor(data=self.data_vec, dims=(self.space_a,))

        return Context()

    def test_mul_scalar_tensor(self, scaler_ctx):
        scalar = 2.5
        result = scalar * scaler_ctx.tensor_sq
        expected_data = scalar * scaler_ctx.data_sq
        assert torch.allclose(result.data, expected_data)
        assert result.dims == scaler_ctx.tensor_sq.dims

    def test_mul_tensor_scalar(self, scaler_ctx):
        scalar = 2.5
        result = scaler_ctx.tensor_sq * scalar
        expected_data = scaler_ctx.data_sq * scalar
        assert torch.allclose(result.data, expected_data)
        assert result.dims == scaler_ctx.tensor_sq.dims

    def test_add_scalar_tensor(self, scaler_ctx):
        # c + T -> c*I + T
        scalar = 2.0
        result = scalar + scaler_ctx.tensor_sq
        eye = torch.eye(2)
        expected_data = scalar * eye + scaler_ctx.data_sq
        assert torch.allclose(result.data, expected_data)
        assert result.dims == scaler_ctx.tensor_sq.dims

    def test_add_tensor_scalar(self, scaler_ctx):
        # T + c -> T + c*I
        scalar = 2.0
        result = scaler_ctx.tensor_sq + scalar
        eye = torch.eye(2)
        expected_data = scaler_ctx.data_sq + scalar * eye
        assert torch.allclose(result.data, expected_data)
        assert result.dims == scaler_ctx.tensor_sq.dims

    def test_sub_scalar_tensor(self, scaler_ctx):
        # c - T -> c*I - T
        scalar = 2.0
        result = scalar - scaler_ctx.tensor_sq
        eye = torch.eye(2)
        expected_data = scalar * eye - scaler_ctx.data_sq
        assert torch.allclose(result.data, expected_data)
        assert result.dims == scaler_ctx.tensor_sq.dims

    def test_sub_tensor_scalar(self, scaler_ctx):
        # T - c -> T - c*I
        scalar = 2.0
        result = scaler_ctx.tensor_sq - scalar
        eye = torch.eye(2)
        expected_data = scaler_ctx.data_sq - scalar * eye
        assert torch.allclose(result.data, expected_data)
        assert result.dims == scaler_ctx.tensor_sq.dims

    def test_truediv_tensor_scalar(self, scaler_ctx):
        scalar = 2.0
        result = scaler_ctx.tensor_sq / scalar
        expected_data = scaler_ctx.data_sq / scalar
        assert torch.allclose(result.data, expected_data)
        assert result.dims == scaler_ctx.tensor_sq.dims

    def test_add_scalar_rank1_fails(self, scaler_ctx):
        # Scalar addition requires at least rank 2 for diagonal broadcasting
        with pytest.raises(ValueError, match="rank 2"):
            _ = 1.0 + scaler_ctx.tensor_vec

    def test_sub_scalar_rank1_fails(self, scaler_ctx):
        with pytest.raises(ValueError, match="rank 2"):
            _ = scaler_ctx.tensor_vec - 1.0

    def test_add_scalar_rank3(self, scaler_ctx):
        dims = (scaler_ctx.space_a, scaler_ctx.space_a, scaler_ctx.space_a)
        data = torch.ones(2, 2, 2)
        tensor = Tensor(data, dims)

        scalar = 5.0
        result = tensor + scalar

        # Expected: T + c*I
        # I is identity on last 2 dims (2, 2) broadcasted to (2, 2, 2)
        eye = torch.eye(2).expand(2, 2, 2)
        expected_data = data + scalar * eye

        assert result.dims == dims
        assert torch.allclose(result.data, expected_data)

    def test_div_dtype_promotion(self, scaler_ctx):
        # Test float / complex -> complex
        t_float = scaler_ctx.tensor_sq
        scalar_complex = 2.0 + 1.0j

        result = t_float / scalar_complex
        assert result.data.is_complex()
        assert torch.allclose(result.data, scaler_ctx.data_sq / scalar_complex)

        # Test complex / float -> complex
        t_complex = Tensor(
            data=scaler_ctx.data_sq.to(torch.complex64), dims=scaler_ctx.tensor_sq.dims
        )
        scalar_float = 2.0

        result2 = t_complex / scalar_float
        assert result2.data.is_complex()
        assert torch.allclose(result2.data, t_complex.data / scalar_float)

    def test_scalar_mixed_type_ops(self, scaler_ctx):
        # Test mixed type operations for scalar-tensor arithmetic

        t_float = scaler_ctx.tensor_sq
        t_complex = Tensor(
            data=scaler_ctx.data_sq.to(torch.complex64), dims=scaler_ctx.tensor_sq.dims
        )
        scalar_float = 2.0
        scalar_complex = 2.0 + 1.0j
        eye = torch.eye(2)

        # 1. Complex Tensor - Float Scalar
        res1 = t_complex - scalar_float
        assert res1.data.dtype == torch.complex64
        assert torch.allclose(res1.data, t_complex.data - scalar_float * eye)

        # 2. Float Tensor - Complex Scalar
        res2 = t_float - scalar_complex
        assert res2.data.dtype == torch.complex64
        assert torch.allclose(res2.data, t_float.data - scalar_complex * eye)

        # 3. Complex Scalar - Float Tensor
        res3 = scalar_complex - t_float
        assert res3.data.dtype == torch.complex64
        # Note: scalar - tensor broadcasts scalar to diagonal
        expected = scalar_complex * eye - t_float.data
        assert torch.allclose(res3.data, expected)


def _simple_hilbert(tag: str, size: int, make_irrep=None) -> HilbertSpace:
    if make_irrep is None:

        def make_irrep(n):
            return (tag, n)

    basis = tuple(
        U1Basis(coef=sy.Integer(1), base=(make_irrep(n),)) for n in range(size)
    )
    return hilbert(basis)


def test_factorize_dim_then_product_dims_roundtrip_hilbert():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 3)

    factorizable = hilbert(
        U1Basis(coef=sy.Integer(1), base=(i, j))
        for i in (0, 1)
        for j in ("a", "b", "c")
    )

    data = torch.arange(
        left.dim * factorizable.dim * right.dim, dtype=torch.float64
    ).reshape(left.dim, factorizable.dim, right.dim)
    tensor = Tensor(data=data, dims=(left, factorizable, right))

    rule = factorizable.factorize((int,), (str,))
    factorized = tensor.factorize_dim(1, rule)
    restored = factorized.product_dims((1, 2))

    assert restored.dims == tensor.dims
    assert torch.equal(restored.data, tensor.data)


def test_product_dims_non_sequential_groups_hilbert():
    irrep_makers = (
        lambda n: n,  # int
        lambda n: f"s{n}",  # str
        lambda n: float(n),  # float
        lambda n: complex(n, 0),  # complex
        lambda n: (n,),  # tuple
        lambda n: bytes([n]),  # bytes
    )
    spaces = tuple(
        _simple_hilbert(f"s{n}", size, irrep_makers[n])
        for n, size in enumerate((2, 3, 4, 2, 5, 3))
    )
    data = torch.arange(
        spaces[0].dim
        * spaces[1].dim
        * spaces[2].dim
        * spaces[3].dim
        * spaces[4].dim
        * spaces[5].dim,
        dtype=torch.float64,
    ).reshape(*(space.dim for space in spaces))
    tensor = Tensor(data=data, dims=spaces)

    out = tensor.product_dims((1, 4), (5, 2))

    expected_dims = (spaces[0], spaces[1] @ spaces[4], spaces[5] @ spaces[2], spaces[3])
    expected_data = data.permute(0, 1, 4, 5, 2, 3).reshape(
        spaces[0].dim,
        spaces[1].dim * spaces[4].dim,
        spaces[5].dim * spaces[2].dim,
        spaces[3].dim,
    )

    assert out.dims == expected_dims
    assert torch.equal(out.data, expected_data)


def test_tensor_mean_reduces_selected_dim():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 3)
    right = _simple_hilbert("right", 4)
    data = torch.randn(left.dim, mid.dim, right.dim, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(left, mid, right))

    out = tensor.mean(1)

    assert out.dims == (left, right)
    assert torch.allclose(out.data, data.mean(dim=1))


def test_tensor_mean_supports_negative_dim():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 3)
    right = _simple_hilbert("right", 4)
    data = torch.randn(left.dim, mid.dim, right.dim, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(left, mid, right))

    out = tensor.mean(-1)

    assert out.dims == (left, mid)
    assert torch.allclose(out.data, data.mean(dim=-1))


def test_tensor_mean_raises_for_out_of_range_dim():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 4)
    tensor = Tensor(
        data=torch.randn(left.dim, right.dim, dtype=torch.float64), dims=(left, right)
    )

    with pytest.raises(IndexError, match="out of range"):
        _ = tensor.mean(2)


def test_tensor_mean_supports_dim_none():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 4)
    data = torch.randn(left.dim, right.dim, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(left, right))

    out = tensor.mean()

    assert out.dims == ()
    assert torch.allclose(out.data, data.mean())


def test_tensor_mean_supports_tuple_dims():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 3)
    right = _simple_hilbert("right", 4)
    data = torch.randn(left.dim, mid.dim, right.dim, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(left, mid, right))

    out = tensor.mean((0, 2))

    assert out.dims == (mid,)
    assert torch.allclose(out.data, data.mean(dim=(0, 2)))


def test_tensor_argmax_reduces_selected_dim():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 3)
    right = _simple_hilbert("right", 4)
    data = torch.randn(left.dim, mid.dim, right.dim, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(left, mid, right))

    out = tensor.argmax(1)

    assert out.dims == (left, right)
    assert torch.equal(out.data, data.argmax(dim=1))


def test_tensor_argmax_supports_negative_dim():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 3)
    right = _simple_hilbert("right", 4)
    data = torch.randn(left.dim, mid.dim, right.dim, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(left, mid, right))

    out = tensor.argmax(-1)

    assert out.dims == (left, mid)
    assert torch.equal(out.data, data.argmax(dim=-1))


def test_tensor_argmax_raises_for_out_of_range_dim():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 4)
    tensor = Tensor(
        data=torch.randn(left.dim, right.dim, dtype=torch.float64), dims=(left, right)
    )

    with pytest.raises(IndexError, match="out of range"):
        _ = tensor.argmax(2)


def test_tensor_argmin_reduces_selected_dim():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 3)
    right = _simple_hilbert("right", 4)
    data = torch.randn(left.dim, mid.dim, right.dim, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(left, mid, right))

    out = tensor.argmin(1)

    assert out.dims == (left, right)
    assert torch.equal(out.data, data.argmin(dim=1))


def test_tensor_argmin_supports_negative_dim():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 3)
    right = _simple_hilbert("right", 4)
    data = torch.randn(left.dim, mid.dim, right.dim, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(left, mid, right))

    out = tensor.argmin(-1)

    assert out.dims == (left, mid)
    assert torch.equal(out.data, data.argmin(dim=-1))


def test_tensor_argmin_raises_for_out_of_range_dim():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 4)
    tensor = Tensor(
        data=torch.randn(left.dim, right.dim, dtype=torch.float64), dims=(left, right)
    )

    with pytest.raises(IndexError, match="out of range"):
        _ = tensor.argmin(2)


def test_one_hot_appends_class_dim():
    sample_space = _simple_hilbert("sample", 4)
    class_space = _simple_hilbert("class", 5)
    data = torch.tensor([0, 2, 4, 1], dtype=torch.long)
    tensor = Tensor(data=data, dims=(sample_space,))

    out = one_hot(tensor, class_space)

    assert out.dims == (sample_space, class_space)
    expected = torch.nn.functional.one_hot(data, num_classes=class_space.dim)
    assert torch.equal(out.data, expected)


def test_one_hot_rejects_non_integer_input():
    sample_space = _simple_hilbert("sample", 3)
    class_space = _simple_hilbert("class", 4)
    tensor = Tensor(
        data=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64), dims=(sample_space,)
    )

    with pytest.raises(TypeError, match="integer-valued"):
        _ = one_hot(tensor, class_space)


def test_one_hot_rejects_out_of_range_indices():
    sample_space = _simple_hilbert("sample", 3)
    class_space = _simple_hilbert("class", 4)
    tensor = Tensor(
        data=torch.tensor([0, 1, 4], dtype=torch.long), dims=(sample_space,)
    )

    with pytest.raises(ValueError, match="out of range"):
        _ = one_hot(tensor, class_space)


def test_kernel_tensor_builds_rank2_tensor_from_kernel():
    left = _simple_hilbert("left", 3)
    right = _simple_hilbert("right", 2)

    out = kernel_tensor(lambda x, y: x.rep[0][1] - 10 * y.rep[0][1], (left, right))

    expected = torch.tensor(
        [[0, -10], [1, -9], [2, -8]],
        dtype=out.data.dtype,
    )
    assert out.dims == (left, right)
    assert out.data.shape == (left.dim, right.dim)
    assert torch.equal(out.data, expected)


def test_kernel_tensor_builds_rank3_tensor_from_kernel():
    a = _simple_hilbert("a", 2)
    b = _simple_hilbert("b", 3)
    c = _simple_hilbert("c", 2)

    out = kernel_tensor(
        lambda x, y, z: x.rep[0][1] + 2 * y.rep[0][1] + 3 * z.rep[0][1],
        (a, b, c),
    )

    expected = torch.empty((a.dim, b.dim, c.dim), dtype=out.data.dtype)
    for i, x in enumerate(a.elements()):
        for j, y in enumerate(b.elements()):
            for k, z in enumerate(c.elements()):
                expected[i, j, k] = x.rep[0][1] + 2 * y.rep[0][1] + 3 * z.rep[0][1]

    assert out.dims == (a, b, c)
    assert torch.equal(out.data, expected)


def test_kernel_tensor_supports_scalar_kernel():
    out = kernel_tensor(lambda: 2 + 3j, ())

    assert out.dims == ()
    assert out.data.shape == torch.Size([])
    assert out.item() == 2 + 3j


def test_allclose_aligns_right_dims():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    space_ab = _space_from_modes(mode_a, mode_b)
    space_ba = _space_from_modes(mode_b, mode_a)

    a_data = torch.randn(space_ab.dim, dtype=torch.float64)
    b_data = torch.empty_like(a_data)
    perm = torch.tensor([3, 4, 0, 1, 2], dtype=torch.long)
    b_data[perm] = a_data

    a = Tensor(data=a_data, dims=(space_ab,))
    b = Tensor(data=b_data, dims=(space_ba,))

    assert allclose(a, b)
    assert a.allclose(b)


def test_allclose_returns_false_for_non_alignable_dims():
    left = _simple_hilbert("left", 3)
    right = _simple_hilbert("right", 3)
    a = Tensor(data=torch.randn(left.dim), dims=(left,))
    b = Tensor(data=torch.randn(right.dim), dims=(right,))

    assert not allclose(a, b)
    assert not a.allclose(b)


def test_allclose_matches_torch_behavior_for_dtype_mismatch():
    left = _simple_hilbert("left", 3)
    a = Tensor(data=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32), dims=(left,))
    b = Tensor(data=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), dims=(left,))

    with pytest.raises(RuntimeError, match="did not match"):
        _ = torch.allclose(a.data, b.data)
    with pytest.raises(RuntimeError, match="did not match"):
        _ = allclose(a, b)
    with pytest.raises(RuntimeError, match="did not match"):
        _ = a.allclose(b)


def test_align_all_aligns_dims():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    space_ab = _space_from_modes(mode_a, mode_b)
    space_ba = _space_from_modes(mode_b, mode_a)

    data = torch.arange(space_ab.dim, dtype=torch.float64)
    # data in BA order: [b0,b1,b2,a0,a1]
    data_ba = data[torch.tensor([2, 3, 4, 0, 1], dtype=torch.long)]
    tensor_ba = Tensor(data=data_ba, dims=(space_ba,))

    out = align_all(tensor_ba, (space_ab,))

    assert out.dims == (space_ab,)
    assert torch.equal(out.data, data)
    assert torch.equal(tensor_ba.align_all((space_ab,)).data, data)


def test_align_all_raises_when_not_alignable():
    left = _simple_hilbert("left", 3)
    right = _simple_hilbert("right", 3)
    tensor = Tensor(data=torch.randn(right.dim), dims=(right,))

    with pytest.raises(ValueError, match="cannot be aligned|Cannot align"):
        _ = align_all(tensor, (left,))


def test_equal_aligns_right_dims():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    space_ab = _space_from_modes(mode_a, mode_b)
    space_ba = _space_from_modes(mode_b, mode_a)

    a_data = torch.randn(space_ab.dim, dtype=torch.float64)
    b_data = torch.empty_like(a_data)
    perm = torch.tensor([3, 4, 0, 1, 2], dtype=torch.long)
    b_data[perm] = a_data

    a = Tensor(data=a_data, dims=(space_ab,))
    b = Tensor(data=b_data, dims=(space_ba,))

    assert equal(a, b)
    assert a.equal(b)


def test_equal_returns_false_for_non_alignable_dims():
    left = _simple_hilbert("left", 3)
    right = _simple_hilbert("right", 3)
    a = Tensor(data=torch.randn(left.dim), dims=(left,))
    b = Tensor(data=torch.randn(right.dim), dims=(right,))

    assert not equal(a, b)
    assert not a.equal(b)


def test_equal_returns_false_when_values_differ():
    left = _simple_hilbert("left", 3)
    a = Tensor(data=torch.tensor([1.0, 2.0, 3.0]), dims=(left,))
    b = Tensor(data=torch.tensor([1.0, 2.0, 4.0]), dims=(left,))

    assert not equal(a, b)
    assert not a.equal(b)


def test_where_selects_between_input_and_other_with_alignment():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    space_ab = _space_from_modes(mode_a, mode_b)
    space_ba = _space_from_modes(mode_b, mode_a)

    mask_data = torch.tensor([True, False, True, False, True], dtype=torch.bool)
    input_data_ab = torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0], dtype=torch.float64)
    other_data_ba = torch.tensor(
        [100.0, 101.0, 102.0, 103.0, 104.0], dtype=torch.float64
    )

    condition = Tensor(data=mask_data, dims=(space_ab,))
    input_tensor = Tensor(data=input_data_ab, dims=(space_ab,))
    other_tensor = Tensor(data=other_data_ba, dims=(space_ba,))

    out = where(condition, input_tensor, other_tensor)

    aligned_other = other_tensor.align_all((space_ab,))
    expected = torch.where(mask_data, input_data_ab, aligned_other.data)
    assert out.dims == (space_ab,)
    assert torch.equal(out.data, expected)


def test_where_condition_only_returns_index_tensors():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    space_a = _space_from_modes(mode_a)
    space_b = _space_from_modes(mode_b)

    mask_data = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    condition = Tensor(data=mask_data, dims=(space_a, space_b))

    out = where(condition)

    expected = torch.where(mask_data)
    assert len(out) == 2
    assert len(expected) == 2
    for actual, exp in zip(out, expected):
        assert actual.dims == (IndexSpace.linear(exp.numel()),)
        assert torch.equal(actual.data, exp)


def test_where_rejects_non_bool_condition():
    mode_a = make_mode("a", 2)
    space_a = _space_from_modes(mode_a)
    condition = Tensor(data=torch.tensor([1, 0], dtype=torch.int64), dims=(space_a,))
    input_tensor = Tensor(data=torch.tensor([1.0, 2.0]), dims=(space_a,))
    other_tensor = Tensor(data=torch.tensor([3.0, 4.0]), dims=(space_a,))

    with pytest.raises(TypeError, match="torch.bool"):
        _ = where(condition, input_tensor, other_tensor)

    with pytest.raises(TypeError, match="torch.bool"):
        _ = where(condition)


def test_nonzero_as_tuple_true_matches_torch():
    a = IndexSpace.linear(2)
    b = IndexSpace.linear(3)
    condition = Tensor(
        data=torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool
        ),
        dims=(a, b),
    )

    out = nonzero(condition, as_tuple=True)
    expected = torch.nonzero(condition.data, as_tuple=True)

    assert len(out) == len(expected)
    for actual, exp in zip(out, expected):
        assert actual.dims == (IndexSpace.linear(exp.numel()),)
        assert torch.equal(actual.data, exp)


def test_tensor_nonzero_as_tuple_true_matches_torch():
    a = IndexSpace.linear(2)
    b = IndexSpace.linear(3)
    condition = Tensor(
        data=torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool
        ),
        dims=(a, b),
    )

    out = condition.nonzero(as_tuple=True)
    expected = torch.nonzero(condition.data, as_tuple=True)

    assert len(out) == len(expected)
    for actual, exp in zip(out, expected):
        assert actual.dims == (IndexSpace.linear(exp.numel()),)
        assert torch.equal(actual.data, exp)


def test_where_uses_symmetric_broadcast_dims():
    a = IndexSpace.linear(2)
    b = IndexSpace.linear(3)

    condition = Tensor(
        data=torch.tensor([[True], [False]], dtype=torch.bool),
        dims=(a, BroadcastSpace()),
    )
    input_tensor = Tensor(
        data=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64),
        dims=(a, b),
    )
    other_tensor = Tensor(
        data=torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float64),
        dims=(BroadcastSpace(), b),
    )

    out = where(condition, input_tensor, other_tensor)

    expected = torch.where(condition.data, input_tensor.data, other_tensor.data)
    assert out.dims == (a, b)
    assert torch.equal(out.data, expected)


def test_where_supports_scalar_condition_broadcast_to_higher_rank_operands():
    a = IndexSpace.linear(2)
    b = IndexSpace.linear(3)

    condition = Tensor(data=torch.tensor(True, dtype=torch.bool), dims=())
    input_tensor = Tensor(
        data=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64),
        dims=(a, b),
    )
    other_tensor = Tensor(
        data=torch.tensor(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=torch.float64
        ),
        dims=(a, b),
    )

    out = where(condition, input_tensor, other_tensor)

    expected = torch.where(condition.data, input_tensor.data, other_tensor.data)
    assert out.dims == (a, b)
    assert torch.equal(out.data, expected)


def test_tensor_where_method_supports_ternary_form():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    space_ab = _space_from_modes(mode_a, mode_b)
    space_ba = _space_from_modes(mode_b, mode_a)

    condition = Tensor(
        data=torch.tensor([True, False, True, False, True], dtype=torch.bool),
        dims=(space_ab,),
    )
    input_tensor = Tensor(
        data=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64),
        dims=(space_ab,),
    )
    other_tensor = Tensor(
        data=torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0], dtype=torch.float64),
        dims=(space_ba,),
    )

    out = condition.where(input_tensor, other_tensor)
    expected = where(condition, input_tensor, other_tensor)

    assert out.dims == expected.dims
    assert torch.equal(out.data, expected.data)


def test_tensor_where_method_supports_condition_only_form():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    space_a = _space_from_modes(mode_a)
    space_b = _space_from_modes(mode_b)

    condition = Tensor(
        data=torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool
        ),
        dims=(space_a, space_b),
    )

    out = condition.where()
    expected = where(condition)

    assert len(out) == len(expected)
    for actual, exp in zip(out, expected):
        assert actual.dims == exp.dims
        assert torch.equal(actual.data, exp.data)


def test_tensor_where_method_rejects_single_argument():
    mode_a = make_mode("a", 2)
    space_a = _space_from_modes(mode_a)
    condition = Tensor(
        data=torch.tensor([True, False], dtype=torch.bool),
        dims=(space_a,),
    )
    input_tensor = Tensor(data=torch.tensor([1.0, 2.0]), dims=(space_a,))

    with pytest.raises(TypeError, match="where\\(\\) or where\\(input, other\\)"):
        _ = condition.where(input=input_tensor)


def test_tensor_item_raises_for_non_scalar():
    left = _simple_hilbert("left", 2)
    tensor = Tensor(data=torch.tensor([1.0, 2.0]), dims=(left,))

    with pytest.raises(ValueError, match="rank-0"):
        _ = tensor.item()


def test_union_dims_prefers_concrete_over_broadcast():
    a = IndexSpace.linear(2)
    b = IndexSpace.linear(3)

    out = union_dims((BroadcastSpace(), b), (a, BroadcastSpace()))

    assert out == (a, b)


def test_union_dims_keeps_left_when_same_rays():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    space_ab = _space_from_modes(mode_a, mode_b)
    space_ba = _space_from_modes(mode_b, mode_a)

    out = union_dims((space_ab,), (space_ba,))

    assert out == (space_ab,)


def test_union_dims_raises_for_incompatible_dims():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 2)

    with pytest.raises(ValueError, match="incompatible"):
        _ = union_dims((left,), (right,))


def test_union_dims_raises_for_rank_mismatch():
    a = IndexSpace.linear(2)
    b = IndexSpace.linear(3)

    with pytest.raises(ValueError, match="same rank"):
        _ = union_dims((a,), (a, b))


def test_union_dims_non_strict_allows_disjoint_spaces():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 2)

    out = union_dims((left,), (right,), allow_merge=True)

    assert out == (left + right,)


def test_equal_matches_torch_behavior_for_dtype_mismatch():
    left = _simple_hilbert("left", 3)
    a = Tensor(data=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32), dims=(left,))
    b = Tensor(data=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), dims=(left,))

    expected = torch.equal(a.data, b.data)
    assert equal(a, b) == expected
    assert a.equal(b) == expected


def test_astype_module_returns_new_tensor_with_converted_dtype():
    left = _simple_hilbert("left", 3)
    tensor = Tensor(
        data=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32), dims=(left,)
    )

    out = astype(tensor, torch.float64)

    assert out is not tensor
    assert out.dims == tensor.dims
    assert out.data.dtype == torch.float64
    assert torch.allclose(out.data, tensor.data.to(dtype=torch.float64))


def test_tensor_astype_returns_new_tensor_with_converted_dtype():
    left = _simple_hilbert("left", 3)
    tensor = Tensor(
        data=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), dims=(left,)
    )

    out = tensor.astype(torch.float32)

    assert out is not tensor
    assert out.dims == tensor.dims
    assert out.data.dtype == torch.float32
    assert torch.allclose(out.data, tensor.data.to(dtype=torch.float32))


def test_all_over_tensor_equality_full_reduction():
    mode_a = make_mode("a", 2)
    mode_b = make_mode("b", 3)
    right = _simple_hilbert("right", 2)
    space_ab = _space_from_modes(mode_a, mode_b)
    space_ba = _space_from_modes(mode_b, mode_a)

    a_data = torch.randn(space_ab.dim, right.dim, dtype=torch.float64)
    b_data = torch.empty_like(a_data)
    perm = torch.tensor([3, 4, 0, 1, 2], dtype=torch.long)
    b_data[perm, :] = a_data

    a = Tensor(data=a_data, dims=(space_ab, right))
    b = Tensor(data=b_data, dims=(space_ba, right))

    compared = a == b
    reduced = tensor_all(compared)

    assert compared.dims == (space_ab, right)
    assert compared.data.dtype == torch.bool
    assert reduced.dims == ()
    assert reduced.data.item() is True


def test_all_over_tensor_equality_dim_and_keepdim():
    left = _simple_hilbert("left", 3)
    right = _simple_hilbert("right", 2)

    a = Tensor(
        data=torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 6.0]], dtype=torch.float64),
        dims=(left, right),
    )
    b = Tensor(
        data=torch.tensor([[1.0, 2.0], [3.0, 0.0], [9.0, 6.0]], dtype=torch.float64),
        dims=(left, right),
    )

    compared = a == b
    reduced = tensor_all(compared, dim=0)
    reduced_keepdim = compared.all(dim=0, keepdim=True)

    assert reduced.dims == (right,)
    assert torch.equal(reduced.data, torch.tensor([True, False]))
    assert isinstance(reduced_keepdim.dims[0], BroadcastSpace)
    assert reduced_keepdim.dims[1] == right
    assert reduced_keepdim.data.shape == (1, right.dim)
    assert torch.equal(reduced_keepdim.data, torch.tensor([[True, False]]))


def test_all_supports_tuple_dims_without_keepdim():
    left = _simple_hilbert("left", 3)
    right = _simple_hilbert("right", 2)
    tensor = Tensor(
        data=torch.tensor([[True, True], [True, False], [True, True]]),
        dims=(left, right),
    )

    out = tensor_all(tensor, dim=(0, 1))

    assert out.dims == ()
    assert out.data.shape == ()
    assert out.data.item() is False


def test_all_supports_tuple_dims_with_keepdim():
    left = _simple_hilbert("left", 3)
    right = _simple_hilbert("right", 2)
    tensor = Tensor(
        data=torch.tensor([[True, True], [True, True], [True, True]]),
        dims=(left, right),
    )

    out = tensor.all(dim=(0, 1), keepdim=True)

    assert len(out.dims) == 2
    assert isinstance(out.dims[0], BroadcastSpace)
    assert isinstance(out.dims[1], BroadcastSpace)
    assert out.data.shape == (1, 1)
    assert out.data.item() is True


def test_all_supports_negative_tuple_dims():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 2)
    right = _simple_hilbert("right", 2)
    tensor = Tensor(
        data=torch.tensor(
            [[[True, True], [True, True]], [[True, True], [True, False]]]
        ),
        dims=(left, mid, right),
    )

    out = tensor.all(dim=(-2, -1))

    assert out.dims == (left,)
    assert torch.equal(out.data, torch.tensor([True, False]))


def test_all_raises_for_out_of_range_dim():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 2)
    tensor = Tensor(
        data=torch.tensor([[True, True], [True, True]]),
        dims=(left, right),
    )

    with pytest.raises(IndexError, match="out of range"):
        _ = tensor.all(dim=2)


def test_all_raises_for_out_of_range_dim_in_tuple():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 2)
    tensor = Tensor(
        data=torch.tensor([[True, True], [True, True]]),
        dims=(left, right),
    )

    with pytest.raises(IndexError, match="out of range"):
        _ = tensor_all(tensor, dim=(0, 2))


def test_all_matches_torch_behavior_dim_none():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 3)
    tensor = Tensor(
        data=torch.tensor([[True, True, True], [True, False, True]]),
        dims=(left, right),
    )

    out = tensor_all(tensor)
    expected = torch.all(tensor.data)

    assert out.dims == ()
    assert torch.equal(out.data, expected)


def test_all_matches_torch_behavior_dim_int_keepdim_false():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 3)
    tensor = Tensor(
        data=torch.tensor([[True, True, True], [True, False, True]]),
        dims=(left, right),
    )

    out = tensor_all(tensor, dim=1, keepdim=False)
    expected = torch.all(tensor.data, dim=1, keepdim=False)

    assert out.dims == (left,)
    assert torch.equal(out.data, expected)


def test_all_matches_torch_behavior_dim_int_keepdim_true():
    left = _simple_hilbert("left", 2)
    right = _simple_hilbert("right", 3)
    tensor = Tensor(
        data=torch.tensor([[True, True, True], [True, False, True]]),
        dims=(left, right),
    )

    out = tensor.all(dim=1, keepdim=True)
    expected = torch.all(tensor.data, dim=1, keepdim=True)

    assert out.data.shape == expected.shape
    assert isinstance(out.dims[1], BroadcastSpace)
    assert torch.equal(out.data, expected)


def test_all_matches_torch_behavior_dim_tuple_keepdim_false():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 2)
    right = _simple_hilbert("right", 2)
    tensor = Tensor(
        data=torch.tensor(
            [[[True, True], [True, True]], [[True, True], [True, False]]]
        ),
        dims=(left, mid, right),
    )

    out = tensor_all(tensor, dim=(1, 2), keepdim=False)
    expected = torch.all(tensor.data, dim=(1, 2), keepdim=False)

    assert out.dims == (left,)
    assert torch.equal(out.data, expected)


def test_all_matches_torch_behavior_dim_tuple_keepdim_true():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 2)
    right = _simple_hilbert("right", 2)
    tensor = Tensor(
        data=torch.tensor(
            [[[True, True], [True, True]], [[True, True], [True, False]]]
        ),
        dims=(left, mid, right),
    )

    out = tensor_all(tensor, dim=(0, 2), keepdim=True)
    expected = torch.all(tensor.data, dim=(0, 2), keepdim=True)

    assert out.data.shape == expected.shape
    assert isinstance(out.dims[0], BroadcastSpace)
    assert isinstance(out.dims[2], BroadcastSpace)
    assert torch.equal(out.data, expected)


def test_all_matches_torch_behavior_negative_tuple_dims():
    left = _simple_hilbert("left", 2)
    mid = _simple_hilbert("mid", 2)
    right = _simple_hilbert("right", 2)
    tensor = Tensor(
        data=torch.tensor(
            [[[True, True], [True, True]], [[True, True], [True, False]]]
        ),
        dims=(left, mid, right),
    )

    out = tensor.all(dim=(-2, -1), keepdim=False)
    expected = torch.all(tensor.data, dim=(-2, -1), keepdim=False)

    assert out.dims == (left,)
    assert torch.equal(out.data, expected)
