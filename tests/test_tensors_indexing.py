import pytest
import torch
from itertools import product
from collections import OrderedDict
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.linalg.tensors import Tensor, where
from qten.symbolics.state_space import (
    BroadcastSpace,
    IndexSpace,
    MomentumSpace,
    brillouin_zone,
)
from qten.symbolics.hilbert_space import U1Basis, HilbertSpace
from qten.geometries.spatials import Lattice
from qten.geometries.boundary import PeriodicBoundary


class TestTensorGetitem:
    @pytest.fixture
    def getitem_ctx(self):
        class Context:
            def __init__(self):
                self.space = IndexSpace.linear(5)
                self.subspace_a = self.space[0:2]
                self.subspace_b = self.space[2:5]
                self.subspace = self.subspace_b

                self.data_mat = torch.arange(25, dtype=torch.float64).reshape(5, 5)
                self.tensor_mat = Tensor(
                    data=self.data_mat, dims=(self.space, self.space)
                )

                self.data_3d = torch.arange(125, dtype=torch.float64).reshape(5, 5, 5)
                self.tensor_3d = Tensor(
                    data=self.data_3d, dims=(self.space, self.space, self.space)
                )

        return Context()

    def test_getitem_normal_returns_tensor(self, getitem_ctx):
        out = getitem_ctx.tensor_mat[1:4, 2:5]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, getitem_ctx.data_mat[1:4, 2:5])
        assert out.dims == (getitem_ctx.space[1:4], getitem_ctx.space[2:5])

    def test_getitem_normal_range_and_none(self, getitem_ctx):
        out = getitem_ctx.tensor_mat[1:5, 0:4]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, getitem_ctx.data_mat[1:5, 0:4])
        assert out.dims == (getitem_ctx.space[1:5], getitem_ctx.space[0:4])

        out2 = getitem_ctx.tensor_mat[None, :, :]
        assert isinstance(out2, Tensor)
        assert out2.data.shape == (1, 5, 5)
        assert out2.dims == (
            BroadcastSpace(),
            getitem_ctx.space,
            getitem_ctx.space,
        )

    def test_getitem_normal_ellipsis_returns_tensor(self, getitem_ctx):
        out = getitem_ctx.tensor_mat[..., :]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, getitem_ctx.data_mat[..., :])
        assert out.dims == (getitem_ctx.space, getitem_ctx.space)

        out2 = getitem_ctx.tensor_mat[..., :, None]
        assert isinstance(out2, Tensor)
        assert torch.equal(out2.data, getitem_ctx.data_mat[..., :, None])
        assert out2.dims == (
            getitem_ctx.space,
            getitem_ctx.space,
            BroadcastSpace(),
        )

    def test_getitem_spatial(self, getitem_ctx):
        out = getitem_ctx.tensor_mat[getitem_ctx.subspace_a, getitem_ctx.subspace_b]
        assert isinstance(out, Tensor)
        assert out.dims == (getitem_ctx.subspace_a, getitem_ctx.subspace_b)
        assert torch.equal(out.data, getitem_ctx.data_mat[0:2, 2:5])

    def test_getitem_statespace(self, getitem_ctx):
        out = getitem_ctx.tensor_mat[getitem_ctx.subspace, getitem_ctx.space]
        assert isinstance(out, Tensor)
        assert out.dims == (getitem_ctx.subspace_b, getitem_ctx.space)
        expected = getitem_ctx.data_mat[2:5, :]
        assert torch.equal(out.data, expected)

    def test_getitem_statespace_int_mix_allowed(self, getitem_ctx):
        out = getitem_ctx.tensor_mat[getitem_ctx.subspace_a, 0]
        assert isinstance(out, Tensor)
        assert out.dims == (getitem_ctx.subspace_a,)
        expected = getitem_ctx.data_mat[0:2, 0]
        assert torch.equal(out.data, expected)

    def test_getitem_3d_hilbert(self, getitem_ctx):
        out = getitem_ctx.tensor_3d[
            getitem_ctx.subspace, getitem_ctx.subspace_a, getitem_ctx.space
        ]
        assert isinstance(out, Tensor)
        assert out.dims == (
            getitem_ctx.subspace_b,
            getitem_ctx.subspace_a,
            getitem_ctx.space,
        )
        expected = getitem_ctx.data_3d[2:5, 0:2, :]
        assert torch.equal(out.data, expected)

    def test_getitem_hilbert_none_inserts_dim(self, getitem_ctx):
        out = getitem_ctx.tensor_mat[
            None, getitem_ctx.subspace_a, getitem_ctx.subspace_b
        ]
        assert isinstance(out, Tensor)
        assert out.dims == (
            BroadcastSpace(),
            getitem_ctx.subspace_a,
            getitem_ctx.subspace_b,
        )
        expected = getitem_ctx.data_mat[0:2, 2:5].unsqueeze(0)
        assert torch.equal(out.data, expected)

    def test_getitem_hilbert_noncontiguous_subspace(self):
        space = IndexSpace.linear(6)
        subspace = IndexSpace(structure=OrderedDict({0: 0, 1: 1, 4: 2, 5: 3}))

        data = torch.arange(36, dtype=torch.float64).reshape(6, 6)
        tensor = Tensor(data=data, dims=(space, space))
        out = tensor[subspace, space]
        expected = data.index_select(0, torch.tensor([0, 1, 4, 5]))

        assert isinstance(out, Tensor)
        assert out.dims == (subspace, space)
        assert torch.equal(out.data, expected)

    def test_getitem_hilbert_invalid_subspace(self, getitem_ctx):
        subspace = IndexSpace(structure=OrderedDict({99: 0}))
        with pytest.raises(IndexError, match="not contained in"):
            _ = getitem_ctx.tensor_mat[subspace, getitem_ctx.space]

    def test_getitem_hilbert_spatial_missing(self, getitem_ctx):
        subspace = IndexSpace(structure=OrderedDict({99: 0}))
        with pytest.raises(IndexError, match="not contained in"):
            _ = getitem_ctx.tensor_mat[subspace, getitem_ctx.space]

    def test_getitem_hilbert_colon_statespace_colon(self, getitem_ctx):
        out = getitem_ctx.tensor_3d[:, getitem_ctx.subspace, :]
        assert isinstance(out, Tensor)
        assert out.dims == (
            getitem_ctx.space,
            getitem_ctx.subspace_b,
            getitem_ctx.space,
        )
        expected = getitem_ctx.data_3d[:, 2:5, :]
        assert torch.equal(out.data, expected)

    def test_getitem_hilbert_ellipsis_colon_statespace_colon(self, getitem_ctx):
        out = getitem_ctx.tensor_3d[..., getitem_ctx.subspace, :]
        assert isinstance(out, Tensor)
        assert out.dims == (
            getitem_ctx.space,
            getitem_ctx.subspace_b,
            getitem_ctx.space,
        )
        expected = getitem_ctx.data_3d[:, 2:5, :]
        assert torch.equal(out.data, expected)

    def test_getitem_hilbert_ellipsis_allowed(self, getitem_ctx):
        out = getitem_ctx.tensor_mat[getitem_ctx.subspace_a, ...]
        assert isinstance(out, Tensor)
        assert out.dims == (getitem_ctx.subspace_a, getitem_ctx.space)
        expected = getitem_ctx.data_mat[0:2, :]
        assert torch.equal(out.data, expected)

    def test_getitem_hilbert_short_key_allowed(self, getitem_ctx):
        out = getitem_ctx.tensor_3d[getitem_ctx.subspace_a]
        assert isinstance(out, Tensor)
        assert out.dims == (
            getitem_ctx.subspace_a,
            getitem_ctx.space,
            getitem_ctx.space,
        )
        expected = getitem_ctx.data_3d[0:2, :, :]
        assert torch.equal(out.data, expected)

    def test_getitem_non_full_slice_on_unsqueezed_axis_keeps_shape_consistent(
        self, getitem_ctx
    ):
        tensor = getitem_ctx.tensor_mat.unsqueeze(0)
        out = tensor[1:]
        expected = getitem_ctx.data_mat.unsqueeze(0)[1:]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert tuple(d.dim for d in out.dims) == tuple(out.data.shape)

    def test_getitem_broadcast_axis_with_singleton_statespace_index_raises(
        self, getitem_ctx
    ):
        tensor = getitem_ctx.tensor_mat.unsqueeze(0)
        singleton = IndexSpace.linear(1)

        with pytest.raises(IndexError, match="Cannot index a BroadcastSpace axis"):
            _ = tensor[singleton]

    def test_getitem_broadcast_axis_with_empty_statespace_index_raises(
        self, getitem_ctx
    ):
        tensor = getitem_ctx.tensor_mat.unsqueeze(0)
        empty = IndexSpace.linear(0)

        with pytest.raises(IndexError, match="Cannot index a BroadcastSpace axis"):
            _ = tensor[empty]

    def test_getitem_broadcast_axis_with_non_singleton_statespace_raises(
        self, getitem_ctx
    ):
        tensor = getitem_ctx.tensor_mat.unsqueeze(0)
        non_singleton = IndexSpace.linear(2)

        with pytest.raises(IndexError, match="Cannot index a BroadcastSpace axis"):
            _ = tensor[non_singleton]

    def test_getitem_with_u1basis_index(self):
        b0 = U1Basis(coef=sy.Integer(0), base=(sy.Integer(0),))
        b1 = U1Basis(coef=sy.Integer(1), base=(sy.Integer(1),))
        space = HilbertSpace.new((b0, b1))
        data = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
        tensor = Tensor(data=data, dims=(space, space, space))

        out = tensor[:, :, b1]
        assert isinstance(out, Tensor)
        assert out.dims == (space, space, HilbertSpace.new((b1,)))
        assert torch.equal(out.data, data[:, :, 1:2])

    def test_getitem_with_momentum_index(self):
        lattice = Lattice(
            basis=ImmutableDenseMatrix([[1]]),
            boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
            unit_cell={"r": ImmutableDenseMatrix([0])},
        )
        momentum_space = brillouin_zone(lattice.dual)
        _k0, k1 = tuple(momentum_space.structure.keys())

        data = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
        tensor = Tensor(
            data=data, dims=(momentum_space, momentum_space, momentum_space)
        )

        out = tensor[:, :, k1]
        expected_dim = MomentumSpace(structure=OrderedDict({k1: 0}))
        expected_idx = momentum_space.structure[k1]
        assert isinstance(out, Tensor)
        assert out.dims == (momentum_space, momentum_space, expected_dim)
        assert torch.equal(out.data, data[:, :, expected_idx : expected_idx + 1])

    def test_getitem_with_indexspace_index(self):
        index_space = IndexSpace.linear(3)
        i1 = tuple(index_space.structure.keys())[1]
        index_subspace = IndexSpace(structure=OrderedDict({i1: 0}))

        data = torch.arange(27, dtype=torch.float64).reshape(3, 3, 3)
        tensor = Tensor(data=data, dims=(index_space, index_space, index_space))

        out = tensor[:, :, index_subspace]
        expected_idx = index_space.structure[i1]
        assert isinstance(out, Tensor)
        assert out.dims == (index_space, index_space, index_subspace)
        assert torch.equal(out.data, data[:, :, expected_idx : expected_idx + 1])


class TestTensorAdvancedGetitem:
    def test_getitem_rejects_multiple_ellipsis_in_normal_indexing(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        data = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        tensor = Tensor(data=data, dims=(a, b))

        with pytest.raises(IndexError, match="single ellipsis"):
            _ = tensor[..., ...]

    def test_getitem_with_tensor_advanced_index_contiguous(self):
        a = IndexSpace.linear(3)
        b = IndexSpace.linear(4)
        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(a, b))

        idx = Tensor(
            data=torch.tensor([2, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        out = tensor[idx, :]

        expected = data[idx.data, :]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (IndexSpace.linear(2), b)

    def test_getitem_with_tensor_advanced_index_scalar_and_vector(self):
        a = IndexSpace.linear(3)
        b = IndexSpace.linear(4)
        sel = IndexSpace.linear(2)
        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(a, b))

        i = Tensor(data=torch.tensor(1, dtype=torch.long), dims=())
        j = Tensor(data=torch.tensor([3, 1], dtype=torch.long), dims=(sel,))

        out = tensor[i, j]
        expected = data[i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (sel,)

    def test_getitem_with_tensor_advanced_index_mixed_rank_broadcast(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(
            data=torch.tensor([[0], [3]], dtype=torch.long),
            dims=(a, BroadcastSpace()),
        )
        j = Tensor(data=torch.tensor([1, 4, 2], dtype=torch.long), dims=(b,))

        out = tensor[i, j]
        expected = data[i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b)

    def test_getitem_with_tensor_advanced_index_separated(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        data = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        tensor = Tensor(data=data, dims=(a, b, c))

        i = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        j = Tensor(
            data=torch.tensor([3, 1], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        out = tensor[i, :, j]

        expected = data[i.data, :, j.data]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (IndexSpace.linear(2), b)

    def test_getitem_with_tensor_advanced_index_and_none_contiguous(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        d = IndexSpace.linear(5)
        data = torch.arange(120, dtype=torch.float64).reshape(2, 3, 4, 5)
        tensor = Tensor(data=data, dims=(a, b, c, d))

        i = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        j = Tensor(
            data=torch.tensor([3, 1], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        out = tensor[:, i, j, None, :]

        expected = data[:, i.data, j.data, None, :]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, IndexSpace.linear(2), BroadcastSpace(), d)

    def test_getitem_with_tensor_advanced_index_and_none_separated(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        data = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        tensor = Tensor(data=data, dims=(a, b, c))

        i = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        j = Tensor(
            data=torch.tensor([3, 1], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )
        out = tensor[None, i, :, j]

        expected = data[None, i.data, :, j.data]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (IndexSpace.linear(2), BroadcastSpace(), b)

    def test_getitem_with_tensor_advanced_index_separated_by_none(self):
        a = IndexSpace.linear(5)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        sel = IndexSpace.linear(2)
        data = torch.arange(60, dtype=torch.float64).reshape(5, 3, 4)
        tensor = Tensor(data=data, dims=(a, b, c))

        i = Tensor(data=torch.tensor([1, 0], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([3, 1], dtype=torch.long), dims=(sel,))
        out = tensor[:, i, None, j]

        expected = data[:, i.data, None, j.data]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        # `None` separates advanced indices, so torch uses separated layout:
        # advanced dims are moved to the front.
        assert out.dims == (sel, a, BroadcastSpace())

    def test_getitem_with_tensor_advanced_index_rejects_invalid_mix(self):
        a = IndexSpace.linear(3)
        b = IndexSpace.linear(4)
        tensor = Tensor(
            data=torch.arange(12, dtype=torch.float64).reshape(3, 4), dims=(a, b)
        )

        state_index = IndexSpace.linear(1)
        tensor_idx = Tensor(
            data=torch.tensor([0], dtype=torch.long), dims=(IndexSpace.linear(1),)
        )
        with pytest.raises(ValueError, match="cannot be mixed"):
            _ = tensor[state_index, tensor_idx]

    def test_getitem_with_tensor_advanced_index_allows_int_and_non_full_slice(self):
        a = IndexSpace.linear(3)
        b = IndexSpace.linear(4)
        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(a, b))
        idx = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )

        out_int = tensor[idx, 1]
        expected_int = data[idx.data, 1]
        assert isinstance(out_int, Tensor)
        assert torch.equal(out_int.data, expected_int)

        out_slice = tensor[idx, 1:3]
        expected_slice = data[idx.data, 1:3]
        assert isinstance(out_slice, Tensor)
        assert torch.equal(out_slice.data, expected_slice)

    def test_getitem_with_tensor_advanced_index_preserves_broadcasted_dims(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(
            data=torch.tensor([[0], [3]], dtype=torch.long),
            dims=(a, BroadcastSpace()),
        )
        j = Tensor(
            data=torch.tensor([[1, 4, 2]], dtype=torch.long),
            dims=(BroadcastSpace(), b),
        )

        out = tensor[i, j]
        expected = data[i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b)

    def test_getitem_with_tensor_advanced_index_raises_for_dim_conflict(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(2)
        c = IndexSpace.linear(2)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(
            data=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            dims=(a, b),
        )
        j = Tensor(
            data=torch.tensor([[1, 0], [4, 2]], dtype=torch.long),
            dims=(a, c.map(lambda n: n + 10)),
        )

        with pytest.raises((ValueError, IndexError)):
            _ = tensor[i, j]

    def test_getitem_with_three_advanced_indices_broadcasted(self):
        xdim = IndexSpace.linear(5)
        ydim = IndexSpace.linear(6)
        zdim = IndexSpace.linear(7)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(4)
        c = IndexSpace.linear(3)

        data = torch.arange(5 * 6 * 7, dtype=torch.float64).reshape(5, 6, 7)
        tensor = Tensor(data=data, dims=(xdim, ydim, zdim))

        i = Tensor(
            data=torch.tensor(
                [[[0, 1, 2]], [[4, 3, 2]]],
                dtype=torch.long,
            ),
            dims=(a, BroadcastSpace(), c),
        )
        j = Tensor(
            data=torch.tensor(
                [[[0], [1], [2], [3]]],
                dtype=torch.long,
            ),
            dims=(BroadcastSpace(), b, BroadcastSpace()),
        )
        k = Tensor(
            data=torch.tensor(
                [
                    [[0, 1, 2], [3, 4, 5], [6, 0, 1], [2, 3, 4]],
                    [[5, 6, 0], [1, 2, 3], [4, 5, 6], [0, 1, 2]],
                ],
                dtype=torch.long,
            ),
            dims=(a, b, c),
        )

        out = tensor[i, j, k]
        expected = data[i.data, j.data, k.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b, c)

    def test_getitem_with_higher_rank_advanced_indices(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(2)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(
            data=torch.tensor(
                [[[0], [3], [1]], [[2], [1], [0]]],
                dtype=torch.long,
            ),
            dims=(a, b, BroadcastSpace()),
        )
        j = Tensor(
            data=torch.tensor(
                [[[0, 1]], [[3, 4]]],
                dtype=torch.long,
            ),
            dims=(a, BroadcastSpace(), c),
        )

        out = tensor[i, j]
        expected = data[i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b, c)

    def test_getitem_with_tensor_advanced_bool_mask_raises_not_implemented(self):
        src = IndexSpace.linear(5)
        data = torch.arange(5, dtype=torch.float64)
        tensor = Tensor(data=data, dims=(src,))

        mask = Tensor(
            data=torch.tensor([True, False, True, False, True], dtype=torch.bool),
            dims=(src,),
        )
        with pytest.raises(
            NotImplementedError, match="Boolean Tensor indexing is not supported yet"
        ):
            _ = tensor[mask]

    def test_getitem_with_tensor_advanced_ellipsis_at_end(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        d = IndexSpace.linear(5)
        sel = IndexSpace.linear(2)

        data = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
        tensor = Tensor(data=data, dims=(a, b, c, d))

        i = Tensor(data=torch.tensor([1, 0], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([3, 1], dtype=torch.long), dims=(sel,))

        out = tensor[..., i, j]
        expected = data[..., i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b, sel)

    def test_getitem_with_tensor_advanced_ellipsis_middle_separated(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        d = IndexSpace.linear(5)
        sel = IndexSpace.linear(2)

        data = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
        tensor = Tensor(data=data, dims=(a, b, c, d))

        i = Tensor(data=torch.tensor([1, 0], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([3, 1], dtype=torch.long), dims=(sel,))

        out = tensor[i, ..., j]
        expected = data[i.data, ..., j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (sel, b, c)

    def test_getitem_with_tensor_advanced_ellipsis_none_then_index(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        c = IndexSpace.linear(4)
        d = IndexSpace.linear(5)
        sel = IndexSpace.linear(2)

        data = torch.arange(2 * 3 * 4 * 5, dtype=torch.float64).reshape(2, 3, 4, 5)
        tensor = Tensor(data=data, dims=(a, b, c, d))

        # Value 4 is valid for axis `d` (size 5), but invalid for axis `c` (size 4).
        # If `...` is expanded too short when `None` is present, this incorrectly
        # targets axis `c` and raises IndexError.
        idx = Tensor(data=torch.tensor([4, 1], dtype=torch.long), dims=(sel,))
        out = tensor[..., None, idx]
        expected = data[..., None, idx.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (a, b, c, BroadcastSpace(), sel)

    def test_getitem_with_tensor_advanced_raises_for_shape_mismatch(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([0, 1], dtype=torch.long), dims=(a,))
        j = Tensor(data=torch.tensor([0, 1, 2], dtype=torch.long), dims=(b,))

        with pytest.raises((ValueError, IndexError)):
            _ = tensor[i, j]

    def test_getitem_with_tensor_advanced_raises_for_out_of_bounds(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)
        sel = IndexSpace.linear(2)

        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([0, 3], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([1, 2], dtype=torch.long), dims=(sel,))

        with pytest.raises(IndexError):
            _ = tensor[i, j]

    def test_getitem_with_tensor_advanced_negative_indices_in_bounds(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)
        sel = IndexSpace.linear(3)

        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([-1, -2, 0], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([1, -1, 2], dtype=torch.long), dims=(sel,))

        out = tensor[i, j]
        expected = data[i.data, j.data]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (sel,)

    def test_getitem_with_tensor_advanced_duplicate_indices(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)
        sel = IndexSpace.linear(4)

        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([1, 1, 2, 1], dtype=torch.long), dims=(sel,))
        j = Tensor(data=torch.tensor([0, 0, 3, 0], dtype=torch.long), dims=(sel,))

        out = tensor[i, j]
        expected = data[i.data, j.data]
        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (sel,)

    def test_getitem_with_tensor_advanced_scalar_index_tensor(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)
        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        scalar_idx = Tensor(data=torch.tensor(1, dtype=torch.long), dims=())
        out = tensor[scalar_idx, :]
        expected = data[scalar_idx.data, :]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (col_src,)

    def test_getitem_with_tensor_advanced_float_index_raises(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)
        bad = IndexSpace.linear(2)
        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        bad_idx = Tensor(
            data=torch.tensor([0.0, 1.0], dtype=torch.float64), dims=(bad,)
        )
        with pytest.raises((IndexError, TypeError, RuntimeError)):
            _ = tensor[bad_idx, :]

    def test_getitem_with_tensor_advanced_empty_index(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)
        empty = IndexSpace.linear(0)

        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([], dtype=torch.long), dims=(empty,))
        j = Tensor(data=torch.tensor([], dtype=torch.long), dims=(empty,))

        out = tensor[i, j]
        expected = data[i.data, j.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (empty,)

    def test_getitem_with_where_indices(self):
        row_src = IndexSpace.linear(3)
        col_src = IndexSpace.linear(4)

        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        mask = Tensor(
            data=torch.tensor(
                [
                    [True, False, False, True],
                    [False, True, False, False],
                    [True, False, True, False],
                ],
                dtype=torch.bool,
            ),
            dims=(row_src, col_src),
        )

        row_idx, col_idx = where(mask)
        out = tensor[row_idx, col_idx]
        expected = data[mask.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (IndexSpace.linear(int(mask.data.sum().item())),)

    def test_getitem_with_tensor_advanced_matches_torch_ellipsis_none_patterns(self):
        # Differential test focused on risky forms involving:
        # Tensor-index, full slice (:), ellipsis (...), and None.
        rank = 4
        axis_size = 5
        dims = tuple(IndexSpace.linear(axis_size) for _ in range(rank))
        data = torch.arange(axis_size**rank, dtype=torch.float64).reshape(
            *(axis_size for _ in range(rank))
        )
        tensor = Tensor(data=data, dims=dims)
        idx = Tensor(
            data=torch.tensor([4, 1], dtype=torch.long),
            dims=(IndexSpace.linear(2),),
        )

        token_alphabet = ("A", "S", "N", "E")  # Tensor, :, None, Ellipsis
        max_key_len = rank + 2
        for key_len in range(1, max_key_len + 1):
            for token_pattern in product(token_alphabet, repeat=key_len):
                if token_pattern.count("E") > 1:
                    continue
                # Focus this sweep on ellipsis+None+advanced interactions.
                if (
                    "A" not in token_pattern
                    or "E" not in token_pattern
                    or "N" not in token_pattern
                ):
                    continue

                key = tuple(
                    idx
                    if t == "A"
                    else slice(None)
                    if t == "S"
                    else None
                    if t == "N"
                    else Ellipsis
                    for t in token_pattern
                )
                torch_key = tuple(k.data if isinstance(k, Tensor) else k for k in key)

                # Compare only valid torch indexing forms.
                try:
                    expected = data[torch_key]
                except Exception:
                    continue

                out = tensor[key]
                assert isinstance(out, Tensor)
                assert torch.equal(out.data, expected)
                assert tuple(dim.dim for dim in out.dims) == tuple(expected.shape)

    def test_getitem_rejects_multiple_ellipsis_in_advanced_indexing(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        data = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        tensor = Tensor(data=data, dims=(a, b))
        idx = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )

        with pytest.raises(IndexError, match="single ellipsis"):
            _ = tensor[..., idx, ...]

    def test_getitem_advanced_too_many_indices_raises(self):
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)
        data = torch.arange(6, dtype=torch.float64).reshape(2, 3)
        tensor = Tensor(data=data, dims=(a, b))
        idx = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )

        with pytest.raises(IndexError, match="Too many indices"):
            _ = tensor[:, :, idx]
