import pytest
import torch

from pyhilbert.tensors import Tensor, where
from pyhilbert.state_space import BroadcastSpace, IndexSpace


class TestTensorAdvancedGetitem:
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

    def test_getitem_with_tensor_advanced_index_rejects_int_and_non_full_slice(self):
        a = IndexSpace.linear(3)
        b = IndexSpace.linear(4)
        data = torch.arange(12, dtype=torch.float64).reshape(3, 4)
        tensor = Tensor(data=data, dims=(a, b))
        idx = Tensor(
            data=torch.tensor([1, 0], dtype=torch.long), dims=(IndexSpace.linear(2),)
        )

        with pytest.raises(TypeError, match="only supports Tensor indices"):
            _ = tensor[idx, 1]

        with pytest.raises(TypeError, match="only supports Tensor indices"):
            _ = tensor[idx, 1:3]

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

        with pytest.raises(ValueError, match="incompatible for broadcast"):
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

    def test_getitem_with_tensor_advanced_bool_mask(self):
        src = IndexSpace.linear(5)
        keep = IndexSpace.linear(3)

        data = torch.arange(5, dtype=torch.float64)
        tensor = Tensor(data=data, dims=(src,))

        mask = Tensor(
            data=torch.tensor([True, False, True, False, True], dtype=torch.bool),
            dims=(src,),
        )
        out = tensor[mask]
        expected = data[mask.data]

        assert isinstance(out, Tensor)
        assert torch.equal(out.data, expected)
        assert out.dims == (keep,)

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

    def test_getitem_with_tensor_advanced_raises_for_shape_mismatch(self):
        row_src = IndexSpace.linear(4)
        col_src = IndexSpace.linear(5)
        a = IndexSpace.linear(2)
        b = IndexSpace.linear(3)

        data = torch.arange(20, dtype=torch.float64).reshape(4, 5)
        tensor = Tensor(data=data, dims=(row_src, col_src))

        i = Tensor(data=torch.tensor([0, 1], dtype=torch.long), dims=(a,))
        j = Tensor(data=torch.tensor([0, 1, 2], dtype=torch.long), dims=(b,))

        with pytest.raises(ValueError, match="not broadcastable in shape"):
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
