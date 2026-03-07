import numpy as np
import plotly.graph_objects as go
import torch

from pyhilbert.hilbert import Mode, hilbert
from pyhilbert.tensors import Tensor
from pyhilbert.utils import FrozenDict


# def create_dummy_tensor(data_like):
#     """Create a Tensor with simple HilbertSpace dims for tests."""
#     if isinstance(data_like, np.ndarray):
#         data = torch.from_numpy(data_like)
#     else:
#         data = data_like

#     m = Mode(count=data.shape[0], attr=FrozenDict({"label": "dummy"}))
#     hspace = hilbert([m])
#     dims = (hspace,) * data.ndim
#     return Tensor(data=data, dims=dims)


# def test_plot_heatmap_complex_matrix_returns_two_traces():
#     mat = np.array(
#         [
#             [1 + 2j, 2 - 1j, 0 + 1j],
#             [-1 + 0j, 0 + 0j, 3 - 2j],
#             [4 + 1j, -2 + 2j, 1 - 3j],
#         ],
#         dtype=np.complex128,
#     )
#     tensor_obj = create_dummy_tensor(mat)

#     fig = tensor_obj.plot("heatmap", title="Complex Heatmap", show=False)

#     assert isinstance(fig, go.Figure)
#     assert len(fig.data) == 2
#     assert fig.layout.title.text == "Complex Heatmap"


# def test_plot_heatmap_high_rank_requires_and_accepts_fixed_indices():
#     # Shape (2, 2, 2): choosing axes=(1, 2) requires one fixed index.
#     data = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
#     tensor_obj = create_dummy_tensor(data)

#     fig = tensor_obj.plot(
#         "heatmap",
#         title="Rank-3 Slice",
#         show=False,
#         axes=(1, 2),
#         fixed_indices=(0,),
#     )

#     assert isinstance(fig, go.Figure)
#     assert len(fig.data) == 1


# def test_plot_spectrum_hermitian_and_nonhermitian():
#     hermitian = np.array(
#         [[2.0, 1.0 - 1.0j, 0.0], [1.0 + 1.0j, 3.0, -2.0j], [0.0, 2.0j, 1.0]],
#         dtype=np.complex128,
#     )
#     nonhermitian = np.array([[0.0, 1.0], [-2.0, 0.5]], dtype=np.float64)

#     h_tensor = create_dummy_tensor(hermitian)
#     nh_tensor = create_dummy_tensor(nonhermitian)

#     h_fig = h_tensor.plot("spectrum", title="Hermitian Spectrum", show=False)
#     nh_fig = nh_tensor.plot("spectrum", title="Non-Hermitian Spectrum", show=False)

#     assert isinstance(h_fig, go.Figure)
#     assert isinstance(nh_fig, go.Figure)
#     assert len(h_fig.data) == 1
#     assert len(nh_fig.data) == 1
#     assert h_fig.layout.title.text == "Hermitian Spectrum"
#     assert nh_fig.layout.title.text == "Non-Hermitian Spectrum"
