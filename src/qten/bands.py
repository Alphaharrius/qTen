from typing import Callable, Dict, Literal, Tuple, Union, cast

import numpy as np
import torch

from .geometries.spatials import Offset, Momentum
from .symbolics.state_space import IndexSpace, MomentumSpace, brillouin_zone
from .symbolics.hilbert_space import (
    HilbertSpace,
    U1Basis,
    FuncOpr,
)
from .linalg.decompose import eigh
from .linalg.tensors import Tensor, mapping_matrix
from .geometries.spatials import ReciprocalLattice
from .geometries.basis_transform import BasisTransform
from .geometries.fourier import fourier_transform
from .symbolics.hilbert_space import Opr
from .utils.collections_ext import matchby


def bandtransform(
    t: Opr,
    tensor: Tensor,
    opt: Literal["left", "right", "both"] = "both",
) -> Tensor:
    """
    Apply a basis transform to a momentum-resolved operator tensor.

    The expected tensor shape is `(K, B_left, B_right)` where `K` is a
    `MomentumSpace` and `B_left`, `B_right` are `HilbertSpace`s. Depending on
    `opt`, this function applies the operator-induced basis transform on the
    left side, right side, or both sides of the band tensor.

    For each transformed side, a k-dependent matrix is built from:
    - the action of `t` on the Hilbert-space basis (`t(space)`), and
    - Fourier transforms that connect Bloch and real-space sectors.

    Momentum handling:
    - The action on `Momentum` is treated as a relabeling/permutation of sectors.
    - We align the k-axis of the transform tensors to the canonical `kspace`
      ordering before multiplication.
    - The input tensor itself is not pre-remapped in k; remapping is used only
      to align transform blocks with each momentum sector.

    Notes
    -----
    This function accepts a general `Opr`, but not every `Opr` is valid here.
    In practice, `t` must act coherently across the real-space and
    momentum-space labels carried by the tensor:

    - `t @ k` must be defined for each `Momentum` in the first tensor axis.
    - `t @ psi` must be defined for each `U1Basis` in the Hilbert-space axes,
      in particular for the `Offset` irrep stored inside each basis state.
    - The Hilbert-space action and momentum action must be dual-compatible, so
      that the Fourier transform remains consistent after applying `t`.
    - After applying `FuncOpr(Offset, Offset.fractional)`, the transformed
      Hilbert space must have the same rays as the original one; otherwise the
      transformed basis does not close on the input band space and this
      function raises `ValueError`.

    Operators that only act on abstract `U1Basis` values or only on `Momentum`
    values are not sufficient. The operator must provide matching actions on
    site offsets and crystal momentum.

    Parameters
    ----------
    `t` : `Opr`
        Operator to apply. It must satisfy the compatibility conditions
        described in the notes below.
    `tensor` : `Tensor`
        Momentum-space tensor with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    `opt` : `Literal["left", "right", "both"]`, default `"both"`
        Which side(s) to transform.

    Returns
    -------
    `Tensor`
        The transformed tensor with the same dimension types.

    Raises
    ------
    `ValueError`
        If `opt` is invalid, if `tensor` is not rank-3 with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`, or if a Hilbert space
        side is not closed under the action of `t`.
    """
    if opt not in ("both", "left", "right"):
        raise ValueError(f"Invalid option {opt} for bandtransform!")
    if not len(tensor.dims) == 3:
        raise ValueError("Input tensor must have exactly 3 dimensions.")
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise ValueError("First dimension of tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise ValueError("Second dimension of tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise ValueError("Third dimension of tensor must be a HilbertSpace.")

    kspace: MomentumSpace = cast(MomentumSpace, tensor.dims[0])

    def build_transform(space: HilbertSpace) -> Tensor:
        fractional = FuncOpr(Offset, Offset.fractional)
        new_space = cast(HilbertSpace, fractional @ t @ space)
        # The transformation will distort the unit-cell of the Hilbert space,
        # we will use fractional to return it to the original unit-cell.
        if not space.same_rays(new_space):
            raise ValueError(
                f"Hilbert space {space} is not closed under the transform {t}!"
            )
        bloch_transform = cast(Tensor, space.cross_gram(new_space)).h(-2, -1)
        left_fourier = fourier_transform(kspace, space, space)  # (K, B, B')
        right_fourier = fourier_transform(kspace, space, space)  # (K, B, B)
        # Keep the transformed unit-cell labels explicit on the region leg so
        # StateSpace auto-alignment does not erase the site permutation.
        # (K, B, B') @ (B', B) @ (B, B)
        transform = (
            left_fourier @ bloch_transform @ right_fourier.h(-2, -1)
        )  # (K, B, B)
        return transform

    mapped_kspace = kspace.map(lambda k: cast(Momentum, t @ k).fractional())

    if opt in ("both", "left"):
        left_fourier = build_transform(cast(HilbertSpace, tensor.dims[1]))  # (K, B, B)
        left_fourier = left_fourier.replace_dim(0, mapped_kspace).align(
            0, kspace
        )  # (K, B, B)
        tensor = cast(Tensor, (left_fourier @ tensor))  # (K, B, B)

    if opt in ("both", "right"):
        right_fourier = build_transform(cast(HilbertSpace, tensor.dims[2]))  # (K, B, B)
        right_fourier = right_fourier.replace_dim(0, mapped_kspace).align(
            0, kspace
        )  # (K, B, B)
        tensor = cast(Tensor, (tensor @ right_fourier.h(-2, -1)))  # (K, B, B)

    return tensor


def bandfold(
    transform: BasisTransform,
    tensor: Tensor,
    opt: Literal["both", "left", "right"] = "both",
) -> Tensor:
    """
    Fold a momentum-resolved band tensor into the Brillouin zone of a
    transformed lattice basis.

    The input tensor is expected to have dimensions
    `(MomentumSpace, HilbertSpace, HilbertSpace)`. The basis transformation is
    applied to the direct lattice underlying the momentum axis, which produces
    a new Brillouin zone and a corresponding momentum remapping. One Hilbert
    space leg is enlarged to match the transformed unit cell, a Fourier-space
    change of basis is applied, and the momentum sectors are then gathered into
    the new momentum grid.

    `opt` selects which Hilbert-space leg defines the enlarged basis:
    - `"left"` uses `tensor.dims[1]`
    - `"right"` and `"both"` use `tensor.dims[2]`

    Parameters
    ----------
    transform : BasisTransform
        Basis change applied to the direct lattice associated with the momentum
        axis.
    tensor : Tensor
        Rank-3 tensor with dimensions
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    opt : Literal["both", "left", "right"], default "both"
        Selects which Hilbert-space leg is rebuilt in the transformed unit
        cell. `"both"` currently follows the right-leg branch.

    Returns
    -------
    Tensor
        Folded tensor on the transformed momentum grid.

    Raises
    ------
    ValueError
        If the tensor is not rank-3, if the momentum space is empty, or if the
        momentum axis does not belong to a single Brillouin zone.
    TypeError
        If the momentum axis is not a `MomentumSpace`, if its underlying space
        is not a `ReciprocalLattice`, or if the selected Hilbert-space leg is
        not a `HilbertSpace`.
    """
    # 1. Parse inputs
    if not tensor.rank() == 3:
        raise ValueError(
            f"Input tensor must be of rank 3, but has rank {tensor.rank()}"
        )
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError(
            "The first dimension of the tensor must be a MomentumSpace, "
            f"but is of type {type(tensor.dims[0])}"
        )
    k_space = cast(MomentumSpace, tensor.dims[0])
    if not k_space.elements():
        raise ValueError("MomentumSpace is empty")
    lattice_set = set(map(lambda k: k.space, k_space))
    if len(lattice_set) != 1:
        raise ValueError("Invalid BZ")
    reciprocal_lattice = lattice_set.pop()
    if not isinstance(reciprocal_lattice, ReciprocalLattice):
        raise TypeError(
            f"Space of momentum should be ReciprocalLattice, but got {type(reciprocal_lattice)}"
        )
    reciprocal_lattice = cast(ReciprocalLattice, reciprocal_lattice)
    lattice = reciprocal_lattice.dual

    # 2. Apply the transformation
    scaled_lattice = transform(lattice)

    # 3. Create new transformed spaces
    scaled_reciprocal_lattice = scaled_lattice.dual
    transformed_unit_cell = tuple(
        sorted(scaled_lattice.unit_cell.values(), key=lambda x: tuple(x.rep))
    )
    # Keep a rebased copy for the current Fourier/matching logic, but return
    # the transformed offsets on the output Hilbert-space labels.
    enlarge_unit_cell = tuple(r.rebase(lattice) for r in transformed_unit_cell)

    # Transform based on opt
    switch_index = -2 if opt == "left" else -1
    target_space = tensor.dims[switch_index]
    if not isinstance(target_space, HilbertSpace):
        raise TypeError(
            f"Dimension at index {switch_index} must be a HilbertSpace, "
            f"but got {type(target_space)}"
        )
    rebased_hilbert = HilbertSpace.new(
        cast(U1Basis, target_space.lookup({Offset: r.fractional()})).replace(r)
        for r in enlarge_unit_cell
    )
    transformed_hilbert = HilbertSpace.new(
        cast(U1Basis, target_space.lookup({Offset: r_lookup.fractional()})).replace(
            r_out
        )
        for r_lookup, r_out in zip(enlarge_unit_cell, transformed_unit_cell)
    )
    # # Transform both sides
    f = fourier_transform(k_space, tensor.dims[switch_index], rebased_hilbert)
    vratio = np.sqrt(len(enlarge_unit_cell) / len(lattice.unit_cell))
    f = f / vratio
    fh = f.h(-2, -1)  # (K, B', B)
    transformed = fh @ tensor @ f  # (K, B', B')
    transformed = transformed.permute(1, 2, 0).unsqueeze(-1)  # (B', B', K, 1)

    # k-mapping
    new_k_space = brillouin_zone(scaled_reciprocal_lattice)
    mapping = matchby(
        k_space,
        new_k_space,
        base_func=lambda k: transform(k).fractional()
        if k.space == reciprocal_lattice
        else k.fractional(),
    )
    k_map = mapping_matrix(k_space, new_k_space, mapping).transpose(0, 1)
    transformed = (k_map @ transformed).squeeze(-1).permute(2, 0, 1)
    for dim in (1, 2):
        if transformed.dims[dim] == rebased_hilbert:
            transformed = transformed.replace_dim(dim, transformed_hilbert)
    return transformed


# TODO: add bandunfold function


def bandfillings(tensor: Tensor, frac: float) -> Tensor:
    """
    Fill a band represented by a band-resolved `Tensor` up to a given filling fraction `frac`.

    The input `tensor` is expected to have dimensions `(MomentumSpace, HilbertSpace, HilbertSpace)`,
    where the first dimension corresponds to momentum sectors and the second two dimensions correspond
    to the band basis. The function fills the bands in each momentum sector according to their energies,
    which are obtained by diagonalizing the band tensor at each momentum point. If one state from a degenerated
    set is filled, all states in that set are filled.

    Parameters
    ----------
    `tensor` : `Tensor`
        A band-resolved tensor with dimensions `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    `frac` : `float`
        The filling fraction, a value between 0 and 1, indicating the fraction of the band to fill.

    Returns
    -------
    `Tensor`
        A tensor of shape `(MomentumSpace, HilbertSpace, IndexSpace)` where `IndexSpace` is a one-dimensional
        state space representing the filled states. The tensor contains the eigenvectors corresponding to the
        filled states in each momentum sector. The dimension of the `IndexSpace` is determined by the maximum number
        of states filled across all momentum sectors. If a momentum sector has fewer filled states than the maximum,
        the remaining entries in the `IndexSpace` for that sector will be zero.
    """
    if tensor.rank() != 3:
        raise ValueError(
            f"Input tensor must be of rank 3, but has rank {tensor.rank()}"
        )
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError("The first dimension of the tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise TypeError("The second dimension of the tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise TypeError("The third dimension of the tensor must be a HilbertSpace.")
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"Filling fraction must be between 0 and 1, got {frac}")

    kspace = cast(MomentumSpace, tensor.dims[0])
    band_space = cast(HilbertSpace, tensor.dims[1])
    eigvals, eigvecs = eigh(tensor)

    nk, nbands = eigvals.data.shape
    total_states = nk * nbands
    target_fill = int(np.floor(frac * total_states + 1e-12))

    if target_fill <= 0:
        return Tensor(
            data=eigvecs.data[..., :0],
            dims=(kspace, band_space, IndexSpace.linear(0)),
        )
    if target_fill >= total_states:
        return Tensor(
            data=eigvecs.data,
            dims=(kspace, band_space, IndexSpace.linear(nbands)),
        )

    flat_vals = eigvals.data.reshape(-1)
    threshold = torch.kthvalue(flat_vals, target_fill).values
    eps = torch.finfo(eigvals.data.dtype).eps
    tol = (abs(threshold).clamp_min(1.0) * eps * max(nbands, 1) * 8).to(
        eigvals.data.dtype
    )
    filled = eigvals.data <= (threshold + tol)

    counts = filled.sum(dim=1)
    max_fill = int(counts.max().item())
    out_dim = IndexSpace.linear(max_fill)

    order = torch.argsort(filled.to(torch.int8), dim=1, descending=True, stable=True)
    packed = torch.gather(
        eigvecs.data,
        2,
        order[:, None, :].expand(-1, eigvecs.data.shape[1], -1),
    )[..., :max_fill]

    valid = (
        torch.arange(max_fill, device=counts.device)[None, :] < counts[:, None]
    ).to(packed.dtype)
    packed = packed * valid[:, None, :]

    return Tensor(data=packed, dims=(kspace, band_space, out_dim))


def bandselect(
    tensor: Tensor,
    **kwargs: Dict[
        str, Union[slice, Tuple[int, ...], Tuple[float, float], Callable[[float], bool]]
    ],
) -> Dict[str, Tensor]:
    """
    Select specific bands from a band-resolved `Tensor` based on criteria provided in `kwargs`.

    The returned `Dict` maps each criterion name to a `Tensor` containing the selected bands that meet that criterion
    with dimensions `(MomentumSpace, HilbertSpace, IndexSpace)`, where `IndexSpace` is a one-dimensional state space
    representing the selected states for that criterion. If a momentum sector has fewer selected states than the maximum
    across all sectors for that criterion, the remaining entries in the `IndexSpace` for that sector will be zero.

    The selection criteria in `kwargs` can be specified as follows:
    - `slice`: Select bands based on their index in the sorted order of energies (e.g., `slice(0, 2)` selects the two lowest-energy bands).
    - `Tuple[int, ...]`: Select specific band indices (e.g., `(0, 2)` selects the lowest and third-lowest energy bands).
    - `Tuple[float, float]`: Select bands based on an energy range (e.g., `(-1.0, 0.0)` selects bands with energies between -1.0 and 0.0).
    - `Callable[[float], bool]`: Select bands based on a custom function that takes the energy as input and returns a boolean
      (e.g., `lambda E: E < 0` selects all bands with negative energy).

    If a criterion have no matching bands in all momentum sectors, the corresponding `Tensor` will have an `IndexSpace` of dimension zero.

    Parameters
    ----------
    `tensor` : `Tensor`
        A band-resolved tensor with dimensions `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    `kwargs` : `Dict[str, Union[slice, Tuple[int, ...], Tuple[float, float], Callable[[float], bool]]]`
        A dictionary mapping criterion names to selection criteria for bands.

    Returns
    -------
    `Dict[str, Tensor]`
        A dictionary mapping each criterion name to a `Tensor` containing the selected bands that meet that criterion,
        with dimensions `(MomentumSpace, HilbertSpace, IndexSpace)`.
    """
    if tensor.rank() != 3:
        raise ValueError(
            f"Input tensor must be of rank 3, but has rank {tensor.rank()}"
        )
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError("The first dimension of the tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise TypeError("The second dimension of the tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise TypeError("The third dimension of the tensor must be a HilbertSpace.")

    kspace = cast(MomentumSpace, tensor.dims[0])
    band_space = cast(HilbertSpace, tensor.dims[1])
    eigvals, eigvecs = eigh(tensor)
    values = eigvals.data
    vectors = eigvecs.data

    nk, nbands = values.shape
    band_indices = torch.arange(nbands, device=values.device)

    def pack(mask: torch.Tensor) -> Tensor:
        counts = mask.sum(dim=1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        out_dim = IndexSpace.linear(max_count)
        if max_count == 0:
            return Tensor(data=vectors[..., :0], dims=(kspace, band_space, out_dim))

        order = torch.argsort(mask.to(torch.int8), dim=1, descending=True, stable=True)
        packed = torch.gather(
            vectors,
            2,
            order[:, None, :].expand(-1, vectors.shape[1], -1),
        )[..., :max_count]
        valid = (
            torch.arange(max_count, device=counts.device)[None, :] < counts[:, None]
        ).to(packed.dtype)
        packed = packed * valid[:, None, :]
        return Tensor(data=packed, dims=(kspace, band_space, out_dim))

    selected: Dict[str, Tensor] = {}
    for name, criterion in kwargs.items():
        mask: torch.Tensor
        if isinstance(criterion, slice):
            picked = band_indices[criterion]
            mask = torch.zeros((nk, nbands), dtype=torch.bool, device=values.device)
            if picked.numel():
                mask[:, picked] = True
        elif isinstance(criterion, tuple):
            if all(isinstance(x, int) and not isinstance(x, bool) for x in criterion):
                mask = torch.zeros((nk, nbands), dtype=torch.bool, device=values.device)
                if criterion:
                    raw_idx = torch.tensor(
                        criterion, dtype=torch.long, device=values.device
                    )
                    if ((raw_idx < -nbands) | (raw_idx >= nbands)).any():
                        raise IndexError(
                            f"Band index out of range in criterion {name!r}"
                        )
                    mask[:, raw_idx % nbands] = True
            elif len(criterion) == 2 and all(
                isinstance(x, (int, float, np.integer, np.floating))
                and not isinstance(x, bool)
                for x in criterion
            ):
                lo, hi = criterion
                mask = (values >= lo) & (values <= hi)
            else:
                raise TypeError(
                    f"Unsupported tuple criterion for {name!r}: {criterion!r}"
                )
        elif callable(criterion):
            mask = torch.tensor(
                [
                    [bool(criterion(v)) for v in row]
                    for row in values.detach().cpu().tolist()
                ],
                dtype=torch.bool,
                device=values.device,
            )
        else:
            raise TypeError(f"Unsupported criterion for {name!r}: {criterion!r}")

        selected[name] = pack(mask)

    return selected
