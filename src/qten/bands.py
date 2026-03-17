from typing import Literal, cast

import numpy as np

from .geometries.spatials import Offset, Momentum
from .symbolics.state_space import MomentumSpace, brillouin_zone
from .symbolics.hilbert_space import (
    HilbertSpace,
    U1Basis,
    FuncOpr,
)
from .linalg.tensors import Tensor, mapping_matrix
from .geometries.spatials import ReciprocalLattice
from .geometries.basis_transform import BasisTransform
from .geometries.fourier import fourier_transform
from .pointgroups.abelian import AffineTransform
from .utils.collections_ext import matchby


def bandaffine(
    t: AffineTransform,
    tensor: Tensor,
    opt: Literal["left", "right", "both"] = "both",
) -> Tensor:
    """
    Apply an affine symmetry action to a momentum-resolved operator tensor.

    The expected tensor shape is `(K, B_left, B_right)` where `K` is a
    `MomentumSpace` and `B_left`, `B_right` are `HilbertSpace`s. Depending on
    `opt`, this function applies the symmetry-induced basis transform on the
    left side, right side, or both sides of the band tensor.

    For each transformed side, a k-dependent matrix is built from:
    - the affine action on the Hilbert space basis (`t(space)`), and
    - Fourier transforms that connect Bloch and real-space sectors.

    Momentum handling:
    - The k action is treated as a relabeling/permutation of sectors.
    - We align the k-axis of the transform tensors to the canonical `kspace`
      ordering before multiplication.
    - The input tensor itself is not pre-remapped in k; remapping is used only
      to align transform blocks with each momentum sector.

    Parameters
    ----------
    `t` : `AffineTransform`
        Affine transformation to apply.
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
        side is not symmetry-compatible with `t`.
    """
    if opt not in ("both", "left", "right"):
        raise ValueError(f"Invalid option {opt} for bandaffine!")
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
        bloch_transform: Tensor = cast(Tensor, space.cross_gram(new_space)).h(
            -2, -1
        )  # (B', B)
        # The transformation will distort the unit-cell of the Hilbert space,
        # we will use fractional to return it to the original unit-cell.
        if not space.same_rays(new_space):
            raise ValueError(
                f"Hilbert space {space} is not symmetric under the transform {t}!"
            )
        left_fourier = fourier_transform(kspace, space, space)  # (K, B, B'=B)
        right_fourier = fourier_transform(kspace, space, space)  # (K, B, B)
        # (K, B, B'=B) @ (B'=B, B) @ (B, B)
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
