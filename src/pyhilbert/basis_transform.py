from dataclasses import dataclass
from sympy import ImmutableDenseMatrix
import sympy as sy
from sympy.matrices.normalforms import smith_normal_decomp  # type: ignore[import-untyped]
from functools import lru_cache
from itertools import product
from typing import List, Tuple, cast, Literal
import numpy as np

from .abstracts import Functional
from .utils import FrozenDict, matchby
from .validations import need_validation
from .validations.symbolics import check_proper_transformation, check_numerical
from .spatials import Lattice, ReciprocalLattice, Offset, Momentum, AffineSpace
from .state_space import MomentumSpace, brillouin_zone
from .hilbert_space import HilbertSpace, U1Basis, hilbert
from .tensors import Tensor, mapping_matrix
from .fourier import fourier_transform


@need_validation(check_proper_transformation("M"), check_numerical("M"))
@dataclass(frozen=True)
class BasisTransform(Functional):
    M: ImmutableDenseMatrix


@lru_cache
def _supercell_shifts(
    dim: int, M: ImmutableDenseMatrix
) -> Tuple[ImmutableDenseMatrix, ...]:
    """
    Generate the integer shifts within the supercell defined by M.
    """
    S, U, V = smith_normal_decomp(M, domain=sy.ZZ)
    Q = V.inv()
    ranges = [range(int(S[i, i])) for i in range(dim)]
    shifts = [ImmutableDenseMatrix([n]) @ Q for n in product(*ranges)]
    return tuple(shifts)


@BasisTransform.register(AffineSpace)
def affine_space_transform(t: BasisTransform, space: AffineSpace) -> AffineSpace:
    """
    Transform an AffineSpace by the basis transformation M.
    """
    new_basis = t.M @ space.basis
    return AffineSpace(basis=new_basis)


@BasisTransform.register(Lattice)
def lattice_transform(t: BasisTransform, lat: Lattice) -> Lattice:
    """
    Generates a Supercell based on the scaling matrix M.
    Automatically populates the new unit cell with original atoms
    to preserve physical density.
    """
    # 1. Validate M
    shifts = _supercell_shifts(lat.dim, t.M)

    # 4. Transform Atoms
    M_inv = t.M.inv()
    new_unit_cell = {}

    # Iterate over existing atoms (or implicit origin)
    if lat.unit_cell:
        items = cast(List[Tuple[str, Offset]], list(lat.unit_cell.items()))
    else:
        default_offset = Offset(
            rep=ImmutableDenseMatrix([0] * lat.dim), space=lat.affine
        )
        items = [("0", default_offset)]

    for label, atom_offset in items:
        atom_vec = atom_offset.rep.reshape(1, lat.dim)
        for i, k in enumerate(shifts):
            # Now both atom_vec and k are 1xN Matrices
            # Formula: new_frac = (old_frac + shift) * M^-1
            new_frac = (atom_vec + k) @ M_inv
            new_frac = new_frac.applyfunc(lambda x: x - sy.floor(x))

            # Generate new label
            new_label = f"{label}_{i}" if len(shifts) > 1 else label
            new_unit_cell[new_label] = new_frac
    new_basis = t.M @ lat.basis
    new_shape = tuple(s // m for s, m in zip(lat.shape, t.M.diagonal()))
    return Lattice(
        basis=new_basis, shape=new_shape, unit_cell=FrozenDict(new_unit_cell)
    )


@BasisTransform.register(ReciprocalLattice)
def reciprocal_lattice_transform(
    t: BasisTransform, lat: ReciprocalLattice
) -> ReciprocalLattice:
    """
    Generate the reciprocal lattice corresponding to the transformed direct lattice.
    """
    dual_lat = lat.dual
    transformed_dual_lat = t(dual_lat)
    return transformed_dual_lat.dual


@BasisTransform.register(Offset)
def offset_transform(t: BasisTransform, r: Offset) -> Offset:
    """

    Transform an Offset by the basis transformation M.
    """
    new_space = t(r.space)
    return r.rebase(new_space)


@BasisTransform.register(Momentum)
def momentum_transform(t: BasisTransform, momentum: Momentum) -> Momentum:
    """
    Docstring for momentum_transform

    Parameters
    ----------
    """
    new_space = t(momentum.space)
    return momentum.rebase(new_space)


def bandfold(
    M: ImmutableDenseMatrix,
    tensor: Tensor,
    opt: Literal["both", "left", "right"] = "both",
) -> Tensor:
    """
    make Tensor with (Momentum, Hilbert, Hilbert) to (scaled Momentum, Hilbert, Hilbert)
    Parameters
    ----------
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
    transform = BasisTransform(M)
    scaled_lattice = transform(lattice)

    # 3. Create new transformed spaces
    scaled_reciprocal_lattice = scaled_lattice.dual
    scaled_offsets = sorted(
        scaled_lattice.unit_cell.values(), key=lambda x: tuple(x.rep)
    )
    enlarge_unit_cell = tuple(r.rebase(lattice.affine) for r in scaled_offsets)

    # Transform based on opt
    switch_index = -2 if opt == "left" else -1
    target_space = tensor.dims[switch_index]
    if not isinstance(target_space, HilbertSpace):
        raise TypeError(
            f"Dimension at index {switch_index} must be a HilbertSpace, "
            f"but got {type(target_space)}"
        )
    rebased_hilbert = hilbert(
        cast(U1Basis, target_space.lookup({Offset: r.fractional()})).replace(r)
        for r in enlarge_unit_cell
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
    return (k_map @ transformed).squeeze(-1).permute(2, 0, 1)


# TODO: add bandunfold function
