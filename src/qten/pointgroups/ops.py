"""
Point-group operations on symbolic bases and tensors.

This module combines point-group transforms with QTen Hilbert spaces and
tensors. The helpers compute joint abelian eigen-bases, project columns into
abelian phase sectors or finite-group irrep sectors, and assemble
representation tensors for point-group actions.

Repository usage
----------------
Use [`joint_point_group_basis()`][qten.pointgroups.ops.joint_point_group_basis]
and the related projection helpers when an existing
[`PointGroupElement`][qten.pointgroups.elements.PointGroupElement],
[`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr], or
[`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] should act on
symbolic Hilbert-space data. Group definitions live in
[`qten.pointgroups.elements`][qten.pointgroups.elements] and
[`qten.pointgroups.finite`][qten.pointgroups.finite].
"""

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any, Optional, Sequence, Tuple, cast

import sympy as sy
import torch

from .basis import PointGroupBasis
from .elements import (
    PointGroupElement,
    PointGroupOpr,
)
from .finite import FinitePointGroup
from ..geometries import Lattice, Offset
from ..linalg.tensors import Tensor, cat, eye, mapping_matrix
from ..phys.spin import Spin, contains_spin, expand_spin, su2_numeric
from ..symbolics import (
    HilbertSpace,
    IndexSpace,
    Multiple,
    U1Basis,
    U1Span,
    hilbert_opr_repr,
)
from ..utils.collections_ext import FrozenDict
from ..utils.devices import Device
from ..precision import get_precision_config


def _relative_tolerance(dtype: torch.dtype, size: int = 1) -> float:
    """Precision- and dimension-aware relative numerical tolerance."""
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    return torch.finfo(real_dtype).eps * max(100, size)


def _same_phase(a: sy.Expr, b: sy.Expr) -> bool:
    diff = sy.simplify(a - b)
    if diff == 0:
        return True

    expanded = sy.simplify(sy.expand_complex(diff))
    if expanded == 0:
        return True

    equals = diff.equals(0)
    return bool(equals)


def _is_unit_modulus(value: sy.Expr) -> bool:
    """Return whether a coefficient is a numerically evaluable U(1) phase."""
    if value.free_symbols:
        return False
    modulus_sq = sy.simplify(sy.conjugate(value) * value)
    if _same_phase(modulus_sq, sy.Integer(1)):
        try:
            complex(sy.N(value))
            return True
        except (TypeError, ValueError):
            return False
    try:
        return abs(abs(complex(sy.N(value))) - 1.0) <= 1e-10
    except (TypeError, ValueError):
        return False


def _phase_basis(opr: PointGroupOpr, phase: sy.Expr) -> PointGroupBasis:
    phase = sy.simplify(phase)
    table = opr.g.basis_table
    if phase in table:
        return cast(PointGroupBasis, table[phase])
    for key, basis in table.items():
        if _same_phase(key, phase):
            return cast(PointGroupBasis, basis)
    raise ValueError(f"Failed to find an PointGroupBasis for phase={phase}.")


def _attach_basis_label(seed: U1Basis | None, basis: PointGroupBasis) -> U1Basis:
    return _attach_sector_label(seed, basis)


def _attach_sector_label(seed: U1Basis | None, label: Any) -> U1Basis:
    if seed is None:
        return U1Basis.new(label)
    try:
        return seed.replace(label)
    except ValueError:
        return U1Basis(coef=seed.coef, base=seed.base + (label,))


def _attach_degeneracy_tag(seed: U1Basis, index: int) -> U1Basis:
    tag = SymmetryDegeneracy(index)
    try:
        return seed.replace(tag)
    except ValueError:
        return U1Basis(coef=seed.coef, base=seed.base + (tag,))


def _column_symmetrize_context(w: Tensor) -> tuple[HilbertSpace, list[U1Basis | None]]:
    if w.rank() != 2:
        raise ValueError("w must be a rank-2 tensor of ambient-space columns.")

    row_dim = w.dims[0]
    if not isinstance(row_dim, HilbertSpace):
        raise ValueError("w.dims[0] must be a HilbertSpace.")
    input_col_dim = w.dims[1]
    if isinstance(input_col_dim, HilbertSpace):
        return row_dim, list(input_col_dim.elements())
    if isinstance(input_col_dim, IndexSpace):
        return row_dim, [None] * input_col_dim.dim
    raise ValueError("w.dims[1] must be either an IndexSpace or a HilbertSpace.")


def _labels_with_degeneracy(raw_labels: list[U1Basis]) -> list[U1Basis]:
    totals: dict[U1Basis, int] = {}
    for label in raw_labels:
        totals[label] = totals.get(label, 0) + 1

    seen: dict[U1Basis, int] = {}
    labels: list[U1Basis] = []
    for label in raw_labels:
        idx = seen.get(label, 0)
        seen[label] = idx + 1
        if totals[label] > 1:
            labels.append(_attach_degeneracy_tag(label, idx))
        else:
            labels.append(label)
    return labels


def _svd_independent_columns(
    columns: Sequence[Tensor], row_dim: HilbertSpace
) -> list[Tensor]:
    if not columns:
        return []
    if len(columns) == 1:
        return [columns[0]]

    stacked = cat(list(columns), dim=-1)
    u, singular_vals, _ = torch.linalg.svd(stacked.data, full_matrices=False)
    if singular_vals.numel() == 0:
        return []
    largest = float(singular_vals[0].item())
    rank_cutoff = (
        _relative_tolerance(stacked.data.dtype, max(stacked.data.shape)) * largest
    )
    rank = int(torch.count_nonzero(singular_vals > rank_cutoff).item())
    if rank == 0:
        return []

    single_col = IndexSpace.linear(1)
    return [
        Tensor(data=u[:, i : i + 1], dims=(row_dim, single_col)) for i in range(rank)
    ]


@dataclass(frozen=True)
class FiniteIrrepSector:
    """Label for a finite-group non-abelian symmetry sector."""

    group: str
    irrep: str
    dim: int


@dataclass(frozen=True)
class SpinorIrrepSector:
    """Label for a finite double-valued point-group symmetry sector."""

    group: str
    irrep: str
    dim: int
    source: str = "spgrep"


@dataclass(frozen=True)
class SpinfulPhaseSector:
    r"""Label for an abelian spinorial sector with phase \(\omega^{2n}=1\)."""

    phase: sy.Expr
    spatial_order: int


@dataclass(frozen=True)
class SymmetryDegeneracy:
    """Typed copy index for repeated symmetry-sector labels."""

    index: int


def point_group_operator_symmetrize(
    group: FinitePointGroup,
    operator: Tensor,
    *,
    fixpoint: Offset | None = None,
    rebase_fixpoint: bool = False,
) -> Tensor:
    r"""
    Average an operator over a finite point group by unitary conjugation.

    This computes
    \(A_G = |G|^{-1}\sum_g D(g) A D(g)^\dagger\). Unlike character
    projection of state vectors, conjugation averaging is valid for spin-1/2
    without double-group character tables: the sign ambiguity of each SU(2)
    lift cancels between \(D(g)\) and \(D(g)^\dagger\).

    The two matrix dimensions of `operator` must describe the same
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]. That space
    must be closed under every point-group operation; otherwise representation
    assembly raises `ValueError`.
    """
    if operator.rank() != 2:
        raise ValueError("operator must be a rank-2 Hilbert-space tensor.")
    row_dim, col_dim = operator.dims
    if not isinstance(row_dim, HilbertSpace) or not isinstance(col_dim, HilbertSpace):
        raise ValueError("operator dimensions must both be HilbertSpace objects.")
    if row_dim != col_dim:
        raise ValueError(
            "operator dimensions must use the same ordered Hilbert-space basis, "
            "including identical U(1) gauges."
        )

    elements = group.elements()
    if not elements:
        raise ValueError("Finite point group contains no elements.")

    averaged: Tensor | None = None
    for element in elements:
        element_opr = PointGroupOpr(element)
        if fixpoint is not None:
            element_opr = element_opr.fixpoint_at(fixpoint, rebase=rebase_fixpoint)
        representation = _hilbert_opr_repr(
            element_opr, row_dim, device=operator.device
        ).to_device(operator.device)
        identity = torch.eye(
            representation.data.shape[0],
            dtype=representation.data.dtype,
            device=representation.data.device,
        )
        tolerance = _relative_tolerance(
            representation.data.dtype, representation.data.shape[0]
        )
        gram = representation.data.conj().T @ representation.data
        if not torch.allclose(gram, identity, rtol=tolerance, atol=tolerance):
            max_error = float(torch.max(torch.abs(gram - identity)).item())
            raise ValueError(
                "Point-group operator averaging requires a unitary Hilbert-space "
                f"representation; max |D†D-I|={max_error:.3e} for {element}."
            )
        transformed = representation @ operator @ representation.h(-2, -1)
        averaged = transformed if averaged is None else averaged + transformed

    assert averaged is not None
    return averaged / len(elements)


def _finite_point_group_column_symmetrize(
    group: FinitePointGroup,
    w: Tensor,
    full_sector: bool = False,
    *,
    fixpoint: Offset | None = None,
    rebase_fixpoint: bool = False,
) -> Tensor:
    r"""
    Symmetrize columns of `w` into irreducible sectors of a finite non-abelian group.

    This function uses the generated group elements and character-table sectors of
    `group` to build projectors
    \(P^\mu = \frac{d_\mu}{|G|}\sum_{g\in G}\chi^\mu(g)^* D(g)\),
    where `D(g)` is the Hilbert-space representation on `w.dims[0]`.

    For each input column, all nonzero irrep projections are collected when
    `full_sector=True`; otherwise only the largest nonzero projected irrep is
    retained. Sector-wise SVD is then applied to remove linear dependence and
    keep independent projected columns.
    """
    row_dim, seeds = _column_symmetrize_context(w)
    spinful = contains_spin(row_dim)
    if spinful:
        if not group.spinor_irreps:
            raise ValueError(
                "No spinor character-table data is available for finite point "
                f"group {group.symbol}."
            )
        irrep_table = group.spinor_irreps["irreps"]
    else:
        if not group.irreps:
            raise ValueError(
                "No character-table data is available for finite point group "
                f"{group.symbol}."
            )
        irrep_table = group.irreps["irreps"]
    if not irrep_table:
        raise ValueError(
            f"Character-table data contains no irreps for point group {group.symbol}."
        )
    single_col = IndexSpace.linear(1)

    elements = group.elements()
    if not elements:
        return Tensor(
            data=w.data.new_empty((row_dim.dim, 0), dtype=w.data.dtype),
            dims=(row_dim, IndexSpace.linear(0)),
        )

    reps: list[Tensor] = []
    for element in elements:
        element_opr = PointGroupOpr(element)
        if fixpoint is not None:
            element_opr = element_opr.fixpoint_at(fixpoint, rebase=rebase_fixpoint)
        reps.append(
            _hilbert_opr_repr(element_opr, row_dim, device=w.device).to_device(w.device)
        )
    working_dtype = reps[0].data.dtype
    tol = _relative_tolerance(working_dtype, row_dim.dim)
    group_order = len(elements)

    sector_projectors: list[tuple[Any, Tensor]] = []
    for irrep_name, irrep_data in irrep_table.items():
        dim = int(irrep_data["dim"])
        if spinful:
            characters = group.spinor_irrep_characters_by_element(irrep_name)
            sector_label: Any = SpinorIrrepSector(
                group=group.symbol or "<anonymous>",
                irrep=irrep_name,
                dim=dim,
            )
        else:
            characters = group.irrep_characters_by_element(irrep_name)
            sector_label = FiniteIrrepSector(
                group=group.symbol or "<anonymous>",
                irrep=irrep_name,
                dim=dim,
            )
        projector = 0 * reps[0]
        for character, rep in zip(characters, reps):
            coefficient = (
                complex(character).conjugate()
                if spinful
                else complex(sy.N(sy.conjugate(character)))
            )
            projector = projector + coefficient * rep
        projector = (dim / group_order) * projector
        sector_projectors.append((sector_label, projector))

    pooled_cols: dict[U1Basis, list[Tensor]] = {}
    label_order: list[U1Basis] = []
    for j, seed in enumerate(seeds):
        col = w[:, j : j + 1].clone().replace_dim(1, single_col).astype(working_dtype)
        input_norm = abs(col.norm().item())
        projection_cutoff = tol * input_norm
        candidates: list[tuple[float, Tensor, U1Basis]] = []
        for sector_label, projector in sector_projectors:
            projected = projector @ col
            projected_norm = projected.norm()
            norm_value = abs(projected_norm.item())
            if norm_value <= projection_cutoff:
                continue
            candidates.append(
                (
                    norm_value,
                    projected / norm_value,
                    _attach_sector_label(seed, sector_label),
                )
            )

        if full_sector:
            selected = candidates
        elif candidates:
            selected = [max(candidates, key=lambda item: item[0])]
        else:
            selected = []
        for _, projected, label in selected:
            if label not in pooled_cols:
                pooled_cols[label] = []
                label_order.append(label)
            pooled_cols[label].append(projected)

    projected_cols: list[Tensor] = []
    raw_labels: list[U1Basis] = []
    for label in label_order:
        independent = _svd_independent_columns(pooled_cols[label], row_dim)
        for col in independent:
            projected_cols.append(col)
            raw_labels.append(label)

    if not projected_cols:
        dtype = reps[0].data.dtype if reps else w.data.dtype
        return Tensor(
            data=w.data.new_empty((row_dim.dim, 0), dtype=dtype),
            dims=(row_dim, IndexSpace.linear(0)),
        )

    out_dim = HilbertSpace.new(_labels_with_degeneracy(raw_labels))
    return cat(projected_cols, dim=-1).replace_dim(-1, out_dim)


def _transform_point_group_basis_direct(
    opr: PointGroupOpr, basis: PointGroupBasis
) -> Multiple[PointGroupBasis]:
    group = opr.g
    if set(group.axes) != set(basis.axes):
        raise ValueError(
            f"Axes of PointGroupElement and PointGroupBasis must match: {group.axes} != {basis.axes}"
        )

    transformed_rep = sy.ImmutableDenseMatrix(
        group.euclidean_repr(basis.order) @ basis.rep
    )
    try:
        scale = next(entry for entry in transformed_rep if entry != 0)
    except StopIteration as exc:
        raise ValueError(f"{basis} is a trivial basis function: zero") from exc
    transformed_basis = PointGroupBasis.from_rep(
        rep=transformed_rep,
        euclidean_basis=group.euclidean_basis(basis.order),
        axes=group.axes,
        order=basis.order,
        group=basis.group,
        irrep=basis.irrep,
        irrep_dim=basis.irrep_dim,
        copy_index=basis.copy_index,
        component_index=basis.component_index,
    )
    return Multiple(sy.Abs(scale), transformed_basis)


def _split_spin_basis(psi: U1Basis) -> tuple[tuple[Any, ...], Spin, sy.Expr]:
    """Split a basis state into (non-spin irreps, Spin, U(1) coef)."""
    spin_reps = [rep for rep in psi.base if type(rep) is Spin]
    if len(spin_reps) != 1:
        raise ValueError(
            f"Expected exactly one Spin irrep in {psi}, found {len(spin_reps)}"
        )
    nons = tuple(rep for rep in psi.base if type(rep) is not Spin)
    return nons, spin_reps[0], psi.coef


def _validate_hilbert_basis(space: HilbertSpace) -> None:
    """Require unique physical rays with numerical unit-modulus gauges."""
    seen_rays: dict[U1Basis, U1Basis] = {}
    for psi in space.elements():
        ray = psi.rays()
        if ray in seen_rays:
            raise ValueError(
                "Hilbert-space basis states must represent unique physical rays; "
                f"{seen_rays[ray]} and {psi} differ only by a gauge coefficient."
            )
        seen_rays[ray] = psi
        if not _is_unit_modulus(psi.coef):
            raise ValueError(
                "Hilbert-space basis coefficients must be numerically evaluable "
                f"unit modulus U(1) phases; got coef={psi.coef} in {psi}"
            )


def _validate_spinful_space(space: HilbertSpace) -> None:
    """Require one spin-1/2 label and a normalized phase on every basis state."""
    _validate_hilbert_basis(space)
    invalid = [
        (psi, sum(type(rep) is Spin for rep in psi.base))
        for psi in space.elements()
        if sum(type(rep) is Spin for rep in psi.base) != 1
    ]
    if invalid:
        psi, count = invalid[0]
        raise ValueError(
            "A spinful Hilbert space must carry exactly one Spin label on every "
            f"basis state; found {count} in {psi}"
        )


def _transform_nons_spin_irreps(
    opr: PointGroupOpr, nons: tuple[Any, ...]
) -> tuple[sy.Expr, tuple[Any, ...]]:
    """Apply the orbital/spatial part of `opr` to non-spin irreps."""
    new_coef: sy.Expr = sy.Integer(1)
    new_base: list[Any] = []
    for rep in nons:
        if type(rep) is PointGroupBasis:
            transformed = _transform_point_group_basis_direct(opr, rep)
            new_coef *= transformed.coef
            new_base.append(transformed.base)
        elif opr.allows(rep):
            ret = opr(rep)
            if isinstance(ret, Multiple):
                new_coef *= ret.coef
                new_base.append(ret.base)
            else:
                new_base.append(ret)
        else:
            new_base.append(rep)
    return sy.simplify(new_coef), tuple(new_base)


def _fold_offset_irreps(nons: tuple[Any, ...]) -> tuple[Any, ...]:
    """Replace exact-type Offset irreps by their intra-cell fractional form."""
    return tuple(rep.fractional() if type(rep) is Offset else rep for rep in nons)


def _space_uses_fractional_offsets(space: HilbertSpace) -> bool:
    """
    True if every Offset is lattice-backed and already intra-cell fractional.

    Unit-cell Hilbert spaces store fractional lattice offsets. PointGroupOpr
    images can leave the unit cell, so D(g) folds them back before lookup.
    Generic AffineSpace offsets are never folded: coordinate range alone does
    not imply periodic boundary conditions.
    """
    saw_offset = False
    for psi in space.elements():
        for rep in psi.base:
            if type(rep) is Offset:
                saw_offset = True
                if not isinstance(rep.space, Lattice) or rep != rep.fractional():
                    return False
    return saw_offset


def spinful_transform_basis(opr: PointGroupOpr, psi: U1Basis) -> U1Span:
    r"""
    Apply \(D_{\mathrm{orb}}(g)\otimes u(g)\) to a single basis state.

    Spatial irreps that `opr` allows are transformed as usual. The
    [`Spin`][qten.phys.spin.Spin] irrep is expanded with the SU(2) factor.
    Returns a [`U1Span`][qten.symbolics.hilbert_space.U1Span] because spin
    mixing generally produces a superposition.
    """
    nons, spin, psi_coef = _split_spin_basis(psi)
    spatial_coef, new_nons = _transform_nons_spin_irreps(opr, nons)
    # Standalone transform keeps the raw geometric image. The Hilbert-space
    # representation can fold unit-cell labels for local/Γ-point matching;
    # nonzero-momentum Bloch translation phases are not represented.

    terms: list[U1Basis] = []
    for amp, spin_out in expand_spin(opr, spin):
        coef = sy.simplify(psi_coef * spatial_coef * amp)
        if coef == 0:
            continue
        terms.append(U1Basis(coef, new_nons + (spin_out,)))

    if not terms:
        raise RuntimeError(f"spinful image of {psi} vanished")

    merged: dict[U1Basis, U1Basis] = {}
    for term in terms:
        ray = term.rays()
        if ray in merged:
            prev = merged[ray]
            merged[ray] = U1Basis(sy.simplify(prev.coef + term.coef), term.base)
        else:
            merged[ray] = term
    return U1Span(tuple(merged.values()))


def spinful_hilbert_opr_repr(
    opr: PointGroupOpr,
    space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> Tensor:
    r"""
    Matrix of \(D(g)=D_{\mathrm{orb}}(g)\otimes u(g)\) on a spinful Hilbert space.

    Fast path: compute the SU(2) factor once, cache the orbital image of each
    distinct non-spin irrep tuple, and scatter numerical amplitudes into a dense
    matrix. This avoids per-basis SymPy Gram assembly used by the earlier
    prototype and is suitable for full finite-group symmetrization.

    When every [`Offset`][qten.geometries.spatials.Offset] label in `space` is
    lattice-backed and already intra-cell fractional, orbital images are
    folded with `Offset.fractional()` before lookup so primitive-lattice
    translations match the stored unit-cell basis. This is a local/Γ-point
    representation; nonzero-momentum Bloch phases require an explicit
    momentum-dependent representation and are not inserted here.
    """
    _validate_spinful_space(space)
    precision = get_precision_config()
    torch_device = device.torch_device() if device is not None else None
    n = space.dim
    data = torch.zeros(
        (n, n),
        dtype=precision.torch_complex,
        device=torch_device,
    )

    # u[out, in] in the ordered basis (Spin.up, Spin.down)
    u = torch.tensor(
        su2_numeric(opr),
        dtype=precision.torch_complex,
        device=torch_device,
    )
    spin_list = (Spin.up, Spin.down)
    spin_index = {Spin.up: 0, Spin.down: 1}

    elements = list(space.elements())
    fold_offsets = _space_uses_fractional_offsets(space)
    # Lookup: (non-spin irrep tuple, Spin) -> (basis index, basis coefficient)
    index_of: dict[tuple[tuple[Any, ...], Spin], tuple[int, complex]] = {}
    parsed: list[tuple[tuple[Any, ...], Spin, complex, int]] = []
    for psi in elements:
        nons, spin, coef = _split_spin_basis(psi)
        j = space.structure[psi]
        numeric_coef = complex(sy.N(coef))
        index_of[(nons, spin)] = (j, numeric_coef)
        parsed.append((nons, spin, numeric_coef, j))

    # Cache orbital images of unique non-spin labels
    nons_image: dict[tuple[Any, ...], tuple[complex, tuple[Any, ...]]] = {}
    for nons, _, _, _ in parsed:
        if nons in nons_image:
            continue
        spatial_coef, new_nons = _transform_nons_spin_irreps(opr, nons)
        if fold_offsets:
            new_nons = _fold_offset_irreps(new_nons)
        nons_image[nons] = (complex(sy.N(spatial_coef)), new_nons)

    for nons, spin_in, psi_coef, j in parsed:
        spatial_coef, new_nons = nons_image[nons]
        s_in = spin_index[spin_in]
        for s_out, spin_out in enumerate(spin_list):
            amp = u[s_out, s_in]
            if amp.abs().item() == 0.0:
                continue
            key = (new_nons, spin_out)
            if key not in index_of:
                raise ValueError(
                    f"Spinful image ({new_nons}, {spin_out}) is not in space "
                    f"{space}. Space is not closed under {opr}."
                )
            i, target_coef = index_of[key]
            data[i, j] += (target_coef.conjugate() * spatial_coef * psi_coef) * amp

    return Tensor(data=data, dims=(space, space))


def _ext_transform_basis(opr: PointGroupOpr, psi: U1Basis) -> U1Basis:
    if any(type(rep) is Spin for rep in psi.base):
        raise NotImplementedError(
            "External one-to-one basis maps cannot express SU(2) spin mixing. "
            "Use spinful_transform_basis / spinful_hilbert_opr_repr instead."
        )
    new_coef: sy.Expr = psi.coef
    new_base: Tuple[Any, ...] = tuple()
    for rep in psi.base:
        if type(rep) is PointGroupBasis:
            transformed = _transform_point_group_basis_direct(opr, rep)
            new_coef *= transformed.coef
            new_rep = transformed.base
        elif opr.allows(rep):
            ret = opr(rep)
            if isinstance(ret, Multiple):
                new_coef *= ret.coef
                new_rep = ret.base
            else:
                new_rep = ret
        else:
            new_rep = rep
        new_base += (new_rep,)
    return U1Basis(new_coef, new_base)


def get_direct_transform(
    opr: PointGroupOpr,
    space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> Tensor:
    r"""
    Build the external basis-mapping tensor from a Hilbert space to its transformed image.

    Unlike [`hilbert_opr_repr()`][qten.symbolics.ops.hilbert_opr_repr], this helper does not require `opr` to preserve the ray structure of
    `space`. Instead it explicitly constructs the transformed output
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] and returns a one-hot mapping matrix with dims `(space, out_space)`.

    When a basis state contains an [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] irrep, that irrep is transformed directly in the Euclidean polynomial basis.
    In particular, no eigen-phase is factored out. For example, a basis
    function `x` rotated by `C4` is mapped to `y` in the output space rather
    than left as `x` with a phase in the tensor data.

    Parameters
    ----------
    opr : PointGroupOpr
        Point-group operator used to transform basis labels.
    space : HilbertSpace
        Input Hilbert space whose ordered basis defines the source axis.
    device : Optional[Device], optional
        Device on which to allocate the returned mapping tensor.

    Returns
    -------
    Tensor
        Rank-2 tensor with dimensions `(space, out_space)` and only `1`
        numerical entries at the mapped basis positions.
    """
    transformed = {psi: _ext_transform_basis(opr, psi) for psi in space.elements()}
    out_space = space.map(lambda psi: transformed[psi])
    return mapping_matrix(space, out_space, transformed, device=device)


def _contains_point_group_basis(space: HilbertSpace) -> bool:
    return any(
        type(rep) is PointGroupBasis for psi in space.elements() for rep in psi.base
    )


def _point_group_basis_scale(
    candidate: PointGroupBasis, target: PointGroupBasis
) -> sy.Expr | None:
    if (
        candidate.axes != target.axes
        or candidate.order != target.order
        or candidate.group != target.group
        or candidate.irrep != target.irrep
        or candidate.irrep_dim != target.irrep_dim
        or candidate.copy_index != target.copy_index
    ):
        return None

    scale = None
    for candidate_entry, target_entry in zip(candidate.rep, target.rep):
        if target_entry != 0:
            entry_scale = sy.simplify(candidate_entry / target_entry)
            scale = entry_scale if scale is None else scale
            if sy.simplify(entry_scale - scale) != 0:
                return None
        elif sy.simplify(candidate_entry) != 0:
            return None

    return scale


def _canonicalize_point_group_basis(
    basis: PointGroupBasis, space: HilbertSpace
) -> Multiple[PointGroupBasis]:
    for psi in space.elements():
        for candidate in psi.base:
            if type(candidate) is not PointGroupBasis:
                continue
            scale = _point_group_basis_scale(basis, candidate)
            if scale is not None:
                return Multiple(scale, candidate)
    raise ValueError(f"Transformed basis {basis} is not represented in {space}.")


def _internal_transform_basis(
    opr: PointGroupOpr, psi: U1Basis, space: HilbertSpace
) -> U1Basis:
    if any(type(rep) is Spin for rep in psi.base):
        raise NotImplementedError(
            "Internal one-to-one basis maps cannot express SU(2) spin mixing. "
            "Use spinful_hilbert_opr_repr / point_group_column_symmetrize on "
            "spinful spaces (which build the full D(g) matrix)."
        )
    new_coef: sy.Expr = psi.coef
    new_base: Tuple[Any, ...] = tuple()
    for rep in psi.base:
        if type(rep) is PointGroupBasis:
            transformed = _transform_point_group_basis_direct(opr, rep)
            canonical = _canonicalize_point_group_basis(transformed.base, space)
            new_coef *= transformed.coef * canonical.coef
            new_rep = canonical.base
        elif opr.allows(rep):
            ret = opr(rep)
            if isinstance(ret, Multiple):
                new_coef *= ret.coef
                new_rep = ret.base
            else:
                new_rep = ret
        else:
            new_rep = rep
        new_base += (new_rep,)
    return U1Basis(sy.simplify(new_coef), new_base)


def _hilbert_opr_repr(
    opr: PointGroupOpr, space: HilbertSpace, *, device: Optional[Device] = None
) -> Tensor:
    _validate_hilbert_basis(space)
    if contains_spin(space):
        if _contains_point_group_basis(space):
            raise NotImplementedError(
                "Combined PointGroupBasis + Spin Hilbert representations "
                "are not implemented yet."
            )
        return spinful_hilbert_opr_repr(opr, space, device=device)
    if not _contains_point_group_basis(space):
        return hilbert_opr_repr(opr, space, device=device)

    ray_to_basis = {psi.rays(): psi for psi in space.elements()}
    precision = get_precision_config()
    torch_device = device.torch_device() if device is not None else None
    data = torch.zeros(
        (space.dim, space.dim),
        dtype=precision.torch_complex,
        device=torch_device,
    )

    for source in space.elements():
        transformed = _internal_transform_basis(opr, source, space)
        target = ray_to_basis.get(transformed.rays())
        if target is None:
            raise ValueError("opr does not preserve the ray structure of space.")

        i = space.structure[target]
        j = space.structure[source]
        matrix_element = sy.simplify(sy.conjugate(target.coef) * transformed.coef)
        data[i, j] += complex(sy.N(matrix_element))

    return Tensor(data=data, dims=(space, space))


def joint_point_group_basis(
    oprs: Sequence[PointGroupElement | PointGroupOpr], order: int
) -> FrozenDict[tuple[sy.Expr, ...], tuple[PointGroupBasis, ...]]:
    """
    Compute common Euclidean eigenfunctions for a commuting family of abelian operators.

    The returned table is keyed by one phase per input operator. Each value is
    the tuple of normalized [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis]
    functions spanning the simultaneous eigenspace for that joint phase sector.

    Parameters
    ----------
    oprs : Sequence[PointGroupElement | PointGroupOpr]
        Non-empty sequence of operators. Affine
        [`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr] inputs contribute
        only their linear part.
    order : int
        Homogeneous polynomial degree used for all Euclidean representations.

    Returns
    -------
    FrozenDict[tuple[sy.Expr, ...], tuple[PointGroupBasis, ...]]
        Mapping from joint phase tuple to the simultaneous eigen-basis
        functions for that sector.

    Raises
    ------
    ValueError
        If `oprs` is empty, if the operators do not share the same ordered
        axes, or if their Euclidean representations at `order` do not commute.
    """
    if not oprs:
        raise ValueError("oprs must be non-empty.")

    groups = tuple(opr.g if isinstance(opr, PointGroupOpr) else opr for opr in oprs)
    axes = groups[0].axes
    if any(g.axes != axes for g in groups[1:]):
        raise ValueError("All operators must share the same ordered axes.")

    transforms = tuple(g.euclidean_repr(order) for g in groups)
    zero = sy.zeros(transforms[0].rows, transforms[0].cols)
    for i, left in enumerate(transforms):
        for right in transforms[i + 1 :]:
            if not sy.simplify(left @ right - right @ left).equals(zero):
                raise ValueError(
                    "All operators must commute in the Euclidean representation "
                    f"of order {order}."
                )

    euclidean_basis = groups[0].euclidean_basis(order)
    ident = sy.ImmutableDenseMatrix.eye(transforms[0].rows)
    all_sector_projectors: list[list[tuple[sy.Expr, sy.ImmutableDenseMatrix]]] = []
    for g, transform in zip(groups, transforms):
        powers = [ident]
        for _ in range(1, g.group_order()):
            powers.append(sy.ImmutableDenseMatrix(sy.simplify(powers[-1] @ transform)))

        sector_projectors: list[tuple[sy.Expr, sy.ImmutableDenseMatrix]] = []
        for phase in g.basis(order):
            projector = sy.zeros(transform.rows, transform.cols)
            for k, power in enumerate(powers):
                projector += sy.simplify((phase ** (-k)) * power)
            sector_projectors.append(
                (
                    sy.simplify(phase),
                    sy.ImmutableDenseMatrix(sy.simplify(projector / g.group_order())),
                )
            )
        all_sector_projectors.append(sector_projectors)

    tbl: dict[tuple[sy.Expr, ...], tuple[PointGroupBasis, ...]] = {}
    for sector_product in product(*all_sector_projectors):
        phases = tuple(phase for phase, _ in sector_product)
        projector = ident
        for _, sector_projector in sector_product:
            projector = sy.ImmutableDenseMatrix(
                sy.simplify(sector_projector @ projector)
            )

        basis_vectors = projector.columnspace()
        if not basis_vectors:
            continue

        labels: list[PointGroupBasis] = []
        seen_reps = set()
        for vec in basis_vectors:
            rep = sy.ImmutableDenseMatrix(vec)
            if all(entry == 0 for entry in rep):
                continue
            basis = PointGroupBasis.from_rep(
                rep=rep,
                euclidean_basis=euclidean_basis,
                axes=axes,
                order=order,
                irrep=phases,
            )
            rep_key = tuple(basis.rep)
            if rep_key in seen_reps:
                continue
            seen_reps.add(rep_key)
            labels.append(basis)

        if labels:
            tbl[phases] = tuple(labels)

    return FrozenDict(tbl)


def _joint_phase_basis(
    oprs: Sequence[PointGroupOpr],
) -> dict[tuple[sy.Expr, ...], PointGroupBasis]:
    """
    Build a representative [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] for each joint phase sector.
    """
    phase_bases: dict[tuple[sy.Expr, ...], PointGroupBasis] = {}
    max_order = prod(opr.g.group_order() for opr in oprs)
    for order in range(max_order):
        table = joint_point_group_basis(oprs, order)
        for phases, bases in table.items():
            if phases in phase_bases or not bases:
                continue
            phase_bases[phases] = bases[0]
    return phase_bases


def point_group_column_symmetrize(
    opr: PointGroupOpr | FinitePointGroup,
    w: Tensor,
    full_sector: bool = False,
    *,
    fixpoint: Offset | None = None,
    rebase_fixpoint: bool = False,
) -> Tensor:
    r"""
    Symmetrize the columns of `w` by projecting each one onto symmetry sectors.

    For a finite-order abelian operator `opr` of order \(n\), each exact
    spinless symmetry sector is labeled by a phase \(\omega^n = 1\). For a
    spin-1/2 representation, this routine uses the safe common period
    \(N=2n\), because a \(2\pi\) proper spin rotation is \(-I\), and labels it by
    [`SpinfulPhaseSector`][qten.pointgroups.ops.SpinfulPhaseSector].
    This function builds the full operator representation `G` on the ambient
    Hilbert space `w.dims[0]` and applies the projector
    \(P_\omega = \frac{1}{N}\sum_{k=0}^{N-1}\omega^{-k}G^k\),

    which is the rendered form of the code-level convention
    `P_omega = (1/N) * sum_{k=0}^{N-1} omega^(-k) G^k`.
    Here \(N=n\) for spinless spaces and \(N=2n\) for spinful spaces; \(2n\)
    need not be the minimal order for operations whose proper spin factor is
    identity.

    If `opr` is a [`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup], this
    routine dispatches to the non-abelian character-projector path
    \(P^\mu = \frac{d_\mu}{|G|}\sum_{g\in G}\chi^\mu(g)^* D(g)\), using the
    packaged ordinary character table for spinless spaces or the packaged
    operation-wise projective spinor characters for spinful spaces.

    The projector is applied to each input column separately. When
    `full_sector` is `True`, every
    nonzero projected sector component is returned. When `full_sector` is
    `False`, only the dominant nonzero sector component of each input column is
    kept, so the output column count does not exceed the input count. Returned
    columns carry the corresponding [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis].

    The output column count can differ from the input one only when
    `full_sector=True`, because symmetry projection may split one approximate
    column into multiple exact sectors.

    Parameters
    ----------
    opr : PointGroupOpr | FinitePointGroup
        Symmetry descriptor. `PointGroupOpr` uses the abelian phase-sector path;
        `FinitePointGroup` uses finite-group irrep projectors.
    w : Tensor
        Rank-2 tensor whose first dimension is a
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] and whose
        columns are vectors to project.
    full_sector : bool, default False
        If `True`, return every nonzero sector component of each input column.
        If `False`, keep only the largest nonzero sector component per input
        column.

    Returns
    -------
    Tensor
        Rank-2 tensor with the same row Hilbert space and a column
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] labelled by
        symmetry-sector basis data.

    Raises
    ------
    ValueError
        If `w` is not rank 2, if `w.dims[0]` is not a `HilbertSpace`, or if
        `w.dims[1]` is neither an `IndexSpace` nor a `HilbertSpace`.
    """
    if isinstance(opr, FinitePointGroup):
        return _finite_point_group_column_symmetrize(
            opr,
            w,
            full_sector=full_sector,
            fixpoint=fixpoint,
            rebase_fixpoint=rebase_fixpoint,
        )

    if fixpoint is not None:
        opr = opr.fixpoint_at(fixpoint, rebase=rebase_fixpoint)

    row_dim, seeds = _column_symmetrize_context(w)

    g_full = _hilbert_opr_repr(opr, row_dim, device=w.device).to_device(w.device)
    spatial_order = opr.g.group_order()
    spinful = contains_spin(row_dim)
    order = 2 * spatial_order if spinful else spatial_order
    ident = eye((row_dim, row_dim)).astype(g_full.data.dtype).to_device(g_full.device)
    single_col = IndexSpace.linear(1)
    tol = _relative_tolerance(g_full.data.dtype, row_dim.dim)

    g_powers: list[Tensor] = [ident]
    for _ in range(1, order):
        g_powers.append(g_powers[-1] @ g_full)

    closure = g_powers[-1] @ g_full
    if not torch.allclose(closure.data, ident.data, rtol=0.0, atol=tol):
        raise ValueError(
            f"Operator representation does not close at expected order {order}"
        )

    sector_projectors: list[tuple[Any, Tensor]] = []
    for m in range(order):
        phase_exact = sy.simplify(sy.exp(2 * sy.pi * sy.I * m / order))
        sector_label: Any
        if spinful:
            sector_label = SpinfulPhaseSector(
                phase=phase_exact,
                spatial_order=spatial_order,
            )
        else:
            sector_label = _phase_basis(opr, phase_exact)
        phase_scalar = complex(sy.N(phase_exact))

        projector = 0 * ident
        for k, g_power in enumerate(g_powers):
            projector = projector + (phase_scalar ** (-k)) * g_power
        sector_projectors.append((sector_label, projector / order))

    projected_cols: list[Tensor] = []
    raw_labels: list[U1Basis] = []
    for j, seed in enumerate(seeds):
        col = (
            w[:, j : j + 1].clone().replace_dim(1, single_col).astype(g_full.data.dtype)
        )
        input_norm = abs(col.norm().item())
        projection_cutoff = tol * input_norm
        candidates: list[tuple[float, Tensor, U1Basis]] = []
        for sector_label, projector in sector_projectors:
            projected = projector @ col

            projected_norm = projected.norm()
            norm_value = abs(projected_norm.item())
            if norm_value <= projection_cutoff:
                continue

            candidates.append(
                (
                    norm_value,
                    projected / norm_value,
                    _attach_sector_label(seed, sector_label),
                )
            )

        if full_sector:
            for _, projected, label in candidates:
                projected_cols.append(projected)
                raw_labels.append(label)
        elif candidates:
            _, projected, label = max(candidates, key=lambda item: item[0])
            projected_cols.append(projected)
            raw_labels.append(label)

    if not projected_cols:
        return Tensor(
            data=w.data.new_empty((row_dim.dim, 0), dtype=g_full.data.dtype),
            dims=(row_dim, IndexSpace.linear(0)),
        )

    out_dim = HilbertSpace.new(_labels_with_degeneracy(raw_labels))
    return cat(projected_cols, dim=-1).replace_dim(-1, out_dim)


def joint_point_group_column_symmetrize(
    oprs: Sequence[PointGroupOpr], w: Tensor, full_sector: bool = False
) -> Tensor:
    """
    Symmetrize columns of `w` into simultaneous sectors of abelian operators.

    The operators in `oprs` are expected to commute on `w.dims[0]`. For each
    operator, this builds the same sector projectors as
    [`point_group_column_symmetrize`][qten.pointgroups.ops.point_group_column_symmetrize], then projects each column onto every joint
    sector in the Cartesian product of those sector decompositions.

    When `full_sector` is `True`, every nonzero joint-sector component is
    returned. When `False`, only the dominant nonzero joint-sector component of
    each input column is kept. Returned columns carry a representative common
    [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] for the corresponding joint phase sector.

    Parameters
    ----------
    oprs : Sequence[PointGroupOpr]
        Non-empty sequence of finite-order abelian operators. They are expected
        to commute on the row Hilbert space of `w`.
    w : Tensor
        Rank-2 tensor whose first dimension is a
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] and whose
        columns are vectors to project.
    full_sector : bool, default False
        If `True`, return every nonzero joint-sector component of each input
        column. If `False`, keep only the largest nonzero joint-sector
        component per input column.

    Returns
    -------
    Tensor
        Rank-2 tensor with the same row Hilbert space and a column
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] labelled by
        representative joint-sector basis data.

    Raises
    ------
    ValueError
        If `oprs` is empty, if `w` is not rank 2, if `w.dims[0]` is not a
        `HilbertSpace`, or if `w.dims[1]` is neither an `IndexSpace` nor a
        `HilbertSpace`.
    """
    if not oprs:
        raise ValueError("oprs must be non-empty.")
    if len(oprs) == 1:
        return point_group_column_symmetrize(oprs[0], w, full_sector=full_sector)
    row_dim, seeds = _column_symmetrize_context(w)
    if contains_spin(row_dim):
        raise NotImplementedError(
            "Joint spinful projection requires double-group compatibility; "
            "spatially commuting operations can have anticommuting spin lifts."
        )

    single_col = IndexSpace.linear(1)
    joint_sector_bases = _joint_phase_basis(oprs)
    all_sector_projectors: list[list[tuple[sy.Expr, Tensor]]] = []
    representations: list[Tensor] = []
    dtype = w.data.dtype
    device = w.device
    for opr in oprs:
        g_full = _hilbert_opr_repr(opr, row_dim, device=w.device).to_device(w.device)
        dtype = g_full.data.dtype
        device = g_full.device
        tolerance = _relative_tolerance(dtype, row_dim.dim)
        for previous in representations:
            commutator = g_full.data @ previous.data - previous.data @ g_full.data
            if not torch.allclose(
                commutator,
                torch.zeros_like(commutator),
                rtol=0.0,
                atol=tolerance,
            ):
                max_error = float(torch.max(torch.abs(commutator)).item())
                raise ValueError(
                    "Joint point-group projection requires commuting Hilbert-space "
                    f"representations; max |[D_i,D_j]|={max_error:.3e}."
                )
        representations.append(g_full)
        order = opr.g.group_order()
        ident = eye((row_dim, row_dim)).astype(dtype).to_device(device)

        g_powers: list[Tensor] = [ident]
        for _ in range(1, order):
            g_powers.append(g_powers[-1] @ g_full)
        closure = g_powers[-1] @ g_full
        if not torch.allclose(closure.data, ident.data, rtol=0.0, atol=tolerance):
            raise ValueError(
                f"Operator representation does not close at expected order {order}"
            )

        sector_projectors: list[tuple[sy.Expr, Tensor]] = []
        for m in range(order):
            phase_exact = sy.simplify(sy.exp(2 * sy.pi * sy.I * m / order))
            phase_scalar = complex(sy.N(phase_exact))

            projector = 0 * ident
            for k, g_power in enumerate(g_powers):
                projector = projector + (phase_scalar ** (-k)) * g_power
            sector_projectors.append((phase_exact, projector / order))
        all_sector_projectors.append(sector_projectors)

    projected_cols: list[Tensor] = []
    raw_labels: list[U1Basis] = []
    for j, seed in enumerate(seeds):
        col = w[:, j : j + 1].clone().replace_dim(1, single_col).astype(dtype)
        input_norm = abs(col.norm().item())
        projection_cutoff = _relative_tolerance(dtype, row_dim.dim) * input_norm
        candidates: list[tuple[float, Tensor, U1Basis]] = []
        for sector_product in product(*all_sector_projectors):
            phases = tuple(sy.simplify(phase) for phase, _ in sector_product)
            basis = joint_sector_bases.get(phases)
            if basis is None:
                continue
            projected = col
            for _, projector in sector_product:
                projected = projector @ projected

            projected_norm = projected.norm()
            norm_value = abs(projected_norm.item())
            if norm_value <= projection_cutoff:
                continue

            candidates.append(
                (
                    norm_value,
                    projected / norm_value,
                    _attach_basis_label(seed, basis),
                )
            )

        if full_sector:
            for _, projected, label in candidates:
                projected_cols.append(projected)
                raw_labels.append(label)
        elif candidates:
            _, projected, label = max(candidates, key=lambda item: item[0])
            projected_cols.append(projected)
            raw_labels.append(label)

    if not projected_cols:
        return Tensor(
            data=w.data.new_empty((row_dim.dim, 0), dtype=dtype),
            dims=(row_dim, IndexSpace.linear(0)),
        )

    out_dim = HilbertSpace.new(_labels_with_degeneracy(raw_labels))
    return cat(projected_cols, dim=-1).replace_dim(-1, out_dim)
