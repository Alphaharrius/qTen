r"""
Point-group operations on symbolic bases and tensors.

This module combines point-group transforms with QTen Hilbert spaces and
tensors. The helpers assemble \(D(g)\) (including the \(SU(2)\) factor on a
spinful space), twirl operators by conjugation, and project columns into
abelian phase sectors, ordinary finite-group irreps, or projective spinor
irreps. [`irrep_content`][qten.pointgroups.ops.irrep_content] counts those
sector labels; [`little_group`][qten.pointgroups.ops.little_group] and
[`little_group_irreps`][qten.pointgroups.ops.little_group_irreps] evaluate
the same **point-group** tables at a caller
[`KPointSet`][qten.geometries.spatials.KPointSet].
[`space_group_character_tables`][qten.pointgroups.ops.space_group_character_tables]
aligns the linear little-group χ to
[`SpaceGroup.little_group`][qten.pointgroups.spacegroup.SpaceGroup.little_group]
order.
[`space_group_irreps`][qten.pointgroups.ops.space_group_irreps] uses Seitz
operators and caller-supplied little-group characters.
[`ebr_decomposition`][qten.pointgroups.ops.ebr_decomposition] tests whether
an HSP irrep vector lies in the elementary-band-representation cone.
[`ebr_lattice_decomposition`][qten.pointgroups.ops.ebr_lattice_decomposition]
allows negative EBR copies (the EBR lattice).
[`ebr_indicator`][qten.pointgroups.ops.ebr_indicator] is the Smith cokernel
of that lattice map.
[`inversion_indicators`][qten.pointgroups.ops.inversion_indicators] is the
class-AII inversion \(\mathbb{Z}_4\times\mathbb{Z}_2^d\) from TRIM parities.
[`axis_stabilizer`][qten.pointgroups.ops.axis_stabilizer] is the real-space
analogue of the little group of a vector.

Repository usage
----------------
Use [`hilbert_repr()`][qten.pointgroups.ops.hilbert_repr] and the related
projection helpers when an existing
[`PointGroupElement`][qten.pointgroups.elements.PointGroupElement],
[`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr], or
[`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] should act on
symbolic Hilbert-space data. Group definitions live in
[`qten.pointgroups.elements`][qten.pointgroups.elements] and
[`qten.pointgroups.finite`][qten.pointgroups.finite].
"""

import warnings
from collections import OrderedDict
from itertools import product
from math import prod
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Tuple, cast

import sympy as sy
import torch
from sympy.matrices.normalforms import smith_normal_decomp

from .basis import PointGroupBasis
from .elements import (
    PointGroupElement,
    PointGroupOpr,
)
from .finite import FinitePointGroup, _matrix_key, _subgroup_from_members
from .spacegroup import SpaceGroup
from .sectors import (
    FiniteIrrepSector,
    JointSpinfulPhaseSector,
    SpinorIrrepSector,
    SpinfulPhaseSector,
    SymmetryDegeneracy,
)
from ..geometries import (
    AffineSpace,
    KPointSet,
    Lattice,
    Momentum,
    Offset,
    ReciprocalLattice,
)
from ..geometries.fourier import _bloch_key, fourier_kernel, fourier_transform
from ..linalg.tensors import Tensor, cat, eye, mapping_matrix
from ..phys.spin import Spin, as_spin, contains_spin, expand_spin, su2_numeric
from ..symbolics import (
    FuncOpr,
    HilbertSpace,
    IndexSpace,
    MomentumBlockSpace,
    MomentumSpace,
    Multiple,
    U1Basis,
    U1Span,
    brillouin_zone,
    hilbert_opr_repr,
    restructure,
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
    raise ValueError(f"Failed to find a PointGroupBasis for phase={phase}.")


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
    \[
    A_G = |G|^{-1}\sum_{g\in G} D(g)\,A\,D(g)^\dagger.
    \]
    The average commutes with every \(D(h)\). Unlike character projection of
    state vectors, conjugation averaging does not need ordinary or spinor
    characters: if a lift is replaced by \(\eta(g)u(g)\) with
    \(\eta(g)\in\{\pm 1\}\), the same factor appears in \(D(g)\) and
    \(D(g)^\dagger\) and cancels.

    The two matrix dimensions of `operator` must describe the same
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]. That space
    must be closed under every point-group operation; otherwise representation
    assembly raises `ValueError`.

    Parameters
    ----------
    group : FinitePointGroup
        Finite point group whose elements generate the average.
    operator : Tensor
        Rank-2 tensor whose two dimensions are the same ordered
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].
    fixpoint : Offset | None, optional
        Desired invariant point. When set, each group element is wrapped as
        a `PointGroupOpr` and recentered with `fixpoint_at` before `D(g)`
        is assembled.
    rebase_fixpoint : bool, default False
        Forwarded to
        [`PointGroupOpr.fixpoint_at`][qten.pointgroups.elements.PointGroupOpr.fixpoint_at]
        as `rebase`.

    Returns
    -------
    Tensor
        The conjugation-averaged operator on the same Hilbert space.

    Raises
    ------
    ValueError
        If `operator` is not a square Hilbert-space tensor, or if some
        `D(g)` is not numerically unitary.
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
    use_spinor = spinful and group.spin != "trivial"
    if use_spinor:
        table = group.spinor_table()
        irrep_table = table["irreps"]
        sector_source = str(table.get("source") or "qten-su2-principal-v1")
    else:
        table = group.ordinary_table()
        irrep_table = table["irreps"]
        sector_source = str(table.get("source") or "bilbao")
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
        if use_spinor:
            characters = group.spinor_irrep_characters_by_element(irrep_name)
            sector_label: Any = SpinorIrrepSector(
                group=group.symbol or "<anonymous>",
                irrep=irrep_name,
                dim=dim,
                source=sector_source,
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
                if use_spinor
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
        projection_cutoff = max(tol, 1e-8) * input_norm
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
    labeled = [(rep, as_spin(rep)) for rep in psi.base]
    spin_reps = [spin for _, spin in labeled if spin is not None]
    if len(spin_reps) != 1:
        raise ValueError(
            f"Expected exactly one Spin irrep in {psi}, found {len(spin_reps)}"
        )
    nons = tuple(rep for rep, spin in labeled if spin is None)
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
        (psi, sum(as_spin(rep) is not None for rep in psi.base))
        for psi in space.elements()
        if sum(as_spin(rep) is not None for rep in psi.base) != 1
    ]
    if invalid:
        psi, count = invalid[0]
        raise ValueError(
            "A spinful Hilbert space must carry exactly one Spin label on every "
            f"basis state; found {count} in {psi}"
        )

    styles: set[str] = set()
    seen_keys: dict[tuple[tuple[Any, ...], Spin], U1Basis] = {}
    for psi in space.elements():
        for rep in psi.base:
            if type(rep) is Spin:
                styles.add("spin")
            elif type(rep) is str and rep in {"up", "down"}:
                styles.add("string")
        nons, spin, _coef = _split_spin_basis(psi)
        key = (nons, spin)
        previous = seen_keys.get(key)
        if previous is not None:
            raise ValueError(
                "Spinful labels collapse to the same (orbital, spin) key; "
                f"{previous} and {psi} both map to {key}."
            )
        seen_keys[key] = psi
    if len(styles) > 1:
        raise ValueError(
            "A spinful Hilbert space cannot mix Spin objects with leftover "
            '"up"/"down" strings.'
        )


def _transform_nons_spin_irreps(
    opr: PointGroupOpr,
    nons: tuple[Any, ...],
    *,
    space: HilbertSpace | None = None,
    fold_offsets: bool = False,
) -> list[tuple[sy.Expr, tuple[Any, ...]]]:
    """Apply the orbital/spatial part of `opr` to non-spin irreps.

    A multidimensional [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis]
    image is expanded in the labels already present in `space`. Standalone
    transforms (no `space`) keep the raw geometric image as one term.
    """
    canonicalize = space is not None and _contains_point_group_basis(space)
    terms: list[tuple[sy.Expr, list[Any]]] = [(sy.Integer(1), [])]
    for rep in nons:
        expansions: list[tuple[sy.Expr, Any]]
        if type(rep) is PointGroupBasis:
            if canonicalize:
                assert space is not None
                image_rep = sy.ImmutableDenseMatrix(
                    opr.g.euclidean_repr(rep.order) @ rep.rep
                )
                expansions = _expand_point_group_image(image_rep, rep, space)
            else:
                transformed = _transform_point_group_basis_direct(opr, rep)
                expansions = [(transformed.coef, transformed.base)]
        elif opr.allows(rep):
            ret = opr(rep)
            if isinstance(ret, Multiple):
                expansions = [(ret.coef, ret.base)]
            else:
                expansions = [(sy.Integer(1), ret)]
        else:
            expansions = [(sy.Integer(1), rep)]
        terms = [
            (sy.simplify(coef * extra_coef), prefix + [extra])
            for coef, prefix in terms
            for extra_coef, extra in expansions
        ]
    result: list[tuple[sy.Expr, tuple[Any, ...]]] = []
    for coef, parts in terms:
        orbital = tuple(parts)
        if fold_offsets:
            orbital = _fold_offset_irreps(orbital)
        result.append((sy.simplify(coef), orbital))
    return result


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

    On a product label \(|\mathrm{orb},s\rangle\),
    \[
    D(g)|\mathrm{orb},s\rangle
    =\sum_{s'}u(g)_{s's}\,|g\cdot\mathrm{orb},\,s'\rangle.
    \]
    Spatial irreps that `opr` allows are transformed as usual. The
    [`Spin`][qten.phys.spin.Spin] irrep is expanded with the \(SU(2)\) factor.
    Returns a [`U1Span`][qten.symbolics.hilbert_space.U1Span] because spin
    mixing generally produces a superposition.
    """
    nons, spin, psi_coef = _split_spin_basis(psi)
    orbital_terms = _transform_nons_spin_irreps(opr, nons)
    # Standalone transform keeps the raw geometric image. The Hilbert-space
    # representation can fold unit-cell labels for local/Γ-point matching;
    # nonzero-momentum Bloch translation phases are not represented.

    terms: list[U1Basis] = []
    for spatial_coef, new_nons in orbital_terms:
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

    Columns of \(u(g)\) are ordered \((\uparrow,\downarrow)\). Fast path:
    compute the \(SU(2)\) factor once, cache the orbital image of each
    distinct non-spin irrep tuple, and scatter numerical amplitudes into a dense
    matrix. This avoids per-basis SymPy Gram assembly used by the earlier
    prototype and is suitable for full finite-group symmetrization.

    When every [`Offset`][qten.geometries.spatials.Offset] label in `space` is
    lattice-backed and already intra-cell fractional, orbital images are
    folded with `Offset.fractional()` before lookup so primitive-lattice
    translations match the stored unit-cell basis. This is a local/Γ-point
    representation; nonzero-momentum Bloch phases require an explicit
    momentum-dependent representation and are not inserted here.

    Parameters
    ----------
    opr : PointGroupOpr
        Point operation, including any affine center already set on it.
    space : HilbertSpace
        Spinful Hilbert space. Every basis state must carry exactly one
        [`Spin`][qten.phys.spin.Spin] label, and the space must be closed
        under `opr`.
    device : Optional[Device], optional
        Device for the returned tensor.

    Returns
    -------
    Tensor
        Square tensor of \(D_{\mathrm{orb}}(g)\otimes u(g)\) on `space`.
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
    nons_image: dict[tuple[Any, ...], list[tuple[complex, tuple[Any, ...]]]] = {}
    for nons, _, _, _ in parsed:
        if nons in nons_image:
            continue
        spatial_terms = _transform_nons_spin_irreps(
            opr, nons, space=space, fold_offsets=fold_offsets
        )
        nons_image[nons] = [
            (complex(sy.N(spatial_coef)), new_nons)
            for spatial_coef, new_nons in spatial_terms
        ]

    for nons, spin_in, psi_coef, j in parsed:
        s_in = spin_index[spin_in]
        for spatial_coef, new_nons in nons_image[nons]:
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
    if any(as_spin(rep) is not None for rep in psi.base):
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

    Spinful Hilbert spaces are rejected: a generic \(SU(2)\) factor maps one
    basis state to a superposition, which this one-to-one mapping cannot
    express. Use [`hilbert_repr`][qten.pointgroups.ops.hilbert_repr] instead.

    When a basis state contains a [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] irrep, that irrep is transformed directly in the Euclidean polynomial basis.
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

    Raises
    ------
    NotImplementedError
        If any basis state in `space` carries a spin-1/2 label.
    """
    transformed = {psi: _ext_transform_basis(opr, psi) for psi in space.elements()}
    out_space = space.map(lambda psi: transformed[psi])
    return mapping_matrix(space, out_space, transformed, device=device)


def _contains_point_group_basis(space: HilbertSpace) -> bool:
    return any(
        type(rep) is PointGroupBasis for psi in space.elements() for rep in psi.base
    )


def _same_point_group_sector(left: PointGroupBasis, right: PointGroupBasis) -> bool:
    return (
        left.axes == right.axes
        and left.order == right.order
        and left.group == right.group
        and left.irrep == right.irrep
        and left.irrep_dim == right.irrep_dim
        and left.copy_index == right.copy_index
    )


def _vector_scale(image: sy.Matrix, target: sy.Matrix) -> sy.Expr | None:
    if len(image) != len(target):
        return None
    scale = None
    for image_entry, target_entry in zip(image, target):
        if target_entry != 0:
            entry_scale = sy.simplify(image_entry / target_entry)
            if scale is None:
                scale = entry_scale
            elif sy.simplify(entry_scale - scale) != 0:
                return None
        elif sy.simplify(image_entry) != 0:
            return None
    return sy.Integer(0) if scale is None else scale


def _point_group_basis_scale(
    candidate: PointGroupBasis, target: PointGroupBasis
) -> sy.Expr | None:
    if not _same_point_group_sector(candidate, target):
        return None
    return _vector_scale(candidate.rep, target.rep)


def _partner_point_group_bases(
    template: PointGroupBasis, space: HilbertSpace
) -> list[PointGroupBasis]:
    partners: list[PointGroupBasis] = []
    seen: set[tuple[Any, ...]] = set()
    for psi in space.elements():
        for candidate in psi.base:
            if type(candidate) is not PointGroupBasis:
                continue
            if not _same_point_group_sector(template, candidate):
                continue
            key = tuple(candidate.rep)
            if key in seen:
                continue
            seen.add(key)
            partners.append(candidate)
    return partners


def _expand_point_group_image(
    image_rep: sy.ImmutableDenseMatrix,
    template: PointGroupBasis,
    space: HilbertSpace,
) -> list[tuple[sy.Expr, PointGroupBasis]]:
    """Expand a Euclidean image in the stored partners of `template`."""
    partners = _partner_point_group_bases(template, space)
    if not partners:
        raise ValueError(
            f"Transformed basis of {template.irrep} is not represented in {space}."
        )
    for partner in partners:
        scale = _vector_scale(image_rep, partner.rep)
        if scale is not None:
            return [(sy.simplify(scale), partner)]

    matrix = sy.Matrix.hstack(*[sy.Matrix(partner.rep) for partner in partners])
    target = sy.Matrix(image_rep)
    try:
        coeffs = matrix.solve_least_squares(target)
    except Exception as exc:
        raise ValueError(
            f"Transformed basis of {template.irrep} is not in the span of "
            f"{template.irrep} in {space}."
        ) from exc
    residual = matrix * coeffs - target
    if any(
        sy.simplify(entry) != 0 and abs(complex(sy.N(entry))) > 1e-8
        for entry in residual
    ):
        raise ValueError(
            f"Transformed basis of {template.irrep} is not in the span of "
            f"{template.irrep} in {space}."
        )
    terms = [
        (sy.simplify(coeff), partner)
        for coeff, partner in zip(coeffs, partners)
        if sy.simplify(coeff) != 0
    ]
    if not terms:
        raise ValueError(f"Transformed basis of {template.irrep} vanished in {space}.")
    return terms


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
    if any(as_spin(rep) is not None for rep in psi.base):
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


def hilbert_repr(
    opr: PointGroupOpr, space: HilbertSpace, *, device: Optional[Device] = None
) -> Tensor:
    r"""
    Assemble the Hilbert-space representation \(D(g)\) of a point operation.

    Spinless spaces use \(D(g)=D_{\mathrm{orb}}(g)\). Spinful spaces use
    \[
    D(g)\,|\mathrm{orb},s\rangle
    =\sum_{s'}u(g)_{s's}\,|g\cdot\mathrm{orb},\,s'\rangle,
    \]
    i.e. \(D(g)=D_{\mathrm{orb}}(g)\otimes u(g)\). Each basis irrep is
    transformed on its own: lattice
    [`Offset`][qten.geometries.spatials.Offset] labels may be folded back into
    the unit cell, [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis]
    polynomials are canonicalized onto labels already present in `space`, and
    [`Spin`][qten.phys.spin.Spin] is expanded by the \(SU(2)\) lift. This
    function has no `fixpoint=`; recenter `opr` with
    [`fixpoint_at`][qten.pointgroups.elements.PointGroupOpr.fixpoint_at]
    first.

    Parameters
    ----------
    opr : PointGroupOpr
        Point operation, including any affine center already set on it.
    space : HilbertSpace
        Ordered basis. Must be closed under `opr`.
    device : Optional[Device], optional
        Device for the returned tensor.

    Returns
    -------
    Tensor
        Square tensor of \(D(g)\) on `space`. On a spinful space this is
        \(D_{\mathrm{orb}}(g)\otimes u(g)\).
    """
    _validate_hilbert_basis(space)
    if contains_spin(space):
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

    fold_offsets = _space_uses_fractional_offsets(space)
    for source in space.elements():
        images = _transform_nons_spin_irreps(
            opr, source.base, space=space, fold_offsets=fold_offsets
        )
        j = space.structure[source]
        for spatial_coef, new_base in images:
            constructed = U1Basis(sy.simplify(source.coef * spatial_coef), new_base)
            target = ray_to_basis.get(constructed.rays())
            if target is None:
                raise ValueError("opr does not preserve the ray structure of space.")
            i = space.structure[target]
            matrix_element = sy.simplify(sy.conjugate(target.coef) * constructed.coef)
            data[i, j] += complex(sy.N(matrix_element))

    return Tensor(data=data, dims=(space, space))


def _unwrapped_offset_image(opr: PointGroupOpr, r: Offset) -> Offset:
    r"""Apply \(\{R\mid\tau\}\) without lattice wrapping.

    Bloch wrap phases need the unwrapped image \(Rr+\tau\). A lattice-backed
    [`Offset`][qten.geometries.spatials.Offset] folds by the supercell in
    `__post_init__`, which would drop the primitive wrap vector.
    """
    affine = AffineSpace(basis=r.space.basis)
    free = (
        r
        if type(r.space) is AffineSpace
        else Offset(sy.ImmutableDenseMatrix(r.rep), affine)
    )
    return opr.rebase(affine) @ free


def _hilbert_opr_repr(
    opr: PointGroupOpr, space: HilbertSpace, *, device: Optional[Device] = None
) -> Tensor:
    return hilbert_repr(opr, space, device=device)


def _point_group_band_blocks(
    opr: PointGroupOpr,
    kspace: MomentumSpace,
    mapped_kspace: MomentumSpace,
    space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> Tensor:
    r"""Assemble \(D(g,k)\) on a unit-cell Hilbert space.

    The leading axis follows `kspace`. `mapped_kspace` must be the same length,
    with entry \(i\) equal to the wrapped image \(g k_i\). On a spinful space
    this is \(e^{-i(gk)\cdot(Rr+\tau)}\,D(g,\Gamma)\), with \(Rr+\tau\)
    not folded by the supercell so the primitive wrap vector stays in the
    phase. Spinless spaces use the transformed (possibly unwrapped) basis and
    the same Fourier kernel.
    """
    if mapped_kspace.dim != kspace.dim:
        raise ValueError(
            "mapped_kspace must have one image momentum for each source momentum."
        )

    if contains_spin(space):
        gamma_repr = hilbert_repr(opr, space, device=device)
        images: list[Offset] = []
        has_offset = True
        for psi in space.elements():
            try:
                images.append(
                    _unwrapped_offset_image(
                        opr, cast(Offset, cast(U1Basis, psi).irrep_of(Offset))
                    )
                )
            except ValueError:
                has_offset = False
                break
        if not has_offset or not images:
            data = gamma_repr.data.unsqueeze(0).expand(kspace.dim, -1, -1).contiguous()
        else:
            phases = fourier_kernel(
                mapped_kspace.elements(), tuple(images), device=device
            )
            data = gamma_repr.data.unsqueeze(0) * phases.unsqueeze(1)
        return Tensor(data=data, dims=(kspace, space, space))

    fractional = FuncOpr(Offset, Offset.fractional)
    raw_space = HilbertSpace.new(
        _internal_transform_basis(opr, psi, space) for psi in space.elements()
    )
    new_space = cast(HilbertSpace, fractional @ raw_space)
    if not space.same_rays(new_space):
        raise ValueError(
            f"Hilbert space {space} is not closed under the transform {opr}!"
        )
    transformed_fourier = fourier_transform(
        mapped_kspace, new_space, raw_space, device=device
    ).replace_dim(2, space)
    home_transform = cast(
        Tensor, space.cross_gram(new_space, device=device)
    ).replace_dim(1, new_space)
    transform = cast(Tensor, home_transform @ transformed_fourier)
    return Tensor(data=transform.data, dims=(kspace, space, space))


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

    For a finite-order abelian operator `opr` of spatial order \(n\), each
    exact spinless sector is labeled by a root of unity \(\zeta^n=1\). For
    spin-1/2 the safe common period is \(N=2n\), because a \(2\pi\) proper
    spin rotation is \(-I\), and the sector is a
    [`SpinfulPhaseSector`][qten.pointgroups.sectors.SpinfulPhaseSector].
    The projector on \(G=D(g)\) is
    \[
    P_\zeta=\frac{1}{N}\sum_{k=0}^{N-1}\zeta^{-k}G^k,\qquad\zeta^N=1,
    \]
    with \(N=n\) spinless and \(N=2n\) spinful. The period \(2n\) need not
    be minimal when the proper spin factor is already the identity.

    If `opr` is a [`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup],
    this routine uses
    \[
    P^\mu=\frac{d_\mu}{|G|}\sum_{g\in G}\chi^\mu(g)^*D(g).
    \]
    Spinless spaces use ordinary (linear) \(\chi\). Spaces that already
    contain [`Spin`][qten.phys.spin.Spin] use the group's \(SU(2)\) section and
    element-wise projective \(\chi\), unless the group was defined with
    `spin="trivial"`. A cyclic
    [`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr] is the
    abelian special case: one-dimensional characters \(\zeta^k\).

    The projector is applied to each input column separately. When
    `full_sector` is `True`, every
    nonzero projected sector component is returned. When `full_sector` is
    `False`, only the dominant nonzero sector component of each input column is
    kept, so the output column count does not exceed the input count.
    Returned columns carry a sector label:
    [`FiniteIrrepSector`][qten.pointgroups.sectors.FiniteIrrepSector] for ordinary
    finite-group irreps,
    [`SpinorIrrepSector`][qten.pointgroups.sectors.SpinorIrrepSector] for
    projective spinor irreps, or
    [`SpinfulPhaseSector`][qten.pointgroups.sectors.SpinfulPhaseSector] /
    [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] for abelian
    phase sectors.

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
    fixpoint : Offset | None, optional
        Desired invariant point. When set, each group element is wrapped as
        a `PointGroupOpr` and recentered with `fixpoint_at` before `D(g)`
        is assembled.
    rebase_fixpoint : bool, default False
        Forwarded to
        [`PointGroupOpr.fixpoint_at`][qten.pointgroups.elements.PointGroupOpr.fixpoint_at]
        as `rebase`.

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


def _require_joint_oprs_in_group(
    oprs: Sequence[PointGroupOpr], group: FinitePointGroup
) -> None:
    """Require every operator's linear part to be an element of `group`."""
    keys = {_matrix_key(element.irrep) for element in group.elements()}
    for opr in oprs:
        if opr.g.axes != group.axes:
            raise ValueError(
                f"Joint operators must use the same ordered axes as {group.symbol}."
            )
        if getattr(opr.g, "spin", group.spin) != group.spin:
            raise ValueError(
                f"Joint operators must use the same spin policy as {group.symbol}."
            )
        if _matrix_key(opr.g.irrep) not in keys:
            raise ValueError(
                f"Joint operator {opr.g} is not an element of the already "
                f"defined point group {group.symbol}."
            )


def joint_point_group_column_symmetrize(
    oprs: Sequence[PointGroupOpr],
    w: Tensor,
    full_sector: bool = False,
    *,
    group: FinitePointGroup | None = None,
) -> Tensor:
    r"""
    Symmetrize columns of `w` into simultaneous sectors of abelian operators.

    The operators in `oprs` are expected to commute on `w.dims[0]`. For each
    operator this builds the same \(P_\zeta\) as
    [`point_group_column_symmetrize`][qten.pointgroups.ops.point_group_column_symmetrize].
    A joint sector is the product projector
    \(P_{\zeta_1}\cdots P_{\zeta_m}\) over the Cartesian product of those
    roots of unity.

    When `full_sector` is `True`, every nonzero joint-sector component is
    returned. When `False`, only the dominant nonzero joint-sector component of
    each input column is kept. Spinless columns carry a representative common
    [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] for the
    corresponding joint phase sector. Spinful columns carry one
    [`JointSpinfulPhaseSector`][qten.pointgroups.sectors.JointSpinfulPhaseSector].

    This helper has no `fixpoint=` argument. Center each operator with
    [`fixpoint_at`][qten.pointgroups.elements.PointGroupOpr.fixpoint_at]
    before calling it.

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
    group : FinitePointGroup | None, optional
        Parent finite point group. Required when `w` is spinful: every operator
        must be an element of this group, so the joint family stays inside one
        double cover.

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
    spinful = contains_spin(row_dim)
    if group is not None:
        _require_joint_oprs_in_group(oprs, group)
    elif spinful:
        raise ValueError(
            "Joint spinful projection requires group= the FinitePointGroup "
            "that contains every operator."
        )

    single_col = IndexSpace.linear(1)
    joint_sector_bases = None if spinful else _joint_phase_basis(oprs)
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
        spatial_order = opr.g.group_order()
        order = 2 * spatial_order if spinful else spatial_order
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
            if spinful:
                label = _attach_sector_label(
                    seed,
                    JointSpinfulPhaseSector(
                        phases=phases,
                        spatial_orders=tuple(opr.g.group_order() for opr in oprs),
                    ),
                )
            else:
                assert joint_sector_bases is not None
                basis = joint_sector_bases.get(phases)
                if basis is None:
                    continue
                label = _attach_basis_label(seed, basis)
            projected = col
            for _, projector in sector_product:
                projected = projector @ projected

            projected_norm = projected.norm()
            norm_value = abs(projected_norm.item())
            if norm_value <= projection_cutoff:
                continue

            candidates.append((norm_value, projected / norm_value, label))

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


def _sector_label(psi: U1Basis) -> FiniteIrrepSector | SpinorIrrepSector | None:
    for rep in psi.base:
        if type(rep) is SpinorIrrepSector or type(rep) is FiniteIrrepSector:
            return rep
    return None


def _bilbao_label(
    group: FinitePointGroup, sector: FiniteIrrepSector | SpinorIrrepSector
) -> str:
    if type(sector) is SpinorIrrepSector:
        table = group.spinor_table()
    else:
        table = group.ordinary_table()
    row = table["irreps"].get(sector.irrep)
    if not isinstance(row, dict):
        return sector.irrep
    label = row.get("bilbao_label")
    return str(label) if label else sector.irrep


def _irrep_row_label(row: dict[str, Any], name: str) -> str:
    label = row.get("bilbao_label")
    return str(label) if label else name


def irrep_content(
    w: Tensor,
    group: FinitePointGroup,
    *,
    fixpoint: Offset | None = None,
    rebase_fixpoint: bool = False,
) -> dict[str, dict[str, int]]:
    r"""Count site-symmetry irreps on the columns of `w`.

    Results are keyed by the packaged `bilbao_label` (Koster \(\Gamma\) names
    such as \(\Gamma_6\), \(\Gamma_8\)).

    With `fixpoint` given, or when some column carries no sector label, the
    count is exact: \(D(g)\) is assembled about `fixpoint` and
    \(n_\mu=|G|^{-1}\sum_g\chi_\mu(g)^*\,\mathrm{Tr}(W^\dagger D(g)W)\) on an
    orthonormal basis \(W\) of the column span. A span that is not invariant
    under the group gives non-integer \(n_\mu\) and a warning.

    Otherwise this only reads the
    [`SpinorIrrepSector`][qten.pointgroups.sectors.SpinorIrrepSector] /
    [`FiniteIrrepSector`][qten.pointgroups.sectors.FiniteIrrepSector] tags left
    by
    [`point_group_column_symmetrize`][qten.pointgroups.ops.point_group_column_symmetrize].
    Each tagged column is one vector of an isotypic block, so a label whose
    column count is not a multiple of the irrep dimension warns and its
    `multiplicity` is rounded down.

    Parameters
    ----------
    w : Tensor
        Rank-2 tensor of columns, typically local trial orbitals.
    group : FinitePointGroup
        Group whose character table supplies `bilbao_label`. Tagged columns
        must have been projected with a group of the same symbol.
    fixpoint : Offset | None, optional
        Affine center of \(D(g)\). Selects the exact trace count.
    rebase_fixpoint : bool, default False
        Forwarded with `fixpoint`.

    Returns
    -------
    dict[str, dict[str, int]]
        Mapping from Bilbao/Koster label to `{"dim", "columns", "multiplicity"}`.
        `columns` is the number of tagged columns, or `multiplicity * dim`
        for the trace count.
    """
    if w.rank() != 2:
        raise ValueError("irrep_content requires a rank-2 tensor of columns.")
    col_dim = w.dims[1]
    tagged = isinstance(col_dim, HilbertSpace) and all(
        _sector_label(cast(U1Basis, psi)) is not None for psi in col_dim.elements()
    )
    if fixpoint is not None or not tagged:
        return _traced_irrep_content(
            w, group, fixpoint=fixpoint, rebase_fixpoint=rebase_fixpoint
        )
    assert isinstance(col_dim, HilbertSpace)

    group_name = group.symbol or "<anonymous>"
    columns: dict[str, int] = {}
    dimensions: dict[str, int] = {}
    for psi in col_dim.elements():
        sector = cast(FiniteIrrepSector | SpinorIrrepSector, _sector_label(psi))
        if sector.group != group_name:
            raise ValueError(
                f"Column is labelled by point group {sector.group!r}, "
                f"not {group_name!r}."
            )
        label = _bilbao_label(group, sector)
        columns[label] = columns.get(label, 0) + 1
        dimensions[label] = int(sector.dim)

    incomplete = {
        label: count
        for label, count in columns.items()
        if dimensions[label] and count % dimensions[label]
    }
    if incomplete:
        detail = ", ".join(
            f"{label}: {count} columns of dim {dimensions[label]}"
            for label, count in incomplete.items()
        )
        warnings.warn(
            "Labelled columns do not form whole irrep copies; pass fixpoint= "
            f"for an exact trace count. {detail}",
            stacklevel=2,
        )
    return {
        label: {
            "dim": dimensions[label],
            "columns": count,
            "multiplicity": count // dimensions[label] if dimensions[label] else count,
        }
        for label, count in columns.items()
    }


def _traced_irrep_content(
    w: Tensor,
    group: FinitePointGroup,
    *,
    fixpoint: Offset | None,
    rebase_fixpoint: bool,
) -> dict[str, dict[str, int]]:
    row_dim = w.dims[0]
    if not isinstance(row_dim, HilbertSpace):
        raise ValueError("w.dims[0] must be a HilbertSpace.")
    representations: list[Tensor] = []
    for element in group.elements():
        opr = PointGroupOpr(element)
        if fixpoint is not None:
            opr = opr.fixpoint_at(fixpoint, rebase=rebase_fixpoint)
        representations.append(
            _hilbert_opr_repr(opr, row_dim, device=w.device).to_device(w.device)
        )
    data = w.data.to(dtype=representations[0].data.dtype)
    multiplicities = _character_multiplicities(data, group, representations)

    use_spinor = contains_spin(row_dim) and group.spin != "trivial"
    table = group.spinor_table() if use_spinor else group.ordinary_table()
    dimensions = {
        _irrep_row_label(row, name): int(row["dim"])
        for name, row in table["irreps"].items()
    }
    return {
        label: {
            "dim": dimensions[label],
            "columns": count * dimensions[label],
            "multiplicity": count,
        }
        for label, count in multiplicities.items()
    }


def _same_momentum(left: Momentum, right: Momentum) -> bool:
    if left.space != right.space:
        return False
    diff = sy.Matrix(left.fractional().rep - right.fractional().rep)
    return all(sy.simplify(entry) == 0 for entry in diff)


def _momentum_rep_key(k: Momentum) -> tuple[Any, ...]:
    frac = k.fractional()
    entries = []
    for j in range(frac.space.dim):
        value = sy.nsimplify(sy.simplify(frac.rep[j, 0]))
        entries.append(0 if value == 0 else value)
    return tuple(entries)


def little_group(group: FinitePointGroup, k: Momentum) -> FinitePointGroup:
    r"""Return the little group \(\{g\in G: R(g)k \equiv k\}\) of `k`.

    Stabilizers are decided on the reciprocal lattice of `k`. If every
    element of `group` fixes `k`, `group` itself is returned so packaged
    character tables stay attached. Otherwise a new
    [`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] is generated
    from the stabilizing elements. A unique packaged crystallographic type
    attaches its ordinary Bilbao table; spinor χ is computed from the live
    \(SU(2)\) section and labelled with Koster \(\Gamma\) names.

    Parameters
    ----------
    group : FinitePointGroup
        Crystal or site point group acting on `k`.
    k : Momentum
        Crystal momentum, typically one point of a
        [`KPointSet`][qten.geometries.spatials.KPointSet].

    Returns
    -------
    FinitePointGroup
        The little group at `k`, with the same `axes` and `spin` as `group`.
    """
    target = k.fractional()
    members = [
        element
        for element in group.elements()
        if _same_momentum(cast(Momentum, PointGroupOpr(element) @ k), target)
    ]
    if not members:
        raise ValueError(f"No little-group elements stabilize {k}.")
    return _subgroup_from_members(group, members)


def _unit_cell_space(region: HilbertSpace) -> HilbertSpace:
    """Fold a localized patch onto the unit-cell Bloch basis.

    [`fourier_transform`][qten.geometries.fourier.fourier_transform] matches
    region states to Bloch states by
    [`_bloch_key`][qten.geometries.fourier._bloch_key] (fractional offset plus
    internal labels). Uniqueness must use that key, not `rays()`: two
    periodic images of the same orbital are distinct rays but one Bloch
    orbital. Concatenated Wannier seeds from two neighborhoods hit this.
    """
    unique: OrderedDict[Any, U1Basis] = OrderedDict()
    for psi in region.elements():
        state = cast(U1Basis, psi)
        try:
            key = _bloch_key(state)
        except ValueError:
            ray = state.rays()
            if ray not in unique:
                unique[ray] = ray
            continue
        if key in unique:
            continue
        try:
            unique[key] = state.replace(state.irrep_of(Offset).fractional())
        except ValueError:
            unique[key] = state.rays()
    if not unique:
        return region
    return HilbertSpace.new(unique.values())


def axis_stabilizer(
    group: FinitePointGroup,
    axis: Sequence[float] | sy.Matrix,
    *,
    oriented: bool = False,
) -> FinitePointGroup:
    r"""Return the subgroup that preserves `axis` as a line (or a ray).

    This is the real-space analogue of
    [`little_group`][qten.pointgroups.ops.little_group]: keep \(g\) with
    \(R(g)v \parallel v\). By default antiparallel images are kept
    (`oriented=False`). `oriented=True` keeps only \(R(g)v=+v\).

    Parameters
    ----------
    group : FinitePointGroup
        Parent point group. Spatial dimension is `len(group.axes)`.
    axis : sequence of float or sympy.Matrix
        Vector in the group's spatial frame (the same Cartesian frame as
        `axis=` on [`pointgroup`][qten.pointgroups.pointgroup]).
    oriented : bool, default False
        If true, keep only elements that preserve the direction of `axis`.

    Returns
    -------
    FinitePointGroup
        The stabilizer. The parent group is returned unchanged when every
        element already preserves `axis`.
    """
    return group.stabilizer(axis, oriented=oriented)


def _momentum_space_from_kpoints(kpts: KPointSet) -> MomentumSpace:
    structure: OrderedDict[Momentum, int] = OrderedDict()
    for k in kpts.points.values():
        key = k.fractional()
        if key not in structure:
            structure[key] = len(structure)
    return MomentumSpace(structure=restructure(structure))


def _reciprocal_of_hilbert(space: HilbertSpace) -> ReciprocalLattice | None:
    found: Lattice | None = None
    for psi in space.elements():
        try:
            offset = cast(U1Basis, psi).irrep_of(Offset)
        except ValueError:
            continue
        if not isinstance(offset.space, Lattice):
            continue
        if found is None:
            found = offset.space
        elif found != offset.space:
            return None
    return None if found is None else found.dual


def _complete_brillouin_zone(k_space: MomentumSpace) -> bool:
    if not k_space.elements():
        return False
    recip = k_space.elements()[0].space
    if not isinstance(recip, ReciprocalLattice):
        return False
    complete = brillouin_zone(recip)
    if k_space.dim != complete.dim:
        return False
    return {_momentum_rep_key(k) for k in k_space.elements()} == {
        _momentum_rep_key(k) for k in complete.elements()
    }


def _interpolate_mesh_to_kpoints(tensor: Tensor, kpts: KPointSet) -> Tensor:
    """Sample or Fourier-interpolate a complete-BZ tensor onto `kpts`."""
    k_space = cast(MomentumSpace, tensor.dims[0])
    recip = k_space.elements()[0].space
    rebased = kpts.rebase(cast(ReciprocalLattice, recip))
    target_space = _momentum_space_from_kpoints(rebased)

    mesh_index = {_momentum_rep_key(k): i for i, k in enumerate(k_space.elements())}
    gather: list[int] = []
    missing = False
    for k in target_space.elements():
        idx = mesh_index.get(_momentum_rep_key(k))
        if idx is None:
            missing = True
            break
        gather.append(idx)
    if not missing:
        data = tensor.data[
            torch.tensor(gather, dtype=torch.long, device=tensor.data.device)
        ]
        return Tensor(data=data, dims=(target_space, tensor.dims[1], tensor.dims[2]))

    if not _complete_brillouin_zone(k_space):
        raise ValueError(
            "little_group_irreps cannot interpolate from an incomplete momentum "
            "mesh. Fourier localized seeds, or pass a complete Brillouin zone."
        )

    lattice = recip.dual
    dim = lattice.dim
    boundary = lattice.boundaries.basis
    boundary_inv = boundary.inv()
    centered_reps = []
    for rep in lattice.boundaries.representatives():
        coeff = boundary_inv @ rep
        shift = sy.ImmutableDenseMatrix(
            [sy.floor(coeff[i, 0] + sy.Rational(1, 2)) for i in range(dim)]
        )
        centered_reps.append(sy.ImmutableDenseMatrix(rep - boundary @ shift))

    real_dtype = (
        torch.float32
        if tensor.data.dtype
        in (torch.float16, torch.bfloat16, torch.float32, torch.complex64)
        else torch.float64
    )
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    device = tensor.data.device
    old_k = torch.tensor(
        [[float(k.rep[i, 0]) for i in range(dim)] for k in k_space.elements()],
        dtype=real_dtype,
        device=device,
    )
    new_k = torch.tensor(
        [[float(k.rep[i, 0]) for i in range(dim)] for k in target_space.elements()],
        dtype=real_dtype,
        device=device,
    )
    translations = torch.tensor(
        [[float(rep[i, 0]) for i in range(dim)] for rep in centered_reps],
        dtype=real_dtype,
        device=device,
    )
    old_kernel = torch.exp(
        (-2j * torch.pi * (old_k @ translations.T)).to(complex_dtype)
    )
    new_kernel = torch.exp(
        (-2j * torch.pi * (new_k @ translations.T)).to(complex_dtype)
    )
    weights = (new_kernel @ old_kernel.conj().T).real / k_space.dim
    interpolated = torch.einsum(
        "nk,k...->n...", weights.to(tensor.data.dtype), tensor.data
    )
    return Tensor(
        data=interpolated, dims=(target_space, tensor.dims[1], tensor.dims[2])
    )


def _region_fourier_transform(
    k_space: MomentumSpace,
    bloch: HilbertSpace,
    region: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> Tensor:
    r"""`fourier_transform` with region cells unwrapped to one periodic image.

    Lattice-backed offsets are folded onto the periodic boundary, so a site one
    cell to the left is stored at cell \(N-1\). The kernel
    \(e^{-2\pi i\kappa\cdot n}\) then differs from the unwrapped value by
    \(e^{-2\pi i\kappa\cdot N}\), which is \(1\) only on the boundary mesh.
    High-symmetry momenta such as \(\kappa=1/2\) are off an odd mesh, so each
    cell is replaced by its image nearest the cell of the first region site.
    The region must fit inside half the periodic boundary around that cell.
    """
    transform = fourier_transform(k_space, bloch, region, device=device)
    offsets = [cast(U1Basis, psi).irrep_of(Offset) for psi in region.elements()]
    if not offsets or not k_space.elements():
        return transform
    lattice = offsets[0].space
    if not isinstance(lattice, Lattice) or any(r.space != lattice for r in offsets):
        return transform

    dim = lattice.dim
    boundary = lattice.boundaries.basis
    boundary_inv = boundary.inv()
    cells = [sy.ImmutableDenseMatrix((r - r.fractional()).rep) for r in offsets]
    reference = cells[0]
    wraps = []
    for cell in cells:
        coeff = boundary_inv @ (cell - reference)
        wrap = sy.ImmutableDenseMatrix(
            [sy.floor(coeff[i, 0] + sy.Rational(1, 2)) for i in range(dim)]
        )
        wraps.append(boundary @ wrap)
    if all(wrap.is_zero_matrix for wrap in wraps):
        return transform

    real_dtype = transform.data.real.dtype
    kappa = torch.tensor(
        [[float(k.rep[i, 0]) for i in range(dim)] for k in k_space.elements()],
        dtype=real_dtype,
        device=transform.data.device,
    )
    shifts = torch.tensor(
        [[float(wrap[i, 0]) for i in range(dim)] for wrap in wraps],
        dtype=real_dtype,
        device=transform.data.device,
    )
    # exp(-2πiκ·(n - Bw)) = exp(-2πiκ·n) · exp(+2πiκ·Bw)
    correction = torch.exp(
        (2j * torch.pi * (kappa @ shifts.T)).to(transform.data.dtype)
    )
    data = transform.data * correction.unsqueeze(1)
    return Tensor(data=data, dims=transform.dims)


def _states_on_kpoints(states: Tensor, kpts: KPointSet) -> tuple[Tensor, HilbertSpace]:
    """Return rank-3 states with dims `(KPoint MomentumSpace, Bloch, columns)`."""
    if states.rank() == 2:
        region = states.dims[0]
        if not isinstance(region, HilbertSpace):
            raise TypeError("Rank-2 states must have a HilbertSpace row axis.")
        bloch = _unit_cell_space(region)
        recip = _reciprocal_of_hilbert(bloch)
        rebased = kpts.rebase(recip) if recip is not None else kpts
        k_space = _momentum_space_from_kpoints(rebased)
        crystal = (
            _region_fourier_transform(k_space, bloch, region, device=states.device)
            @ states
        )
        return cast(Tensor, crystal), bloch

    if states.rank() != 3:
        raise ValueError(
            "little_group_irreps expects rank-2 localized columns or rank-3 "
            "momentum-resolved states."
        )

    leading = states.dims[0]
    bloch = states.dims[1]
    if not isinstance(bloch, HilbertSpace):
        raise TypeError("Rank-3 states must have a HilbertSpace band axis.")
    mesh = states
    if isinstance(leading, MomentumBlockSpace):
        left = leading.left()
        if left.dim != leading.dim:
            raise ValueError(
                "little_group_irreps needs one momentum-block pair per sector."
            )
        mesh = Tensor(data=states.data, dims=(left, bloch, states.dims[2]))
        leading = left
    if not isinstance(leading, MomentumSpace):
        raise TypeError(
            "Rank-3 states must start with a MomentumSpace or MomentumBlockSpace."
        )
    return _interpolate_mesh_to_kpoints(mesh, kpts), bloch


def _orthonormal_columns(data: torch.Tensor) -> torch.Tensor:
    if data.ndim != 2 or data.shape[1] == 0:
        return data
    norms = torch.linalg.vector_norm(data, dim=0)
    scale = norms.amax().clamp_min(1.0)
    tol = torch.finfo(data.real.dtype).eps ** 0.5 * scale * max(data.shape[0], 1)
    keep = norms > tol
    if not bool(keep.any()):
        return data.new_zeros((data.shape[0], 0))
    packed = data[:, keep]
    q_data, _ = torch.linalg.qr(packed, mode="reduced")
    return q_data


_MULTIPLICITY_TOLERANCE = 1e-2


def _project_characters(
    columns: torch.Tensor,
    representations: Sequence[Tensor],
    irreps: Mapping[str, Sequence[complex]],
) -> dict[str, int]:
    order = len(representations)
    if order == 0:
        return {}
    for name, characters in irreps.items():
        if len(characters) != order:
            raise ValueError(
                f"Irrep {name!r} has {len(characters)} characters, "
                f"expected one per little-group element ({order})."
            )
    active = _orthonormal_columns(columns)
    if active.shape[1] == 0:
        return {}

    traces = []
    for rep in representations:
        block = active.conj().T @ (rep.data @ active)
        traces.append(complex(block.trace().item()))

    content: dict[str, int] = {}
    off_integer: dict[str, complex] = {}
    for name, characters in irreps.items():
        overlap = sum(
            complex(character).conjugate() * trace
            for character, trace in zip(characters, traces)
        )
        multiplicity = overlap / order
        rounded = int(round(multiplicity.real))
        if abs(multiplicity - rounded) > _MULTIPLICITY_TOLERANCE:
            off_integer[name] = multiplicity
        content[name] = rounded
    if off_integer:
        detail = ", ".join(
            f"{name}={value.real:.4g}{value.imag:+.2g}j"
            for name, value in off_integer.items()
        )
        warnings.warn(
            "Irrep multiplicities are not integers; the subspace is not "
            "invariant under these operations, or the character table does "
            f"not match D(g). Rounded values are unreliable: {detail}",
            stacklevel=3,
        )
    return {label: value for label, value in content.items() if value}


def _character_multiplicities(
    columns: torch.Tensor,
    group: FinitePointGroup,
    representations: Sequence[Tensor],
) -> dict[str, int]:
    use_spinor = (
        contains_spin(cast(HilbertSpace, representations[0].dims[0]))
        and group.spin != "trivial"
    )
    if use_spinor:
        table = group.spinor_table()
    else:
        table = group.ordinary_table()
    irrep_table = table["irreps"]
    if len(representations) != group.order:
        raise ValueError(
            "little-group representations must be one D(g,k) per group element."
        )
    labeled: dict[str, Sequence[complex]] = {}
    for name, row in irrep_table.items():
        if use_spinor:
            characters = group.spinor_irrep_characters_by_element(name)
        else:
            characters = tuple(
                complex(sy.N(value))
                for value in group.irrep_characters_by_element(name)
            )
        labeled[_irrep_row_label(row, name)] = characters
    return _project_characters(columns, representations, labeled)


def little_group_irreps(
    states: Tensor,
    group: FinitePointGroup,
    kpts: KPointSet,
    *,
    fixpoint: Offset | None = None,
    rebase_fixpoint: bool = True,
) -> dict[str, dict[str, int]]:
    r"""Decompose `states` into little-group irreps at each point of `kpts`.

    Rank-2 localized columns are Fourier-transformed to the exact HSP
    momenta. Rank-3 Wannier/Bloch states are sampled on those momenta, or
    Fourier-interpolated from a complete Brillouin-zone mesh. At each \(k\),
    characters of \(D(g,k)\) on the subspace are projected onto the **little
    group** of \(k\), not onto the parent group.

    Parameters
    ----------
    states : Tensor
        Rank-2 local columns `(HilbertSpace, columns)` or rank-3
        momentum-resolved states `(MomentumSpace, HilbertSpace, columns)`.
    group : FinitePointGroup
        Parent point group. Little groups are subgroups of this group.
    kpts : KPointSet
        Named high-symmetry momenta. They are rebased onto the states'
        reciprocal lattice when that lattice can be inferred.
    fixpoint : Offset | None, optional
        Affine center for \(D(g)\), typically the crystal inversion center
        rather than a Wannier-site origin.
    rebase_fixpoint : bool, default True
        Forwarded to
        [`PointGroupOpr.fixpoint_at`][qten.pointgroups.elements.PointGroupOpr.fixpoint_at].

    Returns
    -------
    dict[str, dict[str, int]]
        `{hsp_name: {bilbao_label: multiplicity}}` with zero-multiplicity
        irreps omitted.

    Notes
    -----
    `group` must map the Bloch unit-cell Hilbert space to itself. A linear
    point operation about `fixpoint` does that only when it is a lattice
    automorphism of that cell. Affine operations that are not of the form
    \(\tau=(I-R)c\) need
    [`space_group_irreps`][qten.pointgroups.ops.space_group_irreps].
    """
    crystal, bloch = _states_on_kpoints(states, kpts)
    k_space = cast(MomentumSpace, crystal.dims[0])
    k_index = {_momentum_rep_key(k): i for i, k in enumerate(k_space.elements())}
    recip = k_space.elements()[0].space if k_space.elements() else None
    rebased = (
        kpts.rebase(cast(ReciprocalLattice, recip))
        if isinstance(recip, ReciprocalLattice)
        else kpts
    )

    result: dict[str, dict[str, int]] = {}
    cache: dict[tuple[Any, ...], dict[str, int]] = {}
    for name, raw_k in rebased.points.items():
        k = raw_k.fractional()
        key = _momentum_rep_key(k)
        if key in cache:
            result[name] = cache[key]
            continue
        idx = k_index.get(key)
        if idx is None:
            raise ValueError(f"High-symmetry point '{name}' is missing after sampling.")
        g_k = little_group(group, k)
        one_k = MomentumSpace(structure=OrderedDict(((k, 0),)))
        representations: list[Tensor] = []
        for element in g_k.elements():
            opr = PointGroupOpr(element)
            if fixpoint is not None:
                opr = opr.fixpoint_at(fixpoint, rebase=rebase_fixpoint)
            mapped = cast(Momentum, opr @ k).fractional()
            mapped_k = MomentumSpace(structure=OrderedDict(((mapped, 0),)))
            blocks = _point_group_band_blocks(
                opr, one_k, mapped_k, bloch, device=states.device
            )
            representations.append(Tensor(data=blocks.data[0], dims=(bloch, bloch)))
        content = _character_multiplicities(crystal.data[idx], g_k, representations)
        cache[key] = content
        result[name] = content
    return result


def _linear_rotation_key(element: PointGroupElement) -> tuple[float, ...]:
    matrix = element.rotation3 if element.rotation3 is not None else element.irrep
    return _matrix_key(matrix)


def space_group_character_tables(
    point_group: FinitePointGroup,
    space_group: SpaceGroup,
    kpts: KPointSet,
    *,
    spinor: bool | None = None,
) -> dict[str, dict[str, tuple[complex, ...]]]:
    r"""Return little-group χ in `space_group.little_group(k)` order.

    Each HSP uses the **linear little group**
    [`little_group`][qten.pointgroups.ops.little_group]`(point_group, k)`,
    not the parent group's irreps. Rows are Koster `bilbao_label` keys.
    Characters are permuted to the Seitz order required by
    [`space_group_irreps`][qten.pointgroups.ops.space_group_irreps].

    These are point-group tables of the linear parts. Nonsymmorphic
    space-group labels still need caller-supplied χ.

    Parameters
    ----------
    point_group : FinitePointGroup
        Parent whose little groups supply the character tables.
    space_group : SpaceGroup
        Seitz representatives; must have the same linear parts as
        `point_group` at each \(k\).
    kpts : KPointSet
        Named momenta.
    spinor : bool, optional
        If omitted, use spinor rows unless `point_group.spin == "trivial"`.

    Returns
    -------
    dict[str, dict[str, tuple[complex, ...]]]
        `tables[hsp][irrep]` ready for `space_group_irreps`.
    """
    use_spinor = point_group.spin != "trivial" if spinor is None else spinor
    tables: dict[str, dict[str, tuple[complex, ...]]] = {}
    for name, raw_k in kpts.points.items():
        k = raw_k.fractional()
        g_k = little_group(point_group, k)
        members = space_group.little_group(k)
        if len(members) != g_k.order:
            raise ValueError(
                f"Space-group little group at {name!r} has {len(members)} "
                f"elements; the linear little group has {g_k.order}."
            )
        by_rotation = {
            _linear_rotation_key(element): index
            for index, element in enumerate(g_k.elements())
        }
        if len(by_rotation) != g_k.order:
            raise ValueError(
                f"Little-group elements at {name!r} are not unique by rotation3."
            )
        indices: list[int] = []
        for opr in members:
            index = by_rotation.get(_linear_rotation_key(opr.g))
            if index is None:
                raise KeyError(
                    f"Seitz linear part at {name!r} is not in the "
                    "point-group little group."
                )
            indices.append(index)
        table = g_k.spinor_table() if use_spinor else g_k.ordinary_table()
        chars: dict[str, tuple[complex, ...]] = {}
        for irrep_name, row in table["irreps"].items():
            if use_spinor:
                full = g_k.spinor_irrep_characters_by_element(irrep_name)
            else:
                full = tuple(
                    complex(sy.N(value))
                    for value in g_k.irrep_characters_by_element(irrep_name)
                )
            chars[_irrep_row_label(row, irrep_name)] = tuple(full[i] for i in indices)
        tables[name] = chars
    return tables


def space_group_irreps(
    states: Tensor,
    group: SpaceGroup,
    kpts: KPointSet,
    tables: Mapping[str, Mapping[str, Sequence[complex]]],
) -> dict[str, dict[str, int]]:
    r"""Decompose `states` with Seitz little-group characters at each HSP.

    Unlike
    [`little_group_irreps`][qten.pointgroups.ops.little_group_irreps], this
    uses the affine translations already stored on
    [`SpaceGroup`][qten.pointgroups.spacegroup.SpaceGroup] elements and
    projects traces onto **caller-supplied** little-group tables. Point-group
    character tables are not used.

    Parameters
    ----------
    states : Tensor
        Rank-2 local columns `(HilbertSpace, columns)` or rank-3
        momentum-resolved states `(MomentumSpace, HilbertSpace, columns)`.
    group : SpaceGroup
        Factor-group Seitz representatives \(\{R\mid\tau\}\).
    kpts : KPointSet
        Named high-symmetry momenta.
    tables : Mapping[str, Mapping[str, Sequence[complex]]]
        `tables[hsp_name][irrep_name]` is \(\chi(g)\) for each little-group
        element of that \(k\), in the order returned by
        [`SpaceGroup.little_group`][qten.pointgroups.spacegroup.SpaceGroup.little_group].

    Returns
    -------
    dict[str, dict[str, int]]
        `{hsp_name: {irrep_name: multiplicity}}` with zero-multiplicity
        irreps omitted.

    Raises
    ------
    KeyError
        If `tables` is missing an HSP name present in `kpts`.
    ValueError
        If a character row length does not match the little group at that \(k\).
    """
    crystal, bloch = _states_on_kpoints(states, kpts)
    k_space = cast(MomentumSpace, crystal.dims[0])
    k_index = {_momentum_rep_key(k): i for i, k in enumerate(k_space.elements())}
    recip = k_space.elements()[0].space if k_space.elements() else None
    rebased = (
        kpts.rebase(cast(ReciprocalLattice, recip))
        if isinstance(recip, ReciprocalLattice)
        else kpts
    )

    result: dict[str, dict[str, int]] = {}
    cache: dict[tuple[Any, ...], dict[str, int]] = {}
    for name, raw_k in rebased.points.items():
        k = raw_k.fractional()
        key = _momentum_rep_key(k)
        if key in cache:
            result[name] = cache[key]
            continue
        idx = k_index.get(key)
        if idx is None:
            raise ValueError(f"High-symmetry point '{name}' is missing after sampling.")
        if name not in tables:
            raise KeyError(
                f"Little-group table is missing high-symmetry point {name!r}."
            )
        members = group.little_group(k)
        one_k = MomentumSpace(structure=OrderedDict(((k, 0),)))
        representations: list[Tensor] = []
        for opr in members:
            mapped = cast(Momentum, opr @ k).fractional()
            mapped_k = MomentumSpace(structure=OrderedDict(((mapped, 0),)))
            blocks = _point_group_band_blocks(
                opr, one_k, mapped_k, bloch, device=states.device
            )
            representations.append(Tensor(data=blocks.data[0], dims=(bloch, bloch)))
        content = _project_characters(crystal.data[idx], representations, tables[name])
        cache[key] = content
        result[name] = content
    return result


def ebr_decomposition(
    content: Mapping[str, Mapping[str, int]],
    ebrs: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    max_copies: int | None = None,
) -> dict[str, int] | None:
    r"""Return nonnegative EBR multiplicities that reproduce `content`, if any.

    Solves \(E n = v\) over nonnegative integers, where \(v\) is the flattened
    HSP irrep vector `content[k][irrep]` and each column of \(E\) is one
    elementary band representation. Missing keys are zero. If several
    solutions exist, the one with smallest \(\ell_1\) norm is returned
    (ties broken by the stored EBR order).

    Parameters
    ----------
    content : Mapping[str, Mapping[str, int]]
        Measured little-group multiplicities `{hsp: {irrep: multiplicity}}`.
    ebrs : Mapping[str, Mapping[str, Mapping[str, int]]]
        `{ebr_name: {hsp: {irrep: multiplicity}}}`.
    max_copies : int | None, optional
        Cap on each EBR coefficient. Default is the largest entry of `content`.

    Returns
    -------
    dict[str, int] | None
        `{ebr_name: copies}` omitting zeros, or `None` if `content` is not
        in the cone.
    """
    names, columns, target = _ebr_integer_system(content, ebrs)
    if not names:
        return {} if not any(target) else None
    cap = max_copies if max_copies is not None else (max(target) if target else 0)
    solution = _nonnegative_integer_combination(columns, target, cap)
    if solution is None:
        return None
    return {name: copies for name, copies in zip(names, solution) if copies}


class EBRIndicator(NamedTuple):
    r"""Cokernel of the EBR map \(E:\mathbb{Z}^{n}\to\mathbb{Z}^{m}\), \(En=v\).

    `torsion` lists Smith invariants \(s_i>1\) with residues \((Uv)_i\bmod s_i\).
    `trivial` means \(v\) is in the EBR lattice and every residue is 0.
    `lattice` is one integer solution (entries may be negative). `cone` is
    the nonnegative solution from
    [`ebr_decomposition`][qten.pointgroups.ops.ebr_decomposition], or `None`.
    """

    in_lattice: bool
    trivial: bool
    torsion: tuple[tuple[int, int], ...]
    lattice: dict[str, int] | None
    cone: dict[str, int] | None


def ebr_lattice_decomposition(
    content: Mapping[str, Mapping[str, int]],
    ebrs: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, int] | None:
    r"""Return any integer EBR multiplicities with \(En=v\), or `None`.

    Negative copies are allowed. This is membership in the EBR *lattice*,
    not the EBR cone. A vector in the lattice but not the cone is
    indicator-trivial and still not an atomic insulator.

    Parameters
    ----------
    content : Mapping[str, Mapping[str, int]]
        Measured little-group multiplicities `{hsp: {irrep: multiplicity}}`.
    ebrs : Mapping[str, Mapping[str, Mapping[str, int]]]
        `{ebr_name: {hsp: {irrep: multiplicity}}}`.

    Returns
    -------
    dict[str, int] | None
        `{ebr_name: copies}` omitting zeros. Copies may be negative.
        `None` if `content` is not in the lattice.
    """
    return ebr_indicator(content, ebrs).lattice


def ebr_indicator(
    content: Mapping[str, Mapping[str, int]],
    ebrs: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> EBRIndicator:
    r"""Symmetry indicator of `content` relative to the EBR lattice of `ebrs`.

    Smith-decomposes the integer matrix \(E\) whose columns are the caller
    EBR vectors. The torsion part of \(\mathrm{coker}\,E\) is the indicator
    group generated by those columns; the residue of \(v\) is the class.
    This is not a space-group catalogue: the group is whatever lattice the
    supplied columns span.

    Parameters
    ----------
    content : Mapping[str, Mapping[str, int]]
        Measured little-group multiplicities `{hsp: {irrep: multiplicity}}`.
    ebrs : Mapping[str, Mapping[str, Mapping[str, int]]]
        `{ebr_name: {hsp: {irrep: multiplicity}}}`.

    Returns
    -------
    EBRIndicator
        Lattice membership, torsion residues, one integer solution, and
        the nonnegative cone solution when it exists.
    """
    names, columns, target = _ebr_integer_system(content, ebrs)
    cone = ebr_decomposition(content, ebrs)
    if not names:
        empty = not any(target)
        return EBRIndicator(
            in_lattice=empty,
            trivial=empty,
            torsion=(),
            lattice={} if empty else None,
            cone=cone,
        )

    matrix = sy.Matrix(list(zip(*columns))) if target else sy.Matrix(0, len(names), [])
    if matrix.rows == 0:
        empty = not any(target)
        return EBRIndicator(
            in_lattice=empty,
            trivial=empty,
            torsion=(),
            lattice={} if empty else None,
            cone=cone,
        )

    smith, left, right = smith_normal_decomp(matrix, domain=sy.ZZ)
    image = left * sy.Matrix(target)
    diagonal = [int(smith[i, i]) for i in range(min(smith.rows, smith.cols))]
    torsion: list[tuple[int, int]] = []
    particular = [0] * smith.cols
    in_lattice = True
    for i, invariant in enumerate(diagonal):
        residue = int(image[i])
        if invariant == 0:
            if residue != 0:
                in_lattice = False
            continue
        if residue % invariant:
            in_lattice = False
        particular[i] = residue // invariant
        if invariant > 1:
            torsion.append((invariant, residue % invariant))
    for i in range(len(diagonal), smith.rows):
        if int(image[i]) != 0:
            in_lattice = False

    lattice: dict[str, int] | None = None
    if in_lattice:
        copies = right * sy.Matrix(particular)
        lattice = {
            name: int(copies[j]) for j, name in enumerate(names) if int(copies[j])
        }
    trivial = in_lattice and all(residue == 0 for _, residue in torsion)
    return EBRIndicator(
        in_lattice=in_lattice,
        trivial=trivial,
        torsion=tuple(torsion),
        lattice=lattice,
        cone=cone,
    )


class InversionIndicators(NamedTuple):
    r"""Class-AII inversion indicators from TRIM parity counts.

    State counts `n_even` and `n_odd` must be even (Kramers). With eight
    3D TRIM this is \(\mathbb{Z}_4\times\mathbb{Z}_2^3\):

    - `z4` \(=\sum_k(n_k^--n_k^+)/4\bmod 4\)
    - `kappa1` \(=\sum_k(n_k^+-n_k^-)/4\bmod 4\) (\(= -z_4\bmod 4\))
    - `nu0` Fu–Kane strong \(\mathbb{Z}_2\)
    - `weak` Fu–Kane weak indices, one per reciprocal primitive direction
      with a \(k_j=1/2\) face
    """

    z4: int
    kappa1: int
    nu0: int
    weak: tuple[int, ...]
    n_trim: int
    filling: int


def inversion_indicators(
    trim: Sequence[tuple[Sequence[object], int, int]],
) -> InversionIndicators:
    r"""Inversion \(\mathbb{Z}_4\) and Fu–Kane \(\mathbb{Z}_2\) from TRIM parities.

    Each entry is `(k_frac, n_even, n_odd)` of *states* (not Kramers pairs)
    at one time-reversal invariant momentum. Every component of `k_frac`
    must be \(0\) or \(1/2\). Filling \(n_{\mathrm{even}}+n_{\mathrm{odd}}\)
    must be the same at every point.

    This is the inversion-only indicator (space group 2, class AII). A
    larger space group's indicator group may be a quotient of this one;
    the caller supplies all \(2^d\) distinct TRIM.

    Parameters
    ----------
    trim : sequence of `(k_frac, n_even, n_odd)`
        One triple per TRIM. Counts are occupied inversion-even and
        inversion-odd states.

    Returns
    -------
    InversionIndicators
        `z4`, `kappa1`, Fu–Kane `nu0` and `weak`, plus `n_trim` and the
        common filling.

    Raises
    ------
    ValueError
        If `trim` is empty, a count is negative or odd, fillings differ,
        a momentum component is not \(0\) or \(1/2\), the TRIM are not the
        \(2^d\) distinct points, or \(\sum_k(n^--n^+)\) is not divisible by 4.
    """
    if not trim:
        raise ValueError("trim must contain at least one momentum.")

    fillings: list[int] = []
    z4_sum = 0
    kappa_sum = 0
    strong = 1
    dim: int | None = None
    parsed: list[tuple[tuple[object, ...], int, int]] = []
    for k_frac, n_even, n_odd in trim:
        even = int(n_even)
        odd = int(n_odd)
        if even < 0 or odd < 0:
            raise ValueError("Parity counts must be nonnegative.")
        if even % 2 or odd % 2:
            raise ValueError("Parity counts must be even (Kramers).")
        coords = tuple(k_frac)
        if dim is None:
            dim = len(coords)
        elif len(coords) != dim:
            raise ValueError("All TRIM momenta must have the same dimension.")
        for component in coords:
            if not _is_trim_component(component):
                raise ValueError("Each TRIM momentum component must be 0 or 1/2.")
        fillings.append(even + odd)
        z4_sum += odd - even
        strong *= -1 if ((odd // 2) % 2) else 1
        parsed.append((coords, even, odd))

    if len(set(fillings)) != 1:
        raise ValueError("Filling n_even + n_odd must be the same at every TRIM.")
    momenta = {tuple(sy.nsimplify(c) for c in coords) for coords, _, _ in parsed}
    if len(momenta) != len(parsed):
        raise ValueError("TRIM momenta must be distinct.")
    if dim is not None and len(parsed) != 2**dim:
        raise ValueError(f"Expected all {2**dim} TRIM in {dim}D, got {len(parsed)}.")
    # Single terms are only ≡ 2 (mod 4); the sum over all TRIM is divisible by 4.
    if z4_sum % 4:
        raise ValueError(
            "sum_k (n_odd - n_even) is not divisible by 4; the parity counts "
            "are inconsistent."
        )
    kappa_sum = -z4_sum // 4
    z4_sum //= 4

    weak: list[int] = []
    for axis in range(dim or 0):
        product_sign = 1
        for coords, _, odd in parsed:
            if _is_half_component(coords[axis]):
                product_sign *= -1 if ((odd // 2) % 2) else 1
        weak.append(0 if product_sign == 1 else 1)

    return InversionIndicators(
        z4=z4_sum % 4,
        kappa1=kappa_sum % 4,
        nu0=0 if strong == 1 else 1,
        weak=tuple(weak),
        n_trim=len(parsed),
        filling=fillings[0],
    )


def _is_trim_component(value: object) -> bool:
    doubled = sy.simplify(2 * sy.nsimplify(value))
    return doubled.is_integer


def _is_half_component(value: object) -> bool:
    return sy.simplify(sy.nsimplify(value) - sy.Rational(1, 2)) == 0


def _ebr_integer_system(
    content: Mapping[str, Mapping[str, int]],
    ebrs: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> tuple[tuple[str, ...], list[tuple[int, ...]], tuple[int, ...]]:
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for table in (content, *ebrs.values()):
        for hsp, row in table.items():
            for irrep in row:
                pair = (hsp, irrep)
                if pair not in seen:
                    seen.add(pair)
                    keys.append(pair)

    target = tuple(int(content.get(hsp, {}).get(irrep, 0)) for hsp, irrep in keys)
    if any(value < 0 for value in target):
        raise ValueError("Irrep multiplicities must be nonnegative.")
    names = tuple(ebrs)
    columns = [
        tuple(int(ebrs[name].get(hsp, {}).get(irrep, 0)) for hsp, irrep in keys)
        for name in names
    ]
    return names, columns, target


def _nonnegative_integer_combination(
    columns: Sequence[tuple[int, ...]],
    target: tuple[int, ...],
    max_copies: int,
) -> tuple[int, ...] | None:
    r"""Smallest-\(\ell_1\) nonnegative integer solution of `columns @ n = target`."""
    n_cols = len(columns)
    best: tuple[int, ...] | None = None
    best_l1: int | None = None
    chosen: list[int] = []

    def rec(index: int, remaining: tuple[int, ...], l1: int) -> None:
        nonlocal best, best_l1
        if best_l1 is not None and l1 >= best_l1:
            return
        if index == n_cols:
            if all(value == 0 for value in remaining):
                best = tuple(chosen)
                best_l1 = l1
            return
        column = columns[index]
        limit = max_copies
        for coeff, leftover in zip(column, remaining):
            if coeff > 0:
                limit = min(limit, leftover // coeff)
        if limit < 0:
            return
        for copies in range(limit + 1):
            nxt = tuple(
                leftover - copies * coeff for leftover, coeff in zip(remaining, column)
            )
            if any(value < 0 for value in nxt):
                continue
            chosen.append(copies)
            rec(index + 1, nxt, l1 + copies)
            chosen.pop()

    rec(0, target, 0)
    return best
