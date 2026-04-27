"""
Point-group operations on symbolic bases and tensors.

This module contains functional helpers that combine abelian point-group
representations with QTen Hilbert spaces and tensors. The functions compute
joint abelian eigen-bases, project Hilbert spaces into symmetry sectors, and
assemble representation tensors for point-group actions.

Repository usage
----------------
Use [`joint_abelian_basis()`][qten.pointgroups.ops.joint_abelian_basis] and the
related projection helpers when an existing
[`AbelianGroup`][qten.pointgroups.abelian.AbelianGroup] or
[`AbelianOpr`][qten.pointgroups.abelian.AbelianOpr] should act on symbolic
Hilbert-space data. The group definitions themselves live in
[`qten.pointgroups.abelian`][qten.pointgroups.abelian].
"""

from itertools import product
from math import prod
from typing import Sequence, cast

import sympy as sy

from .abelian import (
    AbelianBasis,
    AbelianGroup,
    AbelianOpr,
)
from ..linalg.tensors import Tensor, cat, eye
from ..symbolics import HilbertSpace, IndexSpace, U1Basis, hilbert_opr_repr
from ..utils.collections_ext import FrozenDict


def _same_phase(a: sy.Expr, b: sy.Expr) -> bool:
    diff = sy.simplify(a - b)
    if diff == 0:
        return True

    expanded = sy.simplify(sy.expand_complex(diff))
    if expanded == 0:
        return True

    equals = diff.equals(0)
    return bool(equals)


def _phase_basis(opr: AbelianOpr, phase: sy.Expr) -> AbelianBasis:
    phase = sy.simplify(phase)
    table = opr.g.basis_table
    if phase in table:
        return cast(AbelianBasis, table[phase])
    for key, basis in table.items():
        if _same_phase(key, phase):
            return cast(AbelianBasis, basis)
    raise ValueError(f"Failed to find an AbelianBasis for phase={phase}.")


def _attach_basis_label(seed: U1Basis | None, basis: AbelianBasis) -> U1Basis:
    if seed is None:
        return U1Basis.new(basis)
    try:
        return seed.replace(basis)
    except ValueError:
        return U1Basis(coef=seed.coef, base=seed.base + (basis,))


def _attach_degeneracy_tag(seed: U1Basis, index: int) -> U1Basis:
    tag = index
    try:
        return seed.replace(tag)
    except ValueError:
        return U1Basis(coef=seed.coef, base=seed.base + (tag,))


def joint_abelian_basis(
    oprs: Sequence[AbelianGroup | AbelianOpr], order: int
) -> FrozenDict[tuple[sy.Expr, ...], tuple[AbelianBasis, ...]]:
    """
    Compute common Euclidean eigenfunctions for a commuting family of abelian operators.

    The returned table is keyed by one phase per input operator. Each value is
    the tuple of normalized [`AbelianBasis`][qten.pointgroups.abelian.AbelianBasis]
    functions spanning the simultaneous eigenspace for that joint phase sector.

    Parameters
    ----------
    oprs : Sequence[AbelianGroup | AbelianOpr]
        Non-empty sequence of operators. Affine
        [`AbelianOpr`][qten.pointgroups.abelian.AbelianOpr] inputs contribute
        only their linear part.
    order : int
        Homogeneous polynomial degree used for all Euclidean representations.

    Returns
    -------
    FrozenDict[tuple[sy.Expr, ...], tuple[AbelianBasis, ...]]
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

    groups = tuple(opr.g if isinstance(opr, AbelianOpr) else opr for opr in oprs)
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

    tbl: dict[tuple[sy.Expr, ...], tuple[AbelianBasis, ...]] = {}
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

        labels: list[AbelianBasis] = []
        seen_reps = set()
        for vec in basis_vectors:
            rep = sy.ImmutableDenseMatrix(vec)
            if all(entry == 0 for entry in rep):
                continue
            basis = AbelianBasis.from_rep(
                rep=rep,
                euclidean_basis=euclidean_basis,
                axes=axes,
                order=order,
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
    oprs: Sequence[AbelianOpr],
) -> dict[tuple[sy.Expr, ...], AbelianBasis]:
    """
    Build a representative [`AbelianBasis`][qten.pointgroups.abelian.AbelianBasis] for each joint phase sector.
    """
    phase_bases: dict[tuple[sy.Expr, ...], AbelianBasis] = {}
    max_order = prod(opr.g.group_order() for opr in oprs)
    for order in range(max_order):
        table = joint_abelian_basis(oprs, order)
        for phases, bases in table.items():
            if phases in phase_bases or not bases:
                continue
            phase_bases[phases] = bases[0]
    return phase_bases


def abelian_column_symmetrize(
    opr: AbelianOpr, w: Tensor, full_sector: bool = False
) -> Tensor:
    r"""
    Symmetrize the columns of `w` by projecting each one onto every sector of `opr`.

    For a finite-order abelian element `opr` of order \(n\), each exact
    symmetry sector is labeled by a phase \(\omega\) with \(\omega^n = 1\).
    This function
    builds the full operator representation `G` on the ambient Hilbert space
    `w.dims[0]` and applies the projector

    $$
    P_\omega = \frac{1}{n}\sum_{k=0}^{n-1}\omega^{-k}G^k
    $$

    which is the rendered form of the code-level convention
    `P_omega = (1/n) * sum_{k=0}^{n-1} omega^(-k) G^k`.

    The projector is applied to each input column separately. When
    `full_sector` is `True`, every
    nonzero projected sector component is returned. When `full_sector` is
    `False`, only the dominant nonzero sector component of each input column is
    kept, so the output column count does not exceed the input count. Returned
    columns carry the corresponding [`AbelianBasis`][qten.pointgroups.abelian.AbelianBasis].

    The output column count can differ from the input one only when
    `full_sector=True`, because symmetry projection may split one approximate
    column into multiple exact sectors.

    Parameters
    ----------
    opr : AbelianOpr
        Finite-order abelian operator used to build symmetry projectors.
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
    if w.rank() != 2:
        raise ValueError("w must be a rank-2 tensor of ambient-space columns.")

    row_dim = w.dims[0]
    if not isinstance(row_dim, HilbertSpace):
        raise ValueError("w.dims[0] must be a HilbertSpace.")
    input_col_dim = w.dims[1]
    seeds: list[U1Basis | None]
    if isinstance(input_col_dim, HilbertSpace):
        seeds = list(input_col_dim.elements())
    elif isinstance(input_col_dim, IndexSpace):
        seeds = [None] * input_col_dim.dim
    else:
        raise ValueError("w.dims[1] must be either an IndexSpace or a HilbertSpace.")

    g_full = hilbert_opr_repr(opr, row_dim).to_device(w.device)
    order = opr.g.group_order()
    ident = eye((row_dim, row_dim)).astype(g_full.data.dtype).to_device(g_full.device)
    single_col = IndexSpace.linear(1)
    tol = 1e-10

    g_powers: list[Tensor] = [ident]
    for _ in range(1, order):
        g_powers.append(g_powers[-1] @ g_full)

    sector_projectors: list[tuple[AbelianBasis, Tensor]] = []
    for m in range(order):
        phase_exact = sy.simplify(sy.exp(2 * sy.pi * sy.I * m / order))
        sector_basis = _phase_basis(opr, phase_exact)
        phase_scalar = complex(sy.N(phase_exact))

        projector = 0 * ident
        for k, g_power in enumerate(g_powers):
            projector = projector + (phase_scalar ** (-k)) * g_power
        sector_projectors.append((sector_basis, projector / order))

    projected_cols: list[Tensor] = []
    raw_labels: list[U1Basis] = []
    for j, seed in enumerate(seeds):
        col = w[:, j : j + 1].clone().replace_dim(1, single_col)
        candidates: list[tuple[float, Tensor, U1Basis]] = []
        for sector_basis, projector in sector_projectors:
            projected = projector @ col

            projected_norm = projected.norm()
            norm_value = float(projected_norm.item())
            if norm_value <= tol:
                continue

            candidates.append(
                (
                    norm_value,
                    projected / norm_value,
                    _attach_basis_label(seed, sector_basis),
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

    out_dim = HilbertSpace.new(labels)
    return cat(projected_cols, dim=-1).replace_dim(-1, out_dim)


def joint_abelian_column_symmetrize(
    oprs: Sequence[AbelianOpr], w: Tensor, full_sector: bool = False
) -> Tensor:
    """
    Symmetrize columns of `w` into simultaneous sectors of abelian operators.

    The operators in `oprs` are expected to commute on `w.dims[0]`. For each
    operator, this builds the same sector projectors as
    [`abelian_column_symmetrize`][qten.pointgroups.ops.abelian_column_symmetrize], then projects each column onto every joint
    sector in the Cartesian product of those sector decompositions.

    When `full_sector` is `True`, every nonzero joint-sector component is
    returned. When `False`, only the dominant nonzero joint-sector component of
    each input column is kept. Returned columns carry a representative common
    [`AbelianBasis`][qten.pointgroups.abelian.AbelianBasis] for the corresponding joint phase sector.

    Parameters
    ----------
    oprs : Sequence[AbelianOpr]
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
        return abelian_column_symmetrize(oprs[0], w, full_sector=full_sector)
    if w.rank() != 2:
        raise ValueError("w must be a rank-2 tensor of ambient-space columns.")

    row_dim = w.dims[0]
    if not isinstance(row_dim, HilbertSpace):
        raise ValueError("w.dims[0] must be a HilbertSpace.")
    input_col_dim = w.dims[1]
    seeds: list[U1Basis | None]
    if isinstance(input_col_dim, HilbertSpace):
        seeds = list(input_col_dim.elements())
    elif isinstance(input_col_dim, IndexSpace):
        seeds = [None] * input_col_dim.dim
    else:
        raise ValueError("w.dims[1] must be either an IndexSpace or a HilbertSpace.")

    single_col = IndexSpace.linear(1)
    tol = 1e-10

    joint_sector_bases = _joint_phase_basis(oprs)
    all_sector_projectors: list[list[tuple[sy.Expr, Tensor]]] = []
    dtype = w.data.dtype
    device = w.device
    for opr in oprs:
        g_full = hilbert_opr_repr(opr, row_dim).to_device(w.device)
        dtype = g_full.data.dtype
        device = g_full.device
        order = opr.g.group_order()
        ident = eye((row_dim, row_dim)).astype(dtype).to_device(device)

        g_powers: list[Tensor] = [ident]
        for _ in range(1, order):
            g_powers.append(g_powers[-1] @ g_full)

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
        col = w[:, j : j + 1].clone().replace_dim(1, single_col)
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
            norm_value = float(projected_norm.item())
            if norm_value <= tol:
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

    out_dim = HilbertSpace.new(labels)
    return cat(projected_cols, dim=-1).replace_dim(-1, out_dim)
