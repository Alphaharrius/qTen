from typing import cast

import sympy as sy

from .abelian import AbelianBasis, AbelianOpr
from ..linalg.tensors import Tensor, cat, eye
from ..symbolics.hilbert_space import HilbertSpace, U1Basis
from ..symbolics.ops import hilbert_opr_repr
from ..symbolics.state_space import IndexSpace


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


def abelian_column_symmetrize(
    opr: AbelianOpr, w: Tensor, full_sector: bool = False
) -> Tensor:
    """
    Symmetrize the columns of `w` by projecting each one onto every sector of `opr`.

    For a finite-order abelian element `opr` of order `n`, each exact symmetry
    sector is labeled by a phase `omega` with `omega**n = 1`. This function
    builds the full operator representation `G` on the ambient Hilbert space
    `w.dims[0]` and applies the projector

    `P_omega = (1/n) * sum_{k=0}^{n-1} omega^(-k) G^k`

    to each input column separately. When `full_sector` is `True`, every
    nonzero projected sector component is returned. When `full_sector` is
    `False`, only the dominant nonzero sector component of each input column is
    kept, so the output column count does not exceed the input count. Returned
    columns carry the corresponding `AbelianBasis`.

    The output column count can differ from the input one only when
    `full_sector=True`, because symmetry projection may split one approximate
    column into multiple exact sectors.
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
