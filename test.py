import sympy as sy
import torch

import qten
import qten.ops as Q

from qten.bands import bandfillings
from qten.geometries import Lattice, Offset
from qten.phys import Bond, FFObservable
from qten.symbolics import FuncOpr, HilbertSpace, U1Basis, brillouin_zone


def build_honeycomb_soc_bloch(
    *,
    shape: tuple[int, int] = (24, 24),
    t: sy.Rational = sy.Integer(1),
    lambda_soc: sy.Rational = sy.Rational(1, 8),
):
    """Build a spinful honeycomb model with intrinsic SOC (Kane-Mele style)."""
    triangular = sy.ImmutableMatrix(
        [
            [sy.sqrt(3) / 2, 0],
            [-sy.Rational(1, 2), 1],
        ]
    )
    honeycomb = Lattice(
        basis=triangular,
        unit_cell={
            "a": sy.ImmutableMatrix([sy.Rational(1, 3), sy.Rational(2, 3)]),
            "b": sy.ImmutableMatrix([sy.Rational(2, 3), sy.Rational(1, 3)]),
        },
        shape=shape,
    )

    a1, a2 = honeycomb.basis_vectors()
    R_a1 = FuncOpr(Offset, lambda r: r + a1)
    R_a2 = FuncOpr(Offset, lambda r: r + a2)
    R_a1_minus_a2 = FuncOpr(Offset, lambda r: r + a1 - a2)

    a_up = U1Basis.new(honeycomb.at("a"), "up")
    a_dn = U1Basis.new(honeycomb.at("a"), "down")
    b_up = U1Basis.new(honeycomb.at("b"), "up")
    b_dn = U1Basis.new(honeycomb.at("b"), "down")

    tb = FFObservable()

    # Nearest-neighbor hopping (spin-conserving).
    for left, right in ((a_up, b_up), (a_dn, b_dn)):
        tb.add_bond(-t, left, right)
        tb.add_bond(-t, left, R_a2 @ right)
        tb.add_bond(-t, right, R_a1 @ left)

    # Intrinsic SOC: i * lambda_soc * s_z * nu_ij on 2nd-neighbor hops.
    soc_hops = (R_a1, R_a2, R_a1_minus_a2)
    spin_sign = {"up": sy.Integer(1), "down": sy.Integer(-1)}

    for basis, chi in (
        (a_up, sy.Integer(1)),
        (a_dn, sy.Integer(1)),
        (b_up, sy.Integer(-1)),
        (b_dn, sy.Integer(-1)),
    ):
        s = spin_sign[basis.irrep_of(str)]
        coef = sy.I * lambda_soc * s * chi
        for hop in soc_hops:
            tb.add_bond(Bond(coef, (basis, hop @ basis)))

    harmonic = tb.to_tensor()

    bloch_space = HilbertSpace.new([a_up, a_dn, b_up, b_dn])
    k_space = brillouin_zone(honeycomb.dual)
    fourier = Q.fourier_transform(k_space, bloch_space, harmonic.dims[0])
    bloch = fourier @ harmonic @ fourier.h(-2, -1)
    return honeycomb, bloch


def build_diamond_fu_kane_mele_bloch(
    *,
    shape: tuple[int, int, int] = (8, 8, 8),
    t: sy.Rational = sy.Integer(1),
    delta_t: sy.Rational = sy.Integer(0),
    lambda_soc: sy.Rational = sy.Rational(1, 16),
):
    """Build the Fu-Kane-Mele model on the diamond lattice."""
    half = sy.Rational(1, 2)
    quarter = sy.Rational(1, 4)
    fcc = sy.ImmutableMatrix(
        [
            [0, half, half],
            [half, 0, half],
            [half, half, 0],
        ]
    )
    diamond = Lattice(
        basis=fcc,
        unit_cell={
            "A": sy.ImmutableMatrix([0, 0, 0]),
            "B": sy.ImmutableMatrix([quarter, quarter, quarter]),
        },
        shape=shape,
    )

    a1, a2, a3 = diamond.basis_vectors()
    zero = sy.ImmutableMatrix([0, 0, 0])
    e1 = sy.ImmutableMatrix([1, 0, 0])
    e2 = sy.ImmutableMatrix([0, 1, 0])
    e3 = sy.ImmutableMatrix([0, 0, 1])

    def translate(delta: Offset):
        return FuncOpr(Offset, lambda r, delta=delta: r + delta)

    def cell_translation(delta: sy.ImmutableMatrix):
        return translate(Offset(rep=delta, space=diamond.affine))

    R_m1 = translate(-a1)
    R_m2 = translate(-a2)
    R_m3 = translate(-a3)

    A_up = U1Basis.new(diamond.at("A"), "up")
    A_dn = U1Basis.new(diamond.at("A"), "down")
    B_up = U1Basis.new(diamond.at("B"), "up")
    B_dn = U1Basis.new(diamond.at("B"), "down")

    tb = FFObservable()

    # Nearest-neighbor diamond hopping: A(R) connects to four B neighbors.
    # delta_t distorts one of the four tetrahedral bonds, as in the usual
    # Fu-Kane-Mele diamond-lattice model.
    for left, right in ((A_up, B_up), (A_dn, B_dn)):
        tb.add_bond(-(t + delta_t), left, right)
        tb.add_bond(-t, left, R_m1 @ right)
        tb.add_bond(-t, left, R_m2 @ right)
        tb.add_bond(-t, left, R_m3 @ right)

    def add_soc_bond(
        src_up: U1Basis,
        src_dn: U1Basis,
        hop,
        soc_vector: sy.ImmutableMatrix,
    ) -> None:
        dst_up = hop @ src_up
        dst_dn = hop @ src_dn
        spin_bonds = (
            (src_up, dst_up, soc_vector[2]),
            (src_dn, dst_dn, -soc_vector[2]),
            (src_up, dst_dn, soc_vector[0] - sy.I * soc_vector[1]),
            (src_dn, dst_up, soc_vector[0] + sy.I * soc_vector[1]),
        )
        for src, dst, spin_matrix_element in spin_bonds:
            coef = sy.simplify(sy.I * 8 * lambda_soc * spin_matrix_element)
            if coef != 0:
                tb.add_bond(Bond(coef, (src, dst)))

    def add_sublattice_soc(
        src_up: U1Basis,
        src_dn: U1Basis,
        neighbor_cell_shifts: tuple[sy.ImmutableMatrix, ...],
        neighbor_vectors: tuple[sy.ImmutableMatrix, ...],
    ) -> None:
        for i, shift_i in enumerate(neighbor_cell_shifts):
            for j, shift_j in enumerate(neighbor_cell_shifts[i + 1 :], start=i + 1):
                # Fu-Kane-Mele SOC: i * lambda * (d1 x d2) . sigma.
                # Here d1 and d2 are the two nearest-neighbor legs through the
                # intermediate opposite-sublattice site.
                cell_delta = shift_i - shift_j
                hop = cell_translation(cell_delta)
                soc_vector = neighbor_vectors[i].cross(-neighbor_vectors[j])
                add_soc_bond(src_up, src_dn, hop, soc_vector)

    tau_A = sy.ImmutableMatrix([0, 0, 0])
    tau_B = sy.ImmutableMatrix([quarter, quarter, quarter])

    A_to_B_cell_shifts = (zero, -e1, -e2, -e3)
    A_to_B_vectors = tuple(
        fcc @ (tau_B + shift - tau_A) for shift in A_to_B_cell_shifts
    )
    add_sublattice_soc(A_up, A_dn, A_to_B_cell_shifts, A_to_B_vectors)

    B_to_A_cell_shifts = (zero, e1, e2, e3)
    B_to_A_vectors = tuple(
        fcc @ (tau_A + shift - tau_B) for shift in B_to_A_cell_shifts
    )
    add_sublattice_soc(B_up, B_dn, B_to_A_cell_shifts, B_to_A_vectors)

    harmonic = tb.to_tensor()

    bloch_space = HilbertSpace.new([A_up, A_dn, B_up, B_dn])
    k_space = brillouin_zone(diamond.dual)
    fourier = Q.fourier_transform(k_space, bloch_space, harmonic.dims[0])
    bloch = fourier @ harmonic @ fourier.h(-2, -1)
    _ = bloch.plot("bandstructure", backend="plotly")
    return diamond, bloch


def main() -> None:
    lattice, bloch = build_diamond_fu_kane_mele_bloch(delta_t=sy.Rational(1, 4))

    # Inspect the direct gap at half filling from sampled k-points.
    eigvals = torch.linalg.eigvalsh(bloch.data).real
    n_bands = eigvals.shape[-1]
    lower = eigvals[:, n_bands // 2 - 1]
    upper = eigvals[:, n_bands // 2]
    direct_gap = torch.min(upper - lower).item()

    gs = bandfillings(bloch, 0.5)
    c_gs = qten.eye(bloch.dims) - gs @ gs.h(-2, -1)

    print(f"Lattice shape: {lattice.shape}")
    print(f"Bloch data shape: {tuple(bloch.data.shape)}")
    print(f"Min direct gap (sampled): {direct_gap:.6f}")
    print(f"Ground-state correlation shape: {tuple(c_gs.data.shape)}")


if __name__ == "__main__":
    main()
