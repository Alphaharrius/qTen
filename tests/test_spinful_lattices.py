"""Spinful Bloch models on a few lattices (the checks in the repo-root ``test.py``).

Role: end-to-end free-fermion + Fourier + ``bandfillings`` on 1D / 2D / 3D
lattices whose Hilbert space already carries ``Spin``. This is not a substitute
for geometry oracles or SU(2) unit tests.

The root ``test.py`` script is a scratch Kane-Mele / Fu-Kane-Mele builder.
These tests keep the same operations on small meshes: Hermitian Bloch blocks,
half-filling, and the correlation ``I - P``.
"""

import sympy as sy
import torch

import qten
import qten.ops as Q
from qten.bands import bandfillings
from qten.geometries import Lattice, Offset
from qten.phys import Bond, FFObservable, Spin, contains_spin
from qten.symbolics import FuncOpr, HilbertSpace, U1Basis, brillouin_zone


def _spin_sign(basis: U1Basis) -> sy.Integer:
    return sy.Integer(1) if basis.irrep_of(Spin).is_up else sy.Integer(-1)


def _bloch_from_observable(
    lattice: Lattice, tb: FFObservable, cell_states: list[U1Basis]
):
    harmonic = tb.to_tensor()
    bloch_space = HilbertSpace.new(cell_states)
    assert contains_spin(bloch_space)
    k_space = brillouin_zone(lattice.dual)
    fourier = Q.fourier_transform(k_space, bloch_space, harmonic.dims[0])
    bloch = fourier @ harmonic @ fourier.h(-2, -1)
    return bloch


def _assert_spinful_bloch(bloch, *, n_orb: int, min_gap: float | None = None) -> float:
    assert bloch.rank() == 3
    _k_space, row, col = bloch.dims
    assert row.dim == n_orb == col.dim
    assert contains_spin(row)
    hamiltonian = bloch.data
    assert torch.allclose(hamiltonian, hamiltonian.mH, rtol=0, atol=1e-10)

    eigvals = torch.linalg.eigvalsh(hamiltonian).real
    lower = eigvals[:, n_orb // 2 - 1]
    upper = eigvals[:, n_orb // 2]
    gap = float(torch.min(upper - lower).item())
    assert gap >= -1e-10
    if min_gap is not None:
        assert gap > min_gap

    occupied = bandfillings(bloch, 0.5)
    correlation = qten.eye(bloch.dims) - occupied @ occupied.h(-2, -1)
    assert occupied.data.shape[:2] == hamiltonian.shape[:2]
    assert correlation.data.shape == hamiltonian.shape
    assert torch.allclose(correlation.data, correlation.data.mH, rtol=0, atol=1e-10)
    return gap


def _build_ssh_chain_bloch(
    *, shape: tuple[int, ...] = (8,), t1=sy.Integer(1), t2=sy.Rational(1, 2)
):
    chain = Lattice(
        basis=sy.ImmutableMatrix([[1]]),
        unit_cell={
            "a": sy.ImmutableMatrix([0]),
            "b": sy.ImmutableMatrix([sy.Rational(1, 2)]),
        },
        shape=shape,
    )
    (a1,) = chain.basis_vectors()
    hop = FuncOpr(Offset, lambda r: r + a1)

    a_up = U1Basis.new(chain.at("a"), Spin.up)
    a_dn = U1Basis.new(chain.at("a"), Spin.down)
    b_up = U1Basis.new(chain.at("b"), Spin.up)
    b_dn = U1Basis.new(chain.at("b"), Spin.down)

    tb = FFObservable()
    for left, right in ((a_up, b_up), (a_dn, b_dn)):
        tb.add_bond(-t1, left, right)
        tb.add_bond(-t2, right, hop @ left)

    return chain, _bloch_from_observable(chain, tb, [a_up, a_dn, b_up, b_dn])


def _build_honeycomb_kane_mele_bloch(
    *,
    shape: tuple[int, int] = (4, 4),
    t=sy.Integer(1),
    lambda_soc=sy.Rational(1, 8),
):
    honeycomb = Lattice(
        basis=sy.ImmutableMatrix(
            [
                [sy.sqrt(3) / 2, 0],
                [-sy.Rational(1, 2), 1],
            ]
        ),
        unit_cell={
            "a": sy.ImmutableMatrix([sy.Rational(1, 3), sy.Rational(2, 3)]),
            "b": sy.ImmutableMatrix([sy.Rational(2, 3), sy.Rational(1, 3)]),
        },
        shape=shape,
    )
    a1, a2 = honeycomb.basis_vectors()
    hops = (
        FuncOpr(Offset, lambda r: r + a1),
        FuncOpr(Offset, lambda r: r + a2),
        FuncOpr(Offset, lambda r: r + a1 - a2),
    )
    a_up = U1Basis.new(honeycomb.at("a"), Spin.up)
    a_dn = U1Basis.new(honeycomb.at("a"), Spin.down)
    b_up = U1Basis.new(honeycomb.at("b"), Spin.up)
    b_dn = U1Basis.new(honeycomb.at("b"), Spin.down)

    tb = FFObservable()
    for left, right in ((a_up, b_up), (a_dn, b_dn)):
        tb.add_bond(-t, left, right)
        tb.add_bond(-t, left, hops[1] @ right)
        tb.add_bond(-t, right, hops[0] @ left)

    for basis, chi in (
        (a_up, sy.Integer(1)),
        (a_dn, sy.Integer(1)),
        (b_up, sy.Integer(-1)),
        (b_dn, sy.Integer(-1)),
    ):
        coef = sy.I * lambda_soc * _spin_sign(basis) * chi
        for hop in hops:
            tb.add_bond(Bond(coef, (basis, hop @ basis)))

    return honeycomb, _bloch_from_observable(honeycomb, tb, [a_up, a_dn, b_up, b_dn])


def _build_diamond_fu_kane_mele_bloch(
    *,
    shape: tuple[int, int, int] = (2, 2, 2),
    t=sy.Integer(1),
    delta_t=sy.Rational(1, 4),
    lambda_soc=sy.Rational(1, 16),
):
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

    A_up = U1Basis.new(diamond.at("A"), Spin.up)
    A_dn = U1Basis.new(diamond.at("A"), Spin.down)
    B_up = U1Basis.new(diamond.at("B"), Spin.up)
    B_dn = U1Basis.new(diamond.at("B"), Spin.down)

    tb = FFObservable()
    for left, right in ((A_up, B_up), (A_dn, B_dn)):
        tb.add_bond(-(t + delta_t), left, right)
        tb.add_bond(-t, left, translate(-a1) @ right)
        tb.add_bond(-t, left, translate(-a2) @ right)
        tb.add_bond(-t, left, translate(-a3) @ right)

    def add_soc_bond(src_up, src_dn, hop, soc_vector):
        dst_up = hop @ src_up
        dst_dn = hop @ src_dn
        for src, dst, element in (
            (src_up, dst_up, soc_vector[2]),
            (src_dn, dst_dn, -soc_vector[2]),
            (src_up, dst_dn, soc_vector[0] - sy.I * soc_vector[1]),
            (src_dn, dst_up, soc_vector[0] + sy.I * soc_vector[1]),
        ):
            coef = sy.simplify(sy.I * 8 * lambda_soc * element)
            if coef != 0:
                tb.add_bond(Bond(coef, (src, dst)))

    def add_sublattice_soc(src_up, src_dn, cell_shifts, neighbor_vectors):
        for i, shift_i in enumerate(cell_shifts):
            for j, shift_j in enumerate(cell_shifts[i + 1 :], start=i + 1):
                hop = cell_translation(shift_i - shift_j)
                soc_vector = neighbor_vectors[i].cross(-neighbor_vectors[j])
                add_soc_bond(src_up, src_dn, hop, soc_vector)

    tau_A = sy.ImmutableMatrix([0, 0, 0])
    tau_B = sy.ImmutableMatrix([quarter, quarter, quarter])
    a_to_b_shifts = (zero, -e1, -e2, -e3)
    add_sublattice_soc(
        A_up,
        A_dn,
        a_to_b_shifts,
        tuple(fcc @ (tau_B + shift - tau_A) for shift in a_to_b_shifts),
    )
    b_to_a_shifts = (zero, e1, e2, e3)
    add_sublattice_soc(
        B_up,
        B_dn,
        b_to_a_shifts,
        tuple(fcc @ (tau_A + shift - tau_B) for shift in b_to_a_shifts),
    )

    return diamond, _bloch_from_observable(diamond, tb, [A_up, A_dn, B_up, B_dn])


def test_1d_ssh_chain_spinful_bloch_is_hermitian_and_gapped():
    lattice, bloch = _build_ssh_chain_bloch()
    assert lattice.dim == 1
    _assert_spinful_bloch(bloch, n_orb=4, min_gap=0.2)


def test_2d_honeycomb_kane_mele_bloch_is_hermitian():
    lattice, bloch = _build_honeycomb_kane_mele_bloch()
    assert lattice.dim == 2
    _assert_spinful_bloch(bloch, n_orb=4)


def test_3d_diamond_fu_kane_mele_bloch_is_hermitian():
    lattice, bloch = _build_diamond_fu_kane_mele_bloch()
    assert lattice.dim == 3
    _assert_spinful_bloch(bloch, n_orb=4)
