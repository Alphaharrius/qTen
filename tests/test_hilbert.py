import pytest
import sympy as sy
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix

from qten.symbolics.hilbert_space import U1Basis, U1Span, HilbertSpace
from qten.symbolics.ops import region_hilbert
from qten.symbolics.state_space import MomentumSpace, brillouin_zone
from qten.geometries.spatials import Lattice, Offset
from qten.utils.collections_ext import FrozenDict
from qten.geometries.boundary import PeriodicBoundary


@dataclass(frozen=True)
class Orb:
    name: str


def _lattice(basis: ImmutableDenseMatrix, shape: tuple[int, ...]) -> Lattice:
    return Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(*shape)),
        unit_cell={"r": ImmutableDenseMatrix([0] * basis.rows)},
    )


def _state(r: Offset, orb: str = "s", irrep: sy.Expr = sy.Integer(1)) -> U1Basis:
    return U1Basis(coef=irrep, base=(r, Orb(orb)))


def test_u1_state_basic_properties_and_overlap():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)

    psi = _state(r0, "s")
    psi_scaled = _state(r0, "s", sy.Integer(2))

    assert psi.dim == 1
    assert psi.ket(psi_scaled) == sy.Integer(2)
    assert psi.rays() == _state(r0, "s", sy.Integer(1))


def test_u1_state_irrep_access_and_replace():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)

    psi = _state(r0, "p")
    assert psi.irrep_of(Offset) == r0

    replaced = psi.replace(r1)
    assert replaced.irrep_of(Offset) == r1
    assert replaced.rep[1] == Orb("p")


def test_u1_state_rejects_non_unity_type_multiplicity():
    with pytest.raises(ValueError, match="unity multiplicity"):
        U1Basis(coef=sy.Integer(1), base=("a", "b"))


def test_u1_span_addition_and_deduplication():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)

    a = _state(r0, "s")
    b = _state(r1, "s")

    span = a | a
    assert isinstance(span, U1Span)
    assert span.dim == 1

    span2 = span | b
    assert span2.dim == 2


def test_u1_span_is_iterable():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)

    a = _state(r0, "s")
    b = _state(r1, "p")
    span = U1Span((a, b))

    assert list(span) == [a, b]


def test_hilbert_space_creation_and_operations():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (3,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    r2 = Offset(rep=ImmutableDenseMatrix([2]), space=lat.affine)

    s0 = _state(r0, "s")
    s1 = _state(r1, "s")
    s2 = _state(r2, "s")

    hs1 = HilbertSpace.new([s0, s1])
    hs2 = HilbertSpace.new([s1, s2])

    assert isinstance(hs1, HilbertSpace)
    assert hs1.dim == 2
    assert len(hs1.elements()) == 2

    union = hs1 + hs2
    assert list(union.structure.keys()) == [s0, s1, s2]

    union_pipe = hs1 | hs2
    assert list(union_pipe.structure.keys()) == [s0, s1, s2]

    inter = hs1 & hs2
    assert list(inter.structure.keys()) == [s1]

    diff = hs1 - hs2
    assert list(diff.structure.keys()) == [s0]

    union_with_state = hs1 | s2
    assert list(union_with_state.structure.keys()) == [s0, s1, s2]

    prepend_state = s2 | hs1
    assert list(prepend_state.structure.keys()) == [s2, s0, s1]


def test_hilbert_space_group_with_kwargs_selector():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (4,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    r2 = Offset(rep=ImmutableDenseMatrix([2]), space=lat.affine)
    r3 = Offset(rep=ImmutableDenseMatrix([3]), space=lat.affine)

    s0 = _state(r0, "s")
    p1 = _state(r1, "p")
    s2 = _state(r2, "s")
    d3 = _state(r3, "d")
    hs = HilbertSpace.new([s0, p1, s2, d3])

    grouped = hs.group(
        s_band=lambda el: el.irrep_of(Orb) == Orb("s"),
        p_band=Orb("p"),
    )
    assert isinstance(grouped, FrozenDict)
    assert set(grouped.keys()) == {"s_band", "p_band"}
    assert tuple(grouped["s_band"].elements()) == (s0, s2)
    assert tuple(grouped["p_band"].elements()) == (p1,)


def test_hilbert_space_group_raises_on_overlap():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    s0 = _state(r0, "s")
    s1 = _state(r1, "s")
    hs = HilbertSpace.new([s0, s1])

    with pytest.raises(ValueError, match="overlap"):
        hs.group(all_s=Orb("s"), first_only=lambda el: el.irrep_of(Offset) == r0)


def test_hilbert_space_group_by_returns_tuple_of_hilbertspace():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (4,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    r2 = Offset(rep=ImmutableDenseMatrix([2]), space=lat.affine)
    r3 = Offset(rep=ImmutableDenseMatrix([3]), space=lat.affine)

    s0 = _state(r0, "s")
    p1 = _state(r1, "p")
    s2 = _state(r2, "s")
    d3 = _state(r3, "d")
    hs = HilbertSpace.new([s0, p1, s2, d3])

    groups = hs.group_by(Orb)

    assert isinstance(groups, tuple)
    assert all(isinstance(g, HilbertSpace) for g in groups)
    assert len(groups) == 3
    assert tuple(groups[0].elements()) == (s0, s2)
    assert tuple(groups[1].elements()) == (p1,)
    assert tuple(groups[2].elements()) == (d3,)


def test_statespace_getitem_variants():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (3,))
    states = [
        _state(Offset(rep=ImmutableDenseMatrix([i]), space=lat.affine), "s")
        for i in range(3)
    ]
    hs = HilbertSpace.new(states)

    assert hs[0] == states[0]
    assert hs[-1] == states[-1]

    hs_slice = hs[1:3]
    assert isinstance(hs_slice, HilbertSpace)
    assert list(hs_slice.structure.keys()) == [states[1], states[2]]
    assert hs_slice.dim == 2

    hs_range = hs[range(0, 2)]
    assert list(hs_range.structure.keys()) == [states[0], states[1]]

    hs_seq = hs[[2, 0]]
    assert list(hs_seq.structure.keys()) == [states[2], states[0]]

    hs_tuple = hs[(1, -1)]
    assert list(hs_tuple.structure.keys()) == [states[1], states[2]]

    with pytest.raises(IndexError):
        _ = hs[3]

    with pytest.raises(TypeError):
        _ = hs["bad"]

    with pytest.raises(ValueError, match="unique"):
        _ = hs[[0, 0]]

    with pytest.raises(TypeError, match="integers"):
        _ = hs[[0, "bad"]]  # type: ignore[list-item]


def test_hilbert_space_gram_diagonal_for_identical_basis():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (2,))
    a = _state(
        Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine), "s", sy.Integer(2)
    )
    b = _state(
        Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine), "s", sy.Integer(3)
    )
    hs = HilbertSpace.new([a, b])

    gram = hs.cross_gram(hs)
    assert gram.data.shape == (2, 2)
    assert gram.data[0, 0] == 4
    assert gram.data[1, 1] == 9
    assert gram.data[0, 1] == 0


def test_hilbert_space_gram_unitizes_target_dim():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (2,))
    a = _state(
        Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine), "s", sy.Integer(2)
    )
    b = _state(
        Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine), "s", sy.Integer(3)
    )
    hs = HilbertSpace.new([a, b])

    gram = hs.cross_gram(hs)
    assert gram.dims[0] == hs
    assert gram.dims[1] == hs.rays()


def test_hilbert_space_lookup_exact_query_match():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (3,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    r2 = Offset(rep=ImmutableDenseMatrix([2]), space=lat.affine)

    hs = HilbertSpace.new([_state(r0, "s"), _state(r1, "p"), _state(r2, "s")])
    found = hs.lookup({Offset: r1, Orb: Orb("p")})
    assert found == _state(r1, "p")


def test_region_hilbert_matches_region_fractional_offsets():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (4,))

    uc_a = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    uc_b = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lat.affine)
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lat.affine)
    r2 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    r3 = Offset(rep=ImmutableDenseMatrix([sy.Rational(3, 2)]), space=lat.affine)

    bloch_space = HilbertSpace.new(
        [
            _state(uc_a, "s"),
            _state(uc_a, "p"),
            _state(uc_b, "d"),
        ]
    )

    expanded = region_hilbert(bloch_space, [r0, r1, r2, r3])

    assert tuple(expanded.elements()) == (
        _state(r0, "s"),
        _state(r0, "p"),
        _state(r1, "d"),
        _state(r2, "s"),
        _state(r2, "p"),
        _state(r3, "d"),
    )


def test_region_hilbert_raises_for_missing_fractional_offset():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (4,))

    uc_a = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    missing = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lat.affine)

    bloch_space = HilbertSpace.new([_state(uc_a, "s")])

    with pytest.raises(ValueError, match="fractional part"):
        region_hilbert(bloch_space, [missing])


def test_hilbert_space_lookup_errors_for_no_or_multiple_matches():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (2,))
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat.affine)
    hs = HilbertSpace.new([_state(r0, "s"), _state(r1, "s")])

    with pytest.raises(ValueError, match="No state found"):
        hs.lookup({Offset: Offset(rep=ImmutableDenseMatrix([3]), space=lat.affine)})

    with pytest.raises(ValueError, match="Multiple states found"):
        hs.lookup({Orb: Orb("s")})

    with pytest.raises(ValueError, match="cannot be empty"):
        hs.lookup({})


def test_momentum_space_brillouin():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lat = _lattice(basis, (2, 2))
    recip = lat.dual

    ms = brillouin_zone(recip)
    assert isinstance(ms, MomentumSpace)
    assert ms.dim == 4

    assert str(ms) == "MomentumSpace(4)"
    assert "MomentumSpace(4):" in repr(ms)


def test_statespace_type_errors():
    basis = ImmutableDenseMatrix([[1]])
    lat = _lattice(basis, (1,))
    s = _state(Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine), "s")
    hs = HilbertSpace.new([s])
    ms = MomentumSpace(structure={})

    with pytest.raises(ValueError, match="different types"):
        hs + ms

    with pytest.raises(ValueError, match="different types"):
        hs - ms

    with pytest.raises(ValueError, match="different types"):
        hs & ms


def test_hilbert_space_tensor_product_order_and_content():
    left = HilbertSpace.new(
        (
            U1Basis(coef=sy.Integer(1), base=(0,)),
            U1Basis(coef=sy.Integer(1), base=(1,)),
        )
    )
    right = HilbertSpace.new(
        (
            U1Basis(coef=sy.Integer(1), base=(Orb("a"),)),
            U1Basis(coef=sy.Integer(1), base=(Orb("b"),)),
            U1Basis(coef=sy.Integer(1), base=(Orb("c"),)),
        )
    )

    out = left.tensor_product(right)
    expected = tuple(a @ b for a in left.elements() for b in right.elements())

    assert out.dim == left.dim * right.dim
    assert out.elements() == expected


def test_hilbert_space_factorize_success_two_groups():
    h = HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=(i, j))
        for i in (0, 1)
        for j in ("a", "b", "c")
    )

    factorization = h.factorize((int,), (str,))
    left, right = factorization.factorized

    assert left.elements() == (
        U1Basis(coef=sy.Integer(1), base=(0,)),
        U1Basis(coef=sy.Integer(1), base=(1,)),
    )
    assert right.elements() == (
        U1Basis(coef=sy.Integer(1), base=("a",)),
        U1Basis(coef=sy.Integer(1), base=("b",)),
        U1Basis(coef=sy.Integer(1), base=("c",)),
    )
    assert factorization.align_dim.elements() == h.elements()


def test_hilbert_space_factorize_defaults_coef_to_leftmost_factor():
    h = HilbertSpace.new(
        U1Basis(coef=sy.Integer(i + 2), base=(i, j)) for i in (0, 1) for j in ("a", "b")
    )

    factorization = h.factorize((int,), (str,))
    left, right = factorization.factorized

    assert left.elements() == (
        U1Basis(coef=sy.Integer(2), base=(0,)),
        U1Basis(coef=sy.Integer(3), base=(1,)),
    )
    assert right.elements() == (
        U1Basis(coef=sy.Integer(1), base=("a",)),
        U1Basis(coef=sy.Integer(1), base=("b",)),
    )


def test_hilbert_space_factorize_places_coef_on_requested_factor():
    coef_by_orb = {"a": sy.Integer(5), "b": sy.Integer(7)}
    h = HilbertSpace.new(
        U1Basis(coef=coef_by_orb[j], base=(i, j)) for i in (0, 1) for j in ("a", "b")
    )

    factorization = h.factorize((int,), (str,), coef_on=1)
    left, right = factorization.factorized

    assert left.elements() == (
        U1Basis(coef=sy.Integer(1), base=(0,)),
        U1Basis(coef=sy.Integer(1), base=(1,)),
    )
    assert right.elements() == (
        U1Basis(coef=sy.Integer(5), base=("a",)),
        U1Basis(coef=sy.Integer(7), base=("b",)),
    )


def test_hilbert_space_factorize_accepts_negative_coef_index():
    coef_by_orb = {"a": sy.Integer(11), "b": sy.Integer(13)}
    h = HilbertSpace.new(
        U1Basis(coef=coef_by_orb[j], base=(i, j)) for i in (0, 1) for j in ("a", "b")
    )

    positive = h.factorize((int,), (str,), coef_on=1)
    negative = h.factorize((int,), (str,), coef_on=-1)

    assert negative.factorized == positive.factorized


def test_hilbert_space_factorize_rejects_missing_type():
    h = HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=(i, j)) for i in (0, 1) for j in ("a", "b")
    )
    with pytest.raises(ValueError, match="does not match space irrep types"):
        h.factorize((int,))


def test_hilbert_space_factorize_rejects_duplicate_type_request():
    h = HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=(i, j)) for i in (0, 1) for j in ("a", "b")
    )
    with pytest.raises(ValueError, match="must appear exactly once"):
        h.factorize((int,), (int, str))


def test_hilbert_space_factorize_rejects_incomplete_cartesian_product():
    h = HilbertSpace.new(
        (
            U1Basis(coef=sy.Integer(1), base=(1, "a")),
            U1Basis(coef=sy.Integer(1), base=(1, "b")),
            U1Basis(coef=sy.Integer(1), base=(2, "a")),
        )
    )
    with pytest.raises(ValueError, match="complete Cartesian product"):
        h.factorize((int,), (str,))


def test_hilbert_space_factorize_rejects_out_of_range_coef_index():
    h = HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=(i, j)) for i in (0, 1) for j in ("a", "b")
    )
    with pytest.raises(ValueError, match="`coef_on` index 2 is out of range"):
        h.factorize((int,), (str,), coef_on=2)


def test_hilbert_space_factorize_rejects_ambiguous_coef_assignment():
    h = HilbertSpace.new(
        (
            U1Basis(coef=sy.Integer(2), base=(0, "a")),
            U1Basis(coef=sy.Integer(3), base=(0, "b")),
            U1Basis(coef=sy.Integer(5), base=(1, "a")),
            U1Basis(coef=sy.Integer(7), base=(1, "b")),
        )
    )
    with pytest.raises(ValueError, match="does not determine a unique coefficient"):
        h.factorize((int,), (str,), coef_on=0)
