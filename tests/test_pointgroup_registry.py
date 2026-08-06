import numpy as np
import sympy as sy

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice
from qten.pointgroups import (
    PointGroupElement,
    PointGroupOpr,
    FinitePointGroup,
    PointGroupBasis,
    pointgroup,
)
from qten.pointgroups._registry import (
    _double_point_group_data,
    _point_group_data,
    known_point_group_symbols,
)
from qten.phys import su2_of_point_group


def test_existing_affine_pointgroup_queries_return_point_group_element():
    c4 = pointgroup("c4-xy:xy")

    assert isinstance(c4, PointGroupElement)
    assert c4.axes == sy.symbols("x y")
    assert c4.group_order() == 4


def test_named_pointgroup_uses_packaged_generator_data():
    c4v = pointgroup("C4v-xy")

    assert isinstance(c4v, FinitePointGroup)
    assert c4v.symbol == "4mm"
    assert c4v.axes == sy.symbols("x y")
    assert c4v.order == 8
    assert not c4v.is_abelian()
    assert all(isinstance(generator, PointGroupElement) for generator in c4v.generators)


def test_named_pointgroup_supports_hermann_mauguin_symbol():
    group = pointgroup("4mm")

    assert isinstance(group, FinitePointGroup)
    assert group.symbol == "4mm"
    assert group.axes == sy.symbols("x y z")
    assert group.order == 8


def test_known_point_group_symbols_include_crystallographic_classes():
    symbols = set(known_point_group_symbols())

    assert "4mm" in symbols
    assert "mmm" in symbols
    assert "m-3m" in symbols


def test_packaged_data_uses_per_group_records():
    data = _point_group_data()
    records = {record["symbol"]: record for record in data["point_groups"]}
    c4v = records["4mm"]

    assert c4v["aliases"] == ["C4v"]
    assert c4v["source_encoding"] == "gj"
    assert c4v["generators"] == [
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
    ]
    assert c4v["irreps"]["source"] == "bilbao"
    assert c4v["irreps"]["class_labels"] == ["1", "2", "4", "m100", "m1-10"]
    assert c4v["irreps"]["multiplicities"] == [1, 1, 2, 2, 2]
    assert c4v["irreps"]["irreps"]["A1"]["characters"] == [1, 1, 1, 1, 1]
    assert c4v["irreps"]["irreps"]["A2"]["characters"] == [1, 1, 1, -1, -1]
    assert c4v["irreps"]["irreps"]["E"]["dim"] == 2
    assert c4v["irreps"]["irreps"]["E"]["characters"] == [2, -2, 0, 0, 0]


def test_all_packaged_character_tables_are_complete():
    for record in _point_group_data()["point_groups"]:
        table = record["irreps"]
        irreps = table["irreps"]
        group_order = sum(int(value) for value in table["multiplicities"])

        assert sum(int(row["dim"]) ** 2 for row in irreps.values()) == group_order
        assert any(
            all(character == 1 for character in row["characters"])
            for row in irreps.values()
        )


def test_all_packaged_spinor_tables_are_complete_and_orthonormal():
    data = _double_point_group_data()
    records = {record["symbol"]: record for record in data["point_groups"]}

    assert data["schema_version"] == 1
    assert data["source"]["name"] == "spgrep"
    assert data["section_convention"] == "qten-su2-principal-v1"
    assert set(records) == set(known_point_group_symbols())

    for symbol, record in records.items():
        group = pointgroup(symbol)
        assert group.spinor_irreps is not None
        assert record["order"] == group.order
        assert len(record["operations"]) == group.order
        assert len(record["factor_system"]) == group.order
        assert all(len(row) == group.order for row in record["factor_system"])
        assert {int(value) for row in record["factor_system"] for value in row} <= {
            -1,
            1,
        }

        irreps = record["irreps"]
        assert sum(int(irrep["dim"]) ** 2 for irrep in irreps.values()) == group.order
        characters = np.asarray(
            [
                [complex(*value) for value in irrep["characters"]]
                for irrep in irreps.values()
            ]
        )
        assert np.allclose(
            characters.conj() @ characters.T / group.order,
            np.eye(len(irreps)),
            rtol=0,
            atol=1e-12,
        )
        for irrep_name in irreps:
            assert len(group.spinor_irrep_characters_by_element(irrep_name)) == (
                group.order
            )


def test_bilbao_parser_splits_complex_conjugate_irrep_rows():
    data = _point_group_data()
    records = {record["symbol"]: record for record in data["point_groups"]}
    c6_irreps = records["6"]["irreps"]["irreps"]

    assert "^1E2" in c6_irreps
    assert "^2E2" in c6_irreps
    assert c6_irreps["^1E2"]["dim"] == 1
    assert c6_irreps["^2E2"]["dim"] == 1


def test_hexagonal_rotoinversion_classes_align_with_character_tables():
    for symbol in ("6/m", "6/mmm"):
        group = pointgroup(symbol)

        assert isinstance(group, FinitePointGroup)
        assert len(group.element_class_indices()) == group.order

        projector_sum = sum(
            (group.irrep_projector(1, irrep) for irrep in group.irreps["irreps"]),
            sy.zeros(3, 3),
        )
        assert all(
            abs(complex(sy.N(entry))) < 1e-10 for entry in projector_sum - sy.eye(3)
        )


def test_trigonal_and_hexagonal_generators_are_cartesian_spin_rotations():
    symbols = (
        "3",
        "-3",
        "32",
        "3m",
        "-3m",
        "6",
        "-6",
        "6/m",
        "622",
        "6mm",
        "-6m2",
        "6/mmm",
    )
    for symbol in symbols:
        group = pointgroup(symbol)
        for generator in group.generators:
            rotation = generator.irrep
            assert sy.simplify(rotation.T @ rotation) == sy.eye(3)
            lift = su2_of_point_group(generator)
            assert sy.simplify(lift.H @ lift) == sy.eye(2)
            assert sy.simplify(lift.det()) == 1


def test_diagonal_c4v_generator_keeps_b1_b2_geometric_labels():
    x, y = sy.symbols("x y")
    registry_group = pointgroup("C4v-xy")

    assert isinstance(registry_group, FinitePointGroup)
    rotation = sy.ImmutableDenseMatrix([[0, -1], [1, 0]])
    diagonal_mirror = sy.ImmutableDenseMatrix([[0, 1], [1, 0]])
    diagonal_group = FinitePointGroup.from_matrices(
        (rotation, diagonal_mirror),
        axes=(x, y),
        symbol="4mm",
        irreps=registry_group.irreps,
    )

    assert {
        sy.simplify(basis.expr) for basis in diagonal_group.irrep_basis(2, "B1")
    } == {x**2 - y**2}
    assert {
        sy.simplify(basis.expr) for basis in diagonal_group.irrep_basis(2, "B2")
    } == {x * y}


def test_trivial_projector_returns_invariant_sector():
    x, y = sy.symbols("x y")
    c4v = pointgroup("C4v-xy")

    assert isinstance(c4v, FinitePointGroup)
    assert c4v.trivial_projector(1) == sy.ImmutableDenseMatrix.zeros(2, 2)

    invariant_basis = c4v.invariant_basis(2)
    invariant_exprs = {sy.simplify(basis.expr) for basis in invariant_basis}

    assert sy.simplify(x**2 + y**2) in invariant_exprs


def test_c4v_irrep_basis_uses_bilbao_character_table():
    x, y = sy.symbols("x y")
    c4v = pointgroup("C4v-xy")

    assert isinstance(c4v, FinitePointGroup)

    a1 = c4v.irrep_basis(order=2, irrep="A1")
    b1 = c4v.irrep_basis(order=2, irrep="B1")
    b2 = c4v.irrep_basis(order=2, irrep="B2")
    e = c4v.irrep_basis(order=1, irrep="E")

    assert all(isinstance(basis, PointGroupBasis) for basis in a1 + b1 + b2 + e)
    assert {sy.simplify(basis.expr) for basis in a1} == {sy.simplify(x**2 + y**2)}
    assert {sy.simplify(basis.expr) for basis in b1} == {sy.simplify(x**2 - y**2)}
    assert {sy.simplify(basis.expr) for basis in b2} == {sy.simplify(x * y)}
    assert {sy.simplify(basis.expr) for basis in e} == {x, y}


def test_generated_group_element_transforms_point_group_basis():
    x, y = sy.symbols("x y")
    c4v = pointgroup("C4v-xy")

    assert isinstance(c4v, FinitePointGroup)
    rotation = c4v.generators[0]
    e_basis = {basis.expr: basis for basis in c4v.irrep_basis(order=1, irrep="E")}

    x_transformed = rotation @ e_basis[x]
    y_transformed = rotation @ e_basis[y]

    assert sy.simplify(x_transformed.coef) == 1
    assert isinstance(x_transformed.base, PointGroupBasis)
    assert sy.simplify(x_transformed.base.expr - y) == 0
    assert sy.simplify(y_transformed.base.expr + x) == 0


def test_affine_wrapper_transforms_point_group_basis_through_linear_part():
    x, y = sy.symbols("x y")
    c4v = pointgroup("C4v-xy")

    assert isinstance(c4v, FinitePointGroup)
    e_basis = {basis.expr: basis for basis in c4v.irrep_basis(order=1, irrep="E")}
    transformed = PointGroupOpr(c4v.generators[0]) @ e_basis[x]

    assert sy.simplify(transformed.base.expr - y) == 0


def test_manual_c4_and_registry_c4_share_polynomial_basis_but_not_sector_basis():
    x, y = sy.symbols("x y")
    manual_c4 = pointgroup("c4-xy:xy")
    c4v = pointgroup("C4v-xy")

    assert isinstance(manual_c4, PointGroupElement)
    assert isinstance(c4v, FinitePointGroup)

    registry_c4 = c4v.generators[0]
    assert manual_c4.irrep == registry_c4.irrep
    assert manual_c4.axes == registry_c4.axes
    assert manual_c4.euclidean_basis(order=2) == registry_c4.euclidean_basis(order=2)
    assert manual_c4.euclidean_repr(order=2) == registry_c4.euclidean_repr(order=2)

    manual_sector_basis = manual_c4.basis(order=1)
    manual_sector_exprs = {
        sy.simplify(basis.expr) for basis in manual_sector_basis.values()
    }
    finite_sector_exprs = {
        sy.simplify(basis.expr) for basis in c4v.irrep_basis(order=1, irrep="E")
    }

    assert set(manual_sector_basis) == {sy.I, -sy.I}
    assert all(
        sy.simplify(manual_c4.euclidean_repr(1) @ basis.rep - phase * basis.rep)
        == sy.ImmutableDenseMatrix.zeros(2, 1)
        for phase, basis in manual_sector_basis.items()
    )
    assert manual_sector_exprs != finite_sector_exprs
    assert finite_sector_exprs == {x, y}


def test_non_abelian_tetrahedral_point_group_preserves_diamond_lattice_basis():
    x, y, z = sy.symbols("x y z")
    half = sy.Rational(1, 2)
    quarter = sy.Rational(1, 4)
    diamond = Lattice(
        basis=sy.ImmutableDenseMatrix(
            [
                [0, half, half],
                [half, 0, half],
                [half, half, 0],
            ]
        ),
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.diag(3, 3, 3)),
        unit_cell={
            "A": sy.ImmutableDenseMatrix([0, 0, 0]),
            "B": sy.ImmutableDenseMatrix([quarter, quarter, quarter]),
        },
    )
    td = pointgroup("-43m")

    assert isinstance(td, FinitePointGroup)
    assert td.order == 24
    assert not td.is_abelian()
    assert {sy.simplify(basis.expr) for basis in td.irrep_basis(1, "T2")} == {
        x,
        y,
        z,
    }

    sites = (diamond.at("A"), diamond.at("B"))
    for element in td.elements():
        opr = PointGroupOpr(element)
        assert all(opr @ site in diamond for site in sites)
