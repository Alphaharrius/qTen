"""Site-irrep inventory, spinful D(g,k), and little-group characters.

No notebook import and no network. These tests cover the library APIs used
to name frozen filled/empty modes as atomic insulators.
"""

from collections import OrderedDict

import pytest
import torch
import sympy as sy
from sympy import ImmutableDenseMatrix

import qten.ops as Q
from qten.bands import get_band_transform
from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.fourier import fourier_kernel
from qten.geometries.spatials import AffineSpace, KPointSet, Lattice, Momentum, Offset
from qten.linalg.tensors import Tensor
from qten.phys import Spin
from qten.pointgroups import (
    PointGroupOpr,
    SpinorIrrepSector,
    axis_stabilizer,
    hilbert_repr,
    irrep_content,
    little_group,
    little_group_irreps,
    point_group_column_symmetrize,
    pointgroup,
)
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import IndexSpace, MomentumSpace, brillouin_zone


def _gamma8_label(group) -> str:
    for row in group.spinor_table()["irreps"].values():
        if row.get("bilbao_label") == "\u03938":
            return str(row["bilbao_label"])
    raise AssertionError("Td spinor table is missing Γ8")


def _td_tetrahedron_spinful():
    td = pointgroup("-43m")
    affine = AffineSpace(basis=ImmutableDenseMatrix.eye(3))
    center = Offset(ImmutableDenseMatrix.zeros(3, 1), affine)
    vertices = (
        Offset(ImmutableDenseMatrix([1, 1, 1]), affine),
        Offset(ImmutableDenseMatrix([1, -1, -1]), affine),
        Offset(ImmutableDenseMatrix([-1, 1, -1]), affine),
        Offset(ImmutableDenseMatrix([-1, -1, 1]), affine),
    )
    space = HilbertSpace.new(
        U1Basis.new(vertex, spin)
        for vertex in vertices
        for spin in (Spin.up, Spin.down)
    )
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    projected = point_group_column_symmetrize(
        td, seed, full_sector=True, fixpoint=center
    )
    return td, center, space, projected


def _1d_two_site_spinful():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "A": ImmutableDenseMatrix([0]),
            "B": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    space = HilbertSpace.new(
        [
            U1Basis.new(lattice.at("A"), Spin.up),
            U1Basis.new(lattice.at("A"), Spin.down),
            U1Basis.new(lattice.at("B"), Spin.up),
            U1Basis.new(lattice.at("B"), Spin.down),
        ]
    )
    cs = pointgroup("Cs", plane="x")
    mirror = next(element for element in cs.elements() if element.group_order() == 2)
    origin = Offset(ImmutableDenseMatrix([0]), lattice)
    opr = PointGroupOpr(mirror).fixpoint_at(origin, rebase=True)
    return lattice, space, opr


def test_irrep_content_counts_td_gamma6_gamma7_gamma8():
    td, _center, _space, projected = _td_tetrahedron_spinful()
    inventory = irrep_content(projected, td)
    assert inventory == Q.irrep_content(projected, td)
    labels = set(inventory)
    assert "\u03936" in labels
    assert "\u03937" in labels
    assert "\u03938" in labels
    for row in inventory.values():
        assert row["columns"] == row["multiplicity"] * row["dim"]
        assert row["columns"] % row["dim"] == 0


def test_irrep_content_trace_count_matches_labels_with_fixpoint():
    td, center, _space, projected = _td_tetrahedron_spinful()
    assert irrep_content(projected, td, fixpoint=center) == irrep_content(projected, td)


def test_irrep_content_warns_on_partial_labelled_block():
    td, center, space, _projected = _td_tetrahedron_spinful()
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128)[:, :1],
        dims=(space, IndexSpace.linear(1)),
    )
    single = point_group_column_symmetrize(td, seed, full_sector=True, fixpoint=center)
    with pytest.warns(UserWarning, match="whole irrep copies"):
        irrep_content(single, td)
    with pytest.warns(UserWarning, match="not integers"):
        irrep_content(single, td, fixpoint=center)


def test_irrep_content_rejects_labels_from_another_group():
    _td, _center, _space, projected = _td_tetrahedron_spinful()
    with pytest.raises(ValueError, match="labelled by point group"):
        irrep_content(projected, pointgroup("m-3m"))


def test_spinful_dgk_at_gamma_matches_hilbert_repr():
    lattice, space, opr = _1d_two_site_spinful()
    k_space = brillouin_zone(lattice.dual)
    transform = get_band_transform(opr, k_space, space)
    gamma_repr = hilbert_repr(opr, space)

    gamma = next(
        i
        for i, k in enumerate(k_space.elements())
        if all(sy.simplify(k.rep[j, 0]) == 0 for j in range(k.space.dim))
    )
    assert torch.allclose(transform.data[gamma], gamma_repr.data, rtol=0, atol=1e-12)


def test_spinful_dgk_wrap_phase_at_zone_edge():
    lattice, space, opr = _1d_two_site_spinful()
    k_space = brillouin_zone(lattice.dual)
    transform = get_band_transform(opr, k_space, space)
    gamma_repr = hilbert_repr(opr, space)
    images = tuple(opr @ psi.irrep_of(Offset) for psi in space.elements())
    mapped = tuple(pair[0] for pair in transform.dims[0].elements())
    phases = fourier_kernel(mapped, images)
    expected = gamma_repr.data.unsqueeze(0) * phases.unsqueeze(1)
    assert torch.allclose(transform.data, expected, rtol=0, atol=1e-12)

    edge = next(
        i
        for i, k in enumerate(k_space.elements())
        if sy.simplify(k.rep[0, 0] - sy.Rational(1, 2)) == 0
    )
    assert not torch.allclose(transform.data[edge], gamma_repr.data, rtol=0, atol=1e-8)


def test_get_band_transform_spinful_two_site_cell_does_not_raise():
    lattice, space, opr = _1d_two_site_spinful()
    k_space = brillouin_zone(lattice.dual)
    transform = get_band_transform(opr, k_space, space)
    assert transform.data.shape == (k_space.dim, space.dim, space.dim)


def test_little_group_gamma_is_full_group_zone_boundary_is_proper_subgroup():
    c4v = pointgroup("4mm", plane="xy")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    gamma = Momentum(ImmutableDenseMatrix([0, 0]), lattice.dual)
    x_point = Momentum(ImmutableDenseMatrix([sy.Rational(1, 2), 0]), lattice.dual)
    g_gamma = little_group(c4v, gamma)
    g_x = little_group(c4v, x_point)
    assert g_gamma is c4v
    assert g_gamma.order == c4v.order
    assert g_x.order < c4v.order
    assert g_x.order >= 1


def test_little_group_oh_at_cubic_x_is_proper_subgroup():
    oh = pointgroup("m-3m")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0, 0])},
    )
    gamma = Momentum(ImmutableDenseMatrix([0, 0, 0]), lattice.dual)
    x_point = Momentum(ImmutableDenseMatrix([sy.Rational(1, 2), 0, 0]), lattice.dual)
    g_gamma = little_group(oh, gamma)
    g_x = little_group(oh, x_point)
    assert g_gamma is oh
    assert oh.order == 48
    assert g_x.order == 16


def test_little_group_1d_cs_interior_is_identity():
    cs = pointgroup("Cs", plane="x")
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
    )
    gamma = Momentum(ImmutableDenseMatrix([0]), lattice.dual)
    interior = Momentum(ImmutableDenseMatrix([sy.Rational(1, 4)]), lattice.dual)
    edge = Momentum(ImmutableDenseMatrix([sy.Rational(1, 2)]), lattice.dual)
    assert little_group(cs, gamma).order == cs.order
    assert little_group(cs, interior).order == 1
    assert little_group(cs, edge).order == cs.order


def test_little_group_irreps_td_gamma8_at_gamma():
    td, center, space, projected = _td_tetrahedron_spinful()
    gamma8 = _gamma8_label(td)
    spinor_name = next(
        name
        for name, row in td.spinor_table()["irreps"].items()
        if row.get("bilbao_label") == gamma8
    )
    keep = [
        i
        for i, psi in enumerate(projected.dims[1].elements())
        if psi.irrep_of(SpinorIrrepSector).irrep == spinor_name
    ]
    assert keep
    labels = [projected.dims[1].elements()[i] for i in keep]
    columns = Tensor(
        data=projected.data[:, keep],
        dims=(space, HilbertSpace.new(labels)),
    )
    dummy = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2, 2)),
    )
    kpts = KPointSet.from_points(dummy.dual, {"Gamma": (0, 0, 0)})
    gamma = next(iter(kpts.points.values()))
    k_space = MomentumSpace(structure=OrderedDict(((gamma, 0),)))
    states = Tensor(
        data=columns.data.unsqueeze(0),
        dims=(k_space, space, columns.dims[1]),
    )
    content = little_group_irreps(states, td, kpts, fixpoint=center)
    copies = len(keep) // 4
    assert content["Gamma"][gamma8] == copies
    inventory = irrep_content(columns, td)
    assert inventory[gamma8]["multiplicity"] == copies


def test_little_group_irreps_rank2_fourier_to_named_k():
    lattice, space, _opr = _1d_two_site_spinful()
    cs = pointgroup("Cs", plane="x")
    origin = Offset(ImmutableDenseMatrix([0]), lattice)
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    projected = point_group_column_symmetrize(
        cs, seed, full_sector=True, fixpoint=origin
    )
    kpts = KPointSet.from_points(
        lattice.dual,
        {"Gamma": (0,), "X": (sy.Rational(1, 2),)},
    )
    content = little_group_irreps(projected, cs, kpts, fixpoint=origin)
    assert set(content) == {"Gamma", "X"}
    assert content["Gamma"]
    assert content["X"]


def test_little_group_irreps_rank2_folds_periodic_images_of_same_site():
    lattice, space, _opr = _1d_two_site_spinful()
    cs = pointgroup("Cs", plane="x")
    origin = Offset(ImmutableDenseMatrix([0]), lattice)
    translated = HilbertSpace.new(
        [
            U1Basis.new(lattice.at("A", (1,)), Spin.up),
            U1Basis.new(lattice.at("A", (1,)), Spin.down),
            *space.elements(),
        ]
    )
    seed = Tensor(
        data=torch.eye(translated.dim, dtype=torch.complex128),
        dims=(translated, IndexSpace.linear(translated.dim)),
    )
    kpts = KPointSet.from_points(lattice.dual, {"Gamma": (0,)})
    content = little_group_irreps(seed, cs, kpts, fixpoint=origin)
    assert "Gamma" in content
    assert content["Gamma"]


def test_rank2_fourier_off_mesh_uses_unwrapped_cells():
    from qten.pointgroups.ops import _states_on_kpoints

    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(3)),
        unit_cell={"A": ImmutableDenseMatrix([0])},
    )
    # Bond orbital on cells 0 and -1; the -1 site is stored folded at cell 2.
    region = HilbertSpace.new(
        [
            U1Basis.new(lattice.at("A"), Spin.up),
            U1Basis.new(lattice.at("A", (-1,)), Spin.up),
        ]
    )
    seed = Tensor(
        data=torch.ones((2, 1), dtype=torch.complex128) / 2**0.5,
        dims=(region, IndexSpace.linear(1)),
    )
    kpts = KPointSet.from_points(
        lattice.dual,
        {"Gamma": (0,), "X": (sy.Rational(1, 2),), "K": (sy.Rational(1, 3),)},
    )
    crystal, bloch = _states_on_kpoints(seed, kpts)
    assert bloch.dim == 1
    by_name = {
        name: crystal.data[i, 0, 0] for i, name in enumerate(("Gamma", "X", "K"))
    }
    # (1 + e^{2πiκ}) / sqrt(2) for the unwrapped cell -1.
    for name, kappa in (("Gamma", 0.0), ("X", 0.5), ("K", 1 / 3)):
        phase = torch.tensor(2j * torch.pi * kappa, dtype=torch.complex128)
        expected = (1 + torch.exp(phase)) / 2**0.5
        assert torch.allclose(by_name[name], expected, atol=1e-12)


def _2d_axis_two_site_spinful():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={
            "A": ImmutableDenseMatrix([0, 0]),
            "B": ImmutableDenseMatrix([sy.Rational(1, 2), 0]),
        },
    )
    space = HilbertSpace.new(
        U1Basis.new(site, spin)
        for site in lattice.unit_cell.values()
        for spin in (Spin.up, Spin.down)
    )
    origin = Offset(ImmutableDenseMatrix([0, 0]), lattice)
    return lattice, space, origin


def test_axis_stabilizer_1d_cs_line_is_full_group_ray_is_identity():
    cs = pointgroup("Cs", plane="x")
    line = axis_stabilizer(cs, (1,))
    ray = axis_stabilizer(cs, (1,), oriented=True)
    assert line is cs
    assert ray.order == 1


def test_axis_stabilizer_2d_c4v_line_is_proper_subgroup():
    c4v = pointgroup("4mm", plane="xy")
    line = axis_stabilizer(c4v, (1, 0))
    ray = axis_stabilizer(c4v, (1, 0), oriented=True)
    assert line.order == 4
    assert ray.order == 2
    assert Q.axis_stabilizer(c4v, (1, 0)).order == 4
    assert line is not c4v
    assert axis_stabilizer(line, (1, 0)) is line


def test_axis_stabilizer_closes_2d_sites_on_the_axis():
    _lattice, space, origin = _2d_axis_two_site_spinful()
    group = axis_stabilizer(pointgroup("4mm", plane="xy"), (1, 0))
    for element in group.elements():
        opr = PointGroupOpr(element).fixpoint_at(origin, rebase=True)
        hilbert_repr(opr, space)


def test_parent_group_need_not_close_sites_off_the_axis():
    _lattice, space, origin = _2d_axis_two_site_spinful()
    c4v = pointgroup("4mm", plane="xy")
    with pytest.raises(ValueError, match="is not in space"):
        for element in c4v.elements():
            opr = PointGroupOpr(element).fixpoint_at(origin, rebase=True)
            hilbert_repr(opr, space)


def test_little_group_irreps_axis_stabilizer_at_gamma():
    lattice, space, origin = _2d_axis_two_site_spinful()
    group = axis_stabilizer(pointgroup("4mm", plane="xy"), (1, 0))
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    kpts = KPointSet.from_points(lattice.dual, {"Gamma": (0, 0)})
    content = little_group_irreps(seed, group, kpts, fixpoint=origin)
    assert content["Gamma"]
    assert all(label.startswith("\u0393") for label in content["Gamma"])
    assert not any(key.startswith("spinor_") for key in content["Gamma"])


def test_little_group_irreps_zone_boundary_uses_koster_labels():
    c4v = pointgroup("4mm", plane="xy")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    space = HilbertSpace.new(
        U1Basis.new(lattice.at("r"), spin) for spin in (Spin.up, Spin.down)
    )
    origin = Offset(ImmutableDenseMatrix([0, 0]), lattice)
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    kpts = KPointSet.from_points(
        lattice.dual,
        {"Gamma": (0, 0), "X": (sy.Rational(1, 2), 0)},
    )
    content = little_group_irreps(seed, c4v, kpts, fixpoint=origin)
    for name in ("Gamma", "X"):
        assert content[name]
        assert all(label.startswith("\u0393") for label in content[name])
        assert not any(key.startswith("spinor_") for key in content[name])


def test_little_group_spinor_table_has_koster_labels():
    c4v = pointgroup("4mm", plane="xy")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    x_point = Momentum(ImmutableDenseMatrix([sy.Rational(1, 2), 0]), lattice.dual)
    g_x = little_group(c4v, x_point)
    assert g_x.order < c4v.order
    labels = [row.get("bilbao_label") for row in g_x.spinor_table()["irreps"].values()]
    assert all(labels)
    assert all(str(label).startswith("\u0393") for label in labels)


def test_little_group_oh_at_x_spinor_labels_are_koster():
    oh = pointgroup("m-3m")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0, 0])},
    )
    x_point = Momentum(ImmutableDenseMatrix([sy.Rational(1, 2), 0, 0]), lattice.dual)
    g_x = little_group(oh, x_point)
    labels = [row["bilbao_label"] for row in g_x.spinor_table()["irreps"].values()]
    assert all(label.startswith("\u0393") for label in labels)
    assert not any(name.startswith("spinor_") for name in labels)
