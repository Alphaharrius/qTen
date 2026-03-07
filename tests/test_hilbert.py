import pytest
from pyhilbert.hilbert import Mode, hilbert, HilbertSpace, MomentumSpace, brillouin_zone
from pyhilbert.spatials import Lattice
from pyhilbert.boundary import PeriodicBoundary
from pyhilbert.utils import FrozenDict
from sympy import ImmutableDenseMatrix


def test_mode_creation():
    attr = FrozenDict({"a": 1})
    m = Mode(count=2, attr=attr)
    assert m.count == 2
    assert m.dim == 2
    assert m["a"] == 1


def test_mode_getitem():
    m = Mode(count=1, attr=FrozenDict({"a": 1, "b": 2}))
    assert m["a"] == 1

    # Test multiple items getitem
    m_sub = m[("a",)]
    assert m_sub.attr == FrozenDict({"a": 1})

    with pytest.raises(NotImplementedError):
        _ = m[123]


def test_mode_update():
    m = Mode(count=1, attr=FrozenDict({"val": 10}))

    # Update with value
    m2 = m.update(val=20)
    assert m2.attr["val"] == 20
    assert m2 is not m

    # Update with function
    m3 = m.update(val=lambda x: x + 5)
    assert m3.attr["val"] == 15

    # Update unknown key (should remain unchanged or be added? Code: old = updated_attr.get(k, _MISSING).
    # If function, and old is missing, continue.
    # If value, add it.)

    m4 = m.update(new_key=100)
    assert m4.attr["new_key"] == 100

    m5 = m.update(new_key=lambda x: x)  # Should do nothing
    assert "new_key" not in m5.attr


def test_hilbert_space_creation():
    attr1 = FrozenDict({"id": 1})
    m1 = Mode(count=2, attr=attr1)

    attr2 = FrozenDict({"id": 2})
    m2 = Mode(count=3, attr=attr2)

    hs = hilbert([m1, m2])
    assert isinstance(hs, HilbertSpace)
    # The dimension of the StateSpace is the sum of sector dimensions (mode counts),
    # so 2 + 3 = 5.
    assert hs.dim == 5
    assert len(hs.elements()) == 2


def test_hilbert_space_operations():
    m1 = Mode(count=1, attr=FrozenDict({"id": 1}))
    m2 = Mode(count=1, attr=FrozenDict({"id": 2}))
    m3 = Mode(count=1, attr=FrozenDict({"id": 3}))

    s1 = hilbert([m1, m2])
    s2 = hilbert([m2, m3])

    # Union (add / or)
    s_union = s1 + s2
    # Should contain m1, m2, m3
    assert s_union.dim == 3
    assert m1 in s_union.structure
    assert m2 in s_union.structure
    assert m3 in s_union.structure
    # Order should be m1, m2, m3
    keys = list(s_union.structure.keys())
    assert keys == [m1, m2, m3]

    # Intersection (and)
    s_inter = s1 & s2
    assert s_inter.dim == 1
    assert m2 in s_inter.structure
    assert m1 not in s_inter.structure

    # Difference (sub)
    s_diff = s1 - s2
    assert s_diff.dim == 1
    assert m1 in s_diff.structure
    assert m2 not in s_diff.structure

    # Difference reverse
    s_diff2 = s2 - s1
    assert s_diff2.dim == 1
    assert m3 in s_diff2.structure


def test_hilbert_space_update():
    m1 = Mode(count=1, attr=FrozenDict({"val": 1}))
    s = hilbert([m1])

    s2 = s.update(val=2)
    new_m = list(s2.structure.keys())[0]
    assert new_m.attr["val"] == 2


def test_statespace_getitem():
    m1 = Mode(count=2, attr=FrozenDict({"id": 1}))
    m2 = Mode(count=1, attr=FrozenDict({"id": 2}))
    m3 = Mode(count=3, attr=FrozenDict({"id": 3}))
    s = hilbert([m1, m2, m3])

    assert s[0] == m1
    assert s[-1] == m3

    s_slice = s[1:3]
    assert isinstance(s_slice, HilbertSpace)
    assert list(s_slice.structure.keys()) == [m2, m3]
    assert s_slice.dim == 4

    s_range = s[range(0, 2)]
    assert isinstance(s_range, HilbertSpace)
    assert list(s_range.structure.keys()) == [m1, m2]
    assert s_range.dim == 3

    with pytest.raises(IndexError):
        _ = s[3]

    with pytest.raises(TypeError):
        _ = s["bad"]


def test_momentum_space_brillouin():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    recip = lat.dual

    ms = brillouin_zone(recip)
    assert isinstance(ms, MomentumSpace)
    assert ms.dim == 4  # 2x2 points

    assert str(ms) == "MomentumSpace(4)"
    assert "MomentumSpace(4):" in repr(ms)


def test_statespace_errors():
    m1 = Mode(count=1, attr=FrozenDict({"id": 1}))
    s1 = hilbert([m1])

    # Create a MomentumSpace manually to test cross-type operations
    ms = MomentumSpace(structure=s1.structure)

    # Different types addition - raises ValueError
    with pytest.raises(ValueError, match="different types"):
        s1 + ms

    with pytest.raises(ValueError, match="different types"):
        s1 - ms

    with pytest.raises(ValueError, match="different types"):
        s1 & ms


def test_hilbert_update_error():
    # Inject a non-Mode object into HilbertSpace structure to trigger error in _updated
    # This is internal tampering but needed to hit the error branch line 181

    class FakeMode:
        pass

    # We can't easily inject into frozen dataclass or validate types during init easily
    # But HilbertSpace expects keys to be Modes.
    # Let's try to subclass and cheat or use replace if possible, but replace checks types? No.
    pass
    # Skipping this specific error branch as it requires invalid state construction.


def test_hilbert_space_mode_lookup():
    m1 = Mode(count=1, attr=FrozenDict({"id": 1, "type": "a"}))
    m2 = Mode(count=1, attr=FrozenDict({"id": 2, "type": "b"}))
    m3 = Mode(count=1, attr=FrozenDict({"id": 3, "type": "a"}))

    hs = hilbert([m1, m2, m3])

    # Test successful lookup
    found_mode = hs.mode_lookup(id=2)
    assert found_mode is m2

    # Test lookup with multiple attributes
    found_mode_2 = hs.mode_lookup(id=3, type="a")
    assert found_mode_2 is m3

    # Test for no mode found
    with pytest.raises(ValueError, match="No mode found"):
        hs.mode_lookup(id=4)

    # Test for multiple modes found
    with pytest.raises(ValueError, match="Multiple modes found"):
        hs.mode_lookup(type="a")
