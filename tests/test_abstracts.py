import pytest
from dataclasses import dataclass
from qten.abstracts import Functional, Operable, Updatable, operator_eq


@dataclass(frozen=True)
class MockOperable(Operable):
    pass


@dataclass(frozen=True)
class MockUpdatable(Updatable["MockUpdatable"]):
    val: int = 0

    def _updated(self, **kwargs) -> "Updatable":
        new_val = kwargs.get("val", self.val)
        return MockUpdatable(val=new_val)


@dataclass(frozen=True)
class BadUpdatable(Updatable["BadUpdatable"]):
    def _updated(self, **kwargs):
        return self


class _BaseInput:
    pass


class _DerivedInput(_BaseInput):
    pass


@dataclass(frozen=True)
class _MockFunctional(Functional):
    pass


@_MockFunctional.register(_BaseInput)
def _apply_mock_functional_base(functional: _MockFunctional, obj: _BaseInput) -> str:
    return "base"


def test_operable_unimplemented():
    a = MockOperable()
    b = MockOperable()

    # Test all default implementations raise NotImplementedError

    # Arithmetics
    with pytest.raises(NotImplementedError):
        _ = a + b

    with pytest.raises(NotImplementedError):
        _ = -a

    # a - b calls a + (-b). Since -b raises error, this should raise error too.
    with pytest.raises(NotImplementedError):
        _ = a - b

    with pytest.raises(NotImplementedError):
        _ = a * b

    with pytest.raises(NotImplementedError):
        _ = a @ b

    with pytest.raises(NotImplementedError):
        _ = a / b

    with pytest.raises(NotImplementedError):
        _ = a // b

    with pytest.raises(NotImplementedError):
        _ = a**b

    # Comparisons
    # Note: MockOperable is a dataclass, so it implements __eq__ automatically.
    # a == b will return True, not raise NotImplementedError.
    # So we skip testing '==' here or test operator_eq directly.
    with pytest.raises(NotImplementedError):
        operator_eq(a, b)

    with pytest.raises(NotImplementedError):
        _ = a < b

    with pytest.raises(NotImplementedError):
        _ = a <= b

    with pytest.raises(NotImplementedError):
        _ = a > b

    with pytest.raises(NotImplementedError):
        _ = a >= b

    # Logical
    with pytest.raises(NotImplementedError):
        _ = a & b

    with pytest.raises(NotImplementedError):
        _ = a | b


def test_updatable_correct():
    u = MockUpdatable(val=1)
    u2 = u.update(val=2)
    assert u2.val == 2
    assert u2 is not u


def test_updatable_bad_implementation():
    b = BadUpdatable()
    with pytest.raises(RuntimeError, match="must not return self"):
        b.update(foo="bar")


def test_functional_targeted_cache_invalidation():
    functional = _MockFunctional()
    obj = _DerivedInput()

    assert functional(obj) == "base"
    assert (
        _MockFunctional._resolved_methods[(type(obj), type(functional))](
            functional, obj
        )
        == "base"
    )

    @_MockFunctional.register(_DerivedInput)
    def _apply_mock_functional_derived(
        functional: _MockFunctional, obj: _DerivedInput
    ) -> str:
        return "derived"

    assert functional(obj) == "derived"
