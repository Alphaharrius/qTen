from dataclasses import dataclass
from threading import Event, Thread

import pytest

from pyhilbert.validations import need_validation, no_validate


def test_need_validation_rejects_non_dataclass_targets():
    with pytest.raises(TypeError, match="can only be applied to dataclasses"):

        @need_validation()
        class PlainClass:
            pass


def test_need_validation_runs_local_validators_once_on_construction():
    calls: list[int] = []

    def positive(instance: "LocalValidated") -> None:
        calls.append(instance.value)
        if instance.value <= 0:
            raise ValueError("value must be positive")

    @need_validation(positive)
    @dataclass(frozen=True)
    class LocalValidated:
        value: int

    instance = LocalValidated(3)

    assert instance.value == 3
    assert calls == [3]


def test_need_validation_propagates_validator_exceptions():
    def positive(instance: "Validated") -> None:
        if instance.value <= 0:
            raise ValueError("value must be positive")

    @need_validation(positive)
    @dataclass(frozen=True)
    class Validated:
        value: int

    with pytest.raises(ValueError, match="value must be positive"):
        Validated(0)


def test_need_validation_runs_inherited_validators_in_base_to_derived_order():
    calls: list[str] = []

    def base_validator(instance: object) -> None:
        calls.append("base")

    def middle_validator(instance: object) -> None:
        calls.append("middle")

    def leaf_validator(instance: object) -> None:
        calls.append("leaf")

    @need_validation(base_validator)
    @dataclass(frozen=True)
    class Base:
        value: int

    @need_validation(middle_validator)
    @dataclass(frozen=True)
    class Middle(Base):
        pass

    @need_validation(leaf_validator)
    @dataclass(frozen=True)
    class Leaf(Middle):
        pass

    Leaf(1)

    assert calls == ["base", "middle", "leaf"]


def test_need_validation_deduplicates_inherited_validators():
    calls: list[int] = []

    def shared_validator(instance: object) -> None:
        calls.append(1)

    @need_validation(shared_validator)
    @dataclass(frozen=True)
    class Base:
        value: int

    @need_validation(shared_validator)
    @dataclass(frozen=True)
    class Child(Base):
        pass

    Child(1)

    assert calls == [1]


def test_need_validation_runs_after_post_init_normalization():
    calls: list[int] = []

    def validator(instance: "Normalized") -> None:
        calls.append(instance.value)

    @need_validation(validator)
    @dataclass(frozen=True)
    class Normalized:
        value: int

        def __post_init__(self) -> None:
            object.__setattr__(self, "value", abs(self.value))

    instance = Normalized(-4)

    assert instance.value == 4
    assert calls == [4]


def test_need_validation_runs_inherited_validators_when_subclass_has_post_init():
    calls: list[str] = []

    def base_validator(instance: object) -> None:
        calls.append("base")

    def child_validator(instance: object) -> None:
        calls.append("child")

    @need_validation(base_validator)
    @dataclass(frozen=True)
    class Base:
        value: int

    @need_validation(child_validator)
    @dataclass(frozen=True)
    class Child(Base):
        normalized: int = 0

        def __post_init__(self) -> None:
            object.__setattr__(self, "normalized", self.value + 1)

    instance = Child(2)

    assert instance.normalized == 3
    assert calls == ["base", "child"]


def test_no_validate_disables_construction_time_validation():
    calls: list[int] = []

    def positive(instance: "Validated") -> None:
        calls.append(instance.value)
        if instance.value <= 0:
            raise ValueError("value must be positive")

    @need_validation(positive)
    @dataclass(frozen=True)
    class Validated:
        value: int

    with no_validate():
        instance = Validated(0)

    assert instance.value == 0
    assert calls == []


def test_no_validate_works_as_decorator():
    calls: list[int] = []

    def positive(instance: "Validated") -> None:
        calls.append(instance.value)
        if instance.value <= 0:
            raise ValueError("value must be positive")

    @need_validation(positive)
    @dataclass(frozen=True)
    class Validated:
        value: int

    @no_validate()
    def build_invalid() -> Validated:
        return Validated(0)

    instance = build_invalid()

    assert instance.value == 0
    assert calls == []


def test_no_validate_is_nestable():
    calls: list[int] = []

    def positive(instance: "Validated") -> None:
        calls.append(instance.value)
        if instance.value <= 0:
            raise ValueError("value must be positive")

    @need_validation(positive)
    @dataclass(frozen=True)
    class Validated:
        value: int

    with no_validate():
        with no_validate():
            instance = Validated(0)

    assert instance.value == 0
    assert calls == []


def test_no_validate_is_thread_local():
    calls: list[str] = []
    ready = Event()
    done = Event()

    def validator(instance: "Validated") -> None:
        calls.append(instance.name)
        if instance.name == "invalid":
            raise ValueError("invalid instance")

    @need_validation(validator)
    @dataclass(frozen=True)
    class Validated:
        name: str

    def worker() -> None:
        with no_validate():
            ready.set()
            done.wait(timeout=2)
            Validated("invalid")

    thread = Thread(target=worker)
    thread.start()
    ready.wait(timeout=2)

    with pytest.raises(ValueError, match="invalid instance"):
        Validated("invalid")

    done.set()
    thread.join(timeout=2)

    assert calls == ["invalid"]


def test_need_validation_with_empty_validator_list_inherits_ancestor_validators():
    calls: list[str] = []

    def base_validator(instance: object) -> None:
        calls.append("base")

    @need_validation(base_validator)
    @dataclass(frozen=True)
    class Base:
        value: int

    @need_validation()
    @dataclass(frozen=True)
    class Child(Base):
        pass

    Child(1)

    assert calls == ["base"]


def test_need_validation_runs_once_for_leaf_construction():
    calls: list[str] = []

    def base_validator(instance: object) -> None:
        calls.append("base")

    def child_validator(instance: object) -> None:
        calls.append("child")

    @need_validation(base_validator)
    @dataclass(frozen=True)
    class Base:
        value: int

    @need_validation(child_validator)
    @dataclass(frozen=True)
    class Child(Base):
        pass

    Child(1)

    assert calls.count("base") == 1
    assert calls.count("child") == 1
    assert calls == ["base", "child"]


def test_no_validate_does_not_swallow_post_init_exceptions():
    @need_validation()
    @dataclass(frozen=True)
    class Broken:
        value: int

        def __post_init__(self) -> None:
            raise RuntimeError("post-init failed")

    with no_validate():
        with pytest.raises(RuntimeError, match="post-init failed"):
            Broken(1)


def test_need_validation_preserves_wrapped_init_name():
    def validator(instance: object) -> None:
        return None

    @need_validation(validator)
    @dataclass(frozen=True)
    class Wrapped:
        value: int

    assert Wrapped.__init__.__name__ == "__init__"
