import pytest
from pyhilbert.utils import Device, FrozenDict


def test_frozendict_creation_and_access():
    d = FrozenDict({"a": 1, "b": 2})
    assert len(d) == 2
    assert d["a"] == 1
    assert d["b"] == 2

    with pytest.raises(KeyError):
        _ = d["c"]


def test_frozendict_immutability():
    d = FrozenDict({"a": 1})

    # It's a Mapping, so it doesn't have __setitem__ or __delitem__ exposed by default if not implemented.
    # But we should check it doesn't allow modification.
    with pytest.raises(TypeError):
        d["a"] = 2


def test_frozendict_hash():
    d1 = FrozenDict({"a": 1, "b": 2})
    d2 = FrozenDict({"b": 2, "a": 1})  # Order shouldn't matter

    assert hash(d1) == hash(d2)
    assert d1 == d2

    # Can be used as dictionary key
    mapping = {d1: "value"}
    assert mapping[d2] == "value"


def test_frozendict_eq():
    d1 = FrozenDict({"a": 1})
    d2 = {"a": 1}
    assert d1 == d2  # Should compare equal to dict

    d3 = FrozenDict({"a": 2})
    assert d1 != d3


def test_device_new_accepts_supported_forms():
    assert Device.new("cpu") == Device("cpu")
    assert Device.new("gpu") == Device("gpu")
    assert Device.new("gpu:0") == Device("gpu", 0)
    assert Device.new("gpu:12") == Device("gpu", 12)


@pytest.mark.parametrize(
    "name", ["gpu0", "gpu1", "gpu:", "gpu:-1", "gpu:abc", "gpu:1:2"]
)
def test_device_new_rejects_invalid_gpu_forms(name):
    with pytest.raises(ValueError, match=f"Invalid device name: {name}"):
        Device.new(name)
