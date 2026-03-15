import pytest
import numpy as np
import torch
import qten


def test_set_precision_string():
    # 1. Test Single Precision ("32")
    qten.set_precision("32")
    assert torch.get_default_dtype() == torch.float32
    assert torch.tensor([1 + 1j]).dtype == torch.complex64

    # 2. Test Double Precision ("64")
    qten.set_precision("64")
    assert torch.get_default_dtype() == torch.float64
    assert torch.tensor([1 + 1j]).dtype == torch.complex128


def test_set_precision_torch_dtype():
    qten.set_precision(torch.float32)
    assert torch.get_default_dtype() == torch.float32

    qten.set_precision(torch.complex128)
    assert torch.get_default_dtype() == torch.float64


def test_set_precision_numpy_dtype():
    qten.set_precision(np.float32)
    assert torch.get_default_dtype() == torch.float32

    qten.set_precision(np.complex128)
    assert torch.get_default_dtype() == torch.float64


def test_set_precision_no_default_change():
    torch.set_default_dtype(torch.float32)
    qten.set_precision("64", set_torch_default=False)
    assert torch.get_default_dtype() == torch.float32


def test_set_precision_invalid():
    with pytest.raises(ValueError):
        qten.set_precision("16")
