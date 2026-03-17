from typing import List

from multipledispatch import dispatch

import sympy as sy
import torch

from ..symbolics import HilbertSpace
from ..symbolics import U1Basis
from .. import Tensor
from ..precision import get_precision_config

from ._bonds import Bond


class FFObservable:
    _bonds: List[Bond]

    def __init__(self):
        self._bonds = []

    @dispatch(Bond)
    def add_bond(self, bond: Bond):
        if not isinstance(bond, Bond):
            raise TypeError(f"Expected a Bond instance, got {type(bond).__name__}")
        self._bonds.append(bond)

    @dispatch(sy.Number, U1Basis, U1Basis)  # type: ignore[no-redef]
    def add_bond(self, coef: sy.Number, src: U1Basis, dst: U1Basis):
        bond = Bond(coef, (src, dst))
        self.add_bond(bond)

    def to_tensor(self) -> Tensor:
        basis_index: dict[U1Basis, int] = {}
        basis_order: list[U1Basis] = []
        bond_entries: list[tuple[int, int, complex]] = []

        for bond in self._bonds:
            left, right = bond.base
            left_ray = left.rays()
            right_ray = right.rays()

            i = basis_index.setdefault(left_ray, len(basis_order))
            if i == len(basis_order):
                basis_order.append(left_ray)

            j = basis_index.setdefault(right_ray, len(basis_order))
            if j == len(basis_order):
                basis_order.append(right_ray)

            value = complex(bond.coef * left.coef * sy.conjugate(right.coef))
            bond_entries.append((i, j, value))

        space = HilbertSpace.new(basis_order)
        precision = get_precision_config()
        data = torch.zeros((space.dim, space.dim), dtype=precision.torch_complex)

        for i, j, value in bond_entries:
            if i == j:
                data[i, i] += value.real
            else:
                data[i, j] += value
                data[j, i] += value.conjugate()

        return Tensor(data=data, dims=(space, space))
