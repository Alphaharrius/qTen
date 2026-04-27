"""
Free-fermionic observable construction from symbolic bond terms.

This module provides the machinery for collecting weighted transitions between
[`U1Basis`][qten.symbolics.hilbert_space.U1Basis] states and turning them into a
Hermitian matrix tensor. The central object,
[`FFObservable`][qten.phys.FFObservable], stores symbolic
[`Bond`][qten.phys.Bond] terms first, then chooses the minimal
[`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] needed by their ray
representatives during tensor conversion.

Repository usage
----------------
`FFObservable` is intended for small free-fermionic operator assembly where the
user wants to specify transitions symbolically and defer tensor allocation until
the target device and precision are known. The resulting rank-2
[`Tensor`][qten.linalg.tensors.Tensor] belongs to the same Hilbert space on both
legs and is explicitly filled as a Hermitian matrix.
"""

from typing import List, Optional

from multimethod import multimethod

import sympy as sy
from ..utils.devices import Device

import torch

from ..symbolics import HilbertSpace
from ..symbolics import U1Basis
from .. import Tensor
from ..precision import get_precision_config

from ._bonds import Bond


class FFObservable:
    """
    Free-fermionic observable assembled from symbolic `Bond` terms.

    The observable stores a list of weighted basis-state transitions and can
    convert them into a Hermitian matrix representation on the minimal
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] spanned by the bond endpoints after ray reduction.

    Each accumulated [`Bond`][qten.phys.Bond] contributes one directed matrix
    element. During tensor conversion, endpoint basis states are reduced to
    their rays, repeated rays are coalesced into a single Hilbert-space basis,
    and off-diagonal entries are mirrored by complex conjugation.

    Notes
    -----
    Bonds are retained internally in insertion order so the generated Hilbert
    space is deterministic for a fixed sequence of `add_bond` calls.
    """

    _bonds: List[Bond]

    def __init__(self):
        """
        Initialize an empty observable with no bond contributions.

        New terms can be added with [`add_bond()`][qten.phys.FFObservable.add_bond].
        """
        self._bonds = []

    @multimethod
    def add_bond(self, bond: Bond):
        """
        Append a bond contribution to the observable.

        Supported forms
        ---------------
        - `add_bond(bond)`
        - `add_bond(coef, src, dst)`

        Parameters
        ----------
        bond : Bond
            Already-constructed bond term for the single-argument form.

        Returns
        -------
        None
            The observable is updated in place.

        Raises
        ------
        TypeError
            If `bond` is not an instance of `Bond`.
        """
        if not isinstance(bond, Bond):
            raise TypeError(f"Expected a Bond instance, got {type(bond).__name__}")
        self._bonds.append(bond)

    @add_bond.register
    def _(self, coef: sy.Number, src: U1Basis, dst: U1Basis):
        """
        Construct and append a `Bond` from its coefficient and endpoints.

        Parameters
        ----------
        coef : sy.Number
            Symbolic coefficient multiplying the transition.
        src : U1Basis
            Source basis state.
        dst : U1Basis
            Destination basis state.

        Returns
        -------
        None
            The observable is updated in place.
        """
        bond = Bond(coef, (src, dst))
        self.add_bond(bond)

    def to_tensor(self, *, device: Optional[Device] = None) -> Tensor:
        """
        Convert the accumulated bonds into a Hermitian matrix [`Tensor`][qten.linalg.tensors.Tensor].

        Each bond contributes the matrix element induced by its coefficient and
        endpoint basis-state amplitudes. Basis states are first reduced to their
        ray representatives, and the output [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] is built from the
        resulting insertion-ordered unique rays. Off-diagonal entries are
        mirrored by complex conjugation so the returned tensor is Hermitian.

        Parameters
        ----------
        device : Device | None
            Logical device on which to allocate the output tensor data. If
            omitted, PyTorch uses its default device.

        Returns
        -------
        Tensor
            Rank-2 tensor whose dimensions are the minimal Hilbert space
            spanned by the bond endpoint rays.
        """
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
        torch_device = device.torch_device() if device is not None else None
        data = torch.zeros(
            (space.dim, space.dim), dtype=precision.torch_complex, device=torch_device
        )

        for i, j, value in bond_entries:
            if i == j:
                data[i, i] += value.real
            else:
                data[i, j] += value
                data[j, i] += value.conjugate()

        return Tensor(data=data, dims=(space, space))
