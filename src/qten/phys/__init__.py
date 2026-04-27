"""
Physics-facing operator assembly built on symbolic basis states.

This package provides small data structures for describing quantum-mechanical
terms before they are materialized as tensors. Its current focus is the
free-fermionic workflow: collect weighted transitions between symbolic basis
states, reduce those states to a minimal Hilbert space, and build the Hermitian
matrix representation used by the tensor layer.

Two related APIs are exposed here:

- [`Bond`][qten.phys.Bond] stores a weighted directed transition between two
  [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] states.
- [`FFObservable`][qten.phys.FFObservable] accumulates bond terms and converts
  them into a rank-2 [`Tensor`][qten.linalg.tensors.Tensor].

Repository usage
----------------
`qten.phys` sits between [`qten.symbolics`][qten.symbolics] and
[`qten.linalg`][qten.linalg]. Symbolic basis objects carry the state labels and
amplitudes, while the observable builder decides which rays span the output
[`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] and allocates the
corresponding tensor data.
"""

from ._bonds import Bond as Bond
from ._ff_observables import FFObservable as FFObservable
