r"""Quantum geometry and topology of momentum-resolved band Hamiltonians.

The quantum geometric tensor (QGT) is the common parent of the
Fubini--Study metric and Berry curvature.  With the convention used here,

.. math::

    Q_{ij} = \operatorname{Tr}[P (\partial_i P)(\partial_j P)],\qquad
    g_{ij} = \operatorname{Re} Q_{ij},\qquad
    \Omega_{ij} = 2\operatorname{Im} Q_{ij},

where ``P`` projects onto the occupied subspace.  Projector finite differences
make the local quantities invariant under occupied-band gauge rotations.

For a Chern number, the discrete Fukui--Hatsugai--Suzuki (FHS) formula remains
the default: unlike integrating a finite-difference curvature, it is robustly
quantized on a finite mesh.
"""

from dataclasses import dataclass
from typing import Any, Literal
import warnings

import numpy as np
import sympy as sy
import torch
from sympy import ImmutableDenseMatrix

from .geometries import PeriodicBoundary, ReciprocalLattice
from .linalg.tensors import Tensor
from .symbolics import MomentumSpace


@dataclass(frozen=True)
class _Grid:
    data: torch.Tensor
    occupied: torch.Tensor
    projectors: torch.Tensor
    neighbor_x: torch.Tensor
    neighbor_y: torch.Tensor
    output_shape: tuple[int, ...]
    direct_gap: float


def _topology_grid(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None,
    gap_tolerance: float,
) -> _Grid:
    if bloch_hamiltonian.rank() != 3:
        raise ValueError("bloch_hamiltonian must have dimensions (k, band, band).")
    if not isinstance(bloch_hamiltonian.dims[0], MomentumSpace):
        raise TypeError("The first dimension must be a MomentumSpace.")

    data = bloch_hamiltonian.data
    if data.shape[-2] != data.shape[-1]:
        raise ValueError("The Hamiltonian must be square at every momentum.")
    n_bands = data.shape[-1]
    if n_occupied is None:
        n_occupied = n_bands // 2
    if not 0 < n_occupied < n_bands:
        raise ValueError("n_occupied must lie strictly between 0 and n_bands.")

    k_space = bloch_hamiltonian.dims[0]
    momenta = k_space.elements()
    if not momenta or len(momenta[0].rep) != 2:
        raise ValueError("A two-dimensional momentum grid is required.")

    reciprocal_lattice = k_space.extract(ReciprocalLattice)
    direct_boundary = reciprocal_lattice.lattice.boundaries
    if not isinstance(direct_boundary, PeriodicBoundary):
        raise ValueError("The momentum grid requires periodic lattice boundaries.")
    dual_boundary = PeriodicBoundary(ImmutableDenseMatrix(direct_boundary.basis.T))

    def rep_key(rep: ImmutableDenseMatrix) -> tuple[sy.Expr, ...]:
        return tuple(sy.sympify(rep[i, 0]) for i in range(rep.rows))

    representative_to_index: dict[tuple[sy.Expr, ...], int] = {}
    for momentum in momenta:
        integer_rep = ImmutableDenseMatrix(direct_boundary.basis.T @ momentum.rep)
        if any(not sy.sympify(value).is_integer for value in integer_rep):
            raise ValueError(
                "Momentum grid contains a point outside the reciprocal quotient."
            )
        key = rep_key(dual_boundary.wrap(integer_rep))
        if key in representative_to_index:
            raise ValueError(f"Duplicate momentum quotient representative {key}.")
        representative_to_index[key] = k_space.structure[momentum]

    canonical_reps = dual_boundary.representatives()
    if len(representative_to_index) != len(canonical_reps) or any(
        rep_key(rep) not in representative_to_index for rep in canonical_reps
    ):
        raise ValueError("Momentum points do not form a complete reciprocal grid.")
    canonical_position = {
        rep_key(rep): position for position, rep in enumerate(canonical_reps)
    }

    energies, eigenvectors = torch.linalg.eigh(data)
    direct_gap = float(
        (energies[..., n_occupied] - energies[..., n_occupied - 1]).min().item()
    )
    if direct_gap <= gap_tolerance:
        warnings.warn(
            f"Minimum direct gap is {direct_gap:.6e}; the occupied bundle is "
            "not isolated, so its topology and quantum geometry are not well-defined.",
            RuntimeWarning,
            stacklevel=3,
        )

    unit_x = ImmutableDenseMatrix([1, 0])
    unit_y = ImmutableDenseMatrix([0, 1])
    neighbor_keys = (
        [rep_key(dual_boundary.wrap(rep + unit_x)) for rep in canonical_reps],
        [rep_key(dual_boundary.wrap(rep + unit_y)) for rep in canonical_reps],
    )
    canonical_indices = torch.tensor(
        [representative_to_index[rep_key(rep)] for rep in canonical_reps],
        dtype=torch.long,
        device=data.device,
    )
    neighbors = tuple(
        torch.tensor(
            [canonical_position[key] for key in keys],
            dtype=torch.long,
            device=data.device,
        )
        for keys in neighbor_keys
    )
    occupied = eigenvectors[..., :n_occupied]
    occupied = occupied[canonical_indices]
    projectors = occupied @ occupied.conj().transpose(-2, -1)
    if direct_boundary.basis.is_diagonal():
        output_shape = tuple(int(direct_boundary.basis[i, i]) for i in range(2))
    else:
        output_shape = (len(canonical_reps),)
    return _Grid(
        data=data,
        occupied=occupied,
        projectors=projectors,
        neighbor_x=neighbors[0],
        neighbor_y=neighbors[1],
        output_shape=output_shape,
        direct_gap=direct_gap,
    )


def _reshape(values: torch.Tensor, grid: _Grid) -> torch.Tensor:
    return values.reshape(grid.output_shape + values.shape[1:])


def quantum_geometric_tensor(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
) -> np.ndarray:
    r"""Return the gauge-invariant occupied-subspace QGT on a 2-D mesh.

    Central finite differences of the occupied projector are taken along the
    two primitive reciprocal-grid directions.  Consequently the tensor
    components are expressed per grid step, and summing its associated Berry
    curvature performs the corresponding discrete Brillouin-zone integral.

    The result has shape ``grid_shape + (2, 2)``.  For sheared periodic cells,
    ``grid_shape`` is one-dimensional canonical representative order, since a
    rectangular reshape would misrepresent momentum-space adjacency.
    """
    grid = _topology_grid(bloch_hamiltonian, n_occupied, gap_tolerance)
    neighbor_indices = (grid.neighbor_x, grid.neighbor_y)

    # The inverse neighbor permutation gives k-e_i for every canonical k.
    derivatives = []
    for forward in neighbor_indices:
        backward = torch.empty_like(forward)
        backward[forward] = torch.arange(
            forward.numel(), dtype=torch.long, device=forward.device
        )
        derivatives.append((grid.projectors[forward] - grid.projectors[backward]) / 2)

    qgt = torch.empty(
        (grid.projectors.shape[0], 2, 2),
        dtype=grid.data.dtype,
        device=grid.data.device,
    )
    for i in range(2):
        for j in range(2):
            qgt[:, i, j] = torch.einsum(
                "kab,kbc,kca->k", grid.projectors, derivatives[i], derivatives[j]
            )
    return _reshape(qgt, grid).detach().cpu().numpy()


def fubini_study_metric(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
) -> np.ndarray:
    """Return the Fubini--Study metric, the real part of the QGT."""
    return quantum_geometric_tensor(bloch_hamiltonian, n_occupied, gap_tolerance).real


def berry_curvature(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
) -> np.ndarray:
    r"""Return ``Omega_xy = 2 Im(Q_xy)`` on the reciprocal mesh.

    This sign convention agrees with the oriented plaquette used by
    :func:`chern_number` with ``method="fhs"``.
    """
    qgt = quantum_geometric_tensor(bloch_hamiltonian, n_occupied, gap_tolerance)
    return 2.0 * qgt[..., 0, 1].imag


def _discrete_chern_number(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compute the occupied-band Chern number with the discrete FHS formula."""
    grid = _topology_grid(bloch_hamiltonian, n_occupied, gap_tolerance)

    def link(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        determinant = torch.linalg.det(left.conj().transpose(-2, -1) @ right)
        magnitude = determinant.abs()
        if bool((magnitude < 1e-14).any()):
            raise RuntimeError(
                "A neighboring occupied-subspace overlap is singular; "
                "increase the momentum-grid resolution."
            )
        return determinant / magnitude

    link_x = link(grid.occupied, grid.occupied[grid.neighbor_x])
    link_y = link(grid.occupied, grid.occupied[grid.neighbor_y])
    plaquette = (
        link_x
        * link_y[grid.neighbor_x]
        * link_x[grid.neighbor_y].conj()
        * link_y.conj()
    )
    flux = _reshape(torch.angle(plaquette), grid)
    chern = float((flux.sum() / (2.0 * torch.pi)).item())
    return {
        "chern": chern,
        "nearest_integer": int(np.rint(chern)),
        "direct_gap": grid.direct_gap,
        "berry_flux": flux.detach().cpu().numpy(),
    }


def chern_number(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
    *,
    method: Literal["fhs", "qgt"] = "fhs",
) -> dict[str, Any]:
    """Compute a Chern number using FHS or integrated QGT curvature.

    ``method="fhs"`` is recommended and returns ``berry_flux``.
    ``method="qgt"`` returns ``quantum_geometric_tensor``,
    ``fubini_study_metric``, and ``berry_curvature`` as well; its Chern number
    is a finite-difference estimate and need not be exactly quantized.
    """
    if method == "fhs":
        return _discrete_chern_number(bloch_hamiltonian, n_occupied, gap_tolerance)
    if method != "qgt":
        raise ValueError("method must be 'fhs' or 'qgt'.")

    grid = _topology_grid(bloch_hamiltonian, n_occupied, gap_tolerance)
    qgt = quantum_geometric_tensor(bloch_hamiltonian, n_occupied, gap_tolerance)
    curvature = 2.0 * qgt[..., 0, 1].imag
    chern = float(curvature.sum() / (2.0 * np.pi))
    return {
        "chern": chern,
        "nearest_integer": int(np.rint(chern)),
        "direct_gap": grid.direct_gap,
        "quantum_geometric_tensor": qgt,
        "fubini_study_metric": qgt.real,
        "berry_curvature": curvature,
    }


__all__ = [
    "berry_curvature",
    "chern_number",
    "fubini_study_metric",
    "quantum_geometric_tensor",
]
