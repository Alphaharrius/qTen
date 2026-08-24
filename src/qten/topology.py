r"""Quantum geometry and topology of momentum-resolved band Hamiltonians.

This module computes geometric properties of an isolated occupied-band
subspace carried by a rank-3 [`Tensor`][qten.linalg.tensors.Tensor] with dims
``(MomentumSpace, HilbertSpace, HilbertSpace)``. The Hamiltonian is
diagonalized independently at every momentum, and the ``n_occupied``
lowest-energy eigenvectors define the occupied projector (P(k)).

Core API
--------
- [`quantum_geometric_tensor`][qten.topology.quantum_geometric_tensor]
  Gauge-invariant quantum geometric tensor obtained from finite differences
  of the occupied projector.
- [`fubini_study_metric`][qten.topology.fubini_study_metric]
  Symmetric metric given by the real part of the quantum geometric tensor.
- [`berry_curvature`][qten.topology.berry_curvature]
  Local Berry curvature given by its imaginary antisymmetric part.
- [`chern_number`][qten.topology.chern_number]
  First Chern number computed either with discrete FHS link variables or by
  integrating the finite-difference Berry curvature.

Mathematical convention
-----------------------
For occupied projector $P(k)$, QTen uses


$$
Q_{ij}(k) = \operatorname{Tr}\!\left[
    P(k)\,\partial_i P(k)\,\partial_j P(k)
\right],
\qquad
g_{ij}(k) = \operatorname{Re} Q_{ij}(k),
\qquad
\Omega_{ij}(k) = 2\operatorname{Im} Q_{ij}(k).
$$

The curvature sign agrees with the oriented plaquette used by
[`chern_number(..., method="fhs")`][qten.topology.chern_number]. Projector
derivatives make these local quantities invariant under phase changes and
general unitary rotations among occupied eigenvectors.

Momentum-grid convention
------------------------
Finite differences follow the two primitive quotient directions
$e_x=(1,0)$ and $e_y=(0,1)$. Tensor components are therefore expressed per
reciprocal-grid step. Local quantum-geometric results retain the input
``MomentumSpace`` as their first symbolic dimension. Their data therefore use
the flat momentum-space order rather than an unlabeled rectangular reshape.

Numerical methods
-----------------
The default Chern method is the gauge-invariant
Fukui--Hatsugai--Suzuki (FHS) link-variable formula. It is the preferred
finite-mesh topological invariant because a sufficiently resolved, isolated
bundle gives an integer up to floating-point error. Integrating QGT-derived
curvature exposes the connection between local quantum geometry and topology,
but is a central-finite-difference estimate and generally approaches an
integer only as the momentum mesh is refined.
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
from .symbolics import IndexSpace, MomentumSpace


@dataclass(frozen=True)
class _Grid:
    data: torch.Tensor
    occupied: torch.Tensor
    projectors: torch.Tensor
    neighbors: tuple[torch.Tensor, ...]
    canonical_indices: torch.Tensor
    momentum_space: MomentumSpace
    output_shape: tuple[int, ...]
    direct_gap: float

    @property
    def momentum_dim(self) -> int:
        return len(self.neighbors)


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
    if not momenta:
        raise ValueError("The momentum grid must not be empty.")
    momentum_dim = len(momenta[0].rep)
    if momentum_dim not in (1, 2, 3):
        raise ValueError(
            "A one-, two-, or three-dimensional momentum grid is required."
        )

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

    units = tuple(
        ImmutableDenseMatrix(
            [int(component == direction) for component in range(momentum_dim)]
        )
        for direction in range(momentum_dim)
    )
    neighbor_keys = tuple(
        [rep_key(dual_boundary.wrap(rep + unit)) for rep in canonical_reps]
        for unit in units
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
        output_shape = tuple(
            int(direct_boundary.basis[i, i]) for i in range(momentum_dim)
        )
    else:
        output_shape = (len(canonical_reps),)
    return _Grid(
        data=data,
        occupied=occupied,
        projectors=projectors,
        neighbors=neighbors,
        canonical_indices=canonical_indices,
        momentum_space=k_space,
        output_shape=output_shape,
        direct_gap=direct_gap,
    )


def _reshape(values: torch.Tensor, grid: _Grid) -> torch.Tensor:
    return values.reshape(grid.output_shape + values.shape[1:])


def quantum_geometric_tensor(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
) -> Tensor:
    r"""Compute the occupied-subspace quantum geometric tensor on a 1-D, 2-D,
    or 3-D grid.

    The occupied projector is built from the ``n_occupied`` lowest-energy
    eigenvectors at every momentum. Central differences along the two
    primitive reciprocal-grid directions approximate $\partial_xP$ and
    $\partial_yP$, after which
    $Q_{ij}=\operatorname{Tr}[P(\partial_iP)(\partial_jP)]$ is evaluated.

    Parameters
    ----------
    bloch_hamiltonian : Tensor
        Rank-3 Hermitian [`Tensor`][qten.linalg.tensors.Tensor] with dims
        ``(MomentumSpace, HilbertSpace, HilbertSpace)``. The final two data
        axes must be square and represent the Bloch Hamiltonian at each
        momentum.
    n_occupied : int | None, optional
        Number of lowest-energy bands included in the occupied projector.
        Defaults to half the Hamiltonian bands using integer division.
    gap_tolerance : float, optional
        Minimum acceptable direct gap between bands ``n_occupied - 1`` and
        ``n_occupied``. A gap at or below this value emits a
        `RuntimeWarning`, because the selected bundle is not
        numerically isolated. Defaults to ``1e-8``.

    Returns
    -------
    Tensor
        Complex QGT with dims ``(MomentumSpace, IndexSpace(d),
        IndexSpace(d))`` and shape ``(N_k, d, d)``, where ``d`` is the
        momentum-space dimension. The first dimension is the input
        Hamiltonian's momentum space, so momentum labels and their ordering
        are preserved. Components are measured per reciprocal-grid step, not
        per Cartesian inverse-length unit.

    Raises
    ------
    TypeError
        If the first tensor dimension is not a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace].
    ValueError
        If the tensor is not rank 3, its Hamiltonian blocks are not square,
        ``n_occupied`` is invalid, the momentum space is not 1-D, 2-D, or 3-D,
        the boundary is not periodic, or momentum points do not form a unique
        complete reciprocal quotient.

    Notes
    -----
    The projector formulation is invariant under arbitrary momentum-dependent
    unitary rotations within the occupied subspace. It therefore remains
    well-defined when occupied bands cross each other, provided the occupied
    subspace stays separated from the empty bands.

    See Also
    --------
    [`fubini_study_metric`][qten.topology.fubini_study_metric]
        Real part of this tensor.
    [`berry_curvature`][qten.topology.berry_curvature]
        Imaginary antisymmetric part of this tensor.
    [`chern_number`][qten.topology.chern_number]
        Brillouin-zone topological invariant.
    """
    grid = _topology_grid(bloch_hamiltonian, n_occupied, gap_tolerance)
    # The inverse neighbor permutation gives k-e_i for every canonical k.
    derivatives = []
    for forward in grid.neighbors:
        backward = torch.empty_like(forward)
        backward[forward] = torch.arange(
            forward.numel(), dtype=torch.long, device=forward.device
        )
        derivatives.append((grid.projectors[forward] - grid.projectors[backward]) / 2)

    qgt = torch.empty(
        (grid.projectors.shape[0], grid.momentum_dim, grid.momentum_dim),
        dtype=grid.data.dtype,
        device=grid.data.device,
    )
    for i in range(grid.momentum_dim):
        for j in range(grid.momentum_dim):
            qgt[:, i, j] = torch.einsum(
                "kab,kbc,kca->k", grid.projectors, derivatives[i], derivatives[j]
            )
    # The finite-difference calculation uses canonical quotient order. Restore
    # the input MomentumSpace order before attaching its symbolic dimension.
    ordered_qgt = torch.empty_like(qgt)
    ordered_qgt[grid.canonical_indices] = qgt
    component_space = IndexSpace.linear(grid.momentum_dim)
    return Tensor(
        data=ordered_qgt,
        dims=(grid.momentum_space, component_space, component_space),
    )


def fubini_study_metric(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
) -> Tensor:
    r"""Compute the occupied-subspace Fubini--Study metric on a 1-D, 2-D,
    or 3-D grid.

    This function returns $g_{ij}(k)=\operatorname{Re}Q_{ij}(k)$, where the
    QGT is computed by
    [`quantum_geometric_tensor`][qten.topology.quantum_geometric_tensor].

    Parameters
    ----------
    bloch_hamiltonian : Tensor
        Rank-3 Hermitian [`Tensor`][qten.linalg.tensors.Tensor] with dims
        ``(MomentumSpace, HilbertSpace, HilbertSpace)``.
    n_occupied : int | None, optional
        Number of lowest-energy occupied bands. Defaults to half the bands.
    gap_tolerance : float, optional
        Direct-gap warning threshold. Defaults to ``1e-8``.

    Returns
    -------
    Tensor
        Real metric with dims ``(MomentumSpace, IndexSpace(d),
        IndexSpace(d))`` and shape ``(N_k, d, d)``. Components are expressed
        per reciprocal-grid step.

    Raises
    ------
    TypeError
        If the first tensor dimension is not a momentum space.
    ValueError
        If the Hamiltonian, occupied-band selection, or reciprocal grid is
        invalid. See
        [`quantum_geometric_tensor`][qten.topology.quantum_geometric_tensor]
        for the complete validation contract.

    See Also
    --------
    [`quantum_geometric_tensor`][qten.topology.quantum_geometric_tensor]
        Complex parent tensor of the metric and curvature.
    [`berry_curvature`][qten.topology.berry_curvature]
        Berry curvature from the imaginary part of the QGT.
    """
    return quantum_geometric_tensor(bloch_hamiltonian, n_occupied, gap_tolerance).real()


def berry_curvature(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
) -> Tensor:
    r"""Compute occupied-subspace Berry curvature on a 1-D, 2-D, or 3-D grid.

    QTen uses $\Omega_{ij}(k)=2\operatorname{Im}Q_{ij}(k)$. In two dimensions,
    the $xy$ orientation agrees with
    [`chern_number(..., method="fhs")`][qten.topology.chern_number].

    Parameters
    ----------
    bloch_hamiltonian : Tensor
        Rank-3 Hermitian [`Tensor`][qten.linalg.tensors.Tensor] with dims
        ``(MomentumSpace, HilbertSpace, HilbertSpace)``.
    n_occupied : int | None, optional
        Number of lowest-energy occupied bands. Defaults to half the bands.
    gap_tolerance : float, optional
        Direct-gap warning threshold. Defaults to ``1e-8``.

    Returns
    -------
    Tensor
        Real antisymmetric curvature tensor with dims
        ``(MomentumSpace, IndexSpace(d), IndexSpace(d))`` and shape
        ``(N_k, d, d)``. In two dimensions, summing
        ``curvature.data[..., 0, 1]`` and dividing by ``2*pi`` gives the
        central-finite-difference estimate of the first Chern number.

    Raises
    ------
    TypeError
        If the first tensor dimension is not a momentum space.
    ValueError
        If the Hamiltonian, occupied-band selection, or reciprocal grid is
        invalid. See
        [`quantum_geometric_tensor`][qten.topology.quantum_geometric_tensor]
        for the complete validation contract.

    Notes
    -----
    This is QGT-derived curvature, not the compact plaquette flux returned by
    ``chern_number(..., method="fhs")``. Its integral need not be exactly
    quantized on a finite grid.

    See Also
    --------
    [`quantum_geometric_tensor`][qten.topology.quantum_geometric_tensor]
        Complex tensor from which the curvature is derived.
    [`chern_number`][qten.topology.chern_number]
        FHS or curvature-integral Chern number.
    """
    qgt = quantum_geometric_tensor(bloch_hamiltonian, n_occupied, gap_tolerance)
    return Tensor(data=2.0 * qgt.data.imag, dims=qgt.dims)


def _discrete_chern_number(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    gap_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compute the occupied-band Chern number with the discrete FHS formula."""
    grid = _topology_grid(bloch_hamiltonian, n_occupied, gap_tolerance)
    if grid.momentum_dim != 2:
        raise ValueError("The first Chern number requires a two-dimensional grid.")

    def link(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        determinant = torch.linalg.det(left.conj().transpose(-2, -1) @ right)
        magnitude = determinant.abs()
        if bool((magnitude < 1e-14).any()):
            raise RuntimeError(
                "A neighboring occupied-subspace overlap is singular; "
                "increase the momentum-grid resolution."
            )
        return determinant / magnitude

    neighbor_x, neighbor_y = grid.neighbors
    link_x = link(grid.occupied, grid.occupied[neighbor_x])
    link_y = link(grid.occupied, grid.occupied[neighbor_y])
    plaquette = link_x * link_y[neighbor_x] * link_x[neighbor_y].conj() * link_y.conj()
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
    r"""Compute the first Chern number of an occupied band subspace.

    The ``n_occupied`` lowest-energy eigenstates define an occupied bundle over
    a complete two-dimensional periodic momentum grid. Two numerical methods
    are available:

    - ``method="fhs"`` computes normalized determinant link variables between
      neighboring occupied subspaces and sums their oriented plaquette phases.
      This gauge-invariant Fukui--Hatsugai--Suzuki construction is the default
      and the recommended finite-grid topological invariant.
    - ``method="qgt"`` computes the projector quantum geometric tensor, takes
      $\Omega_{xy}=2\operatorname{Im}Q_{xy}$, and evaluates
      $C=(2\pi)^{-1}\sum_k\Omega_{xy}(k)$. It additionally returns all local
      quantum-geometric data.

    Parameters
    ----------
    bloch_hamiltonian : Tensor
        Rank-3 Hermitian [`Tensor`][qten.linalg.tensors.Tensor] with dims
        ``(MomentumSpace, HilbertSpace, HilbertSpace)``. The first data axis
        enumerates a complete two-dimensional reciprocal quotient; the last
        two axes are square Bloch-Hamiltonian matrices.
    n_occupied : int | None, optional
        Number of lowest-energy bands defining the occupied subspace. It must
        lie strictly between zero and the total band count. Defaults to half
        the bands using integer division.
    gap_tolerance : float, optional
        Warning threshold for the minimum direct gap
        $\min_k[E_{n_\mathrm{occupied}}(k)-
        E_{n_\mathrm{occupied}-1}(k)]$. A gap at or below this value emits a
        `RuntimeWarning`, because the occupied bundle is not
        isolated and its Chern number is not well-defined. Defaults to
        ``1e-8``.
    method : {"fhs", "qgt"}, optional
        Numerical construction. ``"fhs"`` uses discrete determinant link
        variables and is robustly quantized on a suitable finite mesh.
        ``"qgt"`` integrates central-finite-difference curvature and exposes
        local quantum geometry. Defaults to ``"fhs"``.

    Returns
    -------
    dict[str, Any]
        Result mapping. Both methods return:

        - ``"chern"``: raw floating-point Chern value.
        - ``"nearest_integer"``: nearest integer obtained with `numpy.rint`.
        - ``"direct_gap"``: minimum occupied-to-empty direct gap.

        For ``method="fhs"`` the mapping also contains ``"berry_flux"``, the
        oriented plaquette phase in radians. Its shape is ``(N_x, N_y)`` for a
        diagonal periodic cell and ``(N_k,)`` in canonical representative
        order for a sheared cell.

        For ``method="qgt"`` the mapping instead contains labeled `Tensor`
        values: ``"quantum_geometric_tensor"`` and
        ``"fubini_study_metric"`` and ``"berry_curvature"`` all have shape
        ``(N_k, 2, 2)``. Each retains the input momentum space as its first
        dimension.

    Raises
    ------
    TypeError
        If the first tensor dimension is not a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace].
    ValueError
        If ``method`` is unsupported; the input is not a rank-3 square Bloch
        Hamiltonian; ``n_occupied`` is outside the valid range; the momentum
        space is not two-dimensional and periodic; or its points do not form a
        unique complete reciprocal quotient.
    RuntimeError
        For ``method="fhs"``, if a neighboring occupied-subspace overlap has
        determinant magnitude below ``1e-14``. This indicates a singular link;
        increasing the momentum-grid resolution may resolve it.

    Warns
    -----
    RuntimeWarning
        If the minimum direct gap is no larger than ``gap_tolerance``.

    Notes
    -----
    The FHS value satisfies

    $$
    C_\mathrm{FHS} = \frac{1}{2\pi}
    \sum_k \operatorname{Arg}\!\left[
      U_x(k)U_y(k+e_x)U_x(k+e_y)^*U_y(k)^*
    \right],
    $$

    where $U_i(k)$ is the phase of the determinant of the occupied-subspace
    overlap between $k$ and $k+e_i$. Determinants make the formula
    invariant under arbitrary unitary changes of occupied-band basis.

    ``nearest_integer`` is a convenience diagnostic, not proof that the bundle
    is isolated or the mesh is sufficiently resolved. Inspect ``direct_gap``
    and, when necessary, repeat the calculation on finer momentum grids.

    For sheared cells, the Chern sum and neighbor graph remain correct, but a
    flat returned array is intentional: canonical quotient order is not a
    rectangular Brillouin-zone heatmap.

    Examples
    --------
    Use the robust finite-grid method:

    ```python
    result = chern_number(hamiltonian, n_occupied=1)
    invariant = result["nearest_integer"]
    flux = result["berry_flux"]
    ```

    Request the differential-geometric decomposition:

    ```python
    geometry = chern_number(hamiltonian, n_occupied=1, method="qgt")
    metric = geometry["fubini_study_metric"]
    curvature = geometry["berry_curvature"]
    ```

    See Also
    --------
    [`quantum_geometric_tensor`][qten.topology.quantum_geometric_tensor]
        Gauge-invariant local quantum geometric tensor.
    [`fubini_study_metric`][qten.topology.fubini_study_metric]
        Metric part of the QGT.
    [`berry_curvature`][qten.topology.berry_curvature]
        Curvature part of the QGT.
    """
    if method == "fhs":
        return _discrete_chern_number(bloch_hamiltonian, n_occupied, gap_tolerance)
    if method != "qgt":
        raise ValueError("method must be 'fhs' or 'qgt'.")

    grid = _topology_grid(bloch_hamiltonian, n_occupied, gap_tolerance)
    if grid.momentum_dim != 2:
        raise ValueError("The first Chern number requires a two-dimensional grid.")
    qgt = quantum_geometric_tensor(bloch_hamiltonian, n_occupied, gap_tolerance)
    curvature = Tensor(data=2.0 * qgt.data.imag, dims=qgt.dims)
    chern = float(curvature.data[..., 0, 1].sum() / (2.0 * np.pi))
    return {
        "chern": chern,
        "nearest_integer": int(np.rint(chern)),
        "direct_gap": grid.direct_gap,
        "quantum_geometric_tensor": qgt,
        "fubini_study_metric": qgt.real(),
        "berry_curvature": curvature,
    }


__all__ = [
    "berry_curvature",
    "chern_number",
    "fubini_study_metric",
    "quantum_geometric_tensor",
]
