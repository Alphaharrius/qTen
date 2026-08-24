r"""Three-dimensional \(\mathbb{Z}_2\) invariants of time-reversal-invariant
insulators.

The occupied subspace of a gapped 3-D Bloch Hamiltonian with even filling
carries four \(\mathbb{Z}_2\) indices \((\nu_0; \nu_1\nu_2\nu_3)\). This module
evaluates them from a rank-3
[`Tensor`][qten.linalg.tensors.Tensor] with dims
``(MomentumSpace, HilbertSpace, HilbertSpace)`` (or a
[`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace] of
diagonal \((k,k)\) blocks whose momenta form a complete 3-D grid).

The input mesh is Fourier-interpolated to a tight-binding hopping tensor so
that time-reversal invariant momenta (TRIM) and Wilson-loop strings can be
sampled independently of whether those points sit on the original grid.

Core API
--------
- [`z2_indices`][qten.topology.z2_indices]
  Fu--Kane inversion parities at the eight TRIM, hybrid-Wannier Wilson loops
  on the six TRIM planes, or both.

Numerical methods
-----------------
- ``method="parity"`` uses Fu--Kane products of inversion eigenvalues at the
  eight TRIM. It requires an inversion operator: either an explicit rank-3
  tensor in the same basis as the Hamiltonian, or orbital
  [`Offset`][qten.geometries.spatials.Offset] labels from which spatial
  inversion about ``inversion_center`` is assembled.
- ``method="wilson"`` tracks hybrid Wannier charge centers on the six TRIM
  planes. It does not need inversion symmetry.
- ``method="auto"`` tries parity first and falls back to Wilson loops if
  inversion cannot be resolved.
- ``method="both"`` runs both constructions. Parity must succeed; the
  returned indices are the parity values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal, overload, TypedDict
import math
import warnings

import sympy as sy
import torch
from sympy import ImmutableDenseMatrix

from ..geometries import Lattice, Momentum, Offset, PeriodicBoundary, ReciprocalLattice
from ..linalg.tensors import Tensor
from ..phys import Spin, as_spin, contains_spin
from ..pointgroups import pointgroup
from ..pointgroups.elements import PointGroupOpr
from ..pointgroups.ops import spinful_hilbert_opr_repr
from ..symbolics import (
    HilbertSpace,
    IndexSpace,
    MomentumBlockSpace,
    MomentumSpace,
    hilbert_opr_repr,
    same_rays,
)


class Z2ParityTrimDiagnostics(TypedDict):
    """Inversion-parity diagnostics at one TRIM."""

    delta: int
    parity_eigenvalues: Tensor
    commutator_error: float
    direct_gap: float


class Z2ParityResult(TypedDict):
    """Result returned by the Fu--Kane inversion-parity method."""

    indices: tuple[int, int, int, int]
    method: Literal["parity"]
    parity_products: dict[tuple[int, int, int], int]
    diagnostics: dict[tuple[int, int, int], Z2ParityTrimDiagnostics]
    direct_gap: float


class Z2WilsonPlaneResult(TypedDict):
    """Hybrid-Wannier data on one TRIM plane."""

    z2: int
    wcc: Tensor
    gap_pos: Tensor
    sweep: Tensor
    min_gap: float
    kramers_resolved: bool


class Z2WilsonResult(TypedDict):
    """Result returned by the Wilson-loop method."""

    indices: tuple[int, int, int, int]
    method: Literal["wilson"]
    planes: dict[tuple[int, float], Z2WilsonPlaneResult]
    axis_z2: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    min_gap: float


class Z2CombinedResult(TypedDict):
    """Result returned when both parity and Wilson-loop methods are run."""

    indices: tuple[int, int, int, int]
    method: Literal["both"]
    parity: Z2ParityResult
    wilson: Z2WilsonResult


def _rep_key(rep: ImmutableDenseMatrix) -> tuple[sy.Expr, ...]:
    return tuple(sy.sympify(rep[i, 0]) for i in range(rep.rows))


def _bloch_momenta(bloch_hamiltonian: Tensor) -> tuple[Momentum, ...]:
    k_dim = bloch_hamiltonian.dims[0]
    if isinstance(k_dim, MomentumSpace):
        return tuple(k_dim.elements())
    if isinstance(k_dim, MomentumBlockSpace):
        momenta: list[Momentum] = []
        for left, right in k_dim.elements():
            if left != right:
                raise ValueError(
                    "Z2 requires a diagonal momentum-block Hamiltonian "
                    "(left and right momenta equal at every block)."
                )
            momenta.append(left)
        return tuple(momenta)
    raise TypeError(
        "The first dimension must be a MomentumSpace or MomentumBlockSpace."
    )


def _unravel_index(index: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    _, n1, n2 = shape
    i2 = index % n2
    i1 = (index // n2) % n1
    i0 = index // (n1 * n2)
    return (i0, i1, i2)


def _fourier_interpolate(
    hoppings: torch.Tensor,
    k_frac: torch.Tensor | Sequence[float],
    rx: torch.Tensor,
    ry: torch.Tensor,
    rz: torch.Tensor,
) -> torch.Tensor:
    real_dtype = hoppings.real.dtype
    k_frac = torch.as_tensor(k_frac, device=hoppings.device, dtype=real_dtype)
    batched = k_frac.ndim == 2
    if not batched:
        k_frac = k_frac[None, :]
    phase = hoppings.new_tensor(-2j * math.pi)
    px = torch.exp(phase * k_frac[:, 0, None] * rx[None, :])
    py = torch.exp(phase * k_frac[:, 1, None] * ry[None, :])
    pz = torch.exp(phase * k_frac[:, 2, None] * rz[None, :])
    values = torch.einsum("xyzab,kx,ky,kz->kab", hoppings, px, py, pz)
    return values if batched else values[0]


def _indices_from_deltas(
    parity_products: dict[tuple[int, int, int], int],
) -> tuple[int, int, int, int]:
    strong = math.prod(parity_products.values())
    weak = tuple(
        math.prod(delta for bits, delta in parity_products.items() if bits[axis] == 1)
        for axis in range(3)
    )
    return (int(strong < 0), *(int(product < 0) for product in weak))


def _largest_gap_position(wcc: torch.Tensor) -> float:
    wcc = wcc.real % 1.0
    wcc, _ = torch.sort(wcc)
    extended = torch.cat((wcc, wcc[:1] + 1.0))
    gaps = extended[1:] - extended[:-1]
    index = int(torch.argmax(gaps).item())
    return float(((extended[index] + 0.5 * gaps[index]) % 1.0).item())


def _sgng(gap_left: float, gap_right: float, wcc_pos: float) -> int:
    dz = (gap_right - gap_left) % 1.0
    dx = (wcc_pos - gap_left) % 1.0
    if dz == 0.0:
        return 1
    if dz <= 0.5:
        crossed = 0.0 < dx < dz
    else:
        crossed = dx > dz
    return -1 if crossed else 1


def _kramers_pairs(wcc: torch.Tensor, tolerance: float) -> bool:
    wcc, _ = torch.sort(wcc.real % 1.0)
    n = int(wcc.numel())
    if n % 2:
        return False
    used = torch.zeros(n, dtype=torch.bool, device=wcc.device)
    for i in range(n):
        if bool(used[i].item()):
            continue
        delta = torch.minimum((wcc - wcc[i]) % 1.0, (wcc[i] - wcc) % 1.0)
        delta = delta.clone()
        delta[i] = torch.inf
        delta[used] = torch.inf
        partner = int(torch.argmin(delta).item())
        if not bool(torch.isfinite(delta[partner]).item()) or float(
            delta[partner].item()
        ) > tolerance:
            return False
        used[i] = True
        used[partner] = True
    return True


def _kramers_spectrum(energies: torch.Tensor, tolerance: float) -> bool:
    values, _ = torch.sort(energies.real)
    if int(values.numel()) % 2:
        return False
    pairs = values.reshape(-1, 2)
    return bool((pairs[:, 1] - pairs[:, 0]).max().item() <= tolerance)


def _time_reversal_matrix(
    space: HilbertSpace, like: torch.Tensor
) -> torch.Tensor | None:
    if not contains_spin(space):
        return None
    theta = like.new_zeros((space.dim, space.dim))
    index_of: dict[tuple[tuple[object, ...], object], int] = {}
    parsed: list[tuple[tuple[object, ...], object, int]] = []
    for psi in space.elements():
        try:
            spin = psi.irrep_of(Spin)
        except ValueError:
            return None
        spin = as_spin(spin)
        if spin is None:
            return None
        nons = tuple(rep for rep in psi.base if as_spin(rep) is None)
        index = space.structure[psi]
        index_of[(nons, spin)] = index
        parsed.append((nons, spin, index))
    for nons, spin, index in parsed:
        partner = Spin.down if spin.is_up else Spin.up
        other = index_of.get((nons, partner))
        if other is None:
            return None
        theta[other, index] = 1.0 if spin.is_up else -1.0
    return theta


def _inversion_element():
    for element in pointgroup("-1").elements():
        dim = element.irrep.rows
        if sy.simplify(element.irrep + sy.eye(dim)) == sy.zeros(dim):
            return element
    raise RuntimeError("Point group -1 does not contain spatial inversion.")


def _as_inversion_center(
    inversion_center: Offset | Sequence[float],
    lattice,
) -> Offset:
    if isinstance(inversion_center, Offset):
        return inversion_center.rebase(lattice)
    values = [sy.sympify(value) for value in inversion_center]
    if len(values) != lattice.dim:
        raise ValueError(
            f"inversion_center must have length {lattice.dim}, got {len(values)}."
        )
    return Offset(rep=ImmutableDenseMatrix(values), space=lattice)


@dataclass
class _Z2Engine:
    hoppings: torch.Tensor
    rx: torch.Tensor
    ry: torch.Tensor
    rz: torch.Tensor
    tau: torch.Tensor
    n_occupied: int
    n_bands: int
    lattice: Lattice
    inversion_hoppings: torch.Tensor | None
    inversion_i0: torch.Tensor | None
    inversion_center_vec: torch.Tensor | None
    positions_cart: torch.Tensor | None
    inversion_error: Exception | None = None
    time_reversal: torch.Tensor | None = None

    def hamiltonians_at(
        self, k_frac: torch.Tensor | Sequence[float]
    ) -> torch.Tensor:
        ham = _fourier_interpolate(self.hoppings, k_frac, self.rx, self.ry, self.rz)
        return 0.5 * (ham + ham.transpose(-1, -2).conj())

    def inversion_at_trim(self, trim_frac: torch.Tensor) -> torch.Tensor:
        if self.inversion_hoppings is not None:
            return _fourier_interpolate(
                self.inversion_hoppings, trim_frac, self.rx, self.ry, self.rz
            )
        if (
            self.inversion_i0 is None
            or self.inversion_center_vec is None
            or self.positions_cart is None
        ):
            if self.inversion_error is not None:
                raise RuntimeError(
                    "Cannot build inversion from orbital Offset labels: "
                    f"{self.inversion_error}"
                ) from self.inversion_error
            raise RuntimeError(
                "Cannot build inversion without Offset-labeled orbitals "
                "or an explicit inversion tensor."
            )
        k_cart = torch.tensor(
            [
                float(c)
                for c in Momentum(
                    rep=ImmutableDenseMatrix(
                        [float(x) for x in trim_frac.detach().cpu().reshape(-1)]
                    ),
                    space=self.lattice.dual,
                ).to_vec()
            ],
            dtype=self.inversion_i0.real.dtype,
            device=self.inversion_i0.device,
        )
        relative = (
            self.positions_cart.to(
                device=self.inversion_i0.device, dtype=self.inversion_i0.real.dtype
            )
            - self.inversion_center_vec.to(
                device=self.inversion_i0.device, dtype=self.inversion_i0.real.dtype
            )
        ) @ k_cart
        return self.inversion_i0 * torch.exp(
            -1j * (relative[:, None] + relative[None, :])
        )

    def run_parity(self, parity_tolerance: float) -> Z2ParityResult:
        parity_products: dict[tuple[int, int, int], int] = {}
        diagnostics: dict[tuple[int, int, int], Z2ParityTrimDiagnostics] = {}
        real_dtype = self.hoppings.real.dtype
        device = self.hoppings.device
        for bits in product((0, 1), repeat=3):
            k_frac = torch.tensor(bits, dtype=real_dtype, device=device) / 2
            ham = self.hamiltonians_at(k_frac)
            inv_matrix = self.inversion_at_trim(k_frac)
            ham = ham.to(dtype=inv_matrix.dtype)
            ham_norm = torch.linalg.matrix_norm(ham).clamp_min(1.0)
            comm = float(
                (
                    torch.linalg.matrix_norm(ham @ inv_matrix - inv_matrix @ ham)
                    / ham_norm
                ).item()
            )
            energies, vectors = torch.linalg.eigh(ham)
            occupied = vectors[:, : self.n_occupied]
            parity_matrix = occupied.conj().transpose(-2, -1) @ inv_matrix @ occupied
            parity_matrix = 0.5 * (
                parity_matrix + parity_matrix.conj().transpose(-2, -1)
            )
            peig = torch.linalg.eigvalsh(parity_matrix).real
            parity_error = float((peig.abs() - 1).abs().max().item())
            if comm > parity_tolerance or parity_error > parity_tolerance:
                raise RuntimeError(
                    f"Inversion is not resolved at TRIM {bits}: "
                    f"commutator={comm:.3e}, parity error={parity_error:.3e}."
                )
            n_odd = int((peig < 0).sum().item())
            if n_odd % 2:
                raise RuntimeError(
                    f"TRIM {bits} has an odd number of negative parities."
                )
            delta = -1 if (n_odd // 2) % 2 else 1
            parity_products[bits] = delta
            if self.n_occupied < self.n_bands:
                gap = float(
                    (energies[self.n_occupied] - energies[self.n_occupied - 1]).item()
                )
            else:
                gap = float("nan")
            diagnostics[bits] = {
                "delta": delta,
                "parity_eigenvalues": Tensor(
                    data=peig.detach(),
                    dims=(IndexSpace.linear(self.n_occupied),),
                ),
                "commutator_error": comm,
                "direct_gap": gap,
            }
        gaps = [item["direct_gap"] for item in diagnostics.values()]
        finite_gaps = [gap for gap in gaps if math.isfinite(gap)]
        direct_gap = min(finite_gaps) if finite_gaps else float("nan")
        return {
            "indices": _indices_from_deltas(parity_products),
            "method": "parity",
            "parity_products": parity_products,
            "diagnostics": diagnostics,
            "direct_gap": direct_gap,
        }

    def _unitary_overlap(
        self, left: torch.Tensor, right: torch.Tensor, dk: torch.Tensor
    ) -> torch.Tensor:
        phases = torch.exp(
            1j
            * (2.0 * math.pi)
            * (self.tau @ dk.to(dtype=self.tau.dtype, device=self.tau.device))
        )
        left = left * phases[:, None].to(dtype=left.dtype)
        matrix = left.conj().transpose(-2, -1) @ right
        u_svd, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u_svd @ vh

    def _wilson_wcc(
        self, k_string: torch.Tensor, loop_axis: int
    ) -> tuple[torch.Tensor, float]:
        ham = self.hamiltonians_at(k_string)
        energies, vectors = torch.linalg.eigh(ham)
        occupied = vectors[..., : self.n_occupied]
        if self.n_occupied < self.n_bands:
            gaps = energies[:, self.n_occupied] - energies[:, self.n_occupied - 1]
            min_gap = float(gaps.min().real.item())
        else:
            min_gap = float("nan")
        n_pts = occupied.shape[0]
        wilson = None
        dk = occupied.new_zeros(3, dtype=self.tau.dtype)
        dk[loop_axis] = 1.0 / n_pts
        for i in range(n_pts):
            step = self._unitary_overlap(occupied[i], occupied[(i + 1) % n_pts], dk)
            wilson = step if wilson is None else wilson @ step
        evals = torch.linalg.eigvals(wilson)
        wcc = (torch.angle(evals) / (2 * math.pi)) % 1.0
        wcc, _ = torch.sort(wcc.real)
        return wcc, min_gap

    def _plane_z2(
        self,
        normal: int,
        trim_value: float,
        n_loop: int,
        n_perp: int,
        kramers_tolerance: float,
    ) -> Z2WilsonPlaneResult:
        loop_axis = (normal + 1) % 3
        sweep_axis = (normal + 2) % 3
        real_dtype = self.hoppings.real.dtype
        device = self.hoppings.device
        sweep = torch.linspace(0.0, 0.5, n_perp, dtype=real_dtype, device=device)
        loop = torch.arange(n_loop, dtype=real_dtype, device=device) / n_loop
        wcc_list: list[torch.Tensor] = []
        min_gap = math.inf
        for ky in sweep:
            k_string = torch.zeros((n_loop, 3), dtype=real_dtype, device=device)
            k_string[:, normal] = trim_value
            k_string[:, sweep_axis] = ky
            k_string[:, loop_axis] = loop
            wcc, gap = self._wilson_wcc(k_string, loop_axis)
            wcc_list.append(wcc)
            if math.isfinite(gap):
                min_gap = min(min_gap, gap)
        kramers_resolved = _kramers_pairs(
            wcc_list[0], kramers_tolerance
        ) and _kramers_pairs(wcc_list[-1], kramers_tolerance)
        gap_pos = torch.tensor(
            [_largest_gap_position(wcc) for wcc in wcc_list],
            dtype=real_dtype,
            device=device,
        )
        invariant = 1
        for left, right, wcc_right in zip(gap_pos, gap_pos[1:], wcc_list[1:]):
            for pos in wcc_right:
                invariant *= _sgng(float(left), float(right), float(pos))
        sweep_space = IndexSpace.linear(n_perp)
        wcc_space = IndexSpace.linear(self.n_occupied)
        return {
            "z2": 1 if invariant == -1 else 0,
            "wcc": Tensor(data=torch.stack(wcc_list), dims=(sweep_space, wcc_space)),
            "gap_pos": Tensor(data=gap_pos, dims=(sweep_space,)),
            "sweep": Tensor(data=sweep, dims=(sweep_space,)),
            "min_gap": min_gap if math.isfinite(min_gap) else float("nan"),
            "kramers_resolved": kramers_resolved,
        }

    def run_wilson(
        self, n_loop: int, n_perp: int, kramers_tolerance: float
    ) -> Z2WilsonResult:
        if n_loop < 8 or n_perp < 5:
            raise ValueError("n_loop must be >= 8 and n_perp must be >= 5.")
        planes: dict[tuple[int, float], Z2WilsonPlaneResult] = {}
        axis_z2 = []
        for normal in range(3):
            zero = self._plane_z2(normal, 0.0, n_loop, n_perp, kramers_tolerance)
            pi = self._plane_z2(normal, 0.5, n_loop, n_perp, kramers_tolerance)
            planes[(normal, 0.0)] = zero
            planes[(normal, 0.5)] = pi
            axis_z2.append((zero["z2"], pi["z2"]))
        strong = [int((z0 + zpi) % 2) for z0, zpi in axis_z2]
        if len(set(strong)) == 1:
            strong_index = strong[0]
        else:
            strong_index = int(sum(strong) >= 2)
        indices = (strong_index, axis_z2[0][1], axis_z2[1][1], axis_z2[2][1])
        min_gap = min(plane["min_gap"] for plane in planes.values())
        result: Z2WilsonResult = {
            "indices": indices,
            "method": "wilson",
            "planes": planes,
            "axis_z2": (axis_z2[0], axis_z2[1], axis_z2[2]),
            "min_gap": min_gap,
        }
        if len(set(strong)) != 1:
            warnings.warn(
                "Strong-index estimates from the three Wilson-loop axes disagree: "
                f"{strong}. Increase n_loop / n_perp.",
                RuntimeWarning,
                stacklevel=3,
            )
        unresolved = [
            key for key, plane in planes.items() if not plane["kramers_resolved"]
        ]
        if unresolved:
            warnings.warn(
                "Kramers pairing of Wannier charge centers is not resolved on "
                f"TRIM planes {unresolved}.",
                RuntimeWarning,
                stacklevel=3,
            )
        return result

    def check_time_reversal(self, *, stacklevel: int = 3) -> None:
        origin = self.hoppings.new_zeros(3, dtype=self.hoppings.real.dtype)
        energies = torch.linalg.eigvalsh(self.hamiltonians_at(origin))
        span = float((energies.max() - energies.min()).clamp_min(1.0).item())
        pair_tol = max(1e-6, 1e-4 * span)
        if not _kramers_spectrum(energies, pair_tol):
            warnings.warn(
                "Eigenvalues at Gamma are not Kramers-degenerate; "
                "the Hamiltonian may break time-reversal, so its Z2 indices "
                "are not well-defined.",
                RuntimeWarning,
                stacklevel=stacklevel,
            )
        if self.time_reversal is None:
            return
        sample = torch.tensor(
            [0.2, 0.3, 0.1],
            dtype=self.hoppings.real.dtype,
            device=self.hoppings.device,
        )
        ham_k = self.hamiltonians_at(sample)
        ham_minus = self.hamiltonians_at(-sample)
        theta = self.time_reversal
        reconstructed = theta @ ham_k.conj() @ theta.conj().transpose(-2, -1)
        ham_norm = torch.linalg.matrix_norm(ham_k).clamp_min(1.0)
        error = float(
            (torch.linalg.matrix_norm(ham_minus - reconstructed) / ham_norm).item()
        )
        if error > 1e-4:
            warnings.warn(
                f"Time-reversal check failed (relative error {error:.3e}); "
                "Z2 indices are not well-defined.",
                RuntimeWarning,
                stacklevel=stacklevel,
            )


def _orbital_geometry(
    space,
    lattice,
    n_bands: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    tau = torch.zeros((n_bands, 3), dtype=dtype, device=device)
    if not isinstance(space, HilbertSpace):
        return tau, None
    states = list(space.elements())
    offsets: list[Offset | None] = []
    missing = 0
    for state in states:
        try:
            offsets.append(state.irrep_of(Offset))
        except ValueError:
            offsets.append(None)
            missing += 1
    if missing == len(states):
        return tau, None
    if missing:
        raise ValueError(
            "Hilbert space has Offset labels on only some orbitals; "
            "label every basis state or none."
        )

    positions = [None] * n_bands
    for state, offset in zip(states, offsets):
        assert offset is not None
        index = space.structure[state]
        positions[index] = torch.tensor(
            [float(c) for c in offset.to_vec()],
            dtype=torch.float64,
        )
        rebased = offset.rebase(lattice)
        for j in range(3):
            tau[index, j] = float(rebased.rep[j])
    if any(item is None for item in positions):
        raise RuntimeError("Offset labels do not cover every orbital index.")
    return tau, torch.stack([item for item in positions if item is not None])


def _spatial_inversion_matrix(
    space: HilbertSpace,
    center: Offset,
    hoppings: torch.Tensor,
) -> torch.Tensor:
    inv_op = PointGroupOpr(_inversion_element()).fixpoint_at(center)
    if contains_spin(space):
        matrix = spinful_hilbert_opr_repr(inv_op, space)
    else:
        matrix = hilbert_opr_repr(inv_op, space)
    return matrix.data.to(device=hoppings.device, dtype=hoppings.dtype)


def _build_engine(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None,
    inversion: Tensor | None,
    inversion_center: Offset | Sequence[float] | None,
) -> _Z2Engine:
    if bloch_hamiltonian.rank() != 3:
        raise ValueError("bloch_hamiltonian must have dimensions (k, band, band).")
    if not isinstance(bloch_hamiltonian.dims[1], HilbertSpace):
        raise TypeError("The second dimension must be a HilbertSpace.")
    if not isinstance(bloch_hamiltonian.dims[2], HilbertSpace):
        raise TypeError("The third dimension must be a HilbertSpace.")
    row_space = bloch_hamiltonian.dims[1]
    if not same_rays(row_space, bloch_hamiltonian.dims[2]):
        raise ValueError(
            "The last two dimensions of the tensor must span the same Hilbert space."
        )
    bloch_hamiltonian = bloch_hamiltonian.align(-1, row_space)

    data = bloch_hamiltonian.data
    if data.shape[-2] != data.shape[-1]:
        raise ValueError("The Hamiltonian must be square at every momentum.")
    n_bands = int(data.shape[-1])

    momenta = _bloch_momenta(bloch_hamiltonian)
    if not momenta or len(momenta[0].rep) != 3:
        raise ValueError("A three-dimensional momentum grid is required.")

    if n_occupied is None:
        n_occupied = n_bands // 2
    n_occupied = int(n_occupied)
    if not 0 < n_occupied < n_bands:
        raise ValueError("n_occupied must lie strictly between 0 and n_bands.")
    if n_occupied % 2:
        raise ValueError("Z2 requires an even occupied count (Kramers pairs).")

    reciprocal_lattice = momenta[0].space
    if not isinstance(reciprocal_lattice, ReciprocalLattice):
        raise TypeError("Momentum points must belong to a ReciprocalLattice.")
    lattice = reciprocal_lattice.lattice
    direct_boundary = lattice.boundaries
    if not isinstance(direct_boundary, PeriodicBoundary):
        raise ValueError("The momentum grid requires periodic lattice boundaries.")
    if not direct_boundary.basis.is_diagonal():
        raise ValueError(
            "Z2 Fourier interpolation requires a diagonal periodic cell; "
            "sheared boundaries are not supported."
        )
    dual_boundary = PeriodicBoundary(ImmutableDenseMatrix(direct_boundary.basis.T))
    grid_shape = tuple(int(n) for n in lattice.shape)
    if len(grid_shape) != 3:
        raise ValueError("A three-dimensional momentum grid is required.")
    n_k = math.prod(grid_shape)
    if n_k != len(momenta):
        raise ValueError(
            f"Found {len(momenta)} momenta, but the lattice shape is {grid_shape}."
        )

    canonical_reps = dual_boundary.representatives()
    if len(canonical_reps) != n_k:
        raise ValueError("Momentum points do not form a complete reciprocal grid.")
    canonical_position = {
        _rep_key(rep): position for position, rep in enumerate(canonical_reps)
    }

    h_grid = data.new_zeros((*grid_shape, n_bands, n_bands))
    seen: set[tuple[int, int, int]] = set()
    for sector, momentum in enumerate(momenta):
        integer_rep = ImmutableDenseMatrix(direct_boundary.basis.T @ momentum.rep)
        if any(not sy.sympify(value).is_integer for value in integer_rep):
            raise ValueError(
                "Momentum grid contains a point outside the reciprocal quotient."
            )
        key = _rep_key(dual_boundary.wrap(integer_rep))
        if key not in canonical_position:
            raise ValueError(
                f"Duplicate or incomplete momentum quotient representative {key}."
            )
        index = _unravel_index(canonical_position[key], grid_shape)
        if index in seen:
            raise ValueError(f"Duplicate momentum grid point {index}.")
        seen.add(index)
        h_grid[index] = data[sector]
    if len(seen) != n_k:
        raise ValueError("Momentum points do not form a complete reciprocal grid.")

    h_grid = 0.5 * (h_grid + h_grid.transpose(-1, -2).conj())
    hoppings = torch.fft.ifftn(h_grid, dim=(0, 1, 2))
    device = hoppings.device
    real_dtype = hoppings.real.dtype
    rx = (torch.fft.fftfreq(grid_shape[0], dtype=real_dtype) * grid_shape[0]).to(device)
    ry = (torch.fft.fftfreq(grid_shape[1], dtype=real_dtype) * grid_shape[1]).to(device)
    rz = (torch.fft.fftfreq(grid_shape[2], dtype=real_dtype) * grid_shape[2]).to(device)

    space = bloch_hamiltonian.dims[1]
    tau, positions_cart = _orbital_geometry(space, lattice, n_bands, device, real_dtype)

    inversion_hoppings = None
    inversion_i0 = None
    inversion_center_vec = None
    inversion_error: Exception | None = None
    if inversion is not None:
        if inversion.rank() != 3:
            raise ValueError("inversion must match bloch_hamiltonian's rank and shape.")
        if not same_rays(inversion.dims[1], row_space) or not same_rays(
            inversion.dims[2], row_space
        ):
            raise ValueError(
                "inversion band axes must span the Hamiltonian Hilbert space."
            )
        inversion = inversion.align(-2, row_space).align(-1, row_space)
        if inversion.data.shape != data.shape:
            raise ValueError("inversion must match bloch_hamiltonian's rank and shape.")
        inv_grid = inversion.data.new_zeros((*grid_shape, n_bands, n_bands))
        for sector, momentum in enumerate(momenta):
            integer_rep = ImmutableDenseMatrix(direct_boundary.basis.T @ momentum.rep)
            key = _rep_key(dual_boundary.wrap(integer_rep))
            index = _unravel_index(canonical_position[key], grid_shape)
            inv_grid[index] = inversion.data[sector]
        inversion_hoppings = torch.fft.ifftn(inv_grid, dim=(0, 1, 2))
    elif isinstance(space, HilbertSpace) and positions_cart is not None:
        if inversion_center is None:
            unique_reps: list[tuple[sy.Expr, ...]] = []
            for state in space.elements():
                frac = state.irrep_of(Offset).rebase(lattice).fractional()
                rep = tuple(sy.simplify(frac.rep[j]) for j in range(3))
                if rep not in unique_reps:
                    unique_reps.append(rep)
            n_sites = len(unique_reps)
            center_rep = sy.Matrix(
                [
                    sy.simplify(sum(rep[j] for rep in unique_reps) / n_sites)
                    for j in range(3)
                ]
            )
            center = Offset(rep=ImmutableDenseMatrix(center_rep), space=lattice)
        else:
            center = _as_inversion_center(inversion_center, lattice)
        try:
            inversion_i0 = _spatial_inversion_matrix(space, center, hoppings)
            inversion_center_vec = torch.tensor(
                [float(c) for c in center.to_vec()], dtype=torch.float64
            )
        except (RuntimeError, ValueError) as exc:
            inversion_i0 = None
            inversion_center_vec = None
            inversion_error = exc

    return _Z2Engine(
        hoppings=hoppings,
        rx=rx,
        ry=ry,
        rz=rz,
        tau=tau,
        n_occupied=n_occupied,
        n_bands=n_bands,
        lattice=lattice,
        inversion_hoppings=inversion_hoppings,
        inversion_i0=inversion_i0,
        inversion_center_vec=inversion_center_vec,
        positions_cart=positions_cart,
        inversion_error=inversion_error,
        time_reversal=_time_reversal_matrix(space, hoppings)
        if isinstance(space, HilbertSpace)
        else None,
    )


@overload
def z2_indices(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    *,
    method: Literal["auto"] = "auto",
    inversion: Tensor | None = None,
    inversion_center: Offset | Sequence[float] | None = None,
    n_loop: int = 32,
    n_perp: int = 17,
    parity_tolerance: float = 1e-5,
    kramers_tolerance: float = 0.08,
    gap_tolerance: float = 1e-8,
) -> Z2ParityResult | Z2WilsonResult: ...


@overload
def z2_indices(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    *,
    method: Literal["parity"],
    inversion: Tensor | None = None,
    inversion_center: Offset | Sequence[float] | None = None,
    n_loop: int = 32,
    n_perp: int = 17,
    parity_tolerance: float = 1e-5,
    kramers_tolerance: float = 0.08,
    gap_tolerance: float = 1e-8,
) -> Z2ParityResult: ...


@overload
def z2_indices(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    *,
    method: Literal["wilson"],
    inversion: Tensor | None = None,
    inversion_center: Offset | Sequence[float] | None = None,
    n_loop: int = 32,
    n_perp: int = 17,
    parity_tolerance: float = 1e-5,
    kramers_tolerance: float = 0.08,
    gap_tolerance: float = 1e-8,
) -> Z2WilsonResult: ...


@overload
def z2_indices(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    *,
    method: Literal["both"],
    inversion: Tensor | None = None,
    inversion_center: Offset | Sequence[float] | None = None,
    n_loop: int = 32,
    n_perp: int = 17,
    parity_tolerance: float = 1e-5,
    kramers_tolerance: float = 0.08,
    gap_tolerance: float = 1e-8,
) -> Z2CombinedResult: ...


def z2_indices(
    bloch_hamiltonian: Tensor,
    n_occupied: int | None = None,
    *,
    method: Literal["auto", "parity", "wilson", "both"] = "auto",
    inversion: Tensor | None = None,
    inversion_center: Offset | Sequence[float] | None = None,
    n_loop: int = 32,
    n_perp: int = 17,
    parity_tolerance: float = 1e-5,
    kramers_tolerance: float = 0.08,
    gap_tolerance: float = 1e-8,
) -> Z2ParityResult | Z2WilsonResult | Z2CombinedResult:
    r"""Compute the 3-D \(\mathbb{Z}_2\) indices of an occupied band subspace.

    The ``n_occupied`` lowest-energy eigenstates, which must form an even
    number of Kramers pairs, define an occupied bundle over a complete
    three-dimensional periodic momentum grid. Two numerical methods are
    available:

    - ``method="parity"`` evaluates Fu--Kane inversion eigenvalues at the eight
      TRIM and returns \((\nu_0; \nu_1\nu_2\nu_3)\) from their products.
    - ``method="wilson"`` computes hybrid Wannier charge centers on the six
      TRIM planes \(k_i=0\) and \(k_i=1/2\). The strong index is
      \(\nu_0=\nu(k_i=0)+\nu(k_i=\pi)\bmod 2\) and the weak indices are the
      three \(k_i=\pi\) plane invariants. If the three axes disagree on
      \(\nu_0\), the majority vote is returned.

    The Hamiltonian is Fourier-interpolated from the supplied mesh, so TRIM
    and Wilson strings need not coincide with sampled \(k\)-points. This is
    the construction used for odd diamond meshes such as \(27^3\). The
    periodic cell must be diagonal in the primitive basis.

    Parameters
    ----------
    bloch_hamiltonian : Tensor
        Rank-3 Hermitian [`Tensor`][qten.linalg.tensors.Tensor] with dims
        ``(MomentumSpace, HilbertSpace, HilbertSpace)``, or with a
        [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace]
        of diagonal \((k,k)\) blocks whose momenta form a complete 3-D
        reciprocal quotient. The last two axes are square Bloch-Hamiltonian
        matrices and must span the same Hilbert space up to ray ordering.
    n_occupied : int | None, optional
        Number of lowest-energy bands defining the occupied subspace. It must
        be even and lie strictly between zero and the total band count.
        Defaults to half the bands using integer division.
    method : {"auto", "parity", "wilson", "both"}, optional
        Numerical construction. ``"auto"`` tries Fu--Kane parity and falls
        back to Wilson loops if inversion cannot be resolved
        (`RuntimeError`). ``"both"`` requires parity to succeed. Defaults to
        ``"auto"``.
    inversion : Tensor | None, optional
        Optional rank-3 inversion tensor with the same shape as
        ``bloch_hamiltonian``. If omitted, spatial inversion is assembled from
        orbital [`Offset`][qten.geometries.spatials.Offset] labels about
        ``inversion_center``.
    inversion_center : Offset | Sequence[float] | None, optional
        Fixed point of spatial inversion, as an `Offset` or a 3-vector in the
        Hamiltonian's direct-lattice coordinates. Defaults to the centroid of
        the unique orbital offsets.
    n_loop : int, optional
        Number of Wilson-loop samples around each closed \(k\)-string.
        Must be at least 8 when Wilson loops are evaluated. Defaults to 32.
    n_perp : int, optional
        Number of hybrid-Wannier samples from a TRIM plane's \(k_\perp=0\)
        edge to \(k_\perp=\pi\). Must be at least 5 when Wilson loops are
        evaluated. Defaults to 17.
    parity_tolerance : float, optional
        Maximum relative \([H,I]\) commutator and inversion-eigenvalue
        deviation accepted at a TRIM. Defaults to ``1e-5``.
    kramers_tolerance : float, optional
        Maximum Wannier-center separation allowed when pairing Kramers
        partners on TRIM-plane endpoints. Defaults to ``0.08``.
    gap_tolerance : float, optional
        Warning threshold for the minimum sampled occupied-to-empty direct
        gap. Defaults to ``1e-8``.

    Returns
    -------
    dict[str, Any]
        Result mapping. Every method returns:

        - ``"indices"``: \((\nu_0, \nu_1, \nu_2, \nu_3)\) as integers in
          \(\{0,1\}\).
        - ``"method"``: the construction that produced those indices.

        For ``method="parity"`` the mapping also contains Fu--Kane
        ``"parity_products"`` at each TRIM, per-TRIM ``"diagnostics"``
        (including a labeled ``"parity_eigenvalues"``
        [`Tensor`][qten.linalg.tensors.Tensor]), and ``"direct_gap"``.

        For ``method="wilson"`` it contains hybrid-Wannier ``"planes"`` whose
        ``"wcc"``, ``"gap_pos"``, and ``"sweep"`` values are labeled
        [`Tensor`][qten.linalg.tensors.Tensor] objects, per-axis plane
        invariants ``"axis_z2"``, and ``"min_gap"``.

        For ``method="both"`` it contains both ``"parity"`` and ``"wilson"``
        sub-results; ``"indices"`` follows the parity values.

    Raises
    ------
    TypeError
        If the first tensor dimension is not a momentum space, or either
        matrix dimension is not a `HilbertSpace`.
    ValueError
        If ``method`` is unsupported; the input is not a rank-3 square Bloch
        Hamiltonian; ``n_occupied`` is invalid; the momentum space is not
        three-dimensional and periodic with a diagonal cell; a
        `MomentumBlockSpace` contains off-diagonal \((k,k')\) blocks; or
        momentum points do not form a unique complete reciprocal quotient.
    RuntimeError
        For ``method="parity"`` or ``method="both"``, if inversion cannot be
        constructed or is not resolved at a TRIM.

    Warns
    -----
    RuntimeWarning
        If the sampled minimum direct gap is no larger than ``gap_tolerance``;
        if ``method="auto"`` falls back from parity to Wilson loops; if the
        three Wilson axes disagree on \(\nu_0\) (majority vote is used); if
        Kramers pairing of Wannier centers is unresolved; if parity and
        Wilson indices disagree; or if the sampled spectrum is not
        time-reversal / Kramers consistent.

    Notes
    -----
    The Fu--Kane strong index is the product of the eight TRIM parity products
    \(\delta(\Gamma_i)\), and each weak index \(\nu_i\) is the product of
    \(\delta\) over the four TRIM with \(k_i=\pi\).

    Examples
    --------
    Use Fu--Kane parities when an inversion tensor is available:

    ```python
    result = z2_indices(hamiltonian, n_occupied=2, inversion=inversion, method="parity")
    nu0, nu1, nu2, nu3 = result["indices"]
    ```

    Fall back to Wilson loops on a system without inversion:

    ```python
    wilson = z2_indices(hamiltonian, n_occupied=2, method="wilson")
    ```

    See Also
    --------
    [`chern_number`][qten.topology.chern_number]
        First Chern number of a 2-D occupied bundle.
    """
    method_name = str(method).lower()
    if method_name not in {"auto", "parity", "wilson", "both"}:
        raise ValueError("method must be 'auto', 'parity', 'wilson', or 'both'.")

    engine = _build_engine(bloch_hamiltonian, n_occupied, inversion, inversion_center)
    engine.check_time_reversal(stacklevel=2)
    parity_result: Z2ParityResult | None = None
    wilson_result: Z2WilsonResult | None = None

    if method_name in {"parity", "both"}:
        parity_result = engine.run_parity(parity_tolerance)
    elif method_name == "auto":
        try:
            parity_result = engine.run_parity(parity_tolerance)
        except RuntimeError as exc:
            warnings.warn(
                f"Parity method unavailable ({exc}). Falling back to Wilson loops.",
                RuntimeWarning,
                stacklevel=2,
            )

    if (
        method_name == "wilson"
        or method_name == "both"
        or (method_name == "auto" and parity_result is None)
    ):
        wilson_result = engine.run_wilson(int(n_loop), int(n_perp), kramers_tolerance)

    if method_name == "both":
        if parity_result is None or wilson_result is None:
            raise RuntimeError("method='both' requires parity and Wilson results.")
        if parity_result["indices"] != wilson_result["indices"]:
            warnings.warn(
                f"Parity {parity_result['indices']} and Wilson "
                f"{wilson_result['indices']} disagree.",
                RuntimeWarning,
                stacklevel=2,
            )
        chosen: Z2ParityResult | Z2WilsonResult | Z2CombinedResult = {
            "indices": parity_result["indices"],
            "method": "both",
            "parity": parity_result,
            "wilson": wilson_result,
        }
    elif parity_result is not None and wilson_result is None:
        chosen = parity_result
    elif wilson_result is not None:
        chosen = wilson_result
    else:
        raise RuntimeError("Z2 calculation produced no result.")

    if chosen["method"] == "parity":
        min_gap = chosen["direct_gap"]
    elif chosen["method"] == "wilson":
        min_gap = chosen["min_gap"]
    else:
        sampled = [chosen["parity"]["direct_gap"], chosen["wilson"]["min_gap"]]
        finite = [gap for gap in sampled if math.isfinite(gap)]
        min_gap = min(finite) if finite else float("nan")
    if math.isfinite(min_gap) and min_gap <= gap_tolerance:
        warnings.warn(
            f"Minimum sampled direct gap is {min_gap:.6e}; the occupied "
            "bundle is not isolated, so its Z2 indices are not well-defined.",
            RuntimeWarning,
            stacklevel=2,
        )
    return chosen


__all__ = [
    "Z2CombinedResult",
    "Z2ParityResult",
    "Z2ParityTrimDiagnostics",
    "Z2WilsonPlaneResult",
    "Z2WilsonResult",
    "z2_indices",
]
