r"""
Band-structure helpers for momentum-resolved QTen tensors.

This module provides utilities for transforming, folding, unfolding, filling,
and selecting bands represented as [`Tensor`][qten.linalg.tensors.Tensor]
objects. The common convention is that a band tensor has dimensions
`(MomentumSpace, HilbertSpace, HilbertSpace)`: the
[`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis indexes
crystal momenta and the two
[`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] axes form the
Hamiltonian or operator matrix at each momentum.

Mathematical convention
-----------------------
A band tensor represents a family of matrices indexed by crystal momentum:
\(H : k \mapsto H(k)\), with
\(H(k)_{ab} = \langle a | H(k) | b \rangle\).

In code this is stored as a rank-3 [`Tensor`][qten.linalg.tensors.Tensor] with
dims `(K, B_left, B_right)`, where `K` is a
[`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and the two
Hilbert-space axes provide the row and column basis labels for each matrix
block.

Geometry transformations act on both parts of this object:
\(k \mapsto k'\) and
\(H(k) \mapsto U(k)\,H(k)\,U(k)^\dagger\).

where the \(k\)-dependent change-of-basis matrix \(U(k)\) is assembled from
symbolic Hilbert-space relabeling and finite Fourier transforms.

Repository usage
----------------
The functions here sit between geometry, symbolic Hilbert-space labels, and
linear algebra. Geometry objects provide real and reciprocal lattice structure,
symbolic state spaces label tensor axes, and linear algebra routines diagonalize
the momentum-sector matrices when filling or selecting bands.
"""

from collections import OrderedDict
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)
import warnings

import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix
from multimethod import multimethod

# TODO: Avoid using torch explicitly here.
import torch

from .geometries import (
    BasisTransform,
    InverseBasisTransform,
    Lattice,
    Momentum,
    Offset,
    ReciprocalLattice,
)
from .geometries.fourier import fourier_transform
from .linalg import eigh
from .linalg._mb_tensor import MomentumBlockTensor
from .linalg.tensors import Tensor
from .precision import get_precision_config
from .symbolics import (
    BzPath,
    FuncOpr,
    HilbertSpace,
    IndexSpace,
    MomentumBlockSpace,
    MomentumSpace,
    Opr,
    StateSpace,
    U1Basis,
    brillouin_zone,
    fractional_opr,
    interpolate_reciprocal_path,
    rebase_opr,
    restructure,
)
from .utils.devices import Device


def _basis_states_at_fractional_offset(
    space: HilbertSpace, offset: Offset
) -> tuple[U1Basis, ...]:
    """Return all basis states whose Offset matches the requested fractional site."""
    target = offset.fractional()
    matches = tuple(
        cast(U1Basis, psi)
        for psi in space.elements()
        if cast(U1Basis, psi).irrep_of(Offset) == target
    )
    if not matches:
        raise ValueError(
            f"No basis states found for fractional offset {target!r} in HilbertSpace."
        )
    return matches


def interpolate_path(
    recip: ReciprocalLattice,
    waypoints: Sequence[Union[Tuple[float, ...], str]],
    n_points: int = 100,
    labels: Optional[Sequence[str]] = None,
    points: Optional[Dict[str, Tuple[float, ...]]] = None,
    waypoint_transform: Optional[Union[BasisTransform, InverseBasisTransform]] = None,
) -> BzPath:
    """
    Build a sampled Brillouin-zone path in a reciprocal lattice.

    This is a backward-compatible wrapper around
    [`interpolate_reciprocal_path`][qten.symbolics.ops.interpolate_reciprocal_path].
    New code may call that symbolic helper directly.

    Parameters
    ----------
    recip : ReciprocalLattice
        Reciprocal lattice in which waypoint coordinates are interpreted.
    waypoints : Sequence[Union[Tuple[float, ...], str]]
        Sequence of explicit fractional coordinates or names looked up in
        `points`.
        For example, `[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]`
        samples a path through three explicit two-dimensional reciprocal
        coordinates, while `["G", "X", "M"]` resolves coordinates from the
        `points` mapping.
    n_points : int
        Number of samples used along the full interpolated path.
    labels : Sequence[str] | None
        Optional display labels for the waypoint ticks.
        For example, `["Γ", "X", "M"]` can label a path whose named inputs are
        `["G", "X", "M"]`.
    points : Dict[str, Tuple[float, ...]] | None
        Optional mapping from waypoint names to fractional reciprocal
        coordinates. For example,
        `{"G": (0.0, 0.0), "X": (0.5, 0.0), "M": (0.5, 0.5)}`.
    waypoint_transform : BasisTransform | InverseBasisTransform | None
        Optional basis transform applied to waypoint fractional coordinates
        before interpolation.
        For [`BasisTransform`][qten.geometries.basis_transform.BasisTransform],
        transformed coordinates are computed as `frac @ M.T`.
        For [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform],
        transformed coordinates are computed as `frac @ M^{-T}`.

    Returns
    -------
    BzPath
        Sampled Brillouin-zone path with momentum space, waypoint labels, and
        path-order metadata.

    Raises
    ------
    ValueError
        If fewer than two waypoints are supplied, if a named waypoint is not
        present in `points`, if waypoint coordinate dimensions do not match
        `recip.dim`, if `n_points` is too small for the number of waypoints, if
        all waypoints are identical, or if `labels` does not match the number
        of waypoints.

    See Also
    --------
    [`interpolate_reciprocal_path(recip, waypoints, n_points, labels, points, waypoint_transform)`][qten.symbolics.ops.interpolate_reciprocal_path]
        Canonical implementation used by this compatibility wrapper.

    Examples
    --------
    ```python
    path = interpolate_path(
        recip,
        waypoints=[(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)],
        labels=["Γ", "X", "M"],
    )
    ```

    ```python
    path = interpolate_path(
        recip,
        waypoints=["G", "X", "M"],
        labels=["Γ", "X", "M"],
        points={"G": (0.0, 0.0), "X": (0.5, 0.0), "M": (0.5, 0.5)},
    )
    ```
    """
    return interpolate_reciprocal_path(
        recip=recip,
        waypoints=waypoints,
        n_points=n_points,
        labels=labels,
        points=points,
        waypoint_transform=waypoint_transform,
    )


def _probe_affine(
    raw_opr: Callable[[Momentum], Momentum],
    recip_lat: ReciprocalLattice,
) -> Tuple[np.ndarray, np.ndarray, ReciprocalLattice]:
    r"""
    Probe *raw_opr* with ``d + 1`` reference momenta to extract its affine
    decomposition
    \(\mathrm{output\_frac} = \mathrm{input\_frac}\, M^{\mathsf{T}} + c\).
    In code, this is represented by the row-vector expression
    `input_frac @ M.T + c`.

    Returns ``(M, c, result_space)`` where *result_space* is the reciprocal
    lattice carried by the output momenta.
    """
    dim = recip_lat.dim
    zero_k = Momentum(rep=ImmutableDenseMatrix([sy.Integer(0)] * dim), space=recip_lat)
    zero_out = raw_opr(zero_k)
    result_space = zero_out.space
    c = np.array([float(zero_out.rep[j, 0]) for j in range(dim)])

    M = np.zeros((dim, dim))
    for i in range(dim):
        e_rep: list[sy.Expr] = [sy.Integer(0)] * dim
        e_rep[i] = sy.Integer(1)
        e_k = Momentum(rep=ImmutableDenseMatrix(e_rep), space=recip_lat)
        e_out = raw_opr(e_k)
        for j in range(dim):
            M[j, i] = float(e_out.rep[j, 0]) - c[j]

    return M, c, result_space


def _momentum_match_indices(
    src: MomentumSpace,
    dest: MomentumSpace,
    transform: Union[np.ndarray, Callable[[Momentum], Momentum]],
    *,
    device: Optional[Device] = None,
) -> Tensor[torch.LongTensor]:
    r"""
    Batch-compute destination indices for a momentum-space mapping via
    integer grid lookup.

    This is the [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace]-specialized counterpart of
    [`match_indices`][qten.symbolics.ops.match_indices]. Instead of evaluating *transform* per element, the
    transformation is applied as a single matrix multiply over all source
    k-points, followed by fractional wrapping and determinant-scaled
    integer snapping (correct for both diagonal and non-diagonal
    boundaries).

    In fractional reciprocal coordinates, affine mappings have the form
    \(\kappa' = M\kappa + c \pmod{1}\). In code, source fractional coordinates are rows, so the matrix product is
    `src_frac @ M.T`, followed by optional `+ c`.
    """
    src_elements = src.elements()
    if not src_elements:
        torch_device = device.torch_device() if device is not None else None
        return Tensor(
            data=cast(
                torch.LongTensor,
                torch.tensor([], dtype=torch.long, device=torch_device),
            ),
            dims=(src,),
        )

    if callable(transform):
        recip_lat = src_elements[0].space
        M, c, _ = _probe_affine(transform, recip_lat)
    else:
        M, c = transform, None

    precision = get_precision_config()
    src_frac = np.array(
        [
            [float(k.rep[j, 0]) for j in range(src_elements[0].space.dim)]
            for k in src_elements
        ],
        dtype=precision.np_float,
    )
    mapped = src_frac @ M.T
    if c is not None:
        mapped = mapped + c
    mapped_wrapped = mapped - np.floor(mapped)

    first_dest_k = next(iter(dest.structure))
    dim = first_dest_k.space.dim
    boundary_np = np.array(
        first_dest_k.space.lattice.boundaries.basis.evalf(),
        dtype=precision.np_float,
    )
    D = abs(int(round(np.linalg.det(boundary_np))))
    mapped_scaled = np.rint(mapped_wrapped * D).astype(np.int64) % D
    dest_items = list(dest.structure.items())
    dest_coords = np.array(
        [
            [int(round(float(k.rep[j, 0]) * D)) % D for j in range(dim)]
            for k, _ in dest_items
        ],
        dtype=np.int64,
    )
    dest_indices = np.array([idx for _, idx in dest_items], dtype=np.int64)

    def _row_keys(arr: np.ndarray) -> np.ndarray:
        arr_c = np.ascontiguousarray(arr)
        return arr_c.view(np.dtype((np.void, arr_c.dtype.itemsize * arr_c.shape[1])))

    src_keys = _row_keys(mapped_scaled).reshape(-1)
    dest_keys = _row_keys(dest_coords).reshape(-1)
    order = np.argsort(dest_keys)
    sorted_dest_keys = dest_keys[order]

    pos = np.searchsorted(sorted_dest_keys, src_keys)
    in_range = pos < sorted_dest_keys.size
    matched = np.zeros_like(in_range, dtype=bool)
    matched[in_range] = sorted_dest_keys[pos[in_range]] == src_keys[in_range]
    if not np.all(matched):
        bad_idx = int(np.flatnonzero(~matched)[0])
        gcoord = tuple(int(x) for x in mapped_scaled[bad_idx])
        raise ValueError(
            f"Source momentum maps to scaled coord {gcoord} (D={D}), "
            f"not in destination BZ."
        )

    indices = dest_indices[order[pos]]

    torch_device = device.torch_device() if device is not None else None
    return Tensor(
        data=cast(
            torch.LongTensor,
            torch.tensor(indices, dtype=torch.long, device=torch_device),
        ),
        dims=(src,),
    )


def _momentum_map(
    kspace: MomentumSpace,
    raw_opr: Callable[[Momentum], Momentum],
) -> MomentumSpace:
    """
    Batch-compute ``kspace.map(lambda k: raw_opr(k).fractional())``.

    *raw_opr* must be the **unwrapped** operator (e.g. ``lambda k: t @ k``).
    Fractional wrapping is applied in bulk via numpy after the linear
    transformation matrix has been determined by probing with ``d + 1``
    reference momenta.
    """
    k_elements = kspace.elements()
    if not k_elements:
        return kspace

    recip_lat = k_elements[0].space
    dim = recip_lat.dim
    M, c, result_space = _probe_affine(raw_opr, recip_lat)
    precision = get_precision_config()
    k_frac = np.array(
        [[float(k.rep[j, 0]) for j in range(dim)] for k in k_elements],
        dtype=precision.np_float,
    )
    new_frac = k_frac @ M.T + c
    new_frac_wrapped = new_frac - np.floor(new_frac)

    boundary_np = np.array(
        result_space.lattice.boundaries.basis.evalf(),
        dtype=precision.np_float,
    )
    D = abs(int(round(np.linalg.det(boundary_np))))
    grid_ints = np.rint(new_frac_wrapped * D).astype(np.int64) % D

    new_structure: OrderedDict[Momentum, int] = OrderedDict()
    for i, (k, idx) in enumerate(kspace.structure.items()):
        rep = ImmutableDenseMatrix(
            [sy.Rational(int(grid_ints[i, j]), D) for j in range(dim)]
        )
        new_structure[Momentum(rep=rep, space=result_space)] = idx

    return MomentumSpace(structure=restructure(new_structure))


def _validate_block_transformable_tensor(
    tensor: Tensor, func_name: str, side: Literal["left", "right"]
) -> tuple[MomentumSpace, HilbertSpace]:
    if tensor.rank() != 3:
        raise ValueError(f"{func_name} requires a rank-3 tensor.")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError(
            f"{func_name} requires the first dimension to be a MomentumSpace."
        )
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise TypeError(
            f"{func_name} requires the second dimension to be a HilbertSpace."
        )
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise TypeError(
            f"{func_name} requires the third dimension to be a HilbertSpace."
        )

    sampled_space = (
        cast(HilbertSpace, tensor.dims[1])
        if side == "left"
        else cast(HilbertSpace, tensor.dims[2])
    )
    return cast(MomentumSpace, tensor.dims[0]), sampled_space


def _get_band_transform_from_spaces(
    t: Opr,
    kspace: MomentumSpace,
    target_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> MomentumBlockTensor:
    mapped_kspace = _momentum_map(kspace, lambda k: cast(Momentum, t @ k))
    if mapped_kspace.dim != kspace.dim:
        raise ValueError(
            "get_band_transform requires a one-to-one momentum map with no sector collisions."
        )

    fractional = FuncOpr(Offset, Offset.fractional)
    raw_space = cast(HilbertSpace, t @ target_space)
    new_space = cast(HilbertSpace, fractional @ raw_space)
    if not target_space.same_rays(new_space):
        raise ValueError(
            f"Hilbert space {target_space} is not closed under the transform {t}!"
        )

    transformed_fourier = fourier_transform(
        mapped_kspace, new_space, raw_space, device=device
    ).replace_dim(2, target_space)
    home_transform = cast(
        Tensor, target_space.cross_gram(new_space, device=device)
    ).replace_dim(1, new_space)
    transform = cast(Tensor, home_transform @ transformed_fourier)

    pair_space = MomentumBlockSpace(
        structure=OrderedDict(
            ((mapped_k, src_k), i)
            for i, (mapped_k, src_k) in enumerate(
                zip(mapped_kspace.elements(), kspace.elements())
            )
        )
    )
    return MomentumBlockTensor(
        data=transform.data,
        dims=(pair_space, target_space, target_space),
    )


@overload
def get_band_transform(
    t: Opr,
    tensor: Tensor,
    side: Literal["left", "right"] = "left",
) -> MomentumBlockTensor:
    """
    Build a band transform from a rank-3 momentum-resolved tensor.
    """
    ...


@overload
def get_band_transform(
    t: Opr,
    kspace: MomentumSpace,
    target_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> MomentumBlockTensor:
    """
    Build a band transform directly from explicit symbolic spaces.

    This overload accepts a
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and the
    sampled [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
    directly, then dispatches to the main
    [`get_band_transform`][qten.bands.get_band_transform] implementation.
    """
    ...


@multimethod
def get_band_transform(
    t: Opr,
    tensor: Tensor,
    side: Literal["left", "right"] = "left",
) -> MomentumBlockTensor:
    r"""
    Construct a reusable one-sided geometric basis-change operator for a
    momentum-resolved band tensor.

    Supported forms
    ---------------
    [`get_band_transform(t, tensor, side=...)`][qten.bands.get_band_transform]
        Build the transform from a rank-3 band tensor with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`, using `side` to choose
        which Hilbert-space leg is sampled.

    [`get_band_transform(t, kspace, target_space, device=...)`][qten.bands.get_band_transform]
        Build the same transform directly from an explicit
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] `kspace`
        and sampled [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
        `target_space`, without first packaging them into a rank-3 tensor.

    Use cases
    ---------
    This helper is useful when the geometric action should be materialized once
    and then reused across multiple band tensors, when only one matrix leg of a
    non-Hermitian or rectangular band object should be transformed, or when the
    transform itself should be inspected as a
    [`MomentumBlockTensor`][qten.MomentumBlockTensor] rather than applied
    immediately.

    For example, a caller may build the left and right transforms separately,
    cache them, and apply them to several tensors sharing the same symbolic
    momentum and Hilbert spaces.

    This function factors one side of the geometric basis change performed by
    [`bandtransform`][qten.bands.bandtransform] into an explicit
    [`MomentumBlockTensor`][qten.MomentumBlockTensor] \(T_g\). For a
    momentum-resolved operator \(H\), the one-sided transformed tensor is
    recovered by \(T_g H\) when `side="left"` and by
    \(H T_g^\dagger\) when `side="right"`. Applying both sides requires
    composing the two separately constructed transforms.

    The leading axis of \(T_g\) is a
    [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace]
    storing ordered pairs `(t @ k, k)`: each block describes the basis map from
    the source momentum sector `k` to the transformed sector `t @ k`.

    Behavior
    --------
    This function does not transform tensor data directly. Instead it builds a
    block operator whose momentum-pair axis records how source sectors feed
    transformed sectors. The Hilbert-space block at each such pair combines:

    1. the symbolic action of `t` on the sampled basis,
    2. fractional wrapping back to the home unit cell, and
    3. the Fourier phase needed to keep Bloch conventions consistent after the
       geometric relabeling.

    The returned transform is therefore the reusable one-sided ingredient of
    [`bandtransform`][qten.bands.bandtransform], not merely a permutation of
    momentum labels.

    Basis sampling
    --------------
    The input tensor may have dims `(K, B_left, B_right)` with potentially
    different left and right Hilbert spaces. The `side` argument chooses which
    matrix leg supplies the canonical Hilbert space used to assemble \(T_g\).

    Parameters
    ----------
    t : Opr
        Operator acting consistently on both
        [`Momentum`][qten.geometries.spatials.Momentum] and the
        [`Offset`][qten.geometries.spatials.Offset]-carrying basis states inside
        the sampled Hilbert space.
    tensor : Tensor
        Rank-3 momentum-space tensor with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    side : Literal["left", "right"], optional
        Which matrix leg to sample when constructing the transform basis.
        `side="left"` uses `tensor.dims[1]`; `side="right"` uses
        `tensor.dims[2]`. The default is `"left"`.

    Returns
    -------
    MomentumBlockTensor
        Block transform tensor with dims
        `(MomentumBlockSpace, B, B)`, where `B` is the sampled Hilbert space
        selected by `side`.

    Raises
    ------
    ValueError
        If `tensor` is not rank 3, if the transformed Hilbert space is not
        closed on the sampled basis after fractional wrapping, or if the
        momentum action of `t` is not one-to-one on the input
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace].
    TypeError
        If the tensor dims do not have the required
        `MomentumSpace/HilbertSpace/HilbertSpace` structure.

    Notes
    -----
    The generated API docs for this module show overload signatures, but the
    prose is rendered from this public implementation docstring. The explicit
    space overload accepts `kspace` and `target_space` directly, then
    dispatches here through the shared construction path.

    Examples
    --------
    Build and apply only the left transform:

    ```python
    T_left = get_band_transform(t, tensor, side="left")
    routed = T_left @ tensor
    ```

    Build both one-sided transforms explicitly and compose them:

    ```python
    T_left = get_band_transform(t, tensor, side="left")
    T_right = get_band_transform(t, tensor, side="right")
    transformed = T_left @ tensor @ T_right.h(-2, -1)
    ```

    Build the transform directly from symbolic spaces:

    ```python
    T_left = get_band_transform(t, kspace, hilbert_space, device=device)
    ```

    See Also
    --------
    [`bandtransform(t, tensor, opt=...)`][qten.bands.bandtransform]
        Public wrapper that applies one-sided or two-sided band transforms.
    [`get_band_fold(transform, tensor, side=...)`][qten.bands.get_band_fold]
        Folding analogue that builds the corresponding block transform for a
        basis-change-induced Brillouin-zone fold.
    """
    kspace, target_space = _validate_block_transformable_tensor(
        tensor, "get_band_transform", side
    )
    return _get_band_transform_from_spaces(
        t, kspace, target_space, device=tensor.device
    )


# mypy does not model multimethod's dynamic .register API.
@get_band_transform.register  # type: ignore[attr-defined]
def _(
    t: Opr,
    kspace: MomentumSpace,
    target_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> MomentumBlockTensor:
    return _get_band_transform_from_spaces(t, kspace, target_space, device=device)


def _get_band_fold_from_spaces(
    transform: BasisTransform,
    k_space: MomentumSpace,
    target_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> MomentumBlockTensor:
    if not k_space.elements():
        raise ValueError("MomentumSpace is empty")
    lattice_set = set(map(lambda k: k.space, k_space))
    if len(lattice_set) != 1:
        raise ValueError("Invalid BZ")
    reciprocal_lattice = lattice_set.pop()
    if not isinstance(reciprocal_lattice, ReciprocalLattice):
        raise TypeError(
            f"Space of momentum should be ReciprocalLattice, but got {type(reciprocal_lattice)}"
        )
    reciprocal_lattice = cast(ReciprocalLattice, reciprocal_lattice)
    lattice = reciprocal_lattice.dual

    scaled_lattice = transform(lattice)
    scaled_reciprocal_lattice = scaled_lattice.dual
    transformed_unit_cell = tuple(
        sorted(scaled_lattice.unit_cell.values(), key=lambda x: tuple(x.rep))
    )
    enlarge_unit_cell = tuple(r.rebase(lattice) for r in transformed_unit_cell)

    rebased_hilbert = HilbertSpace.new(
        psi.replace(r)
        for r in enlarge_unit_cell
        for psi in _basis_states_at_fractional_offset(target_space, r)
    )
    transformed_hilbert = HilbertSpace.new(
        psi.replace(r_out)
        for r, r_out in zip(enlarge_unit_cell, transformed_unit_cell)
        for psi in _basis_states_at_fractional_offset(target_space, r)
    )

    f = fourier_transform(k_space, target_space, rebased_hilbert, device=device)
    vratio = np.sqrt(len(enlarge_unit_cell) / len(lattice.unit_cell))
    fh = (f / vratio).h(-2, -1)

    new_k_space = brillouin_zone(scaled_reciprocal_lattice)
    precision = get_precision_config()
    old_basis_np = np.array(reciprocal_lattice.basis.evalf(), dtype=precision.np_float)
    new_basis_np = np.array(
        scaled_reciprocal_lattice.basis.evalf(), dtype=precision.np_float
    )
    M_rebase = np.linalg.solve(new_basis_np, old_basis_np)
    k_indices = _momentum_match_indices(k_space, new_k_space, M_rebase, device=device)
    new_k_elements = new_k_space.elements()
    pair_space = MomentumBlockSpace(
        structure=OrderedDict(
            (
                (new_k_elements[int(dst_idx)], src_k),
                i,
            )
            for i, (src_k, dst_idx) in enumerate(
                zip(k_space.elements(), k_indices.data.tolist())
            )
        )
    )

    return MomentumBlockTensor(
        data=fh.data,
        dims=(pair_space, transformed_hilbert, target_space),
    )


@overload
def get_band_fold(
    transform: BasisTransform,
    tensor: Tensor,
    side: Literal["left", "right"] = "left",
) -> MomentumBlockTensor:
    """
    Build a band-folding transform from a rank-3 momentum-resolved tensor.
    """
    ...


@overload
def get_band_fold(
    transform: BasisTransform,
    k_space: MomentumSpace,
    target_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> MomentumBlockTensor:
    """
    Build a band-folding transform directly from explicit symbolic spaces.

    This overload accepts a
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and the
    sampled [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
    directly, then dispatches to the main
    [`get_band_fold`][qten.bands.get_band_fold] implementation.
    """
    ...


@multimethod
def get_band_fold(
    transform: BasisTransform,
    tensor: Tensor,
    side: Literal["left", "right"] = "left",
) -> MomentumBlockTensor:
    r"""
    Construct a reusable one-sided Brillouin-zone folding operator for a
    momentum-resolved band tensor.

    Supported forms
    ---------------
    [`get_band_fold(transform, tensor, side=...)`][qten.bands.get_band_fold]
        Build the folding transform from a rank-3 band tensor with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`, using `side` to choose
        which Hilbert-space leg is sampled.

    [`get_band_fold(transform, k_space, target_space, device=...)`][qten.bands.get_band_fold]
        Build the same folding transform directly from an explicit
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] `k_space`
        and sampled [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
        `target_space`, without first packaging them into a rank-3 tensor.

    Use cases
    ---------
    This helper is useful when Brillouin-zone folding should be factored into a
    reusable one-sided operator, when left and right Hilbert-space legs should
    be folded independently, or when the folding map itself should be examined
    as a block tensor before applying it to data.

    Typical workflows include caching folded-cell transforms for repeated use,
    applying folding to only one matrix leg of a tensor, or explicitly
    constructing the left and right folded operators before composing them.

    This function factors one side of the Brillouin-zone folding operation
    into an explicit [`MomentumBlockTensor`][qten.MomentumBlockTensor] \(T_g\).
    For a momentum-resolved operator \(H\), the one-sided folded tensor is
    recovered by \(T_g H\) when `side="left"` and by
    \(H T_g^\dagger\) when `side="right"`. Folding both sides requires
    composing the two separately constructed transforms.

    Each block of \(T_g\) is labelled by a pair
    \((k_{\mathrm{fold}}, k)\) on its leading
    [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace],
    where \(k\) is a momentum of the original Brillouin zone and
    \(k_{\mathrm{fold}}\) is the momentum sector it maps to in the folded
    zone.
    The Hilbert-space legs encode the Fourier-based change of basis between the
    original unit cell and the enlarged transformed cell.

    Behavior
    --------
    Folding changes both the momentum grid and the real-space basis. This
    helper builds the one-sided block operator that performs those two tasks
    together:

    1. each source momentum sector is routed to its folded-zone momentum,
    2. the sampled Hilbert-space basis is enlarged to the transformed unit
       cell, and
    3. the corresponding Fourier change of basis is assembled into each block.

    The result is not just a relabeling of momentum sectors. It is the
    reusable one-sided ingredient of [`bandfold`][qten.bands.bandfold] that
    carries both sector routing and enlarged-cell basis conversion.
    When multiple basis states share the same fractional site within the
    sampled Hilbert space, all of them are preserved in the enlarged folded
    basis.

    Basis sampling
    --------------
    The input tensor may have dims `(K, B_left, B_right)` with potentially
    different left and right Hilbert spaces. The `side` argument chooses which
    leg supplies the canonical basis from which the folding transform is built.

    Parameters
    ----------
    transform : BasisTransform
        Direct-lattice basis transformation that defines the folded Brillouin
        zone.
    tensor : Tensor
        Rank-3 momentum-space tensor with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    side : Literal["left", "right"], optional
        Which matrix leg to sample when constructing the folding basis.
        `side="left"` uses `tensor.dims[1]`; `side="right"` uses
        `tensor.dims[2]`. The default is `"left"`.

    Returns
    -------
    MomentumBlockTensor
        Block folding tensor with dims `(MomentumBlockSpace, B_fold, B)`, where
        `B` is the sampled input Hilbert space selected by `side` and
        `B_fold` is the corresponding enlarged folded-cell Hilbert space.

    Raises
    ------
    ValueError
        If `tensor` is not rank 3, if its momentum axis is empty or
        inconsistent, or if the sampled Hilbert basis has no states at a
        required unit-cell offset during the folding construction.
    TypeError
        If the tensor dims do not have the required
        `MomentumSpace/HilbertSpace/HilbertSpace` structure, or if the momentum
        axis is not backed by a
        [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice].

    Notes
    -----
    The generated API docs for this module show overload signatures, but the
    prose is rendered from this public implementation docstring. The explicit
    space overload accepts `k_space` and `target_space` directly, then
    dispatches here through the shared folding construction path.

    Examples
    --------
    Build and apply only the right folding transform:

    ```python
    T_right = get_band_fold(transform, tensor, side="right")
    routed = tensor @ T_right.h(-2, -1)
    ```

    Build both one-sided folding transforms explicitly:

    ```python
    T_left = get_band_fold(transform, tensor, side="left")
    T_right = get_band_fold(transform, tensor, side="right")
    folded = T_left @ tensor @ T_right.h(-2, -1)
    ```

    Build the folding transform directly from symbolic spaces:

    ```python
    T_left = get_band_fold(transform, k_space, hilbert_space, device=device)
    ```

    See Also
    --------
    [`bandfold(transform, tensor, opt=...)`][qten.bands.bandfold]
        Public wrapper that applies this block folding transform to a band
        tensor.
    [`get_band_transform(t, tensor, side=...)`][qten.bands.get_band_transform]
        Symmetry-transform analogue that constructs a momentum-block transform
        without Brillouin-zone folding.
    """
    k_space, target_space = _validate_block_transformable_tensor(
        tensor, "get_band_fold", side
    )
    return _get_band_fold_from_spaces(
        transform, k_space, target_space, device=tensor.device
    )


# mypy does not model multimethod's dynamic .register API.
@get_band_fold.register  # type: ignore[attr-defined]
def _(
    transform: BasisTransform,
    k_space: MomentumSpace,
    target_space: HilbertSpace,
    *,
    device: Optional[Device] = None,
) -> MomentumBlockTensor:
    return _get_band_fold_from_spaces(transform, k_space, target_space, device=device)


def bandtransform(
    t: Opr,
    tensor: Tensor,
    opt: Literal["left", "right", "both"] = "both",
) -> Tensor:
    r"""
    Apply a basis transform to a momentum-resolved operator tensor.

    The expected tensor shape is `(K, B_left, B_right)` where `K` is a
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and
    `B_left`, `B_right` are
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] axes. This
    function applies the operator-induced basis transform on the selected
    Hilbert-space legs of the band tensor.

    For each transformed side, a k-dependent matrix is built from the action of
    `t` on the Hilbert-space basis and Fourier transforms that connect Bloch and
    real-space sectors.

    Mathematical action
    -------------------
    Let \(B_{\mathrm{left}}\) and \(B_{\mathrm{right}}\) be the input
    Hilbert-space bases and let the corresponding transformed bases be
    \(tB_{\mathrm{left}}\) and \(tB_{\mathrm{right}}\). After wrapping
    transformed sites back to the home unit cell, the finite Fourier transform
    contributes a momentum-dependent phase. The resulting basis-change matrices
    are denoted \(U_t^{(\mathrm{left})}(k)\) and
    \(U_t^{(\mathrm{right})}(k)\). When routed contributions are collapsed back
    onto the transformed momentum grid, the transformed band block is one of:

    `opt="left"`:
    \(H'(t k) = U_t^{(\mathrm{left})}(k)\,H(k)\)

    `opt="right"`:
    \(H'(t k) = H(k)\,U_t^{(\mathrm{right})}(k)^\dagger\)

    `opt="both"`:
    \(H'(t k) = U_t^{(\mathrm{left})}(k)\,H(k)\,U_t^{(\mathrm{right})}(k)^\dagger\)

    Momentum handling
    -----------------
    The action on [`Momentum`][qten.geometries.spatials.Momentum] is treated as
    a relabeling or permutation of sectors. For `opt="both"`, the output
    tensor carries the transformed momentum axis
    `mapped_kspace = {t @ k | k in kspace}`. For `opt="left"` and
    `opt="right"`, the implementation instead preserves a routed
    [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace] pair
    axis so each source block remains attached to its transformed target
    sector. In either case, the selected Hilbert-space transforms are applied
    before any optional collapse back to a plain
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace].

    Notes
    -----
    This function accepts a general [`Opr`][qten.symbolics.hilbert_space.Opr], but not every [`Opr`][qten.symbolics.hilbert_space.Opr] is valid here.
    In practice, `t` must act coherently across the real-space and
    momentum-space labels carried by the tensor:

    `t @ k` must be defined for each
    [`Momentum`][qten.geometries.spatials.Momentum] in the first tensor axis.
    `t @ psi` must be defined for each
    [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] in the Hilbert-space
    axes, in particular for the
    [`Offset`][qten.geometries.spatials.Offset] irrep stored inside each basis
    state.
    The Hilbert-space action and momentum action must be dual-compatible, so
    that the Fourier transform remains consistent after applying `t`.
    For each selected side, after applying
    [`FuncOpr(Offset, Offset.fractional)`][qten.symbolics.hilbert_space.FuncOpr],
    the transformed Hilbert space must have the same rays as the sampled input
    basis on that side. Otherwise the transformed basis does not close on that
    band leg and this function raises `ValueError`.

    Operators that only act on abstract [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] values or only on [`Momentum`][qten.geometries.spatials.Momentum]
    values are not sufficient. The operator must provide matching actions on
    site offsets and crystal momentum.

    Parameters
    ----------
    t : Opr
        Operator to apply. It must satisfy the compatibility conditions
        described in the notes below.
    tensor : Tensor
        Momentum-space tensor with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    opt : Literal["left", "right", "both"], optional
        Which matrix legs to transform. `"left"` applies only the left
        transform, `"right"` applies only the right transform, and `"both"`
        applies independent transforms on both sides. The default is `"both"`.

    Returns
    -------
    Tensor
        If `opt="both"`, returns the transformed tensor with a transformed
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis and
        transformed Hilbert-space matrix legs.
        If `opt="left"` or `opt="right"`, returns the corresponding one-sided
        routed intermediate as a
        [`MomentumBlockTensor`][qten.MomentumBlockTensor] whose leading
        [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace]
        axis stores ordered pairs `(t @ k, k)` or `(k, t @ k)`,
        respectively. In those one-sided modes, the transformed momentum labels
        are carried on the pair axis rather than collapsed back to a plain
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace].

    Raises
    ------
    ValueError
        If `tensor` is not rank 3 with a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis and
        two [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] axes.
        Also raised if a selected transformed Hilbert-space side is not closed
        under the action of `t`, or if the momentum action of `t` is not
        one-to-one on the input momentum space.
    TypeError
        If the tensor dims do not have the required
        `MomentumSpace/HilbertSpace/HilbertSpace` structure.
    """
    if opt not in ("left", "right", "both"):
        raise ValueError(f"opt must be 'left', 'right', or 'both', got {opt!r}.")
    if opt == "left":
        return cast(Tensor, get_band_transform(t, tensor, side="left") @ tensor)
    if opt == "right":
        right_transform = get_band_transform(t, tensor, side="right")
        return cast(Tensor, tensor @ right_transform.h(-2, -1))

    left_transform = get_band_transform(t, tensor, side="left")
    right_transform = get_band_transform(t, tensor, side="right")
    return cast(Tensor, left_transform @ tensor @ right_transform.h(-2, -1))


def bandfold(
    transform: BasisTransform,
    tensor: Tensor,
    opt: Literal["left", "right", "both"] = "both",
) -> Tensor:
    r"""
    Fold a momentum-resolved band tensor into the Brillouin zone of a
    transformed lattice basis.

    The input tensor is expected to have dimensions
    `(MomentumSpace, HilbertSpace, HilbertSpace)`. The basis transformation is
    applied to the direct lattice underlying the
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis, which
    produces a new Brillouin zone and a corresponding momentum remapping. One
    or both [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] legs
    are enlarged to match the transformed unit cell, Fourier-space changes of
    basis are applied, and the momentum sectors are then gathered into the new
    momentum grid. If multiple basis states share the same site offset in the
    sampled Hilbert space, folding preserves all of them at the corresponding
    folded-cell site.

    Mathematical action
    -------------------
    A forward basis transform coarsens the direct lattice basis, so the
    reciprocal Brillouin zone shrinks and multiple old momenta fold onto one
    new momentum sector. If \(F_{\mathrm{left}}(k)\) and
    \(F_{\mathrm{right}}(k)\) are the Fourier-based change-of-basis maps on the
    selected tensor legs, then, after routed contributions are collapsed onto
    the folded momentum grid, the folded block is one of:

    `opt="left"`:
    \(H_{\mathrm{fold}}(k') \mathrel{+}= F_{\mathrm{left}}(k)^\dagger H(k)\)

    `opt="right"`:
    \(H_{\mathrm{fold}}(k') \mathrel{+}= H(k) F_{\mathrm{right}}(k)\)

    `opt="both"`:
    \(H_{\mathrm{fold}}(k') \mathrel{+}= F_{\mathrm{left}}(k)^\dagger H(k) F_{\mathrm{right}}(k)\)

    with \(k' = \mathrm{fold}(k)\).

    Parameters
    ----------
    transform : BasisTransform
        Basis change applied to the direct lattice associated with the momentum
        axis.
    tensor : Tensor
        Rank-3 tensor with dimensions
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    opt : Literal["left", "right", "both"], optional
        Which matrix legs to fold. `"left"` folds only the left leg,
        `"right"` folds only the right leg, and `"both"` folds both legs. The
        default is `"both"`.

    Returns
    -------
    Tensor
        If `opt="both"`, returns the folded tensor on the transformed
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] grid with
        both Hilbert-space legs expressed in the folded-cell basis.
        If `opt="left"` or `opt="right"`, returns the corresponding one-sided
        routed intermediate as a
        [`MomentumBlockTensor`][qten.MomentumBlockTensor] whose leading
        [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace]
        axis stores ordered pairs `(k_fold, k)` or `(k, k_fold)`,
        respectively. In those one-sided modes, accumulation onto the folded
        momentum grid is deferred until the complementary side is composed.

    Raises
    ------
    ValueError
        If the tensor is not rank-3, if the momentum space is empty, or if the
        momentum axis does not belong to a single Brillouin zone. Also raised
        if the sampled Hilbert basis on a selected side has no states at a
        required unit-cell offset during the folding construction.
    TypeError
        If the momentum axis is not a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace], if its
        underlying space is not a
        [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice], or
        if the selected Hilbert-space leg is not a
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].
    """
    if opt not in ("left", "right", "both"):
        raise ValueError(f"opt must be 'left', 'right', or 'both', got {opt!r}.")
    if opt == "left":
        return cast(Tensor, get_band_fold(transform, tensor, side="left") @ tensor)
    if opt == "right":
        right_fold = get_band_fold(transform, tensor, side="right")
        return cast(Tensor, tensor @ right_fold.h(-2, -1))

    left_fold = get_band_fold(transform, tensor, side="left")
    right_fold = get_band_fold(transform, tensor, side="right")
    return cast(Tensor, left_fold @ tensor @ right_fold.h(-2, -1))


def bandunfold(
    inverse_transform: InverseBasisTransform,
    tensor: Tensor,
) -> Tensor:
    r"""
    Unfold a folded momentum-resolved band tensor using an inverse basis transform.

    The input is expected to have dimensions `(MomentumSpace, HilbertSpace,
    HilbertSpace)` where the
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis lives on a
    transformed (folded) Brillouin zone. The inverse transform maps that folded
    lattice back to the primitive one and recovers dimensions
    `(K_primitive, B_primitive, B_primitive)`.

    Mathematical action
    -------------------
    Unfolding routes each primitive momentum \(k\) to its parent folded
    momentum \(\bar{k}\), gathers \(H_{\mathrm{fold}}(\bar{k})\), and then
    projects it back to the primitive-cell basis with a Fourier map \(F(k)\):
    \(H_{\mathrm{unfold}}(k)
    = F(k)\,H_{\mathrm{fold}}(\bar{k})\,F(k)^\dagger\). In code, the parent-sector lookup is `tensor.data[k_indices.data]`, and the
    final basis projection is `f @ gathered @ f.h(-2, -1)`.

    Parameters
    ----------
    inverse_transform : InverseBasisTransform
        Inverse basis transform that maps the folded direct lattice back to the
        primitive lattice.
    tensor : Tensor
        Rank-3 folded band tensor with dimensions
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.

    Returns
    -------
    Tensor
        Unfolded tensor on the primitive Brillouin-zone
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] grid with
        primitive [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
        matrix axes.

    Raises
    ------
    TypeError
        If `inverse_transform` is not an
        [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform],
        if the tensor axes do not have the required symbolic space types, or if
        the momentum axis is not backed by a
        [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice].
    ValueError
        If `tensor` is not rank 3, if the momentum space is empty, or if the
        momentum axis mixes incompatible reciprocal lattices.
    """
    if not isinstance(inverse_transform, InverseBasisTransform):
        raise TypeError(
            "bandunfold requires InverseBasisTransform, "
            f"but got {type(inverse_transform)}"
        )
    if tensor.rank() != 3:
        raise ValueError(
            f"Input tensor must be of rank 3, but has rank {tensor.rank()}"
        )
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError(
            "The first dimension of the tensor must be a MomentumSpace, "
            f"but is of type {type(tensor.dims[0])}"
        )
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise TypeError(
            "The second dimension of the tensor must be a HilbertSpace, "
            f"but is of type {type(tensor.dims[1])}"
        )
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise TypeError(
            "The third dimension of the tensor must be a HilbertSpace, "
            f"but is of type {type(tensor.dims[2])}"
        )

    k_space = cast(MomentumSpace, tensor.dims[0])
    if not k_space.elements():
        raise ValueError("MomentumSpace is empty")
    lattice_set = set(map(lambda k: k.space, k_space))
    if len(lattice_set) != 1:
        raise ValueError("Invalid BZ")
    folded_reciprocal_lattice = lattice_set.pop()
    if not isinstance(folded_reciprocal_lattice, ReciprocalLattice):
        raise TypeError(
            "Space of momentum should be ReciprocalLattice, but got "
            f"{type(folded_reciprocal_lattice)}"
        )
    folded_reciprocal_lattice = cast(ReciprocalLattice, folded_reciprocal_lattice)
    folded_lattice = folded_reciprocal_lattice.dual

    primitive_lattice = cast(Lattice, inverse_transform(folded_lattice))

    folded_hilbert = cast(HilbertSpace, tensor.dims[2])

    primitive_reciprocal_lattice = primitive_lattice.dual
    primitive_k_space = brillouin_zone(primitive_reciprocal_lattice)

    rebased_states = []
    for psi in folded_hilbert:
        u1_psi = cast(U1Basis, psi)
        rebased_states.append(
            u1_psi.replace(u1_psi.irrep_of(Offset).rebase(primitive_lattice))
        )
    rebased_hilbert = HilbertSpace.new(rebased_states)

    primitive_states: "OrderedDict[U1Basis, int]" = OrderedDict()
    for psi in rebased_states:
        primitive_state = psi.replace(psi.irrep_of(Offset).fractional())
        if primitive_state not in primitive_states:
            primitive_states[primitive_state] = len(primitive_states)
    primitive_hilbert = HilbertSpace(structure=primitive_states)

    # Route each primitive-k sector to its folded-k parent.
    precision = get_precision_config()
    primitive_basis_np = np.array(
        primitive_reciprocal_lattice.basis.evalf(), dtype=precision.np_float
    )
    folded_basis_np = np.array(
        folded_reciprocal_lattice.basis.evalf(), dtype=precision.np_float
    )
    M_rebase = np.linalg.solve(folded_basis_np, primitive_basis_np)
    k_indices = _momentum_match_indices(
        primitive_k_space, k_space, M_rebase, device=tensor.device
    )

    gathered = Tensor(
        data=tensor.data[k_indices.data],
        dims=(primitive_k_space, tensor.dims[1], tensor.dims[2]),
    )
    for dim in (1, 2):
        if gathered.dims[dim] == folded_hilbert:
            gathered = gathered.replace_dim(dim, rebased_hilbert)

    f = fourier_transform(
        primitive_k_space, primitive_hilbert, rebased_hilbert, device=tensor.device
    )
    vratio = np.sqrt(rebased_hilbert.dim / primitive_hilbert.dim)
    f = f / vratio
    unfolded = f @ gathered @ f.h(-2, -1)
    return unfolded


def bandcounts(tensor: Tensor) -> Tensor:
    r"""
    Count nonzero columns in each momentum-sector matrix.

    The input tensor is expected to have dimensions
    `(MomentumSpace, HilbertSpace, StateSpace)`. For each momentum sector, the
    trailing two axes are treated as a matrix with rows labelled by the
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] axis and
    columns labelled by the trailing
    [`StateSpace`][qten.symbolics.state_space.StateSpace] axis. A column is
    counted if any entry in that column is nonzero.

    Parameters
    ----------
    tensor : Tensor
        Rank-3 tensor with dimensions `(MomentumSpace, HilbertSpace,
        StateSpace)`.

    Returns
    -------
    Tensor
        Integer-valued tensor with dimensions `(MomentumSpace,)` whose entries
        are the nonzero-column counts of the corresponding momentum blocks.

    Raises
    ------
    ValueError
        If `tensor` is not rank 3.
    TypeError
        If the tensor axes are not `MomentumSpace`, `HilbertSpace`, and
        `StateSpace`, respectively.
    """
    if tensor.rank() != 3:
        raise ValueError(
            f"Input tensor must be of rank 3, but has rank {tensor.rank()}"
        )
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError("The first dimension of the tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise TypeError("The second dimension of the tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], StateSpace):
        raise TypeError("The third dimension of the tensor must be a StateSpace.")

    kspace = cast(MomentumSpace, tensor.dims[0])
    nonzero_columns = torch.any(tensor.data != 0, dim=1)
    counts = nonzero_columns.sum(dim=1, dtype=torch.int64)
    return Tensor(data=counts, dims=(kspace,))


def bandfillings(tensor: Tensor, frac: float) -> Tensor:
    r"""
    Return eigenvectors for occupied bands up to a filling fraction.

    The input tensor is expected to have dimensions
    `(MomentumSpace, HilbertSpace, HilbertSpace)`, where the
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis indexes
    momentum sectors and the two
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] axes form the
    Hamiltonian matrix at each momentum. The tensor is diagonalized at each
    momentum, then eigenvectors with energies below the global filling
    threshold are packed into an output
    [`IndexSpace`][qten.symbolics.state_space.IndexSpace].

    Mathematical convention
    -----------------------
    Each momentum block is diagonalized as \(H(k) V(k) = V(k) E(k)\), and the eigenvectors whose energies fall below the global filling threshold
    are retained. If `frac = f`, the target number of occupied states is

    \(N_{\mathrm{occ}} = \left\lfloor f\,N_k\,N_b \right\rfloor\), where \(N_k\) is the number of momentum sectors and \(N_b\) is the number
    of bands per sector. Degenerate states at the threshold are included
    together.

    Degenerate threshold behavior
    -----------------------------
    If one state in a degenerate set is filled, all states in that set are
    filled. The output index dimension is therefore the maximum number of filled
    states over all momentum sectors, and sectors with fewer filled states are
    padded with zeros.

    Parameters
    ----------
    tensor : Tensor
        Band-resolved tensor with dimensions
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    frac : float
        Filling fraction in the inclusive range `[0, 1]`.

    Returns
    -------
    Tensor
        Eigenvector tensor with dimensions `(MomentumSpace, HilbertSpace,
        IndexSpace)`. For each momentum sector, columns along `IndexSpace`
        contain the eigenvectors selected as filled. The `IndexSpace` size is
        the largest filled count among all momentum sectors; sectors with fewer
        filled bands are padded with zero columns.

    Raises
    ------
    TypeError
        If the tensor axes are not `MomentumSpace`, `HilbertSpace`, and
        `HilbertSpace`, respectively.
    ValueError
        If `tensor` is not rank 3. Also raised if `frac` is outside the
        inclusive range `[0, 1]`.
    """
    if tensor.rank() != 3:
        raise ValueError(
            f"Input tensor must be of rank 3, but has rank {tensor.rank()}"
        )
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError("The first dimension of the tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise TypeError("The second dimension of the tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise TypeError("The third dimension of the tensor must be a HilbertSpace.")
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"Filling fraction must be between 0 and 1, got {frac}")

    kspace = cast(MomentumSpace, tensor.dims[0])
    band_space = cast(HilbertSpace, tensor.dims[1])
    eigvals, eigvecs = eigh(tensor)

    nk, nbands = eigvals.data.shape
    total_states = nk * nbands
    target_fill = int(np.floor(frac * total_states + 1e-12))

    if target_fill <= 0:
        return Tensor(
            data=eigvecs.data[..., :0],
            dims=(kspace, band_space, IndexSpace.linear(0)),
        )
    if target_fill >= total_states:
        return Tensor(
            data=eigvecs.data,
            dims=(kspace, band_space, IndexSpace.linear(nbands)),
        )

    flat_vals = eigvals.data.reshape(-1)
    threshold = torch.kthvalue(flat_vals, target_fill).values
    eps = torch.finfo(eigvals.data.dtype).eps
    tol = (abs(threshold).clamp_min(1.0) * eps * max(nbands, 1) * 8).to(
        eigvals.data.dtype
    )
    filled = eigvals.data <= (threshold + tol)

    counts = filled.sum(dim=1)
    max_fill = int(counts.max().item())
    out_dim = IndexSpace.linear(max_fill)

    order = torch.argsort(filled.to(torch.int8), dim=1, descending=True, stable=True)
    packed = torch.gather(
        eigvecs.data,
        2,
        order[:, None, :].expand(-1, eigvecs.data.shape[1], -1),
    )[..., :max_fill]

    valid = (
        torch.arange(max_fill, device=counts.device)[None, :] < counts[:, None]
    ).to(packed.dtype)
    packed = packed * valid[:, None, :]

    return Tensor(data=packed, dims=(kspace, band_space, out_dim))


def _wannier_reciprocal_lattice(k_space: MomentumSpace) -> ReciprocalLattice:
    if not k_space.elements():
        raise ValueError("MomentumSpace is empty")

    lattice_set = {k.space for k in k_space}
    if len(lattice_set) != 1:
        raise ValueError("Invalid BZ")

    reciprocal_lattice = lattice_set.pop()
    if not isinstance(reciprocal_lattice, ReciprocalLattice):
        raise TypeError(
            "Space of momentum should be ReciprocalLattice, but got "
            f"{type(reciprocal_lattice)}"
        )
    return cast(ReciprocalLattice, reciprocal_lattice)


def _infer_wannier_bridge(
    eigenvectors: Tensor, seed_col_space: HilbertSpace
) -> MomentumBlockTensor | None:
    try:
        old_k_space = cast(MomentumSpace, eigenvectors.dims[0])
        reciprocal_lattice = _wannier_reciprocal_lattice(old_k_space)
        direct_lattice = reciprocal_lattice.dual

        rebased_seed_space = cast(
            HilbertSpace, rebase_opr(direct_lattice.affine) @ seed_col_space
        )
        fractional_seed_space = cast(
            HilbertSpace, fractional_opr() @ rebased_seed_space
        )
    except Exception:
        return None

    if not fractional_seed_space:
        return None

    unique_offsets: "OrderedDict[tuple[sy.Expr, ...], ImmutableDenseMatrix]" = (
        OrderedDict()
    )
    for psi in fractional_seed_space:
        offset = cast(U1Basis, psi).irrep_of(Offset).fractional()
        key = tuple(offset.rep)
        if key not in unique_offsets:
            unique_offsets[key] = offset.rep

    new_lattice = Lattice(
        basis=direct_lattice.basis,
        boundaries=direct_lattice.boundaries,
        unit_cell=OrderedDict(
            (f"r{i}", rep) for i, rep in enumerate(unique_offsets.values())
        ),
    )

    wannier_hilbert = cast(
        HilbertSpace, rebase_opr(new_lattice) @ fractional_seed_space
    )

    new_k_space = brillouin_zone(new_lattice.dual)
    # Build the Bloch matching labels in the same affine space as the region
    # labels so Fourier matching does not depend on Offset ambient-space
    # equality. The output is relabeled back onto the inferred lattice below.
    affine_wannier_hilbert = fractional_seed_space
    f = fourier_transform(
        new_k_space,
        affine_wannier_hilbert,
        rebased_seed_space,
        device=eigenvectors.device,
    )
    fh = f.h(-2, -1).replace_dim(1, seed_col_space).replace_dim(2, wannier_hilbert)

    new_by_rep = {
        tuple(cast(Momentum, new_k).rep): cast(Momentum, new_k)
        for new_k in new_k_space.elements()
    }
    gather_indices = []
    pair_structure: "OrderedDict[tuple[Momentum, Momentum], int]" = OrderedDict()
    for i, old_k in enumerate(old_k_space.elements()):
        new_k = new_by_rep.get(tuple(old_k.rep))
        if new_k is None:
            return None
        gather_indices.append(new_k_space.structure[new_k])
        pair_structure[(old_k, new_k)] = i

    gathered = fh.data[
        torch.tensor(gather_indices, dtype=torch.long, device=fh.data.device)
    ]
    return MomentumBlockTensor(
        data=gathered,
        dims=(
            MomentumBlockSpace(structure=pair_structure),
            seed_col_space,
            wannier_hilbert,
        ),
    )


def svd_projection(
    target: Tensor[Any],
    source: Tensor[Any],
    svd_threshold: float = 1e-1,
    infer_lattice: bool = False,
) -> Tensor[Any]:
    r"""
    Align target states to a source-defined gauge via sectorwise SVD.

    This function computes the sectorwise overlap between target and source
    states, extracts the polar/unitary factor of that overlap via SVD, and
    rotates the target columns into the source-selected gauge.

    Mathematical convention
    -----------------------
    For each momentum sector \(k\), collect the target columns into a matrix
    \(T(k)\) and the source columns into a matrix \(S(k)\). If the target
    Hilbert dimension is \(N\), the target rank is \(r_t\), and the source
    rank is \(r_s\), then
    \(T(k) \in \mathbb{C}^{N \times r_t}\) and
    \(S(k) \in \mathbb{C}^{N \times r_s}\).

    The method proceeds sector by sector:

    1. Form the overlap matrix
       \(M(k) = T(k)^\dagger S(k)\), so
       \(M(k) \in \mathbb{C}^{r_t \times r_s}\).

    2. Compute the singular value decomposition
       \(M(k) = U(k)\,\Sigma(k)\,V(k)^\dagger\).

    3. Discard the singular values and keep only the unitary/polar factor
       \(Q(k) = U(k)\,V(k)^\dagger\).

    4. Rotate the target states by that factor:
       \(T_{\mathrm{proj}}(k) = T(k)\,Q(k)\).

    This output has the same row space as the target states, but its column
    gauge is chosen to optimally match the source states in the orthogonal
    Procrustes sense. Equivalently, \(Q(k)\) solves
    \(\min_Q \|T(k)Q - S(k)\|_F\) over partial isometries \(Q\) of the form
    \(Q = U V^\dagger\) induced by the SVD of \(T(k)^\dagger S(k)\).

    When \(r_t \neq r_s\), the overlap \(M(k)\) is rectangular, so the method
    still makes sense: it returns the best SVD-induced alignment from the
    target column space toward the source column space without requiring equal
    rank.

    If either `target` or `source` uses zero-padded columns to represent
    an inconsistent number of states across the Brillouin zone, those padded
    columns are ignored on a per-momentum basis when forming the SVD. The
    projection therefore acts only on the intersection of nonzero target
    columns and nonzero source columns at each momentum sector.

    In the default branch, or whenever the source metadata is insufficient to
    infer a lattice-backed column space, the result is a plain rank-3
    [`Tensor`][qten.linalg.tensors.Tensor] with the input
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis preserved.

    If `infer_lattice=True` and the source column space is a
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] carrying
    [`Offset`][qten.geometries.spatials.Offset] labels, the function tries to
    build a lattice description directly from those labels.

    A simple example is:

    - suppose the source-side basis contains states such as
      \(|r_1\rangle \otimes |\alpha\rangle\),
      \(|r_2\rangle \otimes |\beta\rangle\),
      \(|r_1\rangle \otimes |\gamma\rangle\);
    - then the new unit cell is built from the distinct site positions
      \(r_1, r_2\) in the order they first appear;
    - the extra labels \(|\alpha\rangle\), \(|\beta\rangle\),
      \(|\gamma\rangle\) are kept, but they are now understood as living on
      that newly built unit cell.

    More concretely, the construction:

    - rebases the source offsets onto the direct lattice of the input momentum
      grid;
    - converts those offsets to fractional coordinates;
    - uses the distinct fractional positions as the sites of a new unit cell;
    - keeps the same overall lattice basis and boundary conditions, so only the
      unit-cell contents are being rebuilt.

    In that branch, the return value becomes a
    [`MomentumBlockTensor`][qten.MomentumBlockTensor]. Its leading
    [`MomentumBlockSpace`][qten.symbolics.state_space.MomentumBlockSpace]
    stores pairs `(k_old, k_new)`, where `k_old` is the original momentum
    sector from the input tensor and `k_new` is the momentum sector in the
    newly created reciprocal lattice with the same fractional momentum
    coordinate.

    The last two tensor legs also become more specific:

    - the middle leg stays in the original target/band Hilbert space;
    - the last leg becomes the Bloch space built from the newly created
      lattice, so its basis states now refer to the newly constructed unit-cell
      sites rather than the original source labels.

    If this lattice inference cannot be carried out, the function simply falls
    back to the plain rank-3 projected tensor.

    Parameters
    ----------
    target : Tensor
        Target states with dims
        `(MomentumSpace, HilbertSpace, IndexSpace)`.
    source : Tensor
        Source states with dims `(MomentumSpace, HilbertSpace, D)`, where `D`
        may
        be an [`IndexSpace`][qten.symbolics.state_space.IndexSpace] or a
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].
    svd_threshold : float, optional
        Warn if the minimum singular value of the overlap drops below this
        threshold, which signals linearly dependent source states or poor projection
        onto the target subspace after zero-padded columns have been ignored.
    infer_lattice : bool, optional
        If `True`, try to build a new unit cell from the source-side offset
        labels by taking their distinct fractional positions. When this
        succeeds, the output is no longer a plain `(k, band, column)` tensor:
        it becomes a
        [`MomentumBlockTensor`][qten.MomentumBlockTensor] whose momentum axis
        records how each original momentum sector matches a momentum sector in
        the reciprocal lattice of that new unit cell, and whose final
        Hilbert-space leg is rebuilt on that lattice. If inference fails, the
        function falls back to the plain-tensor result.

    Returns
    -------
    Tensor
        Projected states. The fallback result has dims
        `(MomentumSpace, HilbertSpace, D)`. The lattice-aware result is a
        [`MomentumBlockTensor`][qten.MomentumBlockTensor] with dims
        `(MomentumBlockSpace(k_old, k_new), HilbertSpace, InferredHilbertSpace)`,
        where `k_old` is the original momentum label, `k_new` is the momentum
        label in the reciprocal lattice built from the new unit cell, and
        `InferredHilbertSpace` is the source-side Bloch space rewritten on that
        lattice.

    Raises
    ------
    ValueError
        If either input tensor is not rank 3.
    TypeError
        If either input tensor does not have
        `MomentumSpace/HilbertSpace/...` leading dimensions.
    """
    if target.rank() != 3 or source.rank() != 3:
        raise ValueError("Both target and source must be rank-3 Tensors.")
    if not isinstance(target.dims[0], MomentumSpace):
        raise TypeError("The first dimension of target must be a MomentumSpace.")
    if not isinstance(source.dims[0], MomentumSpace):
        raise TypeError("The first dimension of source must be a MomentumSpace.")
    if not isinstance(target.dims[1], HilbertSpace):
        raise TypeError("The second dimension of target must be a HilbertSpace.")
    if not isinstance(source.dims[1], HilbertSpace):
        raise TypeError("The second dimension of source must be a HilbertSpace.")

    eps = torch.finfo(target.data.real.dtype).eps

    def _valid_column_mask(data: torch.Tensor) -> torch.Tensor:
        norms = (data.abs() ** 2).sum(dim=1)
        if not norms.numel():
            return torch.zeros_like(norms, dtype=torch.bool)
        scale = norms.amax(dim=1, keepdim=True).clamp_min(1.0)
        # Zero-padded columns are expected to have vanishing norm compared to
        # genuine state columns. Use a noticeably looser threshold than machine
        # epsilon so tiny numerical leakage does not resurrect padded bands.
        tol = scale * (eps**0.5) * max(data.shape[1], 1) * 8
        return norms > tol

    valid_target_mask = _valid_column_mask(target.data)
    valid_source_mask = _valid_column_mask(source.data)

    def _pack_active_columns(
        data: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = torch.argsort(mask.to(torch.int8), dim=1, descending=True, stable=True)
        packed = data.gather(2, order[:, None, :].expand(-1, data.shape[1], -1))
        packed_mask = mask.gather(1, order)
        packed = packed * packed_mask[:, None, :].to(dtype=data.dtype)
        return packed, packed_mask.sum(dim=1), order

    packed_target, target_counts, _ = _pack_active_columns(
        target.data, valid_target_mask
    )
    packed_source, source_counts, source_order = _pack_active_columns(
        source.data, valid_source_mask
    )

    projected_packed = target.data.new_zeros(
        target.data.shape[0],
        target.data.shape[1],
        source.data.shape[2],
    )
    min_svd_val = float("inf")
    count_pairs = torch.stack((target_counts, source_counts), dim=1)

    for target_count, source_count in torch.unique(count_pairs, dim=0):
        n_target = int(target_count.item())
        n_source = int(source_count.item())
        if n_target == 0 or n_source == 0:
            continue

        pair_mask = (target_counts == target_count) & (source_counts == source_count)
        local_target = packed_target[pair_mask, :, :n_target]
        local_source = packed_source[pair_mask, :, :n_source]
        overlap = local_target.conj().transpose(1, 2) @ local_source
        u_data, s_data, vh_data = torch.linalg.svd(overlap, full_matrices=False)
        if s_data.numel():
            min_svd_val = min(min_svd_val, float(s_data.min().item()))

        # Match the Julia wannierprojection contract: ignore padded zero
        # columns, warn on small singular values, and always build the active
        # block from the full thin polar factor of the overlap. This preserves
        # the active source column count while giving an effective support of
        # min(rank(target), rank(source)) sectorwise.
        projected_packed[pair_mask, :, :n_source] = local_target @ (u_data @ vh_data)

    projected_data = target.data.new_zeros(
        target.data.shape[0],
        target.data.shape[1],
        source.data.shape[2],
    )
    projected_data.scatter_(
        2,
        source_order[:, None, :].expand(-1, projected_data.shape[1], -1),
        projected_packed,
    )

    if min_svd_val < svd_threshold:
        warnings.warn(
            f"Precarious SVD projection with minimum singular value of {min_svd_val:.4g}",
            UserWarning,
            stacklevel=2,
        )

    projected = Tensor(
        data=projected_data,
        dims=(target.dims[0], target.dims[1], source.dims[2]),
    )

    if not infer_lattice or not isinstance(source.dims[2], HilbertSpace):
        return projected

    bridge = _infer_wannier_bridge(projected, cast(HilbertSpace, source.dims[2]))
    if bridge is None:
        return projected

    return cast(Tensor[Any], projected @ bridge)


def bandselect(
    tensor: Tensor,
    **kwargs: Dict[
        str, Union[slice, Tuple[int, ...], Tuple[float, float], Callable[[float], bool]]
    ],
) -> Dict[str, Tensor]:
    r"""
    Select specific bands from a band-resolved [`Tensor`][qten.linalg.tensors.Tensor] based on criteria provided in `kwargs`.

    The input [`Tensor`][qten.linalg.tensors.Tensor] is diagonalized at each
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] sector. Each
    keyword argument defines one named selection criterion, and the returned
    dictionary maps each name to a tensor containing the matching eigenvectors.
    Outputs have dimensions `(MomentumSpace, HilbertSpace, IndexSpace)`, where
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] labels the band
    basis and [`IndexSpace`][qten.symbolics.state_space.IndexSpace] labels the
    selected states for each criterion.

    Mathematical convention
    -----------------------
    For each momentum sector, \(H(k) v_n(k) = \epsilon_n(k) v_n(k)\), and each criterion selects a subset of band labels \(n\). The returned
    tensor packs the matching eigenvectors \(v_n(k)\) into an
    [`IndexSpace`][qten.symbolics.state_space.IndexSpace], padding sectors with
    fewer matches by zero columns.

    Supported criteria
    ------------------
    - `slice`: select bands by sorted energy index, such as `slice(0, 2)` for
      the two lowest-energy bands.
    - `Tuple[int, ...]`: select explicit sorted band indices, such as `(0, 2)`
      for the lowest and third-lowest bands.
    - `Tuple[float, float]`: select an inclusive energy range.
    - `Callable[[float], bool]`: select energies for which the callable returns
      `True`.

    If a criterion matches no bands in all momentum sectors, the corresponding
    output tensor has an `IndexSpace` of dimension zero.

    Parameters
    ----------
    tensor : Tensor
        Band-resolved tensor with dimensions
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    kwargs : Dict[str, Union[slice, Tuple[int, ...], Tuple[float, float], Callable[[float], bool]]]
        Named band-selection criteria.

    Returns
    -------
    Dict[str, Tensor]
        Mapping from criterion name to selected eigenvector tensor with
        dimensions `(MomentumSpace, HilbertSpace, IndexSpace)`.

    Raises
    ------
    TypeError
        If the tensor axes are not `(MomentumSpace, HilbertSpace,
        HilbertSpace)`, or if a criterion has an unsupported type.
    ValueError
        If `tensor` is not rank 3.
    IndexError
        If an explicit integer band index is outside the available band range.
    """
    if tensor.rank() != 3:
        raise ValueError(
            f"Input tensor must be of rank 3, but has rank {tensor.rank()}"
        )
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError("The first dimension of the tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise TypeError("The second dimension of the tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise TypeError("The third dimension of the tensor must be a HilbertSpace.")

    kspace = cast(MomentumSpace, tensor.dims[0])
    band_space = cast(HilbertSpace, tensor.dims[1])
    eigvals, eigvecs = eigh(tensor)
    values = eigvals.data
    vectors = eigvecs.data

    nk, nbands = values.shape
    band_indices = torch.arange(nbands, device=values.device)

    def pack(mask: torch.Tensor) -> Tensor:
        counts = mask.sum(dim=1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        out_dim = IndexSpace.linear(max_count)
        if max_count == 0:
            return Tensor(data=vectors[..., :0], dims=(kspace, band_space, out_dim))

        order = torch.argsort(mask.to(torch.int8), dim=1, descending=True, stable=True)
        packed = torch.gather(
            vectors,
            2,
            order[:, None, :].expand(-1, vectors.shape[1], -1),
        )[..., :max_count]
        valid = (
            torch.arange(max_count, device=counts.device)[None, :] < counts[:, None]
        ).to(packed.dtype)
        packed = packed * valid[:, None, :]
        return Tensor(data=packed, dims=(kspace, band_space, out_dim))

    selected: Dict[str, Tensor] = {}
    for name, criterion in kwargs.items():
        mask: torch.Tensor
        if isinstance(criterion, slice):
            picked = band_indices[criterion]
            mask = torch.zeros((nk, nbands), dtype=torch.bool, device=values.device)
            if picked.numel():
                mask[:, picked] = True
        elif isinstance(criterion, tuple):
            if all(isinstance(x, int) and not isinstance(x, bool) for x in criterion):
                mask = torch.zeros((nk, nbands), dtype=torch.bool, device=values.device)
                if criterion:
                    raw_idx = torch.tensor(
                        criterion, dtype=torch.long, device=values.device
                    )
                    if ((raw_idx < -nbands) | (raw_idx >= nbands)).any():
                        raise IndexError(
                            f"Band index out of range in criterion {name!r}"
                        )
                    mask[:, raw_idx % nbands] = True
            elif len(criterion) == 2 and all(
                isinstance(x, (int, float, np.integer, np.floating))
                and not isinstance(x, bool)
                for x in criterion
            ):
                lo, hi = criterion
                mask = (values >= lo) & (values <= hi)
            else:
                raise TypeError(
                    f"Unsupported tuple criterion for {name!r}: {criterion!r}"
                )
        elif callable(criterion):
            mask = torch.tensor(
                [
                    [bool(criterion(v)) for v in row]
                    for row in values.detach().cpu().tolist()
                ],
                dtype=torch.bool,
                device=values.device,
            )
        else:
            raise TypeError(f"Unsupported criterion for {name!r}: {criterion!r}")

        selected[name] = pack(mask)

    return selected


def nearest_bands(
    h_k: Tensor,
    point: Union[str, Sequence[float]] = "Gamma",
    close_to: float = 0.0,
    tol: float = 1e-6,
    points: Optional[Dict[str, Sequence[float]]] = None,
) -> Tensor:
    r"""
    Project a momentum-resolved Hamiltonian onto bands selected at one k-point.

    The input `h_k` is diagonalized at a single anchor momentum \(k_0\).
    Eigenvectors whose anchor eigenvalues lie within `tol` of `close_to` are
    collected into a rectangular matrix \(V\). If the input Hilbert dimension is
    \(N\) and \(S\) bands are selected, then `V` has shape `(N, S)` and the
    returned tensor stores \(V^\dagger H(k) V\) for every momentum \(k\).

    Projection convention
    ---------------------
    At the selected anchor sector, the code computes
    `eigenvalues, eigenvectors = torch.linalg.eigh(H_anchor)`. The columns of
    `eigenvectors` with \(|\epsilon_n(k_0) - \mathrm{close\_to}| \le \mathrm{tol}\)
    form \(V\). The projected block at each momentum is
    \(H_{\mathrm{proj}}(k) = V^\dagger H(k) V\).

    In implementation terms, this projection is the einsum
    `torch.einsum("ia,kab,bj->kij", V_dag, h_k.data, V)`.

    Anchor selection
    ----------------
    - A string `point` is looked up in `points`.
    - `"Gamma"` defaults to the fractional origin when absent from `points`.
    - A coordinate sequence is interpreted directly as fractional coordinates.
    - Fractional-coordinate differences are wrapped by subtracting the nearest
      integer, so equivalent periodic coordinates select the same anchor.

    If no eigenvalue falls inside the tolerance window, the result has two
    zero-dimensional [`IndexSpace`][qten.symbolics.state_space.IndexSpace] axes
    and data shape `(len(kspace), 0, 0)`.

    Notes
    -----
    The selected subspace is fixed by the anchor momentum only. The same
    anchor eigenvector matrix \(V\) is applied to every \(H(k)\); this is a
    projection onto an anchor-defined subspace, not a separately diagonalized
    band selection at each momentum.

    Parameters
    ----------
    h_k : Tensor
        Hamiltonian tensor with dims
        ([`MomentumSpace`][qten.symbolics.state_space.MomentumSpace],
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace],
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]).
    point : str or Sequence[float], default="Gamma"
        Anchor k-point. String labels are resolved through `points`, except
        `"Gamma"` which defaults to the fractional origin.
    close_to : float, default=0.0
        Target eigenvalue for the subspace selection.
    tol : float, default=1e-6
        Half-width of the eigenvalue window around `close_to`.
    points : dict[str, Sequence[float]], optional
        Mapping from labels to fractional coordinates.

    Returns
    -------
    Tensor
        Projected Hamiltonian with dims
        ([`MomentumSpace`][qten.symbolics.state_space.MomentumSpace],
        [`IndexSpace`][qten.symbolics.state_space.IndexSpace],
        [`IndexSpace`][qten.symbolics.state_space.IndexSpace]). The last two
        axes span the selected subspace.

    Raises
    ------
    ValueError
        If `h_k` is not rank 3, if the momentum space is empty, or if the
        anchor coordinate dimension does not match the momentum-space
        dimension.
    TypeError
        If the input dimensions are not
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    KeyError
        If `point` is a string other than `"Gamma"` and is not present in
        `points`.
    """
    if h_k.rank() != 3:
        raise ValueError(f"Input tensor must be of rank 3, but has rank {h_k.rank()}")
    if not isinstance(h_k.dims[0], MomentumSpace):
        raise TypeError("The first dimension of the tensor must be a MomentumSpace.")
    if not isinstance(h_k.dims[1], HilbertSpace):
        raise TypeError("The second dimension of the tensor must be a HilbertSpace.")
    if not isinstance(h_k.dims[2], HilbertSpace):
        raise TypeError("The third dimension of the tensor must be a HilbertSpace.")

    kspace = cast(MomentumSpace, h_k.dims[0])
    k_items = list(kspace.structure.items())
    if not k_items:
        raise ValueError("MomentumSpace is empty")

    dim = k_items[0][0].space.dim
    if isinstance(point, str):
        if points is not None and point in points:
            target_frac = tuple(float(x) for x in points[point])
        elif point == "Gamma":
            target_frac = tuple(0.0 for _ in range(dim))
        else:
            raise KeyError(
                f"Point {point!r} not found in `points`; "
                "provide a `points` mapping or pass explicit fractional coordinates."
            )
    else:
        target_frac = tuple(float(x) for x in point)
    if len(target_frac) != dim:
        raise ValueError(
            f"Anchor point has {len(target_frac)} coordinates but momentum "
            f"space has dimension {dim}."
        )

    precision = get_precision_config()
    k_frac = np.array(
        [[float(k.rep[j, 0]) for j in range(dim)] for k, _ in k_items],
        dtype=precision.np_float,
    )
    k_indices = np.array([idx for _, idx in k_items], dtype=np.int64)
    target_arr = np.asarray(target_frac, dtype=precision.np_float)
    diff = k_frac - target_arr
    diff = diff - np.round(diff)
    dist = np.linalg.norm(diff, axis=1)
    best_row = int(np.argmin(dist))
    anchor_idx = int(k_indices[best_row])

    H_anchor = h_k.data[anchor_idx]
    eigenvalues, eigenvectors = torch.linalg.eigh(H_anchor)

    mask = (eigenvalues - close_to).abs() <= tol
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    n_selected = int(selected.numel())

    V = eigenvectors.index_select(-1, selected)  # (N, H)
    V_dag = V.conj().transpose(-2, -1)  # (H, N)

    projected = torch.einsum("ia,kab,bj->kij", V_dag, h_k.data, V)

    out_space = IndexSpace.linear(n_selected)
    return Tensor(data=projected, dims=(kspace, out_space, out_space))


def proj_wannierization(
    eigenvectors: Tensor[Any],
    seeds: Tensor[Any],
    svd_threshold: float = 1e-1,
    wannierize_lattice: bool = True,
) -> Tensor[Any]:
    r"""
    Perform projective Wannierization from localized real-space trial orbitals.

    This helper implements the standard two-stage projective-Wannier workflow:

    1. Fourier transform localized seed states into momentum space.
    2. At each momentum sector, rotate the target band subspace into the gauge
       that best matches those transformed seeds via the polar/SVD factor of
       the overlap matrix.

    The function is therefore a thin orchestration layer around
    [`fourier_transform()`][qten.geometries.fourier.fourier_transform] and
    [`svd_projection()`][qten.bands.svd_projection].

    Mathematical convention
    -----------------------
    Let
    \(k \in \mathcal{K}\) denote a sampled momentum,
    let \(\{|b\rangle\}_{b=1}^{N_b}\) be the Bloch basis stored on the second
    axis of `eigenvectors`, and let
    \(\{|r\rangle\}_{r=1}^{N_r}\) be the localized basis stored on the first
    axis of `seeds`.

    Suppose the seed tensor stores coefficients
    \(A_{r n}\), where \(n = 1, \dots, N_s\) labels the trial orbitals. In
    matrix form,
    \(A \in \mathbb{C}^{N_r \times N_s}\).

    The discrete Fourier-transform tensor returned by
    [`fourier_transform(k_space, bloch_space, local_seed_space)`][qten.geometries.fourier.fourier_transform]
    is a rank-3 object with entries

    \[
    F(k)_{b r} =
    \delta_{\text{internal}(b),\,\text{internal}(r)}
    \exp(-\mathrm{i}\, k \cdot r),
    \]

    where the Kronecker-style matching factor indicates that only localized and
    Bloch basis states with matching non-offset irreps are connected. In the
    repository's fractional-coordinate convention, this phase is equivalently
    \(\exp(-2\pi\mathrm{i}\,\kappa \cdot n)\).

    For each momentum sector, the real-space seeds are lifted to Bloch form as

    \[
    S(k) = F(k)\,A,
    \]

    with
    \(S(k) \in \mathbb{C}^{N_b \times N_s}\).

    The input `eigenvectors` stores the target subspace as matrices

    \[
    U(k) \in \mathbb{C}^{N_b \times N_t},
    \]

    whose columns span the band manifold to be Wannierized.

    The gauge-fixing step then forms the overlap

    \[
    M(k) = U(k)^\dagger S(k)
    \in \mathbb{C}^{N_t \times N_s},
    \]

    computes its singular value decomposition

    \[
    M(k) = X(k)\,\Sigma(k)\,Y(k)^\dagger,
    \]

    discards the singular values, and keeps only the polar/unitary factor

    \[
    Q(k) = X(k)\,Y(k)^\dagger.
    \]

    The projected Wannier-gauge states are then

    \[
    \widetilde{U}(k) = U(k)\,Q(k).
    \]

    This is exactly the orthogonal-Procrustes solution used by
    [`svd_projection()`][qten.bands.svd_projection]: it preserves the target
    column space while choosing the column gauge that best aligns the target
    states with the Fourier-transformed trial orbitals.

    Step-by-step behavior
    ---------------------
    Given `eigenvectors` and `seeds`, the code performs:

    1. Read the shared momentum grid `k_space = eigenvectors.dims[0]`.
    2. Read the Bloch Hilbert space
       `bloch_space = eigenvectors.dims[1]`.
    3. Read the localized seed row basis
       `local_seed_space = seeds.dims[0]`.
    4. Build the discrete Fourier transform tensor
       `F = fourier_transform(k_space, bloch_space, local_seed_space)`.
    5. Convert localized trial orbitals into momentum-resolved trial orbitals
       by the tensor contraction `crystal_seeds = F @ seeds`.
    6. Call
       `svd_projection(eigenvectors, crystal_seeds, svd_threshold, infer_lattice=wannierize_lattice)`.
    7. Return the projected states.

    Interpretation of the tensor legs
    ---------------------------------
    - `eigenvectors` must have dims
      `(MomentumSpace, HilbertSpace, D_target)`, where the first two axes
      represent momentum and Bloch basis, and the last axis enumerates the
      target band columns.
    - `seeds` must have dims
      `(HilbertSpace_local, D_seed)`, where the first axis is a localized
      real-space basis containing [`Offset`][qten.geometries.spatials.Offset]
      labels, and the second axis enumerates trial orbitals.
    - After Fourier transformation, `crystal_seeds` has dims
      `(MomentumSpace, HilbertSpace, D_seed)`.

    Here `D_target` and `D_seed` may be different state-space types. The most
    common case is that both are
    [`IndexSpace`][qten.symbolics.state_space.IndexSpace], but the seed-column
    space may also be a
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].

    Lattice-aware output
    --------------------
    The `wannierize_lattice` flag is forwarded to
    [`svd_projection()`][qten.bands.svd_projection], but lattice rebuilding is
    only possible when the **column space of the transformed seeds** still
    carries a [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
    with meaningful [`Offset`][qten.geometries.spatials.Offset] labels.

    Concretely:

    - if `seeds.dims[1]` is an
      [`IndexSpace`][qten.symbolics.state_space.IndexSpace], projection still
      works exactly as described above, but no lattice-backed output basis can
      be inferred, so the result remains a plain rank-3 tensor;
    - if `seeds.dims[1]` is a
      [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace], then after
      Fourier transformation the source columns retain those symbolic labels,
      and `svd_projection(..., infer_lattice=True)` may return a
      [`MomentumBlockTensor`][qten.MomentumBlockTensor] whose final Hilbert leg
      has been rebuilt on the inferred Wannier lattice.

    Numerical behavior
    ------------------
    The SVD warning threshold is interpreted exactly as in
    [`svd_projection()`][qten.bands.svd_projection]: if the minimum singular
    value of the overlap becomes smaller than `svd_threshold`, a warning is
    emitted because the trial orbitals may poorly span the target subspace or
    may be nearly linearly dependent after projection. Zero-padded columns, if
    present in the target or source, are ignored sector by sector by the
    underlying projection routine.

    Parameters
    ----------
    eigenvectors : Tensor
        Target band states with dims
        `(MomentumSpace, HilbertSpace, D_target)`.
    seeds : Tensor
        Localized trial orbitals with dims `(HilbertSpace_local, D_seed)`.
        The first axis is the localized real-space basis to be Fourier
        transformed; the second axis enumerates the trial orbitals themselves.
        `D_seed` does not have to be an
        [`IndexSpace`][qten.symbolics.state_space.IndexSpace]: it may also be a
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].
        This distinction controls whether `wannierize_lattice` can do
        anything:

        - if `D_seed` is an
          [`IndexSpace`][qten.symbolics.state_space.IndexSpace], the function
          still performs the full projective Wannierization, but the output
          remains a plain rank-3 tensor because there is no symbolic
          seed-column geometry to rebuild into a Wannier lattice;
        - if `D_seed` is a
          [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace], the
          transformed seed columns retain their symbolic offset labels, so
          `wannierize_lattice=True` may trigger lattice inference in the final
          projection step.
    svd_threshold : float, optional
        Warning threshold passed to
        [`svd_projection()`][qten.bands.svd_projection].
    wannierize_lattice : bool, optional
        Forwarded as `infer_lattice` to
        [`svd_projection()`][qten.bands.svd_projection]. This only has an
        effect when the transformed seed columns carry a
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] label.

    Returns
    -------
    Tensor
        Projected Wannier-gauge states. Usually this is a rank-3 tensor with
        dims `(MomentumSpace, HilbertSpace, D_seed)`. If lattice inference is
        enabled and succeeds, the result may instead be a
        [`MomentumBlockTensor`][qten.MomentumBlockTensor].

    Raises
    ------
    ValueError
        If `eigenvectors` is not rank 3 or `seeds` is not rank 2.
    TypeError
        If `eigenvectors.dims[0]` is not a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace], or if
        `eigenvectors.dims[1]` / `seeds.dims[0]` are not
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].
    """
    if eigenvectors.rank() != 3:
        raise ValueError("eigenvectors must be a rank-3 Tensor.")
    if seeds.rank() != 2:
        raise ValueError("seeds must be a rank-2 Tensor.")
    if not isinstance(eigenvectors.dims[0], MomentumSpace):
        raise TypeError(
            "The first dimension of the eigenvectors must be a MomentumSpace."
        )

    kspace = eigenvectors.dims[0]
    bloch_space = eigenvectors.dims[1]
    local_seed_space = seeds.dims[0]
    if not isinstance(bloch_space, HilbertSpace) or not isinstance(
        local_seed_space, HilbertSpace
    ):
        raise TypeError(
            "The second dimension of eigenvectors and first dimension "
            "of seeds must be HilbertSpace."
        )

    # Perform Fourier transform on local seeds to move them to momentum space
    # `f` has dims `(MomentumSpace, BlochHilbertSpace, LocalSeedHilbertSpace)`.
    f = fourier_transform(
        kspace, bloch_space, local_seed_space, device=eigenvectors.device
    )

    # Map the seeds to crystal momentum seeds
    # f @ local_seeds -> (MomentumSpace, HilbertSpace_out, IndexSpace)
    crystal_seeds = f @ seeds

    return svd_projection(
        eigenvectors,
        crystal_seeds,
        svd_threshold,
        infer_lattice=wannierize_lattice,
    )
