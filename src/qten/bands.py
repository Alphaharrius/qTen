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

$$
H : k \mapsto H(k),
\qquad
H(k)_{ab} = \langle a | H(k) | b \rangle.
$$

In code this is stored as a rank-3 [`Tensor`][qten.linalg.tensors.Tensor] with
dims `(K, B_left, B_right)`, where `K` is a
[`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and the two
Hilbert-space axes provide the row and column basis labels for each matrix
block.

Geometry transformations act on both parts of this object:

$$
k \mapsto k',
\qquad
H(k) \mapsto U(k)\,H(k)\,U(k)^\dagger,
$$

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
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, cast

import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix

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
from .linalg.tensors import Tensor, zeros
from .precision import get_precision_config
from .symbolics import (
    BzPath,
    FuncOpr,
    HilbertSpace,
    IndexSpace,
    MomentumSpace,
    Opr,
    U1Basis,
    brillouin_zone,
    interpolate_reciprocal_path,
    restructure,
)
from .utils.devices import Device


def interpolate_path(
    recip: ReciprocalLattice,
    waypoints: Sequence[Union[Tuple[float, ...], str]],
    n_points: int = 100,
    labels: Optional[Sequence[str]] = None,
    points: Optional[Dict[str, Tuple[float, ...]]] = None,
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
    [`interpolate_reciprocal_path(recip, waypoints, n_points, labels, points)`][qten.symbolics.ops.interpolate_reciprocal_path]
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
    )


def _probe_affine(
    raw_opr: Callable[[Momentum], Momentum],
    recip_lat: ReciprocalLattice,
) -> Tuple[np.ndarray, np.ndarray, ReciprocalLattice]:
    r"""
    Probe *raw_opr* with ``d + 1`` reference momenta to extract its affine
    decomposition
    $$
    \mathrm{output\_frac} = \mathrm{input\_frac}\, M^{\mathsf{T}} + c.
    $$
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

    $$
    \kappa' = M\kappa + c \pmod{1}.
    $$

    In code, source fractional coordinates are rows, so the matrix product is
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


def bandtransform(
    t: Opr,
    tensor: Tensor,
) -> Tensor:
    r"""
    Apply a basis transform to a momentum-resolved operator tensor.

    The expected tensor shape is `(K, B_left, B_right)` where `K` is a
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and
    `B_left`, `B_right` are
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] axes. This
    function applies the operator-induced basis transform on both
    Hilbert-space legs of the band tensor.

    For each transformed side, a k-dependent matrix is built from the action of
    `t` on the Hilbert-space basis and Fourier transforms that connect Bloch and
    real-space sectors.

    Mathematical action
    -------------------
    Let \(B\) be the input Hilbert-space basis and \(tB\) the transformed basis.
    After wrapping transformed sites back to the home unit cell, the finite
    Fourier transform contributes a momentum-dependent phase. The resulting
    basis-change matrix is denoted \(U_t(k)\). The transformed band block is

    $$
    H'(t k) = U_t(k)\,H(k)\,U_t(k)^\dagger.
    $$

    In code, `left_fourier` and `right_fourier` are the two \(U_t(k)\)-style
    maps, and the products are `left_fourier @ tensor` and
    `tensor @ right_fourier.h(-2, -1)`.

    Momentum handling
    -----------------
    - The action on [`Momentum`][qten.geometries.spatials.Momentum] is treated as a relabeling/permutation of sectors.
    - The output tensor carries the transformed momentum axis
      `mapped_kspace = {t @ k | k in kspace}`.
    - Each output k-block is populated from the preimage source block before
      the Hilbert-space conjugation is applied.

    Notes
    -----
    This function accepts a general [`Opr`][qten.symbolics.hilbert_space.Opr], but not every [`Opr`][qten.symbolics.hilbert_space.Opr] is valid here.
    In practice, `t` must act coherently across the real-space and
    momentum-space labels carried by the tensor:

    - `t @ k` must be defined for each [`Momentum`][qten.geometries.spatials.Momentum] in the first tensor axis.
    - `t @ psi` must be defined for each [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] in the Hilbert-space axes,
      in particular for the [`Offset`][qten.geometries.spatials.Offset] irrep stored inside each basis state.
    - The Hilbert-space action and momentum action must be dual-compatible, so
      that the Fourier transform remains consistent after applying `t`.
    - After applying [`FuncOpr(Offset, Offset.fractional)`][qten.symbolics.hilbert_space.FuncOpr], the transformed
      Hilbert space must have the same rays as the original one; otherwise the
      transformed basis does not close on the input band space and this
      function raises `ValueError`.

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

    Returns
    -------
    Tensor
        Transformed tensor with a transformed
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis and
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] matrix axes.

    Raises
    ------
    ValueError
        If `tensor` is not rank 3 with a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis and
        two [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] axes.
        Also raised if a Hilbert-space side is not closed under the action of
        `t`.
    """
    if not len(tensor.dims) == 3:
        raise ValueError("Input tensor must have exactly 3 dimensions.")
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise ValueError("First dimension of tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise ValueError("Second dimension of tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise ValueError("Third dimension of tensor must be a HilbertSpace.")

    kspace: MomentumSpace = cast(MomentumSpace, tensor.dims[0])
    transform_cache: Dict[HilbertSpace, Tensor] = {}

    mapped_kspace = _momentum_map(kspace, lambda k: cast(Momentum, t @ k))

    def build_transform(space: HilbertSpace) -> Tensor:
        cached = transform_cache.get(space)
        if cached is not None:
            return cached

        fractional = FuncOpr(Offset, Offset.fractional)
        raw_space = cast(HilbertSpace, t @ space)
        new_space = cast(HilbertSpace, fractional @ raw_space)
        # The transformation will distort the unit-cell of the Hilbert space,
        # we will use fractional to return it to the original unit-cell.
        if not space.same_rays(new_space):
            raise ValueError(
                f"Hilbert space {space} is not closed under the transform {t}!"
            )
        # `raw_space` keeps the transformed positions before wrapping them back
        # into the home cell; `new_space` is the corresponding wrapped basis.
        # Their difference is the lattice translation whose Bloch phase is
        # encoded by the Fourier transform below.
        transformed_fourier = fourier_transform(
            mapped_kspace, new_space, raw_space, device=tensor.device
        ).replace_dim(2, space)  # (K, B, B')
        # This is the home-cell basis map analogous to the Julia
        # `homefocktransform`: it relabels the wrapped transformed basis back
        # onto the original Hilbert-space labels.
        home_transform = cast(
            Tensor, space.cross_gram(new_space, device=tensor.device)
        ).replace_dim(1, new_space)
        transform = home_transform @ transformed_fourier  # (K, B, B)
        transform_cache[space] = transform
        return transform

    tensor = tensor.replace_dim(0, mapped_kspace)

    left_fourier = build_transform(cast(HilbertSpace, tensor.dims[1]))  # (K, B, B)
    left_fourier = left_fourier.replace_dim(0, mapped_kspace)  # (K, B, B)
    tensor = cast(Tensor, (left_fourier @ tensor))  # (K, B, B)

    right_fourier = build_transform(cast(HilbertSpace, tensor.dims[2]))  # (K, B, B)
    right_fourier = right_fourier.replace_dim(0, mapped_kspace)  # (K, B, B)
    tensor = cast(Tensor, (tensor @ right_fourier.h(-2, -1)))  # (K, B, B)

    return tensor


def bandfold(
    transform: BasisTransform,
    tensor: Tensor,
) -> Tensor:
    r"""
    Fold a momentum-resolved band tensor into the Brillouin zone of a
    transformed lattice basis.

    The input tensor is expected to have dimensions
    `(MomentumSpace, HilbertSpace, HilbertSpace)`. The basis transformation is
    applied to the direct lattice underlying the
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] axis, which
    produces a new Brillouin zone and a corresponding momentum remapping. One
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] leg is enlarged
    to match the transformed unit cell, a Fourier-space change of basis is
    applied, and the momentum sectors are then gathered into the new momentum
    grid.

    Mathematical action
    -------------------
    A forward basis transform coarsens the direct lattice basis, so the
    reciprocal Brillouin zone shrinks and multiple old momenta fold onto one
    new momentum sector. If \(F(k)\) is the Fourier map from the old cell basis
    into the enlarged transformed-cell basis, each block is transformed as

    $$
    H_{\mathrm{fold}}(k') \mathrel{+}=
        F(k)^\dagger H(k) F(k),
    \qquad
    k' = \mathrm{fold}(k).
    $$

    The code-level implementation uses `fh @ tensor @ f` for the block
    transform and `index_add(0, k_indices, transformed)` to accumulate old
    sectors into the folded momentum axis.

    Parameters
    ----------
    transform : BasisTransform
        Basis change applied to the direct lattice associated with the momentum
        axis.
    tensor : Tensor
        Rank-3 tensor with dimensions
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.

    Returns
    -------
    Tensor
        Folded tensor on the transformed
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] grid with
        transformed [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
        matrix axes.

    Raises
    ------
    ValueError
        If the tensor is not rank-3, if the momentum space is empty, or if the
        momentum axis does not belong to a single Brillouin zone.
    TypeError
        If the momentum axis is not a
        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace], if its
        underlying space is not a
        [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice], or
        if the selected Hilbert-space leg is not a
        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].
    """
    # 1. Parse inputs
    if not tensor.rank() == 3:
        raise ValueError(
            f"Input tensor must be of rank 3, but has rank {tensor.rank()}"
        )
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise TypeError(
            "The first dimension of the tensor must be a MomentumSpace, "
            f"but is of type {type(tensor.dims[0])}"
        )
    k_space = cast(MomentumSpace, tensor.dims[0])
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

    # 2. Apply the transformation
    scaled_lattice = transform(lattice)

    # 3. Create new transformed spaces
    scaled_reciprocal_lattice = scaled_lattice.dual
    transformed_unit_cell = tuple(
        sorted(scaled_lattice.unit_cell.values(), key=lambda x: tuple(x.rep))
    )
    # Keep a rebased copy for the current Fourier/matching logic, but return
    # the transformed offsets on the output Hilbert-space labels.
    enlarge_unit_cell = tuple(r.rebase(lattice) for r in transformed_unit_cell)

    # Follow the existing "both" branch behavior by rebuilding the right leg.
    target_space = tensor.dims[-1]
    if not isinstance(target_space, HilbertSpace):
        raise TypeError(
            f"The last dimension must be a HilbertSpace, but got {type(target_space)}"
        )
    rebased_hilbert = HilbertSpace.new(
        cast(U1Basis, target_space.lookup({Offset: r.fractional()})).replace(r)
        for r in enlarge_unit_cell
    )
    transformed_hilbert = HilbertSpace.new(
        cast(U1Basis, target_space.lookup({Offset: r_lookup.fractional()})).replace(
            r_out
        )
        for r_lookup, r_out in zip(enlarge_unit_cell, transformed_unit_cell)
    )
    # # Transform both sides
    f = fourier_transform(k_space, target_space, rebased_hilbert, device=tensor.device)
    vratio = np.sqrt(len(enlarge_unit_cell) / len(lattice.unit_cell))
    f = f / vratio
    fh = f.h(-2, -1)  # (K, B', B)
    transformed = fh @ tensor @ f  # (K, B', B')

    # k-mapping: batch-compute which new-BZ slot each old k-point folds into.
    new_k_space = brillouin_zone(scaled_reciprocal_lattice)

    precision = get_precision_config()
    old_basis_np = np.array(reciprocal_lattice.basis.evalf(), dtype=precision.np_float)
    new_basis_np = np.array(
        scaled_reciprocal_lattice.basis.evalf(), dtype=precision.np_float
    )
    M_rebase = np.linalg.solve(new_basis_np, old_basis_np)

    k_indices = _momentum_match_indices(
        k_space, new_k_space, M_rebase, device=tensor.device
    )

    transformed = (
        zeros((new_k_space, rebased_hilbert, rebased_hilbert), device=tensor.device)
        .astype(transformed.data.dtype)
        .index_add(0, k_indices, transformed)
    )
    for dim in (1, 2):
        if transformed.dims[dim] == rebased_hilbert:
            transformed = transformed.replace_dim(dim, transformed_hilbert)
    return transformed


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

    $$
    H_{\mathrm{unfold}}(k)
        = F(k)\,H_{\mathrm{fold}}(\bar{k})\,F(k)^\dagger.
    $$

    In code, the parent-sector lookup is `tensor.data[k_indices.data]`, and the
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
    Each momentum block is diagonalized as

    $$
    H(k) V(k) = V(k) E(k),
    $$

    and the eigenvectors whose energies fall below the global filling threshold
    are retained. If `frac = f`, the target number of occupied states is

    $$
    N_{\mathrm{occ}} = \left\lfloor f\,N_k\,N_b \right\rfloor,
    $$

    where \(N_k\) is the number of momentum sectors and \(N_b\) is the number
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
    For each momentum sector,

    $$
    H(k) v_n(k) = \epsilon_n(k) v_n(k),
    $$

    and each criterion selects a subset of band labels \(n\). The returned
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
