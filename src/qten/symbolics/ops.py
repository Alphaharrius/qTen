from collections import OrderedDict
from typing import Callable, Dict, Optional, Sequence, TypeVar, Union, cast
import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix
import torch

from ..geometries import Momentum, Offset, ReciprocalLattice
from ..linalg.tensors import Tensor
from . import HilbertSpace, Opr, StateSpace
from .state_space import BzPath, MomentumSpace
from ..utils.devices import Device

T = TypeVar("T")


def region_hilbert(bloch_space: HilbertSpace, region: Sequence[Offset]) -> HilbertSpace:
    """
    Expand a Bloch `HilbertSpace` across a real-space region by unit-cell subgroup.

    The input `bloch_space` is first partitioned by the fractional part of each
    basis state's `Offset` irrep, so all sub-basis states that occupy the same
    position within the unit cell stay grouped together. Each offset in
    `region` is treated as a full target-site offset that already includes its
    unit-cell position. A region site `r` therefore selects the subgroup whose
    fractional offset equals `r.fractional()`, and every state in that subgroup
    is copied with its `Offset` replaced by `r`.

    Parameters
    ----------
    `bloch_space` : `HilbertSpace`
        Source basis to replicate. Each basis element must contain an `Offset`
        irrep. Elements with the same fractional offset are treated as the
        sub-basis of one unit-cell site and are replicated together.
    `region` : `Sequence[Offset]`
        Full target offsets where the matching unit-cell sub-basis should be
        placed.

    Returns
    -------
    `HilbertSpace`
        A Hilbert space containing one copied basis state for each compatible
        pair of region offset and unit-cell subgroup, with output offsets taken
        from `region` after rebasing into the Bloch offset space when needed.

    Raises
    ------
    `ValueError`
        Propagated if a basis element in `bloch_space` does not contain an
        `Offset` irrep to inspect or replace, or if `region` contains a
        fractional offset that does not exist in `bloch_space`.

    Notes
    -----
    Grouping is done by fractional `Offset`, preserving the original basis
    order inside each subgroup. Output order follows `region`.
    """
    grouped_basis: dict[Offset, list] = {}
    bloch_offset_space = None
    for psi in bloch_space:
        offset = psi.irrep_of(Offset)
        if bloch_offset_space is None:
            bloch_offset_space = offset.space
        offset = offset.fractional()
        grouped_basis.setdefault(offset, []).append(psi)

    def iter_region_basis():
        for region_offset in region:
            if (
                bloch_offset_space is not None
                and region_offset.space != bloch_offset_space
            ):
                region_offset = region_offset.rebase(bloch_offset_space)
            group = grouped_basis.get(region_offset.fractional())
            if group is None:
                raise ValueError(
                    "region contains an offset whose fractional part is not present in bloch_space."
                )
            for psi in group:
                yield psi.replace(region_offset)

    return HilbertSpace.new(iter_region_basis())


def hilbert_opr_repr(
    opr: Opr, space: HilbertSpace, *, device: Optional[Device] = None
) -> Tensor:
    """
    Return the matrix representation of an operator on a Hilbert-space basis.

    Let `space = span{ |e_i> }` be the input `HilbertSpace` and let `opr` act
    on each basis state to produce `opr @ space = span{ opr |e_j> }`. This
    function constructs the corresponding representation matrix

    `M_{ij} = ⟨e_i | opr | e_j⟩`,

    implemented as the cross-Gram matrix between the original basis and its
    transformed image. The resulting `Tensor` therefore represents `opr` in the
    basis supplied by `space`.

    Parameters
    ----------
    `opr` : `Opr`
        Operator whose representation is to be computed.
    `space` : `HilbertSpace`
        Basis in which the operator is represented.

    Returns
    -------
    `Tensor`
        Square tensor whose entries are the matrix elements of `opr` in the
        basis `space`.

    Raises
    ------
    `ValueError`
        If `opr` does not preserve the ray structure of `space`, so no
        representation internal to the same projective Hilbert space exists.

    Notes
    -----
    The output dimensions are relabeled back onto `space`, so the returned
    tensor can be interpreted directly as an endomorphism of that Hilbert
    space.
    """
    new_space = opr @ space
    if not space.same_rays(new_space):
        raise ValueError("opr does not preserve the ray structure of space.")
    return space.cross_gram(new_space, device=device).replace_dim(1, space)


def match_indices(
    src: StateSpace[T],
    dest: StateSpace[T],
    matching_func: Callable[[T], T],
    *,
    device: Optional[Device] = None,
) -> Tensor[torch.LongTensor]:
    """
    Build destination indices for matching elements of `src` into `dest`.

    This helper is intended for indexed accumulation patterns such as
    `index_add`, where each element of `src` is matched to exactly one element
    of `dest` and multiple source elements may map to the same destination.

    Parameters
    ----------
    `src` : `StateSpace[T]`
        Source state space whose element order defines the output index order.
    `dest` : `StateSpace[T]`
        Destination state space whose integer positions are used as the
        returned indices.
    `matching_func` : `Callable[[T], T]`
        Function mapping each source element to its matching destination
        element.
    `device` : `Optional[Device]`, optional
        Device to place the returned index tensor on, by default `None` (CPU).

    Returns
    -------
    `Tensor[torch.LongTensor]`
        Rank-1 integer tensor with dims `(src,)`, where each entry is the
        destination index of the corresponding source element.

    Raises
    ------
    `ValueError`
        If any source element maps to an element that is not present in `dest`.
    """
    indices: list[int] = []
    for source_element in src:
        matched = matching_func(source_element)
        if matched not in dest.structure:
            raise ValueError(
                f"Source element {source_element} maps to {matched}, which is not present in destination."
            )
        indices.append(dest.structure[matched])

    return Tensor(
        data=cast(
            torch.LongTensor,
            torch.tensor(
                indices,
                dtype=torch.long,
                device=device.torch_device() if device is not None else None,
            ),
        ),
        dims=(src,),
    )


def interpolate_reciprocal_path(
    recip: ReciprocalLattice,
    waypoints: Sequence[Union[tuple[float, ...], str]],
    n_points: int = 100,
    labels: Optional[Sequence[str]] = None,
    points: Optional[Dict[str, tuple[float, ...]]] = None,
) -> BzPath:
    """Build a dense reciprocal-space sample along a piecewise-linear path."""
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required to define a path.")

    _points: Dict[str, tuple[float, ...]] = points or {}

    resolved_wp: list[tuple[float, ...]] = []
    auto_labels: list[str] = []
    for i, wp in enumerate(waypoints):
        if isinstance(wp, str):
            if wp not in _points:
                raise ValueError(
                    f"Waypoint {i} is the name '{wp}' but it was not found in "
                    f"the points dictionary. Available names: "
                    f"{sorted(_points.keys()) if _points else '(empty)'}."
                )
            resolved_wp.append(_points[wp])
            auto_labels.append(wp)
        else:
            resolved_wp.append(tuple(wp))
            auto_labels.append(str(tuple(wp)))

    dim = recip.dim
    for i, wp in enumerate(resolved_wp):
        if len(wp) != dim:
            raise ValueError(f"Waypoint {i} has {len(wp)} components, expected {dim}.")
    if n_points < len(resolved_wp):
        raise ValueError(
            f"n_points ({n_points}) must be >= number of waypoints ({len(resolved_wp)})."
        )

    basis_mat = np.array(recip.basis.evalf(), dtype=float)
    wp_frac = np.array(resolved_wp, dtype=float)
    wp_cart = wp_frac @ basis_mat.T

    seg_lengths = np.array(
        [
            np.linalg.norm(wp_cart[i + 1] - wp_cart[i])
            for i in range(len(resolved_wp) - 1)
        ]
    )
    total_length = seg_lengths.sum()
    n_segments = len(resolved_wp) - 1

    if total_length < 1e-15:
        raise ValueError("All waypoints are identical; path has zero length.")

    remaining = n_points - n_segments - 1
    interior_per_seg = np.zeros(n_segments, dtype=int)
    if remaining > 0:
        ideal = (seg_lengths / total_length) * remaining
        interior_per_seg = np.floor(ideal).astype(int)
        deficit = remaining - interior_per_seg.sum()
        fracs = ideal - interior_per_seg
        for idx in np.argsort(-fracs)[:deficit]:
            interior_per_seg[idx] += 1

    all_fracs: list[np.ndarray] = []
    waypoint_indices: list[int] = []

    for seg in range(n_segments):
        n_interior = int(interior_per_seg[seg])
        n_seg_points = n_interior + 1
        t_vals = np.linspace(0.0, 1.0, n_seg_points, endpoint=False)
        start = wp_frac[seg]
        end = wp_frac[seg + 1]
        waypoint_indices.append(len(all_fracs))
        for t in t_vals:
            all_fracs.append(start + t * (end - start))

    waypoint_indices.append(len(all_fracs))
    all_fracs.append(wp_frac[-1])

    seen: dict[Momentum, int] = {}
    unique_momenta: list[Momentum] = []
    path_order: list[int] = []

    for frac in all_fracs:
        rep = ImmutableDenseMatrix(
            [sy.Rational(f).limit_denominator(10**9) for f in frac]
        )
        k = Momentum(rep=rep, space=recip)
        if k not in seen:
            seen[k] = len(unique_momenta)
            unique_momenta.append(k)
        path_order.append(seen[k])

    structure: OrderedDict[Momentum, int] = OrderedDict(
        (k, i) for i, k in enumerate(unique_momenta)
    )
    k_space = MomentumSpace(structure=structure)

    all_cart = np.stack(all_fracs) @ basis_mat.T
    diffs = np.diff(all_cart, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    positions = np.concatenate(([0.0], np.cumsum(dists)))

    if labels is None:
        labels = tuple(auto_labels)
    else:
        if len(labels) != len(resolved_wp):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match number of waypoints ({len(resolved_wp)})."
            )
        labels = tuple(labels)

    return BzPath(
        k_space=k_space,
        labels=labels,
        waypoint_indices=tuple(waypoint_indices),
        path_order=tuple(path_order),
        path_positions=tuple(float(p) for p in positions),
    )
