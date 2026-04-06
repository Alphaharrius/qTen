from typing import Callable, Dict, Optional, Sequence, Tuple, TypeVar, Union, cast
from collections import OrderedDict

import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix
from typing import Callable, Optional, Sequence, TypeVar, cast
import torch

from ..geometries import Momentum, Offset, ReciprocalLattice
from ..linalg.tensors import Tensor
from . import HilbertSpace, MomentumSpace, Opr, StateSpace, restructure
from ..utils.devices import Device

T = TypeVar("T")


def _probe_affine(
    raw_opr: Callable[[Momentum], Momentum],
    recip_lat: ReciprocalLattice,
) -> Tuple[np.ndarray, np.ndarray, ReciprocalLattice]:
    """
    Probe *raw_opr* with ``d + 1`` reference momenta to extract its affine
    decomposition ``output_frac = input_frac @ M.T + c``.

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


def _kspace_frac(kspace: MomentumSpace) -> np.ndarray:
    """Return the fractional coordinates of *kspace* as an ``(N, d)`` array."""
    elements = kspace.elements()
    dim = elements[0].space.dim
    return np.array(
        [[float(k.rep[j, 0]) for j in range(dim)] for k in elements],
        dtype=np.float64,
    )


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


def momentum_match_indices(
    src: MomentumSpace,
    dest: MomentumSpace,
    transform: Union[np.ndarray, Callable[[Momentum], Momentum]],
    *,
    device: Optional[Device] = None,
) -> Tensor[torch.LongTensor]:
    """
    Batch-compute destination indices for a momentum-space mapping via
    integer grid lookup.

    This is the ``MomentumSpace``-specialised counterpart of
    `match_indices`.  Instead of evaluating *transform* per element, the
    transformation is applied as a single matrix multiply over all source
    k-points, followed by fractional wrapping and grid snapping.

    Parameters
    ----------
    `src` : `MomentumSpace`
        Source momentum space.
    `dest` : `MomentumSpace`
        Destination momentum space.
    `transform` : `Union[np.ndarray, Callable[[Momentum], Momentum]]`
        Either an ``(d, d)`` numpy matrix that maps source fractional
        coordinates to destination fractional coordinates, or a callable
        (probed with ``d + 1`` reference momenta to extract the affine
        decomposition automatically).  Fractional wrapping is applied
        after the transformation.
    `device` : `Optional[Device]`
        Device for the output tensor.

    Returns
    -------
    `Tensor[torch.LongTensor]`
        Rank-1 integer tensor with dims ``(src,)``, where each entry is
        the destination index of the corresponding source k-point.

    Raises
    ------
    `ValueError`
        If a transformed source k-point does not land on a destination
        grid point.
    """
    if callable(transform):
        recip_lat = next(iter(src.structure)).space
        M, c, _ = _probe_affine(transform, recip_lat)
    else:
        M, c = transform, None

    src_frac = _kspace_frac(src)
    mapped = src_frac @ M.T
    if c is not None:
        mapped = mapped + c
    mapped_wrapped = mapped - np.floor(mapped)

    first_dest_k = next(iter(dest.structure))
    dim = first_dest_k.space.dim
    grid = np.array(first_dest_k.space.shape, dtype=np.int64)
    mapped_grid = np.rint(mapped_wrapped * grid).astype(np.int64) % grid

    lookup: Dict[Tuple[int, ...], int] = {}
    for k, idx in dest.structure.items():
        gcoord = tuple(
            int(round(float(k.rep[j, 0]) * grid[j])) % int(grid[j]) for j in range(dim)
        )
        lookup[gcoord] = idx

    indices: list[int] = []
    for i in range(mapped_grid.shape[0]):
        gcoord = tuple(int(mapped_grid[i, j]) for j in range(dim))
        if gcoord not in lookup:
            raise ValueError(
                f"Source momentum maps to grid {gcoord}, not in destination BZ."
            )
        indices.append(lookup[gcoord])

    torch_device = device.torch_device() if device is not None else None
    return Tensor(
        data=cast(
            torch.LongTensor,
            torch.tensor(indices, dtype=torch.long, device=torch_device),
        ),
        dims=(src,),
    )


def momentum_map(
    kspace: MomentumSpace,
    raw_opr: Callable[[Momentum], Momentum],
) -> MomentumSpace:
    """
    Batch-compute ``kspace.map(lambda k: raw_opr(k).fractional())``.

    *raw_opr* must be the **unwrapped** operator (e.g. ``lambda k: t @ k``).
    Fractional wrapping is applied in bulk via numpy after the linear
    transformation matrix has been determined by probing with ``d + 1``
    reference momenta.

    Parameters
    ----------
    `kspace` : `MomentumSpace`
        Source momentum space to transform.
    `raw_opr` : `Callable[[Momentum], Momentum]`
        Affine momentum operator **without** fractional wrapping.

    Returns
    -------
    `MomentumSpace`
        New momentum space with transformed (and fractionally wrapped)
        k-points, preserving the original index assignment.
    """
    k_elements = kspace.elements()
    if not k_elements:
        return kspace

    recip_lat = k_elements[0].space
    dim = recip_lat.dim
    M, c, result_space = _probe_affine(raw_opr, recip_lat)

    k_frac = _kspace_frac(kspace)
    new_frac = k_frac @ M.T + c
    new_frac_wrapped = new_frac - np.floor(new_frac)

    grid_shape = np.array(result_space.shape, dtype=np.int64)
    grid_ints = np.rint(new_frac_wrapped * grid_shape).astype(np.int64) % grid_shape

    new_structure: OrderedDict[Momentum, int] = OrderedDict()
    for i, (k, idx) in enumerate(kspace.structure.items()):
        rep = ImmutableDenseMatrix(
            [sy.Rational(int(grid_ints[i, j]), int(grid_shape[j])) for j in range(dim)]
        )
        new_structure[Momentum(rep=rep, space=result_space)] = idx

    return MomentumSpace(structure=restructure(new_structure))
