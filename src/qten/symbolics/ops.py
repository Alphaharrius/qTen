"""
Convenience operators for symbolic Hilbert-space transformations.

This module provides small functional constructors for common symbolic
operations: translating or rebasing spatial irreps, expanding Bloch Hilbert
spaces across real-space regions, constructing Hamiltonian representations, and
building basis transforms between position and momentum descriptions.

Repository usage
----------------
Use these helpers when composing [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr]
or converting between [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
representations. The underlying basis, span, and operator classes are defined in
[`qten.symbolics.hilbert_space`][qten.symbolics.hilbert_space].
"""

from typing import Callable, Optional, Sequence, TypeVar, Union, cast, overload
import torch

from ..geometries import AffineSpace, Momentum, Offset, ReciprocalLattice
from ..geometries.spatials import OffsetType
from ..linalg.tensors import Tensor
from . import FuncOpr, HilbertSpace, Opr, StateSpace
from .state_space import MomentumSpace
from ..utils.devices import Device

T = TypeVar("T")
S = TypeVar("S", bound=AffineSpace)


def translate_opr(d: OffsetType) -> FuncOpr[OffsetType]:
    r"""
    Build an operator that translates irreps of the same concrete type by `d`.

    Parameters
    ----------
    d : Offset | Momentum
        Translation to add to the targeted irrep.

    Returns
    -------
    FuncOpr
        Operator applying \(x \mapsto x + d\) to irreps whose concrete type
        matches `type(d)`.
    """
    point_type = cast(type[OffsetType], type(d))
    return FuncOpr(point_type, lambda r: cast(OffsetType, r + d))


def rebase_opr(space: S) -> FuncOpr[OffsetType]:
    r"""
    Build an operator that rebases spatial irreps into `space`.

    For affine spaces this targets [`Offset`][qten.geometries.spatials.Offset] irreps. For reciprocal lattices it
    targets [`Momentum`][qten.geometries.spatials.Momentum] irreps.

    Parameters
    ----------
    space : AffineSpace | ReciprocalLattice
        Target space into which matching spatial irreps are rebased.

    Returns
    -------
    FuncOpr
        Operator applying \(r \mapsto r.\mathrm{rebase}(\mathrm{space})\) to
        matching spatial irreps. In code, this calls `r.rebase(space)`.
    """
    point_type = cast(
        type[OffsetType], Momentum if isinstance(space, ReciprocalLattice) else Offset
    )
    return FuncOpr(point_type, lambda r: cast(OffsetType, r.rebase(space)))


@overload
def fractional_opr() -> FuncOpr[Offset]:
    """
    Return an operator targeting [`Offset`][qten.geometries.spatials.Offset] irreps.
    """
    ...


@overload
def fractional_opr(T: type[OffsetType]) -> FuncOpr[OffsetType]:
    """
    Return an operator targeting irreps of the exact runtime type `T`.
    """
    ...


def fractional_opr(
    T: type[OffsetType] | None = None,
) -> FuncOpr[Offset] | FuncOpr[OffsetType]:
    """
    Build an operator that replaces a spatial irrep by its fractional form.

    Parameters
    ----------
    T : type[Offset] | type[Momentum] | None, optional
        Exact irrep type to target. Because [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] matches exact runtime
        types, reciprocal-space basis states should use
        [`fractional_opr(Momentum)`][qten.symbolics.ops.fractional_opr]. If omitted, [`Offset`][qten.geometries.spatials.Offset] is targeted.

    Returns
    -------
    FuncOpr
        Operator replacing matching spatial irreps by `r.fractional()`.
    """
    if T is None:
        return FuncOpr(Offset, Offset.fractional)
    return FuncOpr(T, lambda r: cast(OffsetType, r.fractional()))


def region_hilbert(bloch_space: HilbertSpace, region: Sequence[Offset]) -> HilbertSpace:
    """
    Expand a Bloch [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] across a real-space region by unit-cell subgroup.

    The input `bloch_space` is first partitioned by the fractional part of each
    basis state's [`Offset`][qten.geometries.spatials.Offset] irrep, so all sub-basis states that occupy the same
    position within the unit cell stay grouped together. Each offset in
    `region` is treated as a full target-site offset that already includes its
    unit-cell position. A region site `r` therefore selects the subgroup whose
    fractional offset equals `r.fractional()`, and every state in that subgroup
    is copied with its [`Offset`][qten.geometries.spatials.Offset] replaced by `r`.

    Parameters
    ----------
    bloch_space : HilbertSpace
        Source basis to replicate. Each basis element must contain an [`Offset`][qten.geometries.spatials.Offset]
        irrep. Elements with the same fractional offset are treated as the
        sub-basis of one unit-cell site and are replicated together.
    region : Sequence[Offset]
        Full target offsets where the matching unit-cell sub-basis should be
        placed.

    Returns
    -------
    HilbertSpace
        A Hilbert space containing one copied basis state for each compatible
        pair of region offset and unit-cell subgroup, with output offsets taken
        from `region` after rebasing into the Bloch offset space when needed.

    Raises
    ------
    ValueError
        Propagated if a basis element in `bloch_space` does not contain an
        Offset irrep to inspect or replace, or if `region` contains a
        fractional offset that does not exist in `bloch_space`.

    Notes
    -----
    Grouping is done by fractional [`Offset`][qten.geometries.spatials.Offset], preserving the original basis
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
    r"""
    Return the matrix representation of an operator on a Hilbert-space basis.

    Let \(\mathrm{space} = \mathrm{span}\{|e_i\rangle\}\) be the input
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] and let `opr`
    act on each basis state to produce
    \(\mathrm{span}\{\mathrm{opr}\,|e_j\rangle\}\). In code, the transformed
    basis is produced by `opr @ space`. This function constructs the
    corresponding representation matrix
    \(M_{ij} = \langle e_i | \mathrm{opr} | e_j \rangle\), implemented as the
    cross-Gram matrix between the original basis and its
    transformed image. The resulting [`Tensor`][qten.linalg.tensors.Tensor] therefore represents `opr` in the
    basis supplied by `space`.

    Parameters
    ----------
    opr : Opr
        Operator whose representation is to be computed.
    space : HilbertSpace
        Basis in which the operator is represented.

    Returns
    -------
    Tensor
        Square tensor whose entries are the matrix elements of `opr` in the
        basis `space`.

    Raises
    ------
    ValueError
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
    src : StateSpace[T]
        Source state space whose element order defines the output index order.
    dest : StateSpace[T]
        Destination state space whose integer positions are used as the
        returned indices.
    matching_func : Callable[[T], T]
        Function mapping each source element to its matching destination
        element.
    device : Optional[Device], optional
        Device to place the returned index tensor on, by default `None` (CPU).

    Returns
    -------
    Tensor[torch.LongTensor]
        Rank-1 integer tensor with dims `(src,)`, where each entry is the
        destination index of the corresponding source element.

    Raises
    ------
    ValueError
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

