from typing import Sequence

from ..geometries import Offset
from . import HilbertSpace


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
