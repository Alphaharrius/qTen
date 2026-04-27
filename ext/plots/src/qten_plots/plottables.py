"""
Plot-ready objects used by QTen plotting backends.

This module defines lightweight containers that carry both geometry data and
backend-independent display preferences. Backend implementations can then read
the same object and translate its styling fields into Matplotlib or Plotly
arguments without changing the underlying QTen geometry objects.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable

from qten.geometries.spatials import Offset
from qten.plottings import Plottable
from ._utils import normalize_pointcloud_marker


@dataclass(frozen=True)
class PointCloud(Plottable):
    """
    Plottable collection of spatial offsets with optional display styling.

    A `PointCloud` groups
    [`Offset`][qten.geometries.spatials.Offset] objects that should be rendered
    together as one scatter trace. The object stores backend-independent style
    hints; each plotting backend maps those hints onto its own marker, color,
    opacity, size, and border options.

    Marker aliases
    --------------
    The `marker` field accepts the common aliases `o`, `circle`, `s`, `square`,
    `d`, `D`, `diamond`, `+`, `plus`, `cross`, and `x`. These are normalized to
    the marker vocabulary supported by each backend.

    Attributes
    ----------
    offsets : FrozenSet[Offset]
        Spatial offsets to render as a single point cloud. The set is normalized
        to a `frozenset` during initialization.
    name : str | None
        Optional trace or legend label.
    color : str | None
        Optional backend color value used for all points in the cloud.
    marker : str | None
        Optional marker shape alias.
    opacity : float | None
        Optional marker opacity in the inclusive range `[0, 1]`.
    size : float | None
        Optional positive marker size.
    border_color : str | None
        Optional marker border color.
    border_width : float | None
        Optional non-negative marker border width.

    Raises
    ------
    TypeError
        If any entry in `offsets` is not an `Offset`.
    ValueError
        If `marker`, `opacity`, `size`, or `border_width` is outside the
        supported range.
    """

    offsets: FrozenSet[Offset] = field(default_factory=frozenset)
    """
    Spatial offsets to render as a single point cloud. The set is normalized
    to a `frozenset` during initialization so the plottable is immutable and
    hashable.
    """
    name: str | None = None
    """Optional trace or legend label shown by plotting backends."""
    color: str | None = None
    """
    Optional backend color value applied uniformly to all points in the cloud.
    """
    marker: str | None = None
    """
    Optional marker shape alias, normalized to the vocabulary supported by the
    selected plotting backend.
    """
    opacity: float | None = None
    """
    Optional marker opacity in the inclusive range `[0, 1]`, forwarded as a
    backend-independent styling hint.
    """
    size: float | None = None
    """Optional positive marker size shared by every rendered point."""
    border_color: str | None = None
    """Optional marker border color used by backends that support outlines."""
    border_width: float | None = None
    """
    Optional non-negative marker border width used by backends that support
    marker outlines.
    """

    def __post_init__(self):
        from qten.geometries.spatials import Offset

        normalized = frozenset(self.offsets)
        if not all(isinstance(offset, Offset) for offset in normalized):
            raise TypeError("PointCloud offsets must all be Offset instances.")
        if self.opacity is not None and not (0.0 <= self.opacity <= 1.0):
            raise ValueError(
                f"PointCloud opacity must lie in [0, 1], got {self.opacity}."
            )
        normalize_pointcloud_marker(self.marker)
        if self.size is not None and self.size <= 0:
            raise ValueError(f"PointCloud size must be positive, got {self.size}.")
        if self.border_width is not None and self.border_width < 0:
            raise ValueError(
                f"PointCloud border_width must be non-negative, got {self.border_width}."
            )
        object.__setattr__(self, "offsets", normalized)

    @classmethod
    def of(
        cls,
        offsets: Iterable[Offset],
        name: str | None = None,
        color: str | None = None,
        marker: str | None = None,
        opacity: float | None = None,
        size: float | None = None,
        border_color: str | None = None,
        border_width: float | None = None,
    ) -> "PointCloud":
        """
        Construct a [`PointCloud`][qten_plots.plottables.PointCloud] from any iterable of offsets.

        This helper is convenient when the offsets are produced by an iterator
        or generator. It freezes the iterable into the immutable set stored by
        the dataclass.

        Parameters
        ----------
        offsets : Iterable[Offset]
            Spatial offsets to render as one point cloud.
        name : str | None
            Optional trace or legend label.
        color : str | None
            Optional backend color value used for all points in the cloud.
        marker : str | None
            Optional marker shape alias.
        opacity : float | None
            Optional marker opacity in the inclusive range `[0, 1]`.
        size : float | None
            Optional positive marker size.
        border_color : str | None
            Optional marker border color.
        border_width : float | None
            Optional non-negative marker border width.

        Returns
        -------
        PointCloud
            New immutable point cloud containing the provided offsets.

        Raises
        ------
        TypeError
            If any entry in `offsets` is not an `Offset`.
        ValueError
            If `marker`, `opacity`, `size`, or `border_width` is outside the
            supported range.
        """
        return cls(
            offsets=frozenset(offsets),
            name=name,
            color=color,
            marker=marker,
            opacity=opacity,
            size=size,
            border_color=border_color,
            border_width=border_width,
        )
