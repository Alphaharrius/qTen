from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
import sympy as sy
from sympy import ImmutableDenseMatrix
from sympy.matrices.normalforms import smith_normal_decomp  # type: ignore[import-untyped]


class BoundaryCondition(ABC):
    """
    Abstract base class for boundary conditions in lattice systems.
    """

    @property
    @abstractmethod
    def basis(self) -> ImmutableDenseMatrix:
        """The matrix of the boundary basis."""
        pass

    @abstractmethod
    def wrap(self, index: ImmutableDenseMatrix) -> ImmutableDenseMatrix:
        """Wrap an index into the valid region of this boundary."""
        pass

    @abstractmethod
    def representatives(self) -> tuple[ImmutableDenseMatrix, ...]:
        """Return one canonical representative per boundary equivalence class."""
        pass


@dataclass(frozen=True)
class PeriodicBoundary(BoundaryCondition):
    """
    Periodic boundary: wraps indices using modulo arithmetic via Smith Normal Form.
    """

    _basis: ImmutableDenseMatrix = field(repr=False)
    _U: ImmutableDenseMatrix = field(init=False, repr=False, compare=False)
    _U_inv: ImmutableDenseMatrix = field(init=False, repr=False, compare=False)
    _periods: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self._basis.rows != self._basis.cols:
            raise ValueError(f"boundary basis must be square, got {self._basis.shape}.")

        S, U, _ = smith_normal_decomp(self._basis, domain=sy.ZZ)
        S = ImmutableDenseMatrix(S)
        U = ImmutableDenseMatrix(U)
        periods = self._snf_periods(S)

        object.__setattr__(self, "_U", U)
        object.__setattr__(self, "_U_inv", ImmutableDenseMatrix(U.inv()))
        object.__setattr__(self, "_periods", periods)

    @property
    def basis(self) -> ImmutableDenseMatrix:
        return self._basis

    def wrap(self, index: ImmutableDenseMatrix) -> ImmutableDenseMatrix:
        expected_shape = (self.basis.rows, 1)
        if index.shape != expected_shape:
            raise ValueError(
                f"index shape {index.shape} does not match expected {expected_shape}."
            )

        if self.basis.is_diagonal():
            wrapped_entries = [
                sy.Mod(index[i, 0], int(self.basis[i, i]))
                for i in range(self.basis.rows)
            ]
            return ImmutableDenseMatrix(self.basis.rows, 1, wrapped_entries)

        coordinates = ImmutableDenseMatrix(self._U @ index)
        wrapped_entries = [
            sy.Mod(coordinates[i, 0], self._periods[i]) for i in range(self.basis.rows)
        ]
        wrapped_coords = ImmutableDenseMatrix(self.basis.rows, 1, wrapped_entries)
        return ImmutableDenseMatrix(self._U_inv @ wrapped_coords)

    def representatives(self) -> tuple[ImmutableDenseMatrix, ...]:
        if self.basis.is_diagonal():
            elements = product(
                *(range(int(self.basis[i, i])) for i in range(self.basis.rows))
            )
            return tuple(
                ImmutableDenseMatrix(self.basis.rows, 1, el) for el in elements
            )

        elements = product(*(range(period) for period in self._periods))
        return tuple(
            ImmutableDenseMatrix(
                self._U_inv @ ImmutableDenseMatrix(self.basis.rows, 1, el)
            )
            for el in elements
        )

    @staticmethod
    def _snf_periods(S: ImmutableDenseMatrix) -> tuple[int, ...]:
        periods: list[int] = []
        for i in range(S.rows):
            invariant = S[i, i]
            period = int(invariant)
            if period == 0:
                raise ValueError(
                    "boundary basis must be full-rank (non-zero SNF invariants)."
                )
            if period < 0:
                raise ValueError(
                    f"SNF invariant factors must be positive, got {invariant}."
                )
            periods.append(period)
        return tuple(periods)
