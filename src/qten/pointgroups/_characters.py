"""Projective character tables from a group's own SU(2) section.

Ordinary Bilbao class tables stay packaged. Spinor characters are computed
from QTen's principal lift so they match ``D(g) = D_orb(g) ⊗ u(g)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import sympy as sy

from ..phys.spin import su2_numeric

if TYPE_CHECKING:
    from .finite import FinitePointGroup


_ATOL = 1e-8


def _identity_index(mult: np.ndarray) -> int:
    order = int(mult.shape[0])
    for index in range(order):
        if np.array_equal(mult[index], np.arange(order)):
            return index
    raise ValueError("Multiplication table has no identity.")


def _inverses(mult: np.ndarray, identity: int) -> np.ndarray:
    order = int(mult.shape[0])
    inverses = np.empty(order, dtype=int)
    for index in range(order):
        matches = np.flatnonzero(mult[index] == identity)
        if matches.size != 1:
            raise ValueError("Multiplication table is not a group.")
        inverses[index] = int(matches[0])
    return inverses


def _conjugacy_classes(
    mult: np.ndarray, inverses: np.ndarray, identity: int
) -> list[list[int]]:
    order = int(mult.shape[0])
    unused = set(range(order))
    classes: list[list[int]] = []
    while unused:
        seed = min(unused)
        members: set[int] = set()
        for conjugator in range(order):
            conjugated = int(mult[int(mult[conjugator, seed]), inverses[conjugator]])
            members.add(conjugated)
        classes.append(sorted(members))
        unused -= members
    classes.sort(key=lambda items: (0 if identity in items else 1, items[0]))
    return classes


def _class_structure_constants(
    mult: np.ndarray, classes: list[list[int]]
) -> np.ndarray:
    class_of = np.empty(mult.shape[0], dtype=int)
    for class_index, members in enumerate(classes):
        for element in members:
            class_of[element] = class_index
    n_class = len(classes)
    structure = np.zeros((n_class, n_class, n_class), dtype=int)
    for left_index, left_class in enumerate(classes):
        for right_index, right_class in enumerate(classes):
            counts = np.zeros(n_class, dtype=int)
            for left in left_class:
                for right in right_class:
                    counts[class_of[int(mult[left, right])]] += 1
            for class_index, members in enumerate(classes):
                size = len(members)
                if counts[class_index] % size:
                    raise ValueError(
                        "Class-algebra structure constants are not integral."
                    )
                structure[left_index, right_index, class_index] = int(
                    counts[class_index] // size
                )
    return structure


def _split_eigenspaces(
    matrices: list[np.ndarray],
    space: np.ndarray,
    *,
    atol: float = _ATOL,
) -> list[np.ndarray]:
    """Split an invariant subspace into simultaneous eigenspaces of `matrices`.

    `space` holds a (possibly non-orthogonal) column basis. Class-algebra
    matrices are real but not Hermitian, so the restriction is obtained by
    least squares rather than a Hermitian projection.
    """
    if space.shape[1] == 1:
        return [space]
    remaining = [space]
    for matrix in matrices:
        split: list[np.ndarray] = []
        for block in remaining:
            restricted, *_ = np.linalg.lstsq(block, matrix @ block, rcond=None)
            values, vectors = np.linalg.eig(restricted)
            used = np.zeros(values.size, dtype=bool)
            for index, value in enumerate(values):
                if used[index]:
                    continue
                mask = np.abs(values - value) <= atol
                used |= mask
                split.append(block @ vectors[:, mask])
        remaining = split
    return remaining


def character_table(mult: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Return irreducible characters on conjugacy classes of a finite group.

    Parameters
    ----------
    mult
        Square multiplication table, ``mult[i, j]`` is the index of ``g_i g_j``.

    Returns
    -------
    characters, classes
        ``characters[mu, k]`` is ``χ^μ(C_k)``. The identity class is first.
        Each class is a sorted list of element indices.
    """
    identity = _identity_index(mult)
    inverses = _inverses(mult, identity)
    classes = _conjugacy_classes(mult, inverses, identity)
    structure = _class_structure_constants(mult, classes)
    n_class = len(classes)
    sizes = np.asarray([len(members) for members in classes], dtype=float)
    order = float(mult.shape[0])

    # Left multiplication: (T_r)_{t,s} = N_{r s t} for C_r C_s = sum_t N_rst C_t.
    class_matrices = [structure[index].T.astype(complex) for index in range(n_class)]
    spaces: list[np.ndarray] | None = None
    for seed in range(8):
        rng = np.random.default_rng(seed)
        combo = sum(
            (complex(rng.normal(), rng.normal()) * matrix for matrix in class_matrices),
            np.zeros((n_class, n_class), dtype=complex),
        )
        _, vectors = np.linalg.eig(combo)
        candidate = _split_eigenspaces(class_matrices, vectors)
        if len(candidate) == n_class and all(
            space.shape[1] == 1 for space in candidate
        ):
            spaces = candidate
            break
    if spaces is None:
        raise ValueError("Failed to split the class algebra into irreps.")

    rows: list[np.ndarray] = []
    for space in spaces:
        if space.shape[1] != 1:
            raise ValueError("Class-algebra eigenspace is not one-dimensional.")
        vector = space[:, 0]
        # Eigenvalues λ_r = |C_r| χ_r / χ(1). Read them from T_r v = λ_r v.
        lambdas = np.empty(n_class, dtype=complex)
        for index, matrix in enumerate(class_matrices):
            image = matrix @ vector
            denom = np.vdot(vector, vector)
            if abs(denom) <= _ATOL:
                raise ValueError("Class-algebra eigenvector vanished.")
            lambdas[index] = np.vdot(vector, image) / denom
            if np.max(np.abs(image - lambdas[index] * vector)) > 1e-6:
                raise ValueError("Class-algebra vector is not an eigenvector.")
        # λ_r = |C_r| χ_r / χ(1)  ⇒  ∑_r |λ_r|² / |C_r| = |G| / d².
        denom = float(np.sum(np.abs(lambdas) ** 2 / sizes).real)
        if denom <= _ATOL:
            raise ValueError("Could not normalize an irreducible character.")
        dim = abs(order / denom) ** 0.5
        dim_int = int(round(dim))
        if abs(dim - dim_int) > 1e-6:
            raise ValueError(f"Non-integral irrep dimension {dim}.")
        characters = lambdas * dim_int / sizes
        # Identity class must be the positive dimension.
        if abs(characters[0] + dim_int) < abs(characters[0] - dim_int):
            characters = -characters
        characters[0] = dim_int
        rows.append(characters)

    table = np.vstack(rows)
    gram = (table * sizes) @ table.conj().T / order
    if not np.allclose(gram, np.eye(len(rows)), rtol=0.0, atol=1e-6):
        raise ValueError("Computed characters are not class-orthogonal.")
    if int(round(float(np.sum(np.abs(table[:, 0]) ** 2).real))) != int(order):
        raise ValueError("Computed irrep dimensions do not satisfy sum(dim^2)=|G|.")
    order_index = np.lexsort(
        (np.round(table[:, 0].real, 6), np.round(np.abs(table).sum(axis=1), 6))
    )
    return table[order_index], classes


def _group_multiplication_table(group: FinitePointGroup) -> np.ndarray:
    from .finite import _matrix_key

    elements = group.elements()
    index = {_matrix_key(element.irrep): i for i, element in enumerate(elements)}
    order = len(elements)
    table = np.empty((order, order), dtype=int)
    for i, left in enumerate(elements):
        for j, right in enumerate(elements):
            table[i, j] = index[_matrix_key(left.invoke(right).irrep)]
    return table


def factor_system_and_lifts(
    group: FinitePointGroup,
) -> tuple[np.ndarray, list[list[list[complex]]]]:
    """Return ω(g, h) and the numeric SU(2) lifts in ``elements()`` order."""
    elements = group.elements()
    lifts = [su2_numeric(element) for element in elements]
    mult = _group_multiplication_table(group)
    order = len(elements)
    omega = np.ones((order, order), dtype=int)
    for i in range(order):
        for j in range(order):
            k = int(mult[i, j])
            product = np.asarray(lifts[i], dtype=complex) @ np.asarray(
                lifts[j], dtype=complex
            )
            target = np.asarray(lifts[k], dtype=complex)
            overlap = complex(np.vdot(target.reshape(-1), product.reshape(-1)) / 2)
            sign = 1 if overlap.real >= 0 else -1
            if abs(abs(overlap) - 1) > 1e-6:
                raise ValueError(
                    "SU(2) lifts are not a section of the point group "
                    f"{group.symbol or '<anonymous>'}."
                )
            omega[i, j] = sign
    return omega, lifts


def _extension_multiplication(mult: np.ndarray, omega: np.ndarray) -> np.ndarray:
    order = int(mult.shape[0])
    hat = np.empty((2 * order, 2 * order), dtype=int)
    for left in range(2 * order):
        left_sign = 1 if left < order else -1
        left_el = left if left < order else left - order
        for right in range(2 * order):
            right_sign = 1 if right < order else -1
            right_el = right if right < order else right - order
            product = int(mult[left_el, right_el])
            sign = left_sign * right_sign * int(omega[left_el, right_el])
            hat[left, right] = product if sign > 0 else product + order
    return hat


def _encode_class_character(value: complex) -> int | float | list[float]:
    if abs(value.imag) < 1e-8:
        real = float(value.real)
        nearest = round(real)
        if abs(real - nearest) < 1e-8:
            return int(nearest)
        return real
    return [round(float(value.real), 12), round(float(value.imag), 12)]


def compute_spinor_irreps(group: FinitePointGroup) -> dict[str, Any]:
    """Build a class-wise spinor table from the group's own SU(2) section."""
    elements = group.elements()
    if not elements:
        raise ValueError("Cannot compute spinor irreps of an empty group.")

    mult = _group_multiplication_table(group)
    omega, _lifts = factor_system_and_lifts(group)
    hat_characters, hat_classes = character_table(
        _extension_multiplication(mult, omega)
    )
    identity = _identity_index(mult)
    minus_identity = identity + len(elements)
    class_of_minus = next(
        index for index, members in enumerate(hat_classes) if minus_identity in members
    )

    projective: list[np.ndarray] = []
    for row in hat_characters:
        dim = complex(row[0]).real
        if abs(complex(row[class_of_minus]).real + dim) > 0.5:
            continue
        projective.append(row)

    if not projective:
        raise ValueError(
            f"No projective irreps found for {group.symbol or '<anonymous>'}."
        )

    hat_class_of = {}
    for hat_class_index, members in enumerate(hat_classes):
        for member in members:
            hat_class_of[member] = hat_class_index

    chi_by_element = np.vstack(
        [
            np.asarray(
                [row[hat_class_of[index]] for index in range(len(elements))],
                dtype=complex,
            )
            for row in projective
        ]
    )

    if group.irreps:
        labels = list(group.irreps["class_labels"])
        label_by_element = group._class_label_index_by_element()
        n_class = len(labels)
    else:
        generated = group.conjugacy_classes()
        labels = [f"C{index}" for index in range(len(generated))]
        label_by_element = tuple(
            next(i for i, members in enumerate(generated) if element in members)
            for element in range(len(elements))
        )
        n_class = len(labels)

    irreps: dict[str, Any] = {}
    for irrep_index, row in enumerate(chi_by_element, start=1):
        totals = np.zeros(n_class, dtype=complex)
        counts = np.zeros(n_class, dtype=int)
        for element_index, character in enumerate(row):
            label_index = label_by_element[element_index]
            totals[label_index] += character
            counts[label_index] += 1
        class_characters = []
        for index in range(n_class):
            if counts[index] == 0:
                class_characters.append(0.0)
                continue
            values = [
                row[element_index]
                for element_index, label_index in enumerate(label_by_element)
                if label_index == index
            ]
            mean = totals[index] / counts[index]
            if any(abs(complex(value) - complex(mean)) > 1e-5 for value in values):
                # Not ω-regular: a projective character vanishes on the class.
                class_characters.append(0.0)
            else:
                class_characters.append(mean)
        if "1" in labels:
            dim = int(round(complex(class_characters[labels.index("1")]).real))
        else:
            dim = int(round(abs(complex(class_characters[0]).real)))
        irreps[f"spinor_{irrep_index}"] = {
            "dim": dim,
            "characters": [
                _encode_class_character(value) for value in class_characters
            ],
        }

    multiplicities = (
        list(group.irreps["multiplicities"])
        if group.irreps
        else [label_by_element.count(index) for index in range(n_class)]
    )
    return {
        "class_labels": labels,
        "multiplicities": multiplicities,
        "irreps": irreps,
        "source": "qten-su2-principal-v1",
    }


def compute_ordinary_irreps(group: FinitePointGroup) -> dict[str, Any]:
    """Build a class-wise ordinary table from the generated group."""
    elements = group.elements()
    if not elements:
        raise ValueError("Cannot compute irreps of an empty group.")

    mult = _group_multiplication_table(group)
    characters, classes = character_table(mult)
    generated = [frozenset(members) for members in group.conjugacy_classes()]
    computed = [frozenset(members) for members in classes]
    try:
        perm = [computed.index(members) for members in generated]
    except ValueError as exc:
        raise ValueError(
            "Computed conjugacy classes do not match the generated group."
        ) from exc
    characters = characters[:, perm]

    labels = [f"C{index}" for index in range(len(generated))]
    identity = next(index for index, members in enumerate(generated) if 0 in members)
    labels[identity] = "1"
    multiplicities = [len(members) for members in generated]

    irreps: dict[str, Any] = {}
    for irrep_index, row in enumerate(characters, start=1):
        dim = int(round(abs(complex(row[identity]).real)))
        irreps[f"irrep_{irrep_index}"] = {
            "dim": dim,
            "characters": [_encode_class_character(complex(value)) for value in row],
        }

    return {
        "class_labels": labels,
        "multiplicities": multiplicities,
        "irreps": irreps,
        "source": "qten-computed",
    }


def parse_class_character(value: Any) -> complex:
    """Decode a packaged class character (int, float, or [re, im])."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    return complex(float(sy.N(value)))
