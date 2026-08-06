"""Generate QTen's offline crystallographic spinor-irrep cache with spgrep.

The generated data contains projective (double-valued) irreducible characters
for the 32 crystallographic point groups.  spgrep is needed only to regenerate
the cache; QTen's runtime reads the packaged JSON without importing spgrep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import spgrep
import sympy as sy
from spgrep import get_crystallographic_pointgroup_spinor_irreps_from_symmetry
from spgrep.rep.representation import get_character

from qten.phys.spin import (
    SU2_SECTION_CONVENTION,
    proper_rotation_matrix,
    su2_from_so3,
)
from qten.pointgroups._registry import _cartesianize_generator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "qten" / "pointgroups" / "data"
POINT_GROUP_DATA_PATH = DATA_DIR / "point_group_data.json"
OUTPUT_PATH = DATA_DIR / "double_point_group_data.json"
SPGREP_URL = "https://github.com/spglib/spgrep"
ROUND_DIGITS = 15


def _matrix_key(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in matrix.reshape(-1))


def _close_group(generators: list[list[list[int]]]) -> list[np.ndarray]:
    """Close exact integer generators in deterministic breadth-first order."""
    generator_arrays = [np.asarray(generator, dtype=int) for generator in generators]
    identity = np.eye(3, dtype=int)
    elements = [identity]
    seen = {_matrix_key(identity)}
    frontier = [identity]
    while frontier:
        current = frontier.pop(0)
        for generator in generator_arrays:
            for candidate in (generator @ current, current @ generator):
                key = _matrix_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                elements.append(candidate)
                frontier.append(candidate)
                if len(elements) > 512:
                    raise ValueError("Point-group closure exceeded 512 elements.")
    return elements


def _lattice(crystal_system: str) -> np.ndarray:
    """Return row-wise standard crystallographic basis vectors for spgrep."""
    if crystal_system in {"trigonal", "hexagonal"}:
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-0.5, np.sqrt(3.0) / 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    return np.eye(3)


def _complex_array(matrix: sy.Matrix) -> np.ndarray:
    return np.asarray(
        [
            [complex(sy.N(matrix[i, j], 17)) for j in range(matrix.cols)]
            for i in range(matrix.rows)
        ],
        dtype=complex,
    )


def _qten_lift(rotation: np.ndarray, crystal_system: str) -> np.ndarray:
    source = sy.ImmutableDenseMatrix(rotation.tolist())
    cartesian = _cartesianize_generator(source, crystal_system)
    proper = proper_rotation_matrix(cartesian)
    return _complex_array(su2_from_so3(proper))


def _section_signs(
    rotations: list[np.ndarray],
    unitary_rotations: np.ndarray,
    crystal_system: str,
) -> list[int]:
    """Return signs eta with U_qten(g) = eta(g) U_spgrep(g)."""
    signs: list[int] = []
    for rotation, spgrep_lift in zip(rotations, unitary_rotations):
        qten_lift = _qten_lift(rotation, crystal_system)
        overlap = np.trace(spgrep_lift.conj().T @ qten_lift) / 2.0
        sign = 1 if overlap.real >= 0 else -1
        if not np.allclose(qten_lift, sign * spgrep_lift, rtol=0, atol=1e-8):
            raise ValueError(
                "spgrep and QTen SU(2) lifts are not related by a sign for "
                f"rotation {rotation.tolist()}."
            )
        signs.append(sign)
    return signs


def _multiplication_table(rotations: list[np.ndarray]) -> np.ndarray:
    index = {_matrix_key(rotation): i for i, rotation in enumerate(rotations)}
    table = np.empty((len(rotations), len(rotations)), dtype=int)
    for i, left in enumerate(rotations):
        for j, right in enumerate(rotations):
            table[i, j] = index[_matrix_key(left @ right)]
    return table


def _signed_factor_system(
    factor_system: np.ndarray,
    section_signs: list[int],
    multiplication_table: np.ndarray,
) -> np.ndarray:
    signs = np.asarray(section_signs, dtype=int)
    order = len(signs)
    normalized = np.empty((order, order), dtype=int)
    for i in range(order):
        for j in range(order):
            k = multiplication_table[i, j]
            raw = complex(factor_system[i, j])
            raw_sign = 1 if raw.real >= 0 else -1
            if not np.allclose(raw, raw_sign, rtol=0, atol=1e-8):
                raise ValueError(f"Spinor factor is not ±1: {raw}")
            normalized[i, j] = signs[i] * signs[j] * raw_sign * signs[k]
    return normalized


def _clean_scalar(value: float) -> float:
    rounded = round(float(value), ROUND_DIGITS)
    return 0.0 if abs(rounded) < 10 ** (-ROUND_DIGITS) else rounded


def _encode_complex(value: complex) -> list[float]:
    return [_clean_scalar(value.real), _clean_scalar(value.imag)]


def _character_signature(character: np.ndarray) -> tuple[float, ...]:
    encoded = [_encode_complex(complex(value)) for value in character]
    return tuple(component for pair in encoded for component in pair)


def _validate_projective_characters(characters: list[np.ndarray], order: int) -> None:
    dimensions = [round(complex(character[0]).real) for character in characters]
    if sum(dimension * dimension for dimension in dimensions) != order:
        raise ValueError("Spinor irrep dimensions do not satisfy sum(dim^2)=|G|.")

    gram = np.asarray(
        [[np.vdot(left, right) / order for right in characters] for left in characters]
    )
    if not np.allclose(gram, np.eye(len(characters)), rtol=0, atol=1e-8):
        raise ValueError("Spinor character rows are not orthonormal.")


def _validate_factor_system(
    rotations: list[np.ndarray],
    unitary_rotations: np.ndarray,
    factor_system: np.ndarray,
    multiplication_table: np.ndarray,
) -> None:
    for i in range(len(rotations)):
        for j in range(len(rotations)):
            k = multiplication_table[i, j]
            expected = factor_system[i, j] * unitary_rotations[k]
            if not np.allclose(
                unitary_rotations[i] @ unitary_rotations[j],
                expected,
                rtol=0,
                atol=1e-8,
            ):
                raise ValueError("spgrep SU(2) matrices violate their factor system.")


def _group_record(record: dict[str, Any]) -> dict[str, Any]:
    rotations = _close_group(record["generators"])
    lattice = _lattice(record["crystal_system"])
    irreps, factor_system, unitary_rotations = (
        get_crystallographic_pointgroup_spinor_irreps_from_symmetry(
            lattice,
            np.asarray(rotations, dtype=int),
            method="Neto",
        )
    )
    multiplication_table = _multiplication_table(rotations)
    _validate_factor_system(
        rotations, unitary_rotations, factor_system, multiplication_table
    )

    section_signs = _section_signs(
        rotations, unitary_rotations, record["crystal_system"]
    )
    qten_factor_system = _signed_factor_system(
        factor_system, section_signs, multiplication_table
    )
    raw_characters = [
        np.asarray(get_character(irrep), dtype=complex)
        * np.asarray(section_signs, dtype=int)
        for irrep in irreps
    ]
    _validate_projective_characters(raw_characters, len(rotations))

    sorted_characters = sorted(
        raw_characters,
        key=lambda character: (
            round(complex(character[0]).real),
            _character_signature(character),
        ),
    )
    normalized_irreps: dict[str, Any] = {}
    for index, character in enumerate(sorted_characters, start=1):
        label = f"spinor_{index}"
        normalized_irreps[label] = {
            "dim": int(round(complex(character[0]).real)),
            "characters": [_encode_complex(complex(value)) for value in character],
        }

    generator_bytes = json.dumps(
        record["generators"], separators=(",", ":"), sort_keys=True
    ).encode()
    return {
        "symbol": record["symbol"],
        "order": len(rotations),
        "generator_sha256": hashlib.sha256(generator_bytes).hexdigest(),
        "operations": [rotation.tolist() for rotation in rotations],
        "factor_system": qten_factor_system.tolist(),
        "irreps": normalized_irreps,
    }


def generate() -> dict[str, Any]:
    point_group_bytes = POINT_GROUP_DATA_PATH.read_bytes()
    point_group_data = json.loads(point_group_bytes)
    records = [_group_record(record) for record in point_group_data["point_groups"]]
    records.sort(key=lambda record: record["symbol"])
    return {
        "schema_version": 1,
        "source": {
            "name": "spgrep",
            "version": spgrep.__version__,
            "url": SPGREP_URL,
        },
        "point_group_data_sha256": hashlib.sha256(point_group_bytes).hexdigest(),
        "section_convention": SU2_SECTION_CONVENTION,
        "character_round_digits": ROUND_DIGITS,
        "point_groups": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if regeneration differs from the packaged cache.",
    )
    args = parser.parse_args()

    generated = json.dumps(generate(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_text() != generated:
            raise SystemExit(f"{OUTPUT_PATH} is stale; regenerate it without --check.")
        return
    OUTPUT_PATH.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
