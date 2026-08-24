"""Build QTen's point-group catalog: generators, Bilbao χ, and spinor χ.

Ordinary class tables are a pinned Bilbao cache. This script never rewrites
those names or numbers. Spinor class tables are computed from QTen's
principal SU(2) lift and stored in the same shape as ``irreps``. Each spinor
row also gets a Representations DPG ``bilbao_label`` (Koster Γ); χ numbers
are never copied from those pages. ``--check`` verifies the generated group
still realizes the packaged ordinary table and that every spinor row is
labeled. spgrep is used only by ``--check-spgrep``; it is never written into
the JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sy

from qten.phys.spin import SU2_SECTION_CONVENTION
from qten.pointgroups._characters import (
    compute_ordinary_irreps,
    compute_spinor_irreps,
    parse_class_character,
)
from qten.pointgroups._registry import (
    _cartesianize_generator,
    _project_generator,
    _axis_symbols,
)
from qten.pointgroups.finite import FinitePointGroup


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "qten" / "pointgroups" / "data"
DATA_PATH = DATA_DIR / "point_group_data.json"
PROVENANCE_PATH = DATA_DIR / "PROVENANCE.md"

TWO_D_PARENTS = ("1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm")
ONE_D_PARENTS = ("1", "m")

C2Z_SPATIAL = [[-1, 0], [0, -1]]
C2Z_ROTATION3 = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
M_1D_SPATIAL = [[-1]]
M_1D_ROTATION3 = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Representations DPG (double point groups). Not BANDREP, not space-group DSG.
DPG_INDEX_URL = (
    "https://cryst.ehu.es/cgi-bin/cryst/programs/"
    "representations_point.pl?tipogrupo=dbg"
)
_GAMMA = "\u0393"
_GAMMA_LABEL_RE = re.compile(rf"^{_GAMMA}(\d+)(?:\^([+-]))?$")
_DPG_GAMMA_HTML_RE = re.compile(
    r"(?:overline|text-decoration:\s*overline|&#772;|̄)[^<]{0,80}"
    r"&Gamma;<sub>(\d+)</sub>"
    r"|"
    r"&Gamma;<sub>(\d+)</sub>[^<]{0,40}"
    r"(?:overline|text-decoration:\s*overline|&#772;|̄)"
    r"|"
    r"&Gamma;<sub>(\d+)</sub>\s*(?:\^|<sup>)([+-])",
    re.IGNORECASE | re.DOTALL,
)
_DPG_GAMMA_PLAIN_RE = re.compile(
    rf"{_GAMMA}\u0305?(\d+)(?:\^([+-]))?|&Gamma;<sub>(\d+)</sub>(?:\^|<sup>)?([+-])?",
    re.IGNORECASE,
)


def _load_catalog() -> dict[str, Any]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _three_d_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in data["point_groups"]
        if record.get("dim", 3) == 3 and record.get("frame", "xyz") == "xyz"
    ]


def _group_from_3d(record: dict[str, Any]) -> FinitePointGroup:
    cartesian = tuple(
        _cartesianize_generator(
            sy.ImmutableDenseMatrix(generator), record["crystal_system"]
        )
        for generator in record["generators"]
    )
    return FinitePointGroup.from_matrices(
        matrices=cartesian,
        axes=_axis_symbols("xyz"),
        symbol=record["symbol"],
        irreps=record.get("irreps"),
        spin="electron",
    )


def _matrix_list(
    matrices: tuple[sy.ImmutableDenseMatrix, ...],
) -> list[list[list[Any]]]:
    encoded: list[list[list[Any]]] = []
    for matrix in matrices:
        rows = []
        for row in range(matrix.rows):
            rows.append(
                [
                    int(entry) if entry.is_integer else str(entry)
                    for entry in (
                        sy.simplify(matrix[row, col]) for col in range(matrix.cols)
                    )
                ]
            )
        encoded.append(rows)
    return encoded


def _format_gamma(index: int, parity: str | None = None) -> str:
    label = f"{_GAMMA}{index}"
    if parity:
        return f"{label}^{parity}"
    return label


def _parse_gamma_label(label: str) -> tuple[int, str | None] | None:
    match = _GAMMA_LABEL_RE.match(label.strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _ordinary_gamma_extent(irreps: dict[str, Any]) -> tuple[int, bool]:
    max_index = 0
    has_parity = False
    for row in irreps.values():
        parsed = _parse_gamma_label(str(row.get("bilbao_label") or ""))
        if parsed is None:
            continue
        max_index = max(max_index, parsed[0])
        if parsed[1]:
            has_parity = True
    return max_index, has_parity


def _is_proper_class(label: str) -> bool:
    return not (label.startswith("m") or label.startswith("-"))


def _row_chi(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [parse_class_character(value) for value in row["characters"]],
        dtype=complex,
    )


def _chi_close(left: np.ndarray, right: np.ndarray, *, atol: float = 1e-6) -> bool:
    return bool(np.allclose(left, right, rtol=0.0, atol=atol))


def _chi_sort_key(chi: np.ndarray) -> tuple[Any, ...]:
    return (round(float(chi[0].real)),) + tuple(
        (-round(float(value.real), 10), -round(float(value.imag), 10)) for value in chi
    )


def _spinor_index(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def _koster_dpg_labels(
    ordinary_irreps: dict[str, Any], spinor_table: dict[str, Any]
) -> dict[str, str]:
    """Koster/Bilbao DPG Γ names from qten class χ (Representations DPG convention).

    Ordinary Γ indices continue. Centrosymmetric groups split each spinor type
    as Γn^+ / Γn^- by χ(i). χ numbers themselves are not rewritten.
    """
    class_labels = list(spinor_table["class_labels"])
    max_index, has_parity = _ordinary_gamma_extent(ordinary_irreps)
    irreps = spinor_table["irreps"]
    inversion = "-1" in class_labels and has_parity
    assigned: dict[str, str] = {}
    next_index = max_index + 1

    if inversion:
        inv_index = class_labels.index("-1")
        proper = [
            index
            for index, label in enumerate(class_labels)
            if _is_proper_class(label)
        ]
        grouped: dict[tuple[Any, ...], list[tuple[str, np.ndarray, str]]] = {}
        for name, row in irreps.items():
            chi = _row_chi(row)
            fingerprint = tuple(
                (round(float(chi[index].real), 8), round(float(chi[index].imag), 8))
                for index in proper
            )
            parity = "+" if chi[inv_index].real > 0 else "-"
            grouped.setdefault(fingerprint, []).append((name, chi, parity))
        ordered = sorted(
            grouped,
            key=lambda fingerprint: _chi_sort_key(
                np.asarray([complex(real, imag) for real, imag in fingerprint])
            ),
        )
        for fingerprint in ordered:
            members = grouped[fingerprint]
            plus = sorted(
                [item for item in members if item[2] == "+"],
                key=lambda item: _spinor_index(item[0]),
            )
            minus = sorted(
                [item for item in members if item[2] == "-"],
                key=lambda item: _spinor_index(item[0]),
            )
            for slot in range(max(len(plus), len(minus))):
                index = next_index
                next_index += 1
                if slot < len(plus):
                    assigned[plus[slot][0]] = _format_gamma(index, "+")
                if slot < len(minus):
                    assigned[minus[slot][0]] = _format_gamma(index, "-")
        return assigned

    items = [(name, _row_chi(row)) for name, row in irreps.items()]
    items.sort(key=lambda item: _chi_sort_key(item[1]) + (_spinor_index(item[0]),))
    for name, _chi in items:
        assigned[name] = _format_gamma(next_index)
        next_index += 1
    return assigned


def _dpg_http_get(url: str, *, timeout: float = 20) -> str | None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "qten-pointgroup-catalog/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if getattr(response, "status", 200) >= 400:
                return None
            return response.read().decode("latin-1", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _parse_dpg_gamma_labels(html: str) -> list[str]:
    """Extract barred (double-valued) Koster Γ labels from a DPG HTML page."""
    labels: list[str] = []
    seen: set[str] = set()
    for match in _DPG_GAMMA_HTML_RE.finditer(html):
        index = next(group for group in match.groups()[:3] if group)
        parity = match.group(4) if match.lastindex and match.lastindex >= 4 else None
        label = _format_gamma(int(index), parity if parity in {"+", "-"} else None)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    if labels:
        return labels
    for match in _DPG_GAMMA_PLAIN_RE.finditer(html):
        index = match.group(1) or match.group(3)
        if not index:
            continue
        parity = match.group(2) or match.group(4)
        # Unbarred Γ1… on the same page are ordinary rows; skip until an overline
        # is present, otherwise keep all Γ after the ordinary max via matching.
        label = _format_gamma(int(index), parity if parity in {"+", "-"} else None)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


class _DPGTableParser(HTMLParser):
    """Collect numeric table rows from Representations DPG HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag == "td" and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _parse_dpg_character_rows(html: str) -> list[dict[str, Any]]:
    parser = _DPGTableParser()
    try:
        parser.feed(html)
    except Exception:
        return []
    parsed: list[dict[str, Any]] = []
    for row in parser.rows:
        joined = " ".join(row)
        gamma_labels = _parse_dpg_gamma_labels(joined)
        if not gamma_labels:
            continue
        numbers: list[complex] = []
        for cell in row:
            stripped = cell.strip().replace(" ", "")
            if not stripped:
                continue
            try:
                if stripped in {"i", "-i", "+i"}:
                    numbers.append(1j if "i" in stripped and "-" not in stripped else -1j)
                    continue
                numbers.append(complex(stripped.replace("i", "j")))
            except ValueError:
                continue
        if numbers:
            parsed.append(
                {
                    "bilbao_label": gamma_labels[0],
                    "characters": numbers,
                    "dim": int(round(abs(numbers[0].real))),
                }
            )
    return parsed


def fetch_dpg_tables(symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Fetch Representations DPG character tables. Empty if the CGI is down."""
    index = _dpg_http_get(DPG_INDEX_URL)
    if not index:
        print("  DPG CGI unavailable (no index)", flush=True)
        return {}
    tables: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        html = _dpg_http_get(f"{DPG_INDEX_URL}&super={urllib.parse.quote(symbol)}")
        if not html or "Internal Server Error" in html or len(html) < 200:
            continue
        rows = _parse_dpg_character_rows(html)
        if rows:
            tables[symbol] = rows
            print(f"  DPG {symbol}: {len(rows)} rows", flush=True)
    return tables


def _eta_align_dpg_labels(
    spinor_table: dict[str, Any], dpg_rows: list[dict[str, Any]]
) -> dict[str, str] | None:
    """Match qten spinor rows to DPG double-valued Γ labels.

    Bilbao DPG pages have no U(g). A double-valued row agrees with qten's
    section up to a global sheet flip on G\\{1} (same idea as ``--check-spgrep``).
    """
    qten_items = [
        (name, _row_chi(row), int(row["dim"]))
        for name, row in spinor_table["irreps"].items()
    ]
    unused = set(range(len(dpg_rows)))
    assigned: dict[str, str] = {}
    for name, chi, dim in qten_items:
        scored = []
        for index in unused:
            dpg = dpg_rows[index]
            dpg_chi = np.asarray(dpg["characters"], dtype=complex)
            if int(dpg.get("dim") or round(abs(dpg_chi[0].real))) != dim:
                continue
            distance = _section_distance(chi, dpg_chi)
            if np.isfinite(distance):
                scored.append((distance, index))
        if not scored:
            return None
        scored.sort()
        distance, match_index = scored[0]
        if distance > 1e-3 * max(1, chi.size**0.5):
            return None
        assigned[name] = str(dpg_rows[match_index]["bilbao_label"])
        unused.remove(match_index)
    if unused:
        return None
    return assigned


def _section_distance(qten_chi: np.ndarray, dpg_chi: np.ndarray) -> float:
    """L2 distance after trying DPG class windows and a global sheet flip."""
    if dpg_chi.size < qten_chi.size:
        return float("inf")
    best = float("inf")
    n_class = qten_chi.size
    for start in range(dpg_chi.size - n_class + 1):
        window = dpg_chi[start : start + n_class]
        if abs(window[0].real - qten_chi[0].real) > 0.5:
            continue
        exact = float(np.linalg.norm(qten_chi - window))
        flipped = np.array(window, copy=True)
        flipped[1:] *= -1
        best = min(best, exact, float(np.linalg.norm(qten_chi - flipped)))
    return best


def _package_spinor_table(
    table: dict[str, Any], labels: dict[str, str], *, label_source: str
) -> dict[str, Any]:
    irreps = {}
    for name, row in table["irreps"].items():
        packed = {
            "bilbao_label": labels[name],
            "characters": row["characters"],
            "dim": row["dim"],
        }
        irreps[name] = packed
    return {
        "class_labels": table["class_labels"],
        "irreps": irreps,
        "label_source": label_source,
        "multiplicities": table["multiplicities"],
        "source": table["source"],
    }


def _inherit_dpg_labels(
    child_table: dict[str, Any], parent_table: dict[str, Any]
) -> dict[str, str]:
    parent_irreps = parent_table["irreps"]
    unused = set(parent_irreps)
    assigned: dict[str, str] = {}
    for name, row in child_table["irreps"].items():
        child_chi = _row_chi(row)
        if name in unused and _chi_close(child_chi, _row_chi(parent_irreps[name])):
            assigned[name] = parent_irreps[name]["bilbao_label"]
            unused.remove(name)
            continue
        match = next(
            (
                parent_name
                for parent_name in unused
                if _chi_close(child_chi, _row_chi(parent_irreps[parent_name]))
            ),
            None,
        )
        if match is None:
            raise SystemExit(
                f"Cannot inherit DPG labels: no parent row matches {name}."
            )
        assigned[name] = parent_irreps[match]["bilbao_label"]
        unused.remove(match)
    return assigned


def _labels_for_spinor_table(
    ordinary_irreps: dict[str, Any],
    spinor_table: dict[str, Any],
    dpg_rows: list[dict[str, Any]] | None,
) -> tuple[dict[str, str], str]:
    if dpg_rows:
        aligned = _eta_align_dpg_labels(spinor_table, dpg_rows)
        if aligned is not None:
            return aligned, "bilbao-dpg"
    return _koster_dpg_labels(ordinary_irreps, spinor_table), "bilbao-dpg-koster"


def _attach_3d_spinor(
    record: dict[str, Any],
    dpg_tables: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    print(f"  spinor {record['symbol']}", flush=True)
    group = _group_from_3d(record)
    table = compute_spinor_irreps(group)
    labels, label_source = _labels_for_spinor_table(
        record["irreps"]["irreps"],
        table,
        (dpg_tables or {}).get(record["symbol"]),
    )
    updated = dict(record)
    updated["dim"] = 3
    updated["frame"] = "xyz"
    updated["spinor_irreps"] = _package_spinor_table(
        table, labels, label_source=label_source
    )
    return updated


def _project_to_xy(record: dict[str, Any]) -> tuple[list, list] | None:
    spatial = []
    rotations = []
    try:
        for generator in record["generators"]:
            cartesian = _cartesianize_generator(
                sy.ImmutableDenseMatrix(generator), record["crystal_system"]
            )
            spatial.append(_project_generator(cartesian, "xy"))
            rotations.append(cartesian)
    except ValueError:
        return None
    return spatial, rotations


def _two_d_record(parent: dict[str, Any]) -> dict[str, Any]:
    symbol = parent["symbol"]
    if symbol == "2":
        spatial = [sy.ImmutableDenseMatrix(C2Z_SPATIAL)]
        rotations = [sy.ImmutableDenseMatrix(C2Z_ROTATION3)]
    else:
        projected = _project_to_xy(parent)
        if projected is None:
            raise ValueError(f"Cannot embed {symbol} into the xy plane.")
        spatial, rotations = projected
    group = FinitePointGroup.from_matrices(
        matrices=spatial,
        axes=_axis_symbols("xy"),
        symbol=symbol,
        irreps=parent.get("irreps"),
        rotation3s=rotations,
        spin="electron",
    )
    print(f"  spinor {symbol} (2D xy)", flush=True)
    table = compute_spinor_irreps(group)
    return {
        "aliases": list(parent.get("aliases", [])),
        "crystal_system": parent["crystal_system"],
        "dim": 2,
        "frame": "xy",
        "generators": {
            "spatial": _matrix_list(tuple(spatial)),
            "rotation3": _matrix_list(tuple(rotations)),
        },
        "irreps": parent.get("irreps"),
        "source_encoding": parent.get("source_encoding"),
        "spinor_irreps": _package_spinor_table(
            table,
            _inherit_dpg_labels(table, parent["spinor_irreps"]),
            label_source=str(
                parent["spinor_irreps"].get("label_source", "bilbao-dpg-koster")
            ),
        ),
        "symbol": symbol,
    }


def _one_d_record(parent: dict[str, Any]) -> dict[str, Any]:
    symbol = parent["symbol"]
    if symbol == "1":
        spatial = [sy.ImmutableDenseMatrix([[1]])]
        rotations = [sy.ImmutableDenseMatrix(I3)]
    elif symbol == "m":
        spatial = [sy.ImmutableDenseMatrix(M_1D_SPATIAL)]
        rotations = [sy.ImmutableDenseMatrix(M_1D_ROTATION3)]
    else:
        raise ValueError(f"No 1D embedding for {symbol}.")
    group = FinitePointGroup.from_matrices(
        matrices=spatial,
        axes=_axis_symbols("x"),
        symbol=symbol,
        irreps=parent.get("irreps"),
        rotation3s=rotations,
        spin="electron",
    )
    print(f"  spinor {symbol} (1D x)", flush=True)
    table = compute_spinor_irreps(group)
    return {
        "aliases": list(parent.get("aliases", [])),
        "crystal_system": parent["crystal_system"],
        "dim": 1,
        "frame": "x",
        "generators": {
            "spatial": _matrix_list(tuple(spatial)),
            "rotation3": _matrix_list(tuple(rotations)),
        },
        "irreps": parent.get("irreps"),
        "source_encoding": parent.get("source_encoding"),
        "spinor_irreps": _package_spinor_table(
            table,
            _inherit_dpg_labels(table, parent["spinor_irreps"]),
            label_source=str(
                parent["spinor_irreps"].get("label_source", "bilbao-dpg-koster")
            ),
        ),
        "symbol": symbol,
    }


def generate() -> dict[str, Any]:
    data = _load_catalog()
    parents = {record["symbol"]: record for record in _three_d_records(data)}
    print("Fetching Bilbao REPRESENTATIONS DPG", flush=True)
    dpg_tables = fetch_dpg_tables(list(parents))
    if not dpg_tables:
        print(
            "  DPG CGI down; assigning Koster Γ labels from the qten section",
            flush=True,
        )
    print("Computing 3D spinor class tables", flush=True)
    three_d = [
        _attach_3d_spinor(record, dpg_tables) for record in parents.values()
    ]
    labeled = {record["symbol"]: record for record in three_d}
    print("Computing 2D/1D embeddings", flush=True)
    embedded = [_two_d_record(labeled[symbol]) for symbol in TWO_D_PARENTS]
    embedded.extend(_one_d_record(labeled[symbol]) for symbol in ONE_D_PARENTS)
    records = three_d + embedded
    records.sort(key=lambda record: (int(record.get("dim", 3)), record["symbol"]))
    return {
        "schema_version": 3,
        "section_convention": SU2_SECTION_CONVENTION,
        "source": data.get("source"),
        "point_groups": records,
    }


def relabel_existing(
    data: dict[str, Any],
    dpg_tables: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Write DPG labels onto packaged spinor χ without recomputing numbers."""
    dpg_tables = dpg_tables or {}
    three_d = []
    for record in _three_d_records(data):
        table = record["spinor_irreps"]
        labels, label_source = _labels_for_spinor_table(
            record["irreps"]["irreps"],
            table,
            dpg_tables.get(record["symbol"]),
        )
        updated = dict(record)
        updated["spinor_irreps"] = _package_spinor_table(
            table, labels, label_source=label_source
        )
        three_d.append(updated)
    labeled = {record["symbol"]: record for record in three_d}
    others = []
    for record in data["point_groups"]:
        if record.get("dim", 3) == 3 and record.get("frame", "xyz") == "xyz":
            continue
        parent = labeled[record["symbol"]]
        table = record["spinor_irreps"]
        updated = dict(record)
        updated["spinor_irreps"] = _package_spinor_table(
            table,
            _inherit_dpg_labels(table, parent["spinor_irreps"]),
            label_source=str(
                parent["spinor_irreps"].get("label_source", "bilbao-dpg-koster")
            ),
        )
        others.append(updated)
    records = three_d + others
    records.sort(key=lambda record: (int(record.get("dim", 3)), record["symbol"]))
    return {
        "schema_version": data.get("schema_version", 3),
        "section_convention": data.get(
            "section_convention", SU2_SECTION_CONVENTION
        ),
        "source": data.get("source"),
        "point_groups": records,
    }


def _write_provenance() -> None:
    PROVENANCE_PATH.write_text(
        "\n".join(
            [
                "# Point-Group Data Provenance",
                "",
                "The catalog in `point_group_data.json` has one record shape for",
                "ordinary and spinor characters: generators plus class-wise χ.",
                "",
                "## Ordinary irreducible representations",
                "",
                "3D generators come from pymatgen-core `v2026.5.18`:",
                "",
                "- https://raw.githubusercontent.com/materialsproject/pymatgen-core/v2026.5.18/src/pymatgen/symmetry/symm_data.json",
                "",
                "Ordinary `irreps` are a pinned Bilbao class-table cache:",
                "",
                "- https://cryst.ehu.es/cgi-bin/rep/programs/sam/point.py?num=<point-group-number>&sg=<representative-space-group>",
                "",
                "Linear characters of the 32 crystallographic point groups do not",
                "depend on QTen's SU(2) lift, so these names and numbers stay frozen.",
                "A catalog rebuild only recomputes `spinor_irreps`. `--check` confirms",
                "that the generated group still realizes the packaged ordinary table.",
                "Schoenflies aliases are added by qten.",
                "",
                "## Spinor irreducible representations",
                "",
                "`spinor_irreps` uses the same `class_labels` / `{name: {dim, characters}}`",
                "shape. The numbers are computed from QTen's principal SU(2) lift",
                f"`{SU2_SECTION_CONVENTION}` of each element's `rotation3`. They are not",
                "copied from Bilbao double-group pages (those tables have no U(g)).",
                "",
                "Each spinor row has `bilbao_label`, the Koster Γ name used by",
                "Representations DPG (Elcoro et al., J. Appl. Cryst. 50, 1457 (2017)):",
                "",
                "- https://cryst.ehu.es/cgi-bin/cryst/programs/representations_point.pl?tipogrupo=dbg",
                "",
                "A rebuild fetches those pages when the CGI is up, η-aligns class rows",
                "to QTen's SU(2) section (same idea as `--check-spgrep`), and writes",
                "the names. χ stays the live lift. If the CGI is down, labels are the",
                "same Koster continuation of the ordinary Γ indices, with centrosymmetric",
                "irreps split as Γn^+ / Γn^- by χ(i). `--check` still verifies live-lift",
                "χ and that every spinor row has `bilbao_label`.",
                "",
                "2D and 1D records add `dim`, `frame`, and paired `spatial` / `rotation3`",
                "generators. Spin still uses the stored 3D rotation, never a padded 2D",
                "matrix. Their DPG labels are inherited from the parent 3D table.",
                "",
                "## spgrep",
                "",
                "spgrep is a development-only checker (`--check-spgrep`): after η-aligning",
                "its SU(2) section to QTen's lift, class averages are compared. spgrep",
                "numbers are never written into this JSON.",
                "",
                "- https://github.com/spglib/spgrep",
                "",
                "@article{spgrep,",
                "    doi = {10.21105/joss.05269},",
                "    url = {https://doi.org/10.21105/joss.05269},",
                "    year = {2023},",
                "    publisher = {The Open Journal},",
                "    volume = {8},",
                "    number = {85},",
                "    pages = {5269},",
                "    author = {Shinohara, Kohei and Togo, Atsushi and Tanaka, Isao},",
                "    title = {spgrep: On-the-fly generator of space-group irreducible representations},",
                "    journal = {J. Open Source Softw.}",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _rows_match(
    left: list[list[complex]], right: list[list[complex]], *, atol: float
) -> bool:
    unused = set(range(len(right)))
    for row in left:
        match = next(
            (
                index
                for index in unused
                if np.allclose(row, right[index], rtol=0.0, atol=atol)
            ),
            None,
        )
        if match is None:
            return False
        unused.remove(match)
    return not unused


def _check_ordinary_json(record: dict[str, Any]) -> None:
    table = record.get("irreps") or {}
    symbol = record["symbol"]
    if table.get("source") != "bilbao":
        raise SystemExit(f"{symbol} ordinary table source must be 'bilbao'.")
    labels = table.get("class_labels") or []
    multiplicities = [int(value) for value in table.get("multiplicities", ())]
    irreps = table.get("irreps") or {}
    if len(labels) != len(multiplicities):
        raise SystemExit(
            f"{symbol} ordinary class_labels and multiplicities are different lengths."
        )
    order = sum(multiplicities)
    if sum(int(row["dim"]) ** 2 for row in irreps.values()) != order:
        raise SystemExit(
            f"Ordinary dimensions do not satisfy sum(dim^2)=|G| for {symbol}."
        )


def _check_ordinary_realizes_group(
    group: FinitePointGroup, record: dict[str, Any]
) -> None:
    table = record["irreps"]
    order = sum(int(value) for value in table["multiplicities"])
    if group.order != order:
        raise SystemExit(
            f"Generated order {group.order} does not match ordinary |G|={order} "
            f"for {record['symbol']}."
        )
    try:
        class_by_element = group.element_class_indices()
    except ValueError as exc:
        raise SystemExit(
            f"Could not align generated classes to Bilbao labels for {record['symbol']}."
        ) from exc
    if len(class_by_element) != group.order:
        raise SystemExit(
            f"Ordinary class alignment length mismatches |G| for {record['symbol']}."
        )

    packaged_rows = [
        [
            complex(sy.N(character))
            for character in group.irrep_characters_by_element(name)
        ]
        for name in table["irreps"]
    ]
    computed = compute_ordinary_irreps(group)
    generated_index = group._generated_class_indices()
    computed_rows = [
        [
            parse_class_character(row["characters"][generated_index[index]])
            for index in range(group.order)
        ]
        for row in computed["irreps"].values()
    ]
    if not _rows_match(packaged_rows, computed_rows, atol=1e-6):
        raise SystemExit(
            f"Packaged Bilbao χ does not match the generated group for {record['symbol']}."
        )


def check_self(data: dict[str, Any], *, live: bool = True) -> None:
    three_d = _three_d_records(data)
    if len(three_d) != 32:
        raise SystemExit(f"Expected 32 3D records, found {len(three_d)}.")
    embedded_2d = [record for record in data["point_groups"] if record.get("dim") == 2]
    embedded_1d = [record for record in data["point_groups"] if record.get("dim") == 1]
    if {record["symbol"] for record in embedded_2d} != set(TWO_D_PARENTS):
        raise SystemExit("2D catalog symbols do not match the embedding table.")
    if {record["symbol"] for record in embedded_1d} != set(ONE_D_PARENTS):
        raise SystemExit("1D catalog symbols do not match the embedding table.")

    for record in data["point_groups"]:
        _check_ordinary_json(record)
        irreps = record.get("irreps") or {}
        spinor = record.get("spinor_irreps")
        if not spinor:
            raise SystemExit(f"{record['symbol']} is missing spinor_irreps.")
        if spinor.get("class_labels") != irreps.get("class_labels"):
            raise SystemExit(
                f"Spinor class labels differ from ordinary labels for {record['symbol']}."
            )
        order = sum(int(value) for value in irreps.get("multiplicities", ()))
        if sum(int(row["dim"]) ** 2 for row in spinor["irreps"].values()) != order:
            raise SystemExit(
                f"Spinor dimensions do not satisfy sum(dim^2)=|G| for {record['symbol']}."
            )
        labels = [
            str(row.get("bilbao_label") or "")
            for row in spinor["irreps"].values()
        ]
        if any(not label for label in labels):
            raise SystemExit(
                f"{record['symbol']} is missing spinor bilbao_label on at least one row."
            )
        if len(set(labels)) != len(labels):
            raise SystemExit(
                f"{record['symbol']} spinor bilbao_label values are not unique."
            )
        if record.get("dim", 3) < 3:
            generators = record["generators"]
            if not isinstance(generators, dict):
                raise SystemExit(
                    f"{record['symbol']} dim={record['dim']} needs spatial/rotation3 generators."
                )
            if len(generators["spatial"]) != len(generators["rotation3"]):
                raise SystemExit(
                    f"{record['symbol']} spatial and rotation3 generators differ in length."
                )
        if record.get("dim", 3) == 3:
            group = _group_from_3d(record)
        else:
            generators = record["generators"]
            group = FinitePointGroup.from_matrices(
                matrices=tuple(
                    sy.ImmutableDenseMatrix(matrix) for matrix in generators["spatial"]
                ),
                axes=_axis_symbols(record["frame"]),
                symbol=record["symbol"],
                irreps=record.get("irreps"),
                rotation3s=tuple(
                    sy.ImmutableDenseMatrix(matrix)
                    for matrix in generators["rotation3"]
                ),
                spin="electron",
            )
        _check_ordinary_realizes_group(group, record)
        if not live:
            continue
        live_table = compute_spinor_irreps(group)
        packaged_rows = [
            [parse_class_character(value) for value in row["characters"]]
            for row in spinor["irreps"].values()
        ]
        live_rows = [
            [parse_class_character(value) for value in row["characters"]]
            for row in live_table["irreps"].values()
        ]
        if not _rows_match(packaged_rows, live_rows, atol=1e-6):
            raise SystemExit(
                f"Packaged spinor table does not match the live lift for {record['symbol']}."
            )

    three_d_by_symbol = {record["symbol"]: record for record in three_d}
    for record in data["point_groups"]:
        if record.get("dim", 3) == 3:
            continue
        parent = three_d_by_symbol[record["symbol"]]
        expected = _inherit_dpg_labels(
            record["spinor_irreps"], parent["spinor_irreps"]
        )
        for name, row in record["spinor_irreps"]["irreps"].items():
            if row.get("bilbao_label") != expected[name]:
                raise SystemExit(
                    f"{record['symbol']} dim={record['dim']} {name} DPG label "
                    f"{row.get('bilbao_label')!r} does not inherit {expected[name]!r}."
                )


def _close_integer_group(generators: list[list[list[int]]]) -> list[np.ndarray]:
    arrays = [np.asarray(generator, dtype=int) for generator in generators]
    identity = np.eye(3, dtype=int)
    elements = [identity]
    seen = {tuple(int(value) for value in identity.reshape(-1))}
    frontier = [identity]
    while frontier:
        current = frontier.pop(0)
        for generator in arrays:
            for candidate in (generator @ current, current @ generator):
                key = tuple(int(value) for value in candidate.reshape(-1))
                if key in seen:
                    continue
                seen.add(key)
                elements.append(candidate)
                frontier.append(candidate)
    return elements


def _section_signs(
    rotations: list[np.ndarray],
    unitary_rotations: np.ndarray,
    crystal_system: str,
) -> list[int]:
    from qten.phys.spin import proper_rotation_matrix, su2_from_so3

    signs: list[int] = []
    for rotation, spgrep_lift in zip(rotations, unitary_rotations):
        cartesian = _cartesianize_generator(
            sy.ImmutableDenseMatrix(rotation.tolist()), crystal_system
        )
        qten_lift = np.asarray(
            [
                [complex(sy.N(entry)) for entry in row]
                for row in su2_from_so3(proper_rotation_matrix(cartesian)).tolist()
            ],
            dtype=complex,
        )
        overlap = np.trace(spgrep_lift.conj().T @ qten_lift) / 2.0
        sign = 1 if overlap.real >= 0 else -1
        if not np.allclose(qten_lift, sign * spgrep_lift, rtol=0, atol=1e-8):
            raise ValueError(
                "spgrep and QTen SU(2) lifts are not related by a sign for "
                f"rotation {rotation.tolist()}."
            )
        signs.append(sign)
    return signs


def check_spgrep(data: dict[str, Any]) -> None:
    from spgrep import get_crystallographic_pointgroup_spinor_irreps_from_symmetry
    from spgrep.rep.representation import get_character

    for record in _three_d_records(data):
        print(f"  spgrep {record['symbol']}", flush=True)
        rotations = _close_integer_group(record["generators"])
        if record["crystal_system"] in {"trigonal", "hexagonal"}:
            lattice = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [-0.5, np.sqrt(3.0) / 2.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        else:
            lattice = np.eye(3)
        irreps, _factor_system, unitary_rotations = (
            get_crystallographic_pointgroup_spinor_irreps_from_symmetry(
                lattice,
                np.asarray(rotations, dtype=int),
                method="Neto",
            )
        )
        signs = _section_signs(rotations, unitary_rotations, record["crystal_system"])
        aligned = [
            np.asarray(get_character(irrep), dtype=complex) * np.asarray(signs)
            for irrep in irreps
        ]
        group = _group_from_3d(record)
        group = FinitePointGroup(
            generators=group.generators,
            axes=group.axes,
            symbol=group.symbol,
            irreps=group.irreps,
            spinor_irreps=record["spinor_irreps"],
            spin=group.spin,
        )
        class_index = group._class_label_index_by_element()
        n_class = len(record["irreps"]["class_labels"])
        from qten.pointgroups.finite import _matrix_key

        qten_keys = {
            _matrix_key(element.irrep): index
            for index, element in enumerate(group.elements())
        }
        spgrep_rows = []
        for character in aligned:
            values: list[list[complex]] = [[] for _ in range(n_class)]
            for cryst_index, rotation in enumerate(rotations):
                cartesian = _cartesianize_generator(
                    sy.ImmutableDenseMatrix(rotation.tolist()),
                    record["crystal_system"],
                )
                qten_index = qten_keys[_matrix_key(cartesian)]
                values[class_index[qten_index]].append(complex(character[cryst_index]))
            row = []
            for class_values in values:
                if not class_values:
                    row.append(0j)
                    continue
                mean = sum(class_values) / len(class_values)
                if any(abs(value - mean) > 1e-5 for value in class_values):
                    row.append(0j)
                else:
                    row.append(mean)
            spgrep_rows.append(row)
        packaged_rows = [
            [parse_class_character(value) for value in row["characters"]]
            for row in record["spinor_irreps"]["irreps"].values()
        ]
        if not _rows_match(packaged_rows, spgrep_rows, atol=1e-6):
            raise SystemExit(
                f"spgrep class averages do not match QTen spinor χ for {record['symbol']}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate the packaged catalog: ordinary Bilbao χ still matches "
            "the generated group, spinor χ matches the live QTen lift, and "
            "every spinor row has a DPG bilbao_label."
        ),
    )
    parser.add_argument(
        "--check-spgrep",
        action="store_true",
        help="Also compare 3D class averages to η-aligned spgrep characters.",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="With --check, skip recomputing spinor tables from the lift.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help=(
            "Fetch Representations DPG (or fall back to Koster Γ continuation) "
            "and write bilbao_label onto existing spinor χ without recomputing."
        ),
    )
    args = parser.parse_args()

    if args.check or args.check_spgrep:
        data = _load_catalog()
        check_self(data, live=not args.skip_live)
        if args.check_spgrep:
            check_spgrep(data)
        return

    if args.labels_only:
        data = _load_catalog()
        print("Fetching Bilbao REPRESENTATIONS DPG", flush=True)
        dpg_tables = fetch_dpg_tables(
            [record["symbol"] for record in _three_d_records(data)]
        )
        if not dpg_tables:
            print(
                "  DPG CGI down; assigning Koster Γ labels from the qten section",
                flush=True,
            )
        generated = relabel_existing(data, dpg_tables)
    else:
        generated = generate()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(
        json.dumps(generated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_provenance()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
