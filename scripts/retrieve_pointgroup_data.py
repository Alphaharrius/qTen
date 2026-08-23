"""Retrieve and normalize crystallographic point-group generator data.

This is a one-time data harvesting helper. It downloads pymatgen's structured
symmetry metadata and writes a compact qten-owned JSON file containing only the
point-group fields needed at runtime.
"""

from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from urllib.request import urlopen


PYMATGEN_VERSION = "v2026.5.18"
SYMM_DATA_URL = (
    "https://raw.githubusercontent.com/materialsproject/pymatgen-core/"
    f"{PYMATGEN_VERSION}/src/pymatgen/symmetry/symm_data.json"
)
BILBAO_POINT_TABLE_URL = "https://cryst.ehu.es/cgi-bin/rep/programs/sam/point.py"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "qten" / "pointgroups" / "data"
DATA_PATH = DATA_DIR / "point_group_data.json"
PROVENANCE_PATH = DATA_DIR / "PROVENANCE.md"


SCHOENFLIES_ALIASES = {
    "C1": "1",
    "Ci": "-1",
    "S2": "-1",
    "C2": "2",
    "Cs": "m",
    "C1h": "m",
    "C2h": "2/m",
    "D2": "222",
    "V": "222",
    "C2v": "mm2",
    "D2h": "mmm",
    "Vh": "mmm",
    "C4": "4",
    "S4": "-4",
    "C4h": "4/m",
    "D4": "422",
    "C4v": "4mm",
    "D2d": "-42m",
    "D4h": "4/mmm",
    "C3": "3",
    "C3i": "-3",
    "S6": "-3",
    "D3": "32",
    "C3v": "3m",
    "D3d": "-3m",
    "C6": "6",
    "C3h": "-6",
    "C6h": "6/m",
    "D6": "622",
    "C6v": "6mm",
    "D3h": "-6m2",
    "D6h": "6/mmm",
    "T": "23",
    "Th": "m-3",
    "O": "432",
    "Td": "-43m",
    "Oh": "m-3m",
}


BILBAO_POINT_GROUPS = {
    "1": {"num": 1, "sg": 1},
    "-1": {"num": 2, "sg": 2},
    "2": {"num": 3, "sg": 3},
    "m": {"num": 4, "sg": 6},
    "2/m": {"num": 5, "sg": 10},
    "222": {"num": 6, "sg": 16},
    "mm2": {"num": 7, "sg": 25},
    "mmm": {"num": 8, "sg": 47},
    "4": {"num": 9, "sg": 75},
    "-4": {"num": 10, "sg": 81},
    "4/m": {"num": 11, "sg": 83},
    "422": {"num": 12, "sg": 89},
    "4mm": {"num": 13, "sg": 99},
    "-42m": {"num": 14, "sg": 111},
    "4/mmm": {"num": 15, "sg": 123},
    "3": {"num": 16, "sg": 143},
    "-3": {"num": 17, "sg": 147},
    "32": {"num": 18, "sg": 149},
    "3m": {"num": 19, "sg": 156},
    "-3m": {"num": 20, "sg": 162},
    "6": {"num": 21, "sg": 168},
    "-6": {"num": 22, "sg": 174},
    "6/m": {"num": 23, "sg": 175},
    "622": {"num": 24, "sg": 177},
    "6mm": {"num": 25, "sg": 183},
    "-6m2": {"num": 26, "sg": 187},
    "6/mmm": {"num": 27, "sg": 191},
    "23": {"num": 28, "sg": 195},
    "m-3": {"num": 29, "sg": 200},
    "432": {"num": 30, "sg": 207},
    "-43m": {"num": 31, "sg": 215},
    "m-3m": {"num": 32, "sg": 221},
}


def _fetch_json(url: str) -> dict:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("ISO-8859-1")


def _cell_text(raw: str) -> str:
    raw = re.sub(r"<\s*sub\s*>(.*?)<\s*/\s*sub\s*>", r"\1", raw, flags=re.I | re.S)
    raw = re.sub(r"<\s*sup\s*>(.*?)<\s*/\s*sup\s*>", r"^\1", raw, flags=re.I | re.S)
    raw = re.sub(r"<[^>]+>", "", raw)
    return " ".join(unescape(raw).split())


def _cell_values(raw: str) -> list[str]:
    raw = re.sub(r"<\s*br\s*/?\s*>", "\n", raw, flags=re.I)
    return [_cell_text(part) for part in raw.split("\n")]


def _parse_number(text: str) -> int | str:
    try:
        return int(text)
    except ValueError:
        return text


def _parse_bilbao_character_table(html: str) -> dict:
    match = re.search(
        r"<table[^>]*>\s*<caption[^>]*>.*?Character Table.*?</table>",
        html,
        flags=re.I | re.S,
    )
    if match is None:
        raise ValueError("Failed to locate Bilbao character table.")

    table = match.group(0)
    rows = re.findall(r"<tr>(.*?)</tr>", table, flags=re.I | re.S)
    parsed_rows = []
    for row in rows:
        raw_cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)
        cell_values = [_cell_values(cell) for cell in raw_cells]
        width = max((len(values) for values in cell_values), default=1)
        for i in range(width):
            parsed_rows.append(
                [values[i] if len(values) > 1 else values[0] for values in cell_values]
            )
    parsed_rows = [row for row in parsed_rows if row]
    if len(parsed_rows) < 2:
        raise ValueError("Bilbao character table did not contain enough rows.")

    header = parsed_rows[0]
    has_functions_column = header[-1].lower() == "functions"
    class_labels = header[2:-1] if has_functions_column else header[2:]
    first_data_row = 1
    if parsed_rows[1][0].rstrip(".").lower() == "mult":
        multiplicities = [_parse_number(value) for value in parsed_rows[1][2:-1]]
        first_data_row = 2
    else:
        # Bilbao omits the multiplicity row for abelian groups because every
        # conjugacy class contains one element.
        multiplicities = [1] * len(class_labels)

    irreps = {}
    for row in parsed_rows[first_data_row:]:
        label = row[0]
        if len(row) < len(class_labels) + 2:
            continue
        characters = [_parse_number(value) for value in row[2 : 2 + len(class_labels)]]
        irreps[label] = {
            "bilbao_label": row[1],
            "dim": characters[0],
            "characters": characters,
            "functions": row[2 + len(class_labels)]
            if len(row) > 2 + len(class_labels)
            else "",
        }

    return {
        "source": "bilbao",
        "class_labels": class_labels,
        "multiplicities": multiplicities,
        "irreps": irreps,
    }


def _fetch_bilbao_irreps(symbol: str) -> dict:
    params = BILBAO_POINT_GROUPS[symbol]
    url = f"{BILBAO_POINT_TABLE_URL}?num={params['num']}&sg={params['sg']}"
    table = _parse_bilbao_character_table(_fetch_text(url))
    table["url"] = url
    return table


def _aliases_by_symbol() -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for alias, symbol in SCHOENFLIES_ALIASES.items():
        aliases.setdefault(symbol, []).append(alias)
    return {symbol: sorted(values) for symbol, values in aliases.items()}


def _point_group_records(raw: dict, *, with_bilbao: bool) -> list[dict]:
    aliases = _aliases_by_symbol()
    generator_matrices = raw["generator_matrices"]
    crystal_systems = raw["point_group_crystal_system_map"]

    records = []
    for symbol, encoding in sorted(raw["point_group_encoding"].items()):
        records.append(
            {
                "symbol": symbol,
                "aliases": aliases.get(symbol, []),
                "crystal_system": crystal_systems[symbol],
                "source_encoding": encoding,
                "generators": [generator_matrices[key] for key in encoding],
                "irreps": _fetch_bilbao_irreps(symbol) if with_bilbao else {},
            }
        )
    return records


def main() -> None:
    raw = _fetch_json(SYMM_DATA_URL)
    with_bilbao = True

    data = {
        "schema_version": 2,
        "source": {
            "name": "pymatgen-core",
            "version": PYMATGEN_VERSION,
            "url": SYMM_DATA_URL,
        },
        "point_groups": _point_group_records(raw, with_bilbao=with_bilbao),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    PROVENANCE_PATH.write_text(
        "\n".join(
            [
                "# Point-Group Data Provenance",
                "",
                "The normalized point-group generator data in `point_group_data.json`",
                f"was generated from pymatgen-core `{PYMATGEN_VERSION}`:",
                "",
                f"- {SYMM_DATA_URL}",
                f"- {BILBAO_POINT_TABLE_URL}?num=<point-group-number>&sg=<representative-space-group>",
                "",
                "The runtime JSON is normalized into one qten-owned record per",
                "crystallographic point group. Each record contains:",
                "",
                "- `symbol`",
                "- `aliases`",
                "- `crystal_system`",
                "- `source_encoding`",
                "- `generators`",
                "- `irreps` from Bilbao character tables",
                "",
                "Schoenflies aliases are added by qten for convenient lookup.",
                "Preserve pymatgen attribution and license terms when redistributing this data.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
