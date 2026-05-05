#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


def _extract_torch_spec(pyproject: dict) -> str | None:
    optional = pyproject.get("project", {}).get("optional-dependencies", {})
    specs = {
        dep
        for deps in optional.values()
        for dep in deps
        if isinstance(dep, str) and re.match(r"^torch([<>=!~].*)?$", dep)
    }
    if not specs:
        return None
    if len(specs) != 1:
        raise SystemExit(f"Expected one torch spec, found: {sorted(specs)}")
    return next(iter(specs))


def _replace_optional(text: str, pattern: str, repl: str) -> tuple[str, int]:
    updated, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE | re.DOTALL)
    return updated, count


def _replace_once(text: str, pattern: str, repl: str) -> str:
    updated, count = _replace_optional(text, pattern, repl)
    if count != 1:
        raise SystemExit(f"Expected one match for pattern: {pattern}")
    return updated


def rewrite_pyproject(path: Path) -> None:
    original = path.read_text()
    parsed = tomllib.loads(original)
    torch_spec = _extract_torch_spec(parsed)

    if torch_spec is None:
        if any(
            isinstance(dep, str) and re.match(r"^torch([<>=!~].*)?$", dep)
            for dep in parsed.get("project", {}).get("dependencies", [])
        ):
            return
        raise SystemExit("No torch dependency found under [project.optional-dependencies].")

    if torch_spec not in parsed.get("project", {}).get("dependencies", []):
        original = _replace_once(
            original,
            r'(?ms)^dependencies = \[\n(?P<body>.*?)^\]',
            lambda m: f'dependencies = [\n{m.group("body")}    "{torch_spec}",\n]',
        )

    original, _ = _replace_optional(
        original,
        r'(?ms)^\[project\.optional-dependencies\]\n.*?(?=^\[)',
        "",
    )
    original, _ = _replace_optional(
        original,
        r'(?ms)^conflicts = \[\n.*?^\]\n\n',
        "",
    )
    original, _ = _replace_optional(
        original,
        r'(?ms)^torch = \[\n.*?^\]\n\n',
        "",
    )
    original, _ = _replace_optional(
        original,
        r'(?ms)^torchvision = \[\n.*?^\]\n\n',
        "",
    )

    for index_name in ("pytorch-cu126", "pytorch-cu128", "pytorch-cu129", "pytorch-cu130", "pytorch-cpu"):
        original, _ = _replace_optional(
            original,
            rf'(?ms)^\[\[tool\.uv\.index\]\]\nname = "{re.escape(index_name)}"\n.*?(?=^\[\[tool\.uv\.index\]\]|^\[)',
            "",
        )

    path.write_text(original.rstrip() + "\n")


def rewrite_tox_ini(path: Path) -> None:
    original = path.read_text()
    updated = original.replace("uv sync --active --extra cpu", "uv sync --active")
    path.write_text(updated)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pyproject.toml")
    target = target.resolve()
    rewrite_pyproject(target)
    tox_ini = target.parent / "tox.ini"
    if tox_ini.exists():
        rewrite_tox_ini(tox_ini)


if __name__ == "__main__":
    main()
