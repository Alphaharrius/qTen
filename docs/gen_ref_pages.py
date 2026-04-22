from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path("reference")
ROOT_SUMMARY = Path("SUMMARY.md")
PACKAGES = (
    ("qten", ROOT / "src" / "qten"),
    ("qten_plots", ROOT / "ext" / "plots" / "src" / "qten_plots"),
)


def write_module_page(module_name: str, doc_path: Path, source_path: Path) -> None:
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        title = module_name.split(".")[-1]
        fd.write(f"# `{module_name}`\n\n")
        if source_path.name == "__init__.py":
            fd.write(f"Package reference for `{module_name}`.\n\n")
        else:
            fd.write(f"Module reference for `{module_name}`.\n\n")
        fd.write(f"::: {module_name}\n")

    mkdocs_gen_files.set_edit_path(doc_path, source_path.relative_to(ROOT))


nav = mkdocs_gen_files.Nav()

for package_name, package_root in PACKAGES:
    for source_path in sorted(package_root.rglob("*.py")):
        module_parts = source_path.relative_to(package_root).with_suffix("").parts
        if module_parts[-1] == "__main__":
            continue
        if module_parts[-1].startswith("_") and module_parts[-1] != "__init__":
            continue

        if module_parts[-1] == "__init__":
            doc_rel_path = Path(package_name, *module_parts[:-1], "index.md")
            nav_parts = (package_name, *module_parts[:-1])
            module_name = ".".join((package_name, *module_parts[:-1]))
        else:
            doc_rel_path = Path(package_name, *module_parts).with_suffix(".md")
            nav_parts = (package_name, *module_parts)
            module_name = ".".join((package_name, *module_parts))

        doc_path = REFERENCE_ROOT / doc_rel_path
        write_module_page(module_name, doc_path, source_path)
        nav[nav_parts] = Path("reference") / doc_rel_path

with mkdocs_gen_files.open(REFERENCE_ROOT / "index.md", "w") as fd:
    fd.write("# API Reference\n\n")
    fd.write(
        "Generated module and package reference for the `qten` core package "
        "and the `qten_plots` extension package.\n"
    )

with mkdocs_gen_files.open(ROOT_SUMMARY, "w") as nav_file:
    nav_file.write("* [Home](index.md)\n")
    nav_file.write("* [Build Docs](build-docs.md)\n")
    nav_file.write("* [API Reference](reference/index.md)\n")
    for line in nav.build_literate_nav():
        nav_file.write(f"    {line}")
