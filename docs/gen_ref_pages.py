from __future__ import annotations

import ast
import posixpath
from pathlib import Path

import mkdocs_gen_files


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path("reference")
ROOT_SUMMARY = Path("SUMMARY.md")
PACKAGES = (
    ("qten", ROOT / "src" / "qten"),
    ("qten_plots", ROOT / "ext" / "plots" / "src" / "qten_plots"),
)
PLOT_METHOD_INDEX_MODULES = {"qten.plottings", "qten_plots"}


def relative_doc_link(from_doc_path: Path, target_doc_path: Path) -> str:
    return posixpath.relpath(
        target_doc_path.as_posix(),
        start=from_doc_path.parent.as_posix(),
    )


def write_module_page(module_name: str, doc_path: Path, source_path: Path) -> None:
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"# `{module_name}`\n\n")
        if source_path.name == "__init__.py":
            fd.write(f"Package reference for `{module_name}`.\n\n")
        else:
            fd.write(f"Module reference for `{module_name}`.\n\n")
        if module_name in PLOT_METHOD_INDEX_MODULES:
            plot_methods_link = relative_doc_link(doc_path, Path("plot-methods.md"))
            fd.write("## Registered Plot Methods\n\n")
            fd.write(
                "For the user-facing `obj.plot(...)` methods registered by "
                f"the plotting extension, see [Plot Methods]({plot_methods_link}).\n\n"
            )
        fd.write(f"::: {module_name}\n")
        if source_path.name == "__init__.py":
            exports = public_reexports(source_path)
            if exports:
                fd.write("\n## Exported API\n\n")
                for export_name in exports:
                    fd.write(f"::: {module_name}.{export_name}\n\n")

    mkdocs_gen_files.set_edit_path(doc_path, source_path.relative_to(ROOT))


def public_reexports(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text())
    exports: list[str] = []
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            export_name = alias.asname or alias.name
            if export_name.startswith("_"):
                continue
            if export_name in seen:
                continue
            seen.add(export_name)
            exports.append(export_name)
    return exports


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
    nav_file.write("* [Plot Methods](plot-methods.md)\n")
    nav_file.write("* [API Reference](reference/index.md)\n")
    for line in nav.build_literate_nav():
        nav_file.write(f"    {line}")
