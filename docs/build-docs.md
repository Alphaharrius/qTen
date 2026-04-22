# Build The HTML Docs

Run all commands from the repository root.

## Install Dependencies

The docs build imports the actual Python packages, so the environment needs the
core package dependencies as well as the MkDocs toolchain.

For a CPU-only local environment:

```bash
uv sync --extra cpu --group dev --group docs
```

If you are already using one of the CUDA extras in this repository, replace
`--extra cpu` with the CUDA extra you normally develop against.

## Preview Locally

Start the live-reloading docs server:

```bash
uv run mkdocs serve
```

MkDocs will print a local URL, usually `http://127.0.0.1:8000/`.

## Build Static HTML

Generate the static HTML site:

```bash
uv run mkdocs build
```

The built site is written to `site/`. The main entry point is:

```text
site/index.html
```

## How The API Pages Are Generated

The file `docs/gen_ref_pages.py` walks both package trees:

- `src/qten`
- `ext/plots/src/qten_plots`

For each importable module, it generates a Markdown stub that contains a
`mkdocstrings` directive. During the MkDocs build, `mkdocstrings` imports that
module and renders the API documentation from its NumPy-style docstrings and
signatures.
