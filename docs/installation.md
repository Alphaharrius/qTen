# Installation

QTen requires Python 3.11 or newer.

## Install PyTorch

Install PyTorch first, choosing the CPU or CUDA build that matches your platform.
Use the official PyTorch installation command for your system.

For a basic `uv` environment:

```bash
uv add torch
```

## Install QTen

With `uv`:

```bash
uv add qten
```

With `pip`:

```bash
pip install qten
```

## Optional Plotting Support

Install `qten-plots` if you want the registered `obj.plot(...)` helpers.

With `uv`:

```bash
uv add qten-plots
```

With `pip`:

```bash
pip install qten-plots
```

## Development Install

From a clone of this repository, create a CPU-only development environment with:

```bash
uv sync --extra cpu --group dev
```

For CUDA development, replace `cpu` with one of the CUDA extras configured by
the repository:

```bash
uv sync --extra cu126 --group dev
uv sync --extra cu128 --group dev
uv sync --extra cu129 --group dev
uv sync --extra cu130 --group dev
```

To include the documentation toolchain as well:

```bash
uv sync --extra cpu --group dev --group docs
```
