#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
MAIN_DIST_DIR="${DIST_DIR}/qten"
PLOTS_DIST_DIR="${DIST_DIR}/qten-plots"
REPOSITORY="pypi"
UPLOAD=1

usage() {
    cat <<'EOF'
Usage: scripts/publish_pypi.sh [--testpypi] [--skip-upload]

Builds both distributions:
  - qten
  - qten-plots

Then validates them with twine and uploads them to PyPI or TestPyPI.

Environment:
  PYPI_TOKEN        Required for PyPI uploads
  TEST_PYPI_TOKEN   Required for TestPyPI uploads

Examples:
  scripts/publish_pypi.sh --skip-upload
  scripts/publish_pypi.sh --testpypi
  PYPI_TOKEN=... scripts/publish_pypi.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --testpypi)
            REPOSITORY="testpypi"
            shift
            ;;
        --skip-upload)
            UPLOAD=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH." >&2
    exit 1
fi

mkdir -p "${MAIN_DIST_DIR}" "${PLOTS_DIST_DIR}"
rm -f "${MAIN_DIST_DIR}"/* "${PLOTS_DIST_DIR}"/*

echo "Building qten"
(
    cd "${ROOT_DIR}"
    uv run --group dev python -m build --sdist --wheel --outdir "${MAIN_DIST_DIR}"
)

echo "Building qten-plots"
(
    cd "${ROOT_DIR}/ext/plots"
    uv run --group dev python -m build --sdist --wheel --outdir "${PLOTS_DIST_DIR}"
)

shopt -s nullglob
artifacts=(
    "${MAIN_DIST_DIR}"/*.tar.gz
    "${MAIN_DIST_DIR}"/*.whl
    "${PLOTS_DIST_DIR}"/*.tar.gz
    "${PLOTS_DIST_DIR}"/*.whl
)
shopt -u nullglob

if [[ ${#artifacts[@]} -eq 0 ]]; then
    echo "No distribution artifacts were produced." >&2
    exit 1
fi

echo "Validating artifacts with twine"
uv run --group dev twine check "${artifacts[@]}"

if [[ "${UPLOAD}" -eq 0 ]]; then
    echo "Build completed. Upload skipped."
    printf '%s\n' "${artifacts[@]}"
    exit 0
fi

if [[ "${REPOSITORY}" == "testpypi" ]]; then
    REPOSITORY_URL="https://test.pypi.org/legacy/"
    TOKEN="${TEST_PYPI_TOKEN:-}"
else
    REPOSITORY_URL="https://upload.pypi.org/legacy/"
    TOKEN="${PYPI_TOKEN:-}"
fi

if [[ -z "${TOKEN}" ]]; then
    echo "Missing token for ${REPOSITORY}. Set the appropriate environment variable." >&2
    if [[ "${REPOSITORY}" == "testpypi" ]]; then
        echo "Expected TEST_PYPI_TOKEN." >&2
    else
        echo "Expected PYPI_TOKEN." >&2
    fi
    exit 1
fi

echo "Uploading qten first"
TWINE_USERNAME="__token__" \
TWINE_PASSWORD="${TOKEN}" \
uv run --group dev twine upload \
    --repository-url "${REPOSITORY_URL}" \
    "${MAIN_DIST_DIR}"/*

echo "Uploading qten-plots second"
TWINE_USERNAME="__token__" \
TWINE_PASSWORD="${TOKEN}" \
uv run --group dev twine upload \
    --repository-url "${REPOSITORY_URL}" \
    "${PLOTS_DIST_DIR}"/*

echo "Publish completed to ${REPOSITORY}."
