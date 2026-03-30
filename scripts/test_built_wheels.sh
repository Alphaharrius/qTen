#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_PARENT="${ROOT_DIR}/.tmp"
TMP_DIR=""
KEEP_TMP=0
INSTALL_PLOTS=1
USE_CUDA=0
CUDA_VERSION_OVERRIDE=""

usage() {
    cat <<'EOF'
Usage: scripts/test_built_wheels.sh [--keep] [--no-plots] [--cuda] [--cuda-version X.Y] [--] [pytest args...]

Builds local wheel artifacts, creates an isolated uv virtual environment under
.tmp/, installs the wheel(s), and runs the repository tests against the
installed packages instead of the source checkout.

Options:
  --keep       Keep the temporary environment directory after the run
  --no-plots   Do not build or install qten-plots; skips tests/test_plots.py
  --cuda       Preinstall a CUDA-enabled torch wheel chosen from nvidia-smi
  --cuda-version
               Override CUDA detection with an explicit version like 12.8
  -h, --help   Show this help text

Examples:
  scripts/test_built_wheels.sh
  scripts/test_built_wheels.sh -- -q tests/test_abelian.py
  scripts/test_built_wheels.sh --no-plots -- -q
  scripts/test_built_wheels.sh --cuda -- -q tests/test_gpu.py
EOF
}

version_to_int() {
    local version="$1"
    local major minor
    IFS='.' read -r major minor <<<"${version}"
    if [[ -z "${major:-}" || -z "${minor:-}" ]]; then
        return 1
    fi
    printf '%d%02d\n' "${major}" "${minor}"
}

detect_cuda_version() {
    if [[ -n "${CUDA_VERSION_OVERRIDE}" ]]; then
        printf '%s\n' "${CUDA_VERSION_OVERRIDE}"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required when --cuda is used." >&2
        return 1
    fi
    nvidia-smi | sed -nE 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/p' | head -n 1
}

select_torch_cuda_tag() {
    local detected_version="$1"
    local detected_int supported_version supported_tag supported_int
    local supported=(
        "13.0 cu130"
        "12.9 cu129"
        "12.8 cu128"
        "12.6 cu126"
    )

    detected_int="$(version_to_int "${detected_version}")" || return 1
    for entry in "${supported[@]}"; do
        read -r supported_version supported_tag <<<"${entry}"
        supported_int="$(version_to_int "${supported_version}")" || return 1
        if (( detected_int >= supported_int )); then
            printf '%s\n' "${supported_tag}"
            return 0
        fi
    done

    echo "CUDA ${detected_version} is older than the supported PyTorch CUDA wheel channels in this script." >&2
    return 1
}

PYTEST_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep)
            KEEP_TMP=1
            shift
            ;;
        --no-plots)
            INSTALL_PLOTS=0
            shift
            ;;
        --cuda)
            USE_CUDA=1
            shift
            ;;
        --cuda-version)
            USE_CUDA=1
            CUDA_VERSION_OVERRIDE="${2:-}"
            if [[ -z "${CUDA_VERSION_OVERRIDE}" ]]; then
                echo "--cuda-version requires a value like 12.8." >&2
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            PYTEST_ARGS=("$@")
            break
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH." >&2
    exit 1
fi

mkdir -p "${TMP_PARENT}" "${ROOT_DIR}/dist/qten"
TMP_DIR="$(mktemp -d "${TMP_PARENT}/wheel-test.XXXXXX")"
VENV_DIR="${TMP_DIR}/.venv"
MAIN_DIST_DIR="${ROOT_DIR}/dist/qten"
MAIN_WHEEL=""
PLOTS_DIST_DIR="${ROOT_DIR}/dist/qten-plots"
PLOTS_WHEEL=""
MAIN_VERSION="$(sed -nE 's/^version = "([^"]+)"/\1/p' "${ROOT_DIR}/pyproject.toml" | head -n 1)"
PLOTS_VERSION="$(sed -nE 's/^version = "([^"]+)"/\1/p' "${ROOT_DIR}/ext/plots/pyproject.toml" | head -n 1)"
TORCH_SPEC="$(sed -nE '/^dependencies = \[/,/^\]/ s/^[[:space:]]*"((torch[^"]*))",?/\1/p' "${ROOT_DIR}/pyproject.toml" | head -n 1)"
TORCH_CUDA_VERSION=""
TORCH_CUDA_TAG=""
TORCH_INDEX_URL=""

if [[ -z "${MAIN_VERSION}" ]]; then
    echo "Failed to read qten version from ${ROOT_DIR}/pyproject.toml." >&2
    exit 1
fi

if [[ ${INSTALL_PLOTS} -eq 1 && -z "${PLOTS_VERSION}" ]]; then
    echo "Failed to read qten-plots version from ${ROOT_DIR}/ext/plots/pyproject.toml." >&2
    exit 1
fi

if [[ ${USE_CUDA} -eq 1 && -z "${TORCH_SPEC}" ]]; then
    echo "Failed to read the torch dependency from ${ROOT_DIR}/pyproject.toml." >&2
    exit 1
fi

cleanup() {
    if [[ ${KEEP_TMP} -eq 0 && -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
        rm -rf "${TMP_DIR}"
    fi
}
trap cleanup EXIT

echo "Building qten wheel"
uv run --group dev python -m build --wheel --outdir "${MAIN_DIST_DIR}" "${ROOT_DIR}"

MAIN_WHEEL="${MAIN_DIST_DIR}/qten-${MAIN_VERSION}-py3-none-any.whl"
if [[ -z "${MAIN_WHEEL}" ]]; then
    echo "Failed to locate the built qten wheel in ${MAIN_DIST_DIR}." >&2
    exit 1
fi
if [[ ! -f "${MAIN_WHEEL}" ]]; then
    echo "Expected qten wheel not found: ${MAIN_WHEEL}" >&2
    exit 1
fi

if [[ ${INSTALL_PLOTS} -eq 1 ]]; then
    mkdir -p "${PLOTS_DIST_DIR}"
    echo "Building qten-plots wheel"
    uv run --group dev python -m build --wheel --outdir "${PLOTS_DIST_DIR}" "${ROOT_DIR}/ext/plots"
    PLOTS_WHEEL="${PLOTS_DIST_DIR}/qten_plots-${PLOTS_VERSION}-py3-none-any.whl"
    if [[ -z "${PLOTS_WHEEL}" ]]; then
        echo "Failed to locate the built qten-plots wheel in ${PLOTS_DIST_DIR}." >&2
        exit 1
    fi
    if [[ ! -f "${PLOTS_WHEEL}" ]]; then
        echo "Expected qten-plots wheel not found: ${PLOTS_WHEEL}" >&2
        exit 1
    fi
fi

echo "Creating isolated uv environment in ${TMP_DIR}"
uv venv "${VENV_DIR}"

if [[ ${USE_CUDA} -eq 1 ]]; then
    TORCH_CUDA_VERSION="$(detect_cuda_version)"
    if [[ -z "${TORCH_CUDA_VERSION}" ]]; then
        echo "Failed to detect a CUDA version from nvidia-smi." >&2
        exit 1
    fi
    TORCH_CUDA_TAG="$(select_torch_cuda_tag "${TORCH_CUDA_VERSION}")"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

    echo "Installing ${TORCH_SPEC} from ${TORCH_INDEX_URL} for CUDA ${TORCH_CUDA_VERSION}"
    uv pip install --python "${VENV_DIR}/bin/python" --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"
fi

INSTALL_ARGS=(
    --python "${VENV_DIR}/bin/python"
    pytest
    "${MAIN_WHEEL}"
)

if [[ ${INSTALL_PLOTS} -eq 1 ]]; then
    INSTALL_ARGS+=("${PLOTS_WHEEL}")
fi

echo "Installing wheels into isolated environment"
uv pip install "${INSTALL_ARGS[@]}"

TEST_TARGETS=("${ROOT_DIR}/tests")
if [[ ${INSTALL_PLOTS} -eq 0 ]]; then
    TEST_TARGETS+=("--ignore=${ROOT_DIR}/tests/test_plots.py")
fi

echo "Running tests against installed wheel(s)"
PYTEST_CMD=("${VENV_DIR}/bin/python" -m pytest -q "${TEST_TARGETS[@]}")
if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    PYTEST_CMD+=("${PYTEST_ARGS[@]}")
fi

(
    cd "${TMP_DIR}"
    "${PYTEST_CMD[@]}"
)

if [[ ${KEEP_TMP} -eq 1 ]]; then
    echo "Kept temporary environment at ${TMP_DIR}"
fi
