#!/bin/bash
# ==============================================================================
# Humanized Bass Transcription Pipeline (pYIN / MusicXML Pro Edition)
# Native Apple Silicon (M1/M2/M3) & Linux Support
# ==============================================================================
set -euo pipefail

OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

echo "Detected OS: ${OS_TYPE} | Architecture: ${ARCH_TYPE}"

PYTHON_BIN=""

# ------------------------------------------------------------------------------
# 1. Platform-Specific System Dependency Provisioning
# ------------------------------------------------------------------------------
if [[ "$OS_TYPE" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    if ! brew ls --versions python@3.11 &> /dev/null; then
        echo "Installing Python 3.11..."
        brew install python@3.11 || true
    fi

    if [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
        PYTHON_BIN="/opt/homebrew/opt/python@3.11/bin/python3.11"
    else
        PYTHON_BIN="$(command -v python3.11 || command -v python3)"
    fi

elif [[ "$OS_TYPE" == "Linux" ]]; then
    if command -v python3.11 &> /dev/null; then
        PYTHON_BIN="$(command -v python3.11)"
    elif command -v python3 &> /dev/null; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "Installing Python 3..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y python3 python3-pip ffmpeg
        fi
        PYTHON_BIN="$(command -v python3)"
    fi
else
    echo "FATAL: Unsupported Operating System: ${OS_TYPE}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Virtual Environment Setup & Dependencies
# ------------------------------------------------------------------------------
ENV_DIR="${PWD}/.bass_pipeline_env"

if [ ! -d "$ENV_DIR" ]; then
    echo "Provisioning isolated Python environment..."
    "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/pip" install --upgrade pip --quiet
"$ENV_DIR/bin/pip" install --quiet -r requirements.txt

# ------------------------------------------------------------------------------
# 3. Execution
# ------------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    echo "Usage: ./humanbass.sh [-l|--all-levels] <path_to_stem_folder_1> ..."
    exit 1
fi

"$ENV_DIR/bin/python" src/main.py "$@"
