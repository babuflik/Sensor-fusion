#!/usr/bin/env bash
set -euo pipefail

# Creates a virtual environment in `.venv` (or $VENV_DIR) and installs
# `requirements.txt`. If you source this script, it will activate the venv
# in your current shell. If you run it normally, it prints the activation
# command when finished.

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

if [ ! -x "$(command -v "$PYTHON")" ]; then
  echo "Python executable '$PYTHON' not found." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

echo "Upgrading pip and installing requirements (if present)"
"$VENV_DIR/bin/pip" install -U pip
if [ -f requirements.txt ]; then
  "$VENV_DIR/bin/pip" install -r requirements.txt
else
  echo "No requirements.txt found in $(pwd); skipping pip install"
fi

# If the script is sourced, $0 != $BASH_SOURCE evaluates true when sourced
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  # being sourced; activate in current shell
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  echo "Activated virtualenv: $VENV_DIR"
else
  echo "Virtualenv ready at: $VENV_DIR"
  echo "Activate it with: source $VENV_DIR/bin/activate"
  echo "To create+activate in one step, run: source tools/setup_venv.sh"
fi
