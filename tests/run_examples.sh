#!/bin/bash


# Cause the script to exit if a single command fails
set -eo pipefail
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT/examples"
PYTHON="${PYTHON:-python}"
# rm catboostclassifier.py
rm keras_example.py # Keras example crashes randomly
rm custom_searcher_example.py # So we don't need to install HEBO during CI
for f in *.py; do echo "running $f" && $PYTHON "$f" || exit 1 ; done

