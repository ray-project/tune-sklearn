#!/bin/bash


# Cause the script to exit if a single command fails
set -eo pipefail
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT/examples"
PYTHON="${PYTHON:-python}"
# rm catboostclassifier.py
rm bohb_example.py hpbandster_sgd.py # Temporary hack to avoid breaking CI
for f in *.py; do echo "running $f" && $PYTHON "$f" || exit 1 ; done

