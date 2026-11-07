#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  echo "Set up SolveMCQ's Python environment using the Browser app instructions in README.md."
  read "?Press Enter to close."
  exit 1
fi
.venv/bin/python -m solvemcq.ui
