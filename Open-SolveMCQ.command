#!/bin/zsh
cd -- "${0:A:h}" || exit 1
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ ! -x .venv/bin/python ]]; then
  echo "Set up SolveMCQ's Python environment using the Browser app instructions in README.md."
  read "?Press Enter to close."
  exit 1
fi
.venv/bin/python -m solvemcq.ui
