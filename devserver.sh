#!/bin/sh
source .venv/bin/activate
export CHAINLIT_PORT="${PORT:-8000}"
export DEBUG=1
export WATCH=1
export TOKENIZERS_PARALLELISM=true

printenv

python main.py
