#!/bin/sh
source .venv/bin/activate
export CHAINLIT_PORT="${PORT:-8000}"
export DEBUG=1
export WATCH=1

printenv

python main.py