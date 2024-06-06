#!/bin/sh
source .venv/bin/activate
export OLLAMA_MODEL="llama3"
export CHAINLIT_PORT="${PORT:-8000}"
export DEBUG=1
export WATCH=1
export HF_HUB_ENABLE_HF_TRANSFER=1

printenv

nohup ollama serve &
sleep 2 && ollama pull $OLLAMA_MODEL

python -Xfrozen_modules=off main.py