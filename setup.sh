if [ ! -f .env ]; then
    echo ".env not found, see README.md"
    exit 1
fi

# setup venv
python -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt

# patch socketio packet json encoding so that json._default_encoder is used
sed -i "s/, separators=(',', ':'))/)/g" .venv/lib/python*/site-packages/socketio/packet.py

# start ollama an ensure model
export OLLAMA_MODEL="llama3"
nohup ollama serve &
sleep 2 && ollama pull $OLLAMA_MODEL

# disabled dataset fetching since on deployment corpus is mounted
# you can prefetch specific datasets running the following
# export HF_HUB_ENABLE_HF_TRANSFER=1
# python hf_fetch.py
