# MedBot
A prototype of a chatbot based-off [MedRAG](https://huggingface.co/datasets/MedRAG/).

## Setup

### .env
0. Create a .env file
0. Define your HuggingFace API Key as follows
```
HF_TOKEN=...
```

### setup.sh
0. Run `./setup.sh` to configure the environment and fetch dependencies

### devserver.sh
0. Run `./devserver.sh` to start the server

### Run locally 
1. Download https://www.ollama.com/ and then `ollama` command line tool
2. ```ollama pull llama3```
3. run `chainlit run main.py -w` command from your project root

### For big files commit

### Example:
-  `brew install git-lfs`
-  `git lfs install`
-  `git lfs track "my_index_file_large.faiss"`
-  `git add my_index_file_large.faiss`
-  `git commit -m "Add large file`