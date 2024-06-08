# MedBot
A prototype of a chatbot based-off [MedRAG](https://huggingface.co/datasets/MedRAG/).

## Setup

### .env
0. Create a .env file
0. Define your OpenAI API Key as follows
```
OPENAI_API_KEY=sk-...
```
2. Download https://www.ollama.com/ and then `ollama` command line tool
3. ```ollama run llama3```
4.  run `chainlit run main.py -w` command from your project root


## For big file commit!

### Example:
-  `brew install git-lfs`
-  `git lfs install`
-  `git lfs track "my_index_file_large.faiss"`
-  `git add my_index_file_large.faiss`
-  `git commit -m "Add large file`