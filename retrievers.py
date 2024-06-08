import os

from pydantic import BaseModel, ConfigDict

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import os
import faiss
import json
import torch
import tqdm
import numpy as np
import json

class CompositeRetriever(BaseRetriever):
    
    retrievers: List[BaseRetriever] = []
    def __init__(self, retrievers: List[BaseRetriever]):
        super().__init__()
        self.retrievers = retrievers

    def _get_relevant_documents(
        self, query: str, *args, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        matching_documents = []
        for ret in self.retrievers:
            docs = ret.get_relevant_documents(query=query, run_manager=run_manager)
            matching_documents.append(docs)
        return matching_documents                

class CustomSentenceTransformer(SentenceTransformer):
    def _load_auto_model(self, model_name_or_path):
      transformer_model = Transformer(model_name_or_path)
      # change the default pooling "MEAN" to "CLS"
      # See https://datascience.stackexchange.com/questions/66207/what-is-purpose-of-the-cls-token-and-why-is-its-encoding-output-important
      pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
      return [transformer_model, pooling_model]

class MedCptEmbeddings(BaseModel, Embeddings): 
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str = "ncbi/MedCPT"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    query_embedding_function: CustomSentenceTransformer = CustomSentenceTransformer("ncbi/MedCPT-Query-Encoder", device=device)
    article_embedding_function: CustomSentenceTransformer = CustomSentenceTransformer("ncbi/MedCPT-Article-Encoder", device=device)

    def __init__(self):
        super().__init__()

        self.query_embedding_function.eval()
        self.article_embedding_function.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            return [self.article_embedding_function.encode(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        with torch.no_grad():
            return self.query_embedding_function.encode(question)

def _idx2txt(indices, chunk_dir):
   return [json.loads(open(os.path.join(chunk_dir, i["source"]+".jsonl")).read().strip().split('\n')[i["index"]]) for i in indices]

class MedRag():
    
    def __init__(self, dataset="textbooks", corpus_dir = './corpus'):
        super().__init__()

        import hf_fetch
        assert dataset in hf_fetch._datasets

        self.dataset_name = dataset
        self.embeddings = MedCptEmbeddings()

        self.index_dir = os.path.join(corpus_dir, dataset, "index", "ncbi/MedCPT-Article-Encoder")
        self.chunk_dir = os.path.join(corpus_dir, dataset, "chunk")

        self.index = faiss.read_index(os.path.join(self.index_dir, "faiss.index"))
        self.metadatas = [json.loads(line) for line in open(os.path.join(self.index_dir, "metadatas.jsonl")).read().strip().split('\n')]

    def retrieve(self, question, num_snippets=32):
        with torch.no_grad():
            query_embed = self.embeddings.query_embedding_function.encode(question)
            result = self.index.search(np.array([query_embed]), k=num_snippets)

            indices = [self.metadatas[i] for i in result[1][0]]
            texts = _idx2txt(indices, self.chunk_dir)
            scores = result[0][0].tolist()
            return texts, scores

    def query(self, question, num_snippets=32):
        retrieved_snippets, scores = self.retrieve(question, num_snippets)
        contexts = [
            "Document [{:d}] (Title: {:s}) (Score: {:.2f}) {:s}".format(
                idx, 
                retrieved_snippets[idx]["title"], 
                scores[idx],
                retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))] if len(retrieved_snippets) > 0 else []
        return "\n".join(contexts)

class DocumentEncoder(json.JSONEncoder):

  def __init__(self):
    self.__dict__.update(json._default_encoder.__dict__)
    self._default_encoder = json._default_encoder

  def default(self, obj):
    if isinstance(obj, Document):
      return {
        "page_content": obj.page_content,
        "metadata": obj.metadata
      }
    return self._default_encoder.default(self, obj)

# Workaround chainlit/socketio serialization
json._default_encoder = DocumentEncoder()

class MedRagRetriever(BaseRetriever):
    medrag: MedRag = None

    def __init__(self, dataset="textbooks", corpus_dir = './corpus'):
        super().__init__()
        self.medrag = MedRag(dataset, corpus_dir)

    def _get_relevant_documents(
        self, query: str, *args, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        retrieved_snippets, scores = self.medrag.retrieve(query, num_snippets=3) # top 3 snippets
        documents = [
            Document(
                page_content=retrieved_snippets[idx]["content"], 
                metadata={"title":retrieved_snippets[idx]["title"], "idx": idx, "score": scores[idx]}
            ) for idx in range(len(retrieved_snippets))] if len(retrieved_snippets) > 0 else []
        
        return documents

    def embeddings(self):
        return self.medrag.embeddings()

if __name__ == "__main__":
    print(MedRagRetriever(dataset="textbooks", corpus_dir = './corpus').get_relevant_documents("covid"))