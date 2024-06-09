from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_text_splitters.character import CharacterTextSplitter

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter

from history import get_session_history

from retrievers import MedRagRetriever

import chainlit as cl

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from typing import Dict


@cl.on_chat_start
async def on_chat_start():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Your name is MedBot. 
You are a smart assistant for question-answering tasks about health topics.
Use the context content provided to answer the question but do not refer to it for example do not use phrases like 'based on the context', 'according to the context', 'information provided earlier', 'given the context you provided', etc.
Summarize the contents providing the most educational answer possible.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.
You are not a real doctor or healthcare professional.
Carry on the conversation with the user with follow up questions about their health or health topics.
Ask if the user has a history or family history with the illness or related symptoms.
Always append the following disclaimer at the end of your message: > __**Note:** The content on this site is for informational or educational purposes only, might not be factual and does not substitute professional medical advice or consultations with healthcare professionals.__

<context>
{context}
</context>
                """,
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", 'Question: {input}'),
        ]
    )

    model = ChatOllama(streaming=True, model="llama3")
    qa = create_stuff_documents_chain(model, prompt)

    retriever = MedRagRetriever(dataset="textbooks", corpus_dir = './corpus')

    embeddings = retriever.embeddings()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.6)
    pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, redundant_filter, relevant_filter])
    compressed_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

    chain = create_retrieval_chain(compressed_retriever, qa)
    
    runnable = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    out_msg = cl.Message(content="")
    
    async for chunk in runnable.astream(
        {"input": message.content },
        config=RunnableConfig(
                callbacks=[cl.LangchainCallbackHandler()], 
                configurable={'session_id': cl.user_session.get("id")}
            ),
    ):
        # Ensure the chunk is JSON serializable because some chunks are not, they are objects of langchain like HumanMessage or something. We don't really need to stream them to user
        if isinstance(chunk, dict):
            if 'answer' in chunk:
                chunk_str = chunk['answer']
                await out_msg.stream_token(chunk_str)
    await out_msg.send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
