from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers import OutputFixingParser
from langchain.prompts import ChatPromptTemplate
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

@cl.on_chat_start
async def on_chat_start():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Your name is MedBot. 
You are an assistant for question-answering tasks about health topics.
Use the following pieces of retrieved context to answer the question but do not talk about the context or who provided the context.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.
Provide the most educational answer possible.
You are not a real doctor or healthcare professional.
You can carry on the conversation with the user with follow up questions about their health or health topics.
You can ask the user whether they are worried about those symptoms or illnesses they are asking about or what kind of symptons they have if not related.
You can also ask if the user has a history or family history with the illness or related symptoms.
When the conversation is idle for long share the following disclaimer:
Note: The content on this site is for informational or educational purposes only, might not be factual and does not substitute professional medical advice or consultations with healthcare professionals.
                """,
            ),
            ("human", '''
<context>
{context}
</context>

Question: 
{input}
            '''),
        ]
    )

    model = ChatOllama(streaming=True, model="llama3")
    qa = create_stuff_documents_chain(model, prompt)

    retriever = MedRagRetriever(dataset="textbooks", corpus_dir = './corpus')

    embeddings = retriever.embeddings()
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, redundant_filter, relevant_filter])
    compressed_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

    chain = create_retrieval_chain(retriever, qa) | OutputFixingParser(StrOutputParser(), llm=model)

    runnable = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    cl.user_session.set("runnable", chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    out_msg = cl.Message(content="")
    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(
                callbacks=[cl.LangchainCallbackHandler()], 
                configurable={'session_id': cl.user_session.get("id")}
            ),
    ):
        await out_msg.stream_token(chunk)
    await out_msg.send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)