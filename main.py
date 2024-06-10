from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.schema.runnable.config import RunnableConfig
from langchain_text_splitters.character import CharacterTextSplitter

from langchain.retrievers import MergerRetriever
from langchain_community.retrievers import WikipediaRetriever, PubMedRetriever
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_transformers.embeddings_redundant_filter import (
    EmbeddingsRedundantFilter,
)

from history import get_session_history

from retrievers import MedRagRetriever, MedCptEmbeddings
from prompts import main_prompt

from agents.yt_agent import YTAgent
import chainlit as cl

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



def agent_yt_execute(diagnosis):
    '''

    :param message: user input form Chainlit
    :return: result after running via CrewAI CoT using specified llm
    '''

    simple_yt_agent = YTAgent()
    crew = simple_yt_agent.get_crew()
    result = crew.kickoff(
        inputs = {'query': diagnosis}
    )
    return result

@cl.on_chat_start
async def on_chat_start():

    model = ChatOllama(streaming=True, model="llama3")
    qa = create_stuff_documents_chain(model, main_prompt)

    corpus_dir = "./corpus"
    retriever = MergerRetriever(
        retrievers=[
            MedRagRetriever(dataset="statpearls", corpus_dir=corpus_dir),
            MedRagRetriever(dataset="textbooks", corpus_dir=corpus_dir),
            PubMedRetriever(),
            WikipediaRetriever(),
        ]
    )

    embeddings = MedCptEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.6)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )

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
        {"input": message.content},
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler()],
            configurable={"session_id": cl.user_session.get("id")},
        ),
    ):
        # Ensure the chunk is JSON serializable because some chunks are not, they are objects of langchain like HumanMessage or something. We don't really need to stream them to user
        if isinstance(chunk, dict):
            if "answer" in chunk:
                chunk_str = chunk["answer"]
                await out_msg.stream_token(chunk_str)
    await out_msg.send()
    # kind of Q&A by CrewAI agents, after the summary provided to user
    ask_if_yotube_interested(out_msg.content)



@cl.step
async def ask_if_yotube_interested(diagnosis):
    res = await cl.AskUserMessage(content="Would you like to also add some Youtube video related to your situation?", timeout=10).send()
    if res:
        res_as_text = res['output'].lower()
        if 'yup' in res_as_text or 'yes' in res_as_text or 'sure' in res_as_text:
            videos = agent_yt_execute(diagnosis)
            await cl.Message(
                content=f"Here are some YT videos {videos}"
            ).send()
    else:
        await cl.Message(
            content="It looks like you are not interested. It's fine let's continue chatting"
        ).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
