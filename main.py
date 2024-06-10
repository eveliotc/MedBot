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

from agents.rag_agent import RagAgent
import chainlit as cl

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



def agent_execute(message):
    '''

    :param message: user input form Chainlit
    :return: result after running via CrewAI CoT using specified llm
    '''

    simple_agent = RagAgent()
    crew = simple_agent.get_crew()
    result = crew.kickoff(
        inputs = {'query': message}
    )
    return result

@cl.on_chat_start
async def on_chat_start():

    # await cl.sleep(5)
    # res = await cl.AskUserMessage(content="Hi what is your name", timeout=10).send()
    # if res:
    #  await cl.Message(
    #     content=f"Your name is: {res['output']}",
    # ).send()
    # else:
    #     await cl.Message(
    #         content=f"I look like I did not receive any information",
    #     ).send()



    model = ChatOllama(streaming=True, model="llama3")
    qa = create_stuff_documents_chain(model, main_prompt)

    corpus_dir = "./corpus"
    retriever = MergerRetriever(
        retrievers=[
            #MedRagRetriever(dataset="statpearls", corpus_dir=corpus_dir),
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


@cl.step
async def tool():
    # Simulate a running task
    await cl.sleep(2)

    return "Here are some Yotube videos for you"

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    out_msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"input": message.content},
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler()],
            configurable={"session_id": cl.user_session.get("id")},
            #history=get_session_history(cl.user_session.get("id"))
        ),
    ):
        # Ensure the chunk is JSON serializable because some chunks are not, they are objects of langchain like HumanMessage or something. We don't really need to stream them to user
        if isinstance(chunk, dict):
            if "answer" in chunk:
                chunk_str = chunk["answer"]
                await out_msg.stream_token(chunk_str)
    await out_msg.send()

    await cl.Message(content="btw here are some ytube videos").send()


'''
#this is Crew AI implementation based on user input. Will merge with rest of the code later
@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content
    response = agent_execute(user_input)
    await cl.Message(response).send()
'''

if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
