from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable doctor who provides accurate and eloquent yet simple answers to health related questions based on given context.",
            ),
            ("human", '''
Here are the relevant documents: 
{context}

Here is the question:
{question}
            '''),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    out_msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content, "context": "Patient should take vitamin c."},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await out_msg.stream_token(chunk)

    await out_msg.send()