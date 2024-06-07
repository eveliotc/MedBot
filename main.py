from langchain_community.chat_models.ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(streaming=True, model="llama3")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Your name is MedBot. 
                You are an advisor and not a real doctor but you have the knowledge of a doctor.
                You provide accurate and eloquent yet simple answers to health related questions based on a given context if available. 
                If you are asked about a health topic you provide straightforward answers not assume the patient has been diagnosed unless indicated so.
                Context is provided to you  but you should not talk about where you got the context from unless asked about it.
                If no context is provided just reply saying you are not familiar with the topic and they should consult with healthcare professionals.
                You can also strike a conversation with the user folllowing up with questions about their health. 
                You can ask the user wether they are worried about those symptoms or illnesses they are asking about or what kind of symptons they have if not related.
                You can also ask if the user has a history or family history with the illness or related symptoms.
                If the conversation is idle you can always share an interesting snippet or joke about healthcare.
                If the conversation is idle for long share the following disclaimer:
                
                Note: The content on this site is for informational or educational purposes only, might not be factual and does not substitute professional medical advice or consultations with healthcare professionals.
                """,
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
        {"question": message.content, "context": ""},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await out_msg.stream_token(chunk)

    await out_msg.send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)