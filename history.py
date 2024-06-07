
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return ChatMessageHistory() # Ephemeral