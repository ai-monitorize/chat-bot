from typing import Optional

from watson.chat import Chat, UserMessage, SystemMessage
from watson.llms.chat import ChatLLM


class ChatBot:
    def __init__(self, llm: ChatLLM, system_message: Optional[SystemMessage]):
        self.llm = llm
        self.chat = Chat(system_message)

    def run(self, message: UserMessage):
        self.chat.add_message(message)
        return self.llm.invoke(self.chat, stream=True)
