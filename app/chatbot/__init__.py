from app.llms.chat import ChatLLM


class ChatBot:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def run(self, chat):
        return self.llm.invoke(chat, stream=True)
