from app.llms import BaseLLM


class ChatLLM(BaseLLM):
    def invoke(self, *args, **kwargs):
        ...
