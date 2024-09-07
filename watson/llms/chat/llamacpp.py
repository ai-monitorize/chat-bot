from typing import Optional

from llama_cpp import Llama
from watson.chat import Chat
from watson.llms.chat.base import ChatLLM


class LLamaCppChat(ChatLLM):
    def __init__(self, model_id: str, file: Optional[str], ctx_size: Optional[int] = 4096,
                 n_gpu_layers: Optional[int] = -1):
        self.model = self.model = Llama.from_pretrained(
            repo_id=model_id,
            filename=file,
            n_ctx=ctx_size,
            n_gpu_layers=n_gpu_layers
        )

    def invoke(self, *args, **kwargs):
        if kwargs["stream"]:
            return self._stream(*args)
        else:
            return self._run(*args)

    def _stream(self, chat: Chat):
        stream = self.model.create_chat_completion(
            messages=chat.messages,
            stream=True
        )

        for chunk in stream:
            if not chunk["choices"]:
                continue
            elif "content" not in chunk["choices"][0]["delta"]:
                continue
            yield chunk["choices"][0]["delta"]["content"]

    def _run(self, chat: Chat):
        output = self.model.create_chat_completion(messages=chat.messages)

        if output["choices"]:
            return output["choices"][0]["message"]["content"]
        else:
            return ""
