from llama_cpp import Llama
from app.chat.chat import Chat


class Llm:
    def __init__(self, model_id: str, file: str) -> None:
        self.model = Llama.from_pretrained(repo_id=model_id, filename=file, n_gpu_layers=-1)

    def generate_text_stream(self, chat: Chat) -> (str, bool):
        stream = self.model.create_chat_completion(
            messages=chat.messages,
            stream=True
        )

        generated_text = ''
        for output in stream:
            choice = output['choices'][0]['delta']
            text_chunk = choice['content'] if choice.__contains__('content') else ''
            generated_text += text_chunk
            yield text_chunk, False

        return generated_text, True

    def generate_text(self, chat: Chat) -> str:
        output = self.model.create_chat_completion(messages=chat.messages)

        return output['choices'][0]['message']['content']
