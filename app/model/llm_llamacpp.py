from llama_cpp import Llama


class ChatMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class Chat:
    def __init__(self, message: ChatMessage) -> None:
        self.chat = [message]

    def add_message(self, message: ChatMessage) -> None:
        if len(self.chat) > 1:
            self.chat[1] = message
        else:
            self.chat.append(message)


class Llm:
    def __init__(self, model_path: str) -> None:
        self.model = Llama(model_path=model_path, n_gpu_layers=-1)

    def generate_text(self, chat: Chat) -> str:
        chat_history = "\n".join([f"{msg.role}: {msg.content}" for msg in chat.chat])

        # Generate text using the model
        output = self.model(chat_history, max_tokens=1024, temperature=0.1, top_p=0.9)

        return output['choices'][0]['text']
