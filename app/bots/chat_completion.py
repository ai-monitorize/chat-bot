from app.chat.chat import Chat
from app.model.llm_llamacpp import Llm

llm = None
chat = None


def initialize():
    global llm, chat
    if llm is None:
        llm = Llm(
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
        )
    if chat is None:
        chat = Chat()
        chat.add_system_message(
            "You are an assistant that answers users' questions. "
            "If you encounter questions about monitoring and alerting, "
            "answer with a formatted list containing the provided alerts."
        )


def process(prompt):
    initialize()
    chat.add_user_message(prompt)
    for text_chunk, is_finished in llm.generate_text_stream(chat):
        if is_finished:
            chat.add_assistant_message(text_chunk)
        else:
            yield text_chunk
