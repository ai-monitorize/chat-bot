import torch

from app.rag.retriever import retrieve
from app.model.llm import ChatMessage, Chat, Llm


def main():
    initial_message = ChatMessage("system",
                                  "Answer user's question using alerts given in the context. In the context are currently active alerts which describe the current status of the system. Answer with a formated list of active alerts, and with a short description for each active alert.")
    chat = Chat(initial_message)
    llm = Llm("meta-llama/Meta-Llama-3-8B-Instruct", torch.float16)

    while True:
        prompt = input("prompt: ")

        if prompt == "quit":
            break
        else:
            context = retrieve(prompt)
            user_message = ChatMessage("user", "Context:\n" + context + "\nQuestion: " + prompt)
            chat.add_message(user_message)
            for assistant_answer in llm.generate_text(chat):
                print(assistant_answer)


main()
