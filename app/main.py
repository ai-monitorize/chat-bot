from app.rag.retriever import retrieve
from app.model.llm_llamacpp import ChatMessage, Chat, Llm


def main():
    initial_message = ChatMessage("system",
                                  "You are a helpful assistant. Users can ask you a various range of question and you should answer them.")
    chat = Chat(initial_message)
    llm = Llm("C:\\Users\\vlada\\.cache\\huggingface\\hub\\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\\snapshots\\9a8dec50f04fa8fad1dc1e7bc20a84a512e2bb01\\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
)
    # llm = Llm("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")

    while True:
        prompt = input("prompt: ")

        if prompt == "quit":
            break
        else:
            context = retrieve(prompt)
            user_message = ChatMessage("user", "Context:\n" + str(context) + "\nQuestion: " + prompt)
            chat.add_message(user_message)
            # for assistant_answer in llm.generate_text(chat):
            print(llm.generate_text(chat))


main()
