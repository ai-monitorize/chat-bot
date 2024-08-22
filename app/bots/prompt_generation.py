# from app.chat.chat import Chat, ChatMessage
# from app.model.llm_llamacpp import Llm
#
# llm = Llm(
#     "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
#     "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
# )
# chat = Chat()
# chat.add_system_message(
#     "From the given prompt generate 5 prompts which have the same meaning and semantics."
#     "The idea is to have 5 more prompts which would help me do better semantic search."
#     "Each prompt should be generated in a separate row."
#     "I don't want to have any other text generated from you apart from the prompts")
#
#
# def process(prompt):
#     chat.add_message(ChatMessage('user', prompt), 1)
#     return llm.generate_text(chat)
