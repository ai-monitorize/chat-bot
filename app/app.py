import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import uuid

from app.chat import Chat
from app.chatbot import ChatBot
from app.llms.chat import LLamaCppChat

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

llm = LLamaCppChat(model_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                   file="Meta-Llama-3.1-8B-Instruct-Q6_K.gguf")
chat_bot = ChatBot(llm)


@app.post("/v1/chat")
async def process_chat(request: Request):
    data = await request.json()
    print(data)
    messages = data.get("messages", "")
    uid = str(uuid.uuid4())

    if messages.__len__() == 0:
        return {"error": "No messages provided."}

    def event_stream():

        # initial_message = SystemMessage(
        #     "You are an assistant named Watson. You are supposed to answer users' questions. "
        #     "If you encounter questions about monitoring and alerting,"
        #     "and you have real information about currently active alerts, then "
        #     "answer with a formatted list containing the provided alerts."
        # )
        for text_chunk in chat_bot.run(Chat(messages)):
            message = {
                'uuid': uid,
                'content': text_chunk,
                'role': 'ASSISTANT'
            }
            yield json.dumps(message) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")
