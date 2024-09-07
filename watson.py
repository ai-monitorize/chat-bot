import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import uuid

from watson.chat.base import SystemMessage, UserMessage
from watson.chatbot import ChatBot
from watson.llms.chat import LLamaCppChat

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.post("/v1/chat")
async def process_chat(request: Request):
    data = await request.json()
    print(data)
    prompt: str = data.get("prompt", "")
    uid = str(uuid.uuid4())

    if not prompt:
        return {"error": "No prompt provided."}

    def event_stream():
        llm = LLamaCppChat(model_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                           file="Meta-Llama-3.1-8B-Instruct-Q6_K.gguf")
        initial_message = SystemMessage(
            "You are an assistant named Watson. You are supposed to answer users' questions. "
            "If you encounter questions about monitoring and alerting,"
            "and you have real information about currently active alerts, then "
            "answer with a formatted list containing the provided alerts."
        )
        chat_bot = ChatBot(llm, initial_message)

        user_message = UserMessage(prompt)
        for text_chunk in chat_bot.run(user_message):
            message = {
                'id': uid,
                'text': text_chunk,
                'role': 'assistant'
            }
            yield json.dumps(message) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")
