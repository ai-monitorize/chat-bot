import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.bots import chat_completion as chat_completion_bot
import uuid

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
    prompt = data.get("prompt", "")
    uid = str(uuid.uuid4())

    if not prompt:
        return {"error": "No prompt provided."}

    def event_stream():
        for text_chunk in chat_completion_bot.process(prompt):
            message = {
                'id': uid,
                'text': text_chunk,
                'role': 'assistant'
            }
            yield json.dumps(message) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")