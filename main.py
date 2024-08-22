from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from app.bots import chat_completion as chat_completion_bot

app = FastAPI()

@app.post("/v1/chat")
async def process_chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    if not prompt:
        return {"error": "No prompt provided."}

    def event_stream():
        for text_chunk in chat_completion_bot.process(prompt):
            yield text_chunk

    return StreamingResponse(event_stream(), media_type="text/plain")