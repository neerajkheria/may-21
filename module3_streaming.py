import os
import json
import asyncio

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi import HTTPException

from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from openai import AsyncOpenAI

load_dotenv()

app = FastAPI(
    title="AI Streaming API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


class QueryRequest(BaseModel):

    query: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.3


async def stream_ai_response(
    query: str,
    model: str,
    max_tokens: int,
    temperature: float
):

    try:

        yield (
            f"data: "
            f"{json.dumps({'event': 'start'})}\n\n"
        )

        token_count = 0

        stream = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer support agent."
                    )
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        async for chunk in stream:

            delta = chunk.choices[0].delta

            if delta.content:

                token_count += 1

                payload = {
                    "event": "token",
                    "token": delta.content,
                    "index": token_count
                }

                yield f"data: {json.dumps(payload)}\n\n"

                await asyncio.sleep(0)

        yield (
            f"data: "
            f"{json.dumps({'event': 'done'})}\n\n"
        )

    except Exception as e:

        payload = {
            "event": "error",
            "message": str(e)
        }

        yield f"data: {json.dumps(payload)}\n\n"


@app.post("/stream")
async def stream_response(request: QueryRequest):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    return StreamingResponse(
        stream_ai_response(
            request.query,
            request.model,
            request.max_tokens,
            request.temperature
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "streaming": True
    }