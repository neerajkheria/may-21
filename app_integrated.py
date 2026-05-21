import os
import json
import asyncio
import hashlib
import logging

from dotenv import load_dotenv

from fastapi import FastAPI

from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from openai import (
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
    APIStatusError
)

import redis

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="TechMart AI Support"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

sync_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

SYSTEM_PROMPT = (
    "You are a TechMart support agent."
)

CACHE_TTL = 3600


def cache_key(query: str):

    return (
        "ai:" +
        hashlib.sha256(
            query.strip().lower().encode()
        ).hexdigest()
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=16
    ),
    retry=retry_if_exception_type(
        (
            RateLimitError,
            APIStatusError
        )
    )
)
def sync_openai(query: str, model: str):

    response = sync_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ],
        max_tokens=300,
        temperature=0.3
    )

    return response.choices[0].message.content


class QueryRequest(BaseModel):

    query: str
    model: str = "gpt-3.5-turbo"


@app.post("/ask")
async def ask(req: QueryRequest):

    key = cache_key(req.query)

    cached = redis_client.get(key)

    if cached:

        return {
            "response": json.loads(cached),
            "source": "cache"
        }

    text = sync_openai(req.query, req.model)

    redis_client.setex(
        key,
        CACHE_TTL,
        json.dumps(text)
    )

    return {
        "response": text,
        "source": "openai"
    }


@app.post("/stream")
async def stream(req: QueryRequest):

    async def generator():

        stream_response = (
            await async_client.chat.completions.create(
                model=req.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": req.query
                    }
                ],
                max_tokens=400,
                temperature=0.3,
                stream=True
            )
        )

        async for chunk in stream_response:

            delta = chunk.choices[0].delta

            if delta.content:

                yield (
                    f"data: "
                    f"{json.dumps({'token': delta.content})}\n\n"
                )

                await asyncio.sleep(0)

        yield (
            f"data: "
            f"{json.dumps({'done': True})}\n\n"
        )

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/health")
async def health():

    redis_ok = redis_client.ping()

    return {
        "api": "ok",
        "redis": redis_ok,
        "version": "3.0.0"
    }