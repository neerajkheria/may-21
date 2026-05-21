import redis
import hashlib
import json
import time
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

CACHE_TTL = 3600

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def make_cache_key(query: str, model="gpt-3.5-turbo"):

    normalized = query.strip().lower()

    raw = f"{model}::{normalized}"

    return "ai_cache:" + hashlib.sha256(raw.encode()).hexdigest()


def get_ai_response(query: str, model="gpt-3.5-turbo"):

    cache_key = make_cache_key(query, model)

    start = time.perf_counter()

    cached = redis_client.get(cache_key)

    if cached:

        elapsed = (time.perf_counter() - start) * 1000

        return {
            "response": json.loads(cached),
            "source": "CACHE_HIT",
            "latency_ms": round(elapsed, 2),
            "cost": 0.0
        }

    completion = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer support agent "
                    "for TechMart."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],
        max_tokens=300,
        temperature=0.3
    )

    response_text = completion.choices[0].message.content

    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    cost = (
        prompt_tokens * 0.0000005 +
        completion_tokens * 0.0000015
    )

    payload = {
        "text": response_text,
        "tokens": prompt_tokens + completion_tokens
    }

    redis_client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(payload)
    )

    elapsed = (time.perf_counter() - start) * 1000

    return {
        "response": payload,
        "source": "OPENAI_API",
        "latency_ms": round(elapsed, 2),
        "cost": round(cost, 6)
    }


def get_cache_stats():

    info = redis_client.info()

    return {
        "total_keys": redis_client.dbsize(),
        "memory_used_mb": round(
            info["used_memory"] / 1024 / 1024,
            2
        ),
        "connected_clients": info["connected_clients"],
        "hits": info.get("keyspace_hits", 0),
        "misses": info.get("keyspace_misses", 0)
    }


if __name__ == "__main__":

    test_queries = [
        "What is your return policy?",
        "How do I track my order?",
        "What is your return policy?",
        "How do I reset my password?",
        "How do I track my order?"
    ]

    print("=" * 60)
    print("REDIS CACHE DEMO")
    print("=" * 60)

    for query in test_queries:

        print(f"\nQuery: {query}")

        result = get_ai_response(query)

        print(f"Source : {result['source']}")
        print(f"Latency : {result['latency_ms']} ms")
        print(f"Cost : ${result['cost']}")
        print(
            f"Response : "
            f"{result['response']['text'][:100]}"
        )

    print("\nCACHE STATS")
    print(json.dumps(get_cache_stats(), indent=2))