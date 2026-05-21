import os
import time
import logging
import threading

from enum import Enum

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi import HTTPException

from pydantic import BaseModel

from openai import (
    OpenAI,
    RateLimitError,
    APIStatusError,
    APITimeoutError
)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

RETRYABLE_ERRORS = (
    RateLimitError,
    APIStatusError,
    APITimeoutError
)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=30
    ),
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    before_sleep=before_sleep_log(
        logger,
        logging.WARNING
    ),
    reraise=True
)
def call_openai_with_retry(
    query: str,
    model="gpt-3.5-turbo"
):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a support agent."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        max_tokens=200,
        temperature=0.3,
        timeout=15.0
    )

    return response.choices[0].message.content


SYSTEM_MSG = (
    "You are a TechMart support assistant."
)

primary_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,
    max_tokens=300,
    timeout=20
)

fallback_llm_1 = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=300,
    timeout=15
)

fallback_llm_2 = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=200,
    timeout=10
)

resilient_llm = primary_llm.with_fallbacks([
    fallback_llm_1,
    fallback_llm_2
])


def get_resilient_response(query: str):

    messages = [
        SystemMessage(content=SYSTEM_MSG),
        HumanMessage(content=query)
    ]

    try:

        result = resilient_llm.invoke(messages)

        model_used = result.response_metadata.get(
            "model_name",
            "unknown"
        )

        return {
            "text": result.content,
            "model_used": model_used,
            "status": "ok"
        }

    except Exception as e:

        return {
            "text": "Service unavailable.",
            "model_used": "none",
            "status": "error",
            "error": str(e)
        }


class CircuitState(Enum):

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:

    def __init__(
        self,
        failure_threshold=5,
        recovery_timeout=60
    ):

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED

        self.failure_count = 0

        self.last_failure_time = None

        self._lock = threading.Lock()

    def call(
        self,
        func,
        *args,
        fallback_func=None,
        **kwargs
    ):

        with self._lock:

            if self.state == CircuitState.OPEN:

                if (
                    time.time() -
                    self.last_failure_time
                ) > self.recovery_timeout:

                    self.state = CircuitState.HALF_OPEN

                else:

                    if fallback_func:
                        return fallback_func(*args)

                    raise RuntimeError(
                        "Circuit breaker OPEN"
                    )

        try:

            result = func(*args, **kwargs)

            with self._lock:

                self.failure_count = 0
                self.state = CircuitState.CLOSED

            return result

        except Exception:

            with self._lock:

                self.failure_count += 1

                self.last_failure_time = time.time()

                if (
                    self.failure_count >=
                    self.failure_threshold
                ):

                    self.state = CircuitState.OPEN

            raise


circuit_breaker = CircuitBreaker()

app_retry = FastAPI(
    title="Resilient AI API"
)


class QueryRequest(BaseModel):

    query: str
    strategy: str = "langchain_fallback"


def static_fallback(query: str):

    return (
        "Support system overloaded. "
        "Please try again later."
    )


@app_retry.post("/query")
async def handle_query(request: QueryRequest):

    if not request.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Empty query"
        )

    try:

        if request.strategy == "tenacity_retry":

            text = circuit_breaker.call(
                call_openai_with_retry,
                request.query,
                fallback_func=static_fallback
            )

            return {
                "response": text,
                "strategy": "retry+circuit_breaker"
            }

        result = get_resilient_response(
            request.query
        )

        return result

    except RetryError:

        return {
            "response": static_fallback(
                request.query
            )
        }


@app_retry.get("/circuit-status")
async def circuit_status():

    return {
        "state": circuit_breaker.state.value,
        "failure_count": circuit_breaker.failure_count,
        "threshold": circuit_breaker.failure_threshold
    }