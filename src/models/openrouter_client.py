import os
import time
import requests

from dotenv import load_dotenv


load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


MODEL_IDS = {
    "qwen3_30b_a3b_instruct_2507": (
        "qwen/qwen3-30b-a3b-instruct-2507"
    ),
    "granite4_micro": (
        "ibm-granite/granite-4.0-h-micro"
    ),
    "gemma3_4b": (
        "google/gemma-3-4b-it"
    ),
}


PROVIDERS = {
    "qwen3_30b_a3b_instruct_2507": "siliconflow",
    "granite4_micro": "cloudflare",
    "gemma3_4b": "deepinfra",
}


def call_model(
    model_name,
    prompt,
    temperature=0.0,
    top_p=1.0,
    max_new_tokens=16,
):
    """
    Chiama un modello via OpenRouter.

    Restituisce:
    - raw_output
    - latency_seconds
    - prompt_tokens
    - completion_tokens
    - total_tokens
    - provider
    """

    if OPENROUTER_API_KEY is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY non trovata nel file .env"
        )

    if model_name not in MODEL_IDS:
        raise ValueError(
            f"Modello sconosciuto: {model_name}"
        )

    model_id = MODEL_IDS[model_name]

    headers = {
        "Authorization": (
            f"Bearer {OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "reasoning": {
            "enabled": False
        },
    }

    provider_name = PROVIDERS.get(
        model_name
    )

    if provider_name is not None:

        payload["provider"] = {
            "order": [
                provider_name
            ],
            "allow_fallbacks": False,
        }

    start = time.perf_counter()

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    latency = (
        time.perf_counter()
        - start
    )

    if not response.ok:

        raise RuntimeError(
            f"OpenRouter error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    raw_output = (
        data["choices"][0]["message"].get(
            "content"
        )
    )

    usage = data.get(
        "usage",
        {}
    )

    return {
        "raw_output": raw_output,
        "latency_seconds": latency,
        "prompt_tokens": usage.get(
            "prompt_tokens"
        ),
        "completion_tokens": usage.get(
            "completion_tokens"
        ),
        "total_tokens": usage.get(
            "total_tokens"
        ),
        "provider": data.get(
            "provider"
        ),
    }