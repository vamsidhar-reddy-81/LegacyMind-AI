from __future__ import annotations

import os


def call_llm(system_prompt: str, user_prompt: str) -> str | None:
    """Return an LLM response when credentials are configured, otherwise None."""
    if os.getenv("LEGACYMIND_FAST_MODE", "1") == "1":
        return None
    if _has_azure_config():
        return _call_azure(system_prompt, user_prompt)
    if os.getenv("OPENAI_API_KEY"):
        return _call_openai(system_prompt, user_prompt)
    return None


def _call_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _call_azure(system_prompt: str, user_prompt: str) -> str:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def _has_azure_config() -> bool:
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    return all(os.getenv(name) for name in required)
