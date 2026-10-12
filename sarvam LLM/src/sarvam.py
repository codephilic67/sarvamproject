"""
Sarvam AI Chat Completions client.

Docs: https://docs.sarvam.ai/api-reference/chat/chat-completions-v1
Auth:  https://docs.sarvam.ai/api-reference/authentication

Uses POST https://api.sarvam.ai/v1/chat/completions with model sarvam-105b.
Prefer the official `api-subscription-key` header. Never expose the key.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import httpx

SARVAM_BASE_URL = os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai").rstrip("/")
SARVAM_CHAT_PATH = "/v1/chat/completions"
DEFAULT_MODEL = "sarvam-105b"
ALLOWED_MODELS = ("sarvam-105b", "sarvam-105b-conversations")
DEFAULT_TIMEOUT = 90.0
DOCS_URL = "https://docs.sarvam.ai/api-reference/chat/chat-completions-v1"

# Structured Outputs schema for QA dataset generation (docs: response_format json_schema)
QA_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_qa_examples",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "evidence_quote": {"type": "string"},
                            "question_type": {
                                "type": "string",
                                "enum": [
                                    "factual",
                                    "definition",
                                    "procedural",
                                    "comparison",
                                    "reasoning",
                                    "other",
                                ],
                            },
                            "source_chunk_id": {"type": "string"},
                        },
                        "required": [
                            "question",
                            "answer",
                            "evidence_quote",
                            "question_type",
                            "source_chunk_id",
                        ],
                    },
                }
            },
            "required": ["examples"],
        },
    },
}


class SarvamError(Exception):
    """API or configuration error for Sarvam."""


class SarvamAuthError(SarvamError):
    pass


class SarvamTimeoutError(SarvamError):
    pass


class SarvamRateLimitError(SarvamError):
    pass


class SarvamResponseError(SarvamError):
    pass


def get_api_key(
    explicit: Optional[str] = None,
    streamlit_secrets: Any = None,
) -> Optional[str]:
    """
    Resolve key from: explicit → SARVAM_API_KEY env → Streamlit secrets.
    Never log or display the value.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    env_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if env_key:
        return env_key
    if streamlit_secrets is not None:
        try:
            key = streamlit_secrets.get("SARVAM_API_KEY", "")
            if key and str(key).strip():
                return str(key).strip()
        except Exception:
            pass
    return None


def has_api_key(streamlit_secrets: Any = None, session_key: Optional[str] = None) -> bool:
    if session_key and str(session_key).strip():
        return True
    return get_api_key(streamlit_secrets=streamlit_secrets) is not None


def _safe_error_message(resp: httpx.Response) -> str:
    """Extract error.code / error.message without leaking request bodies."""
    try:
        payload = resp.json()
        err = payload.get("error") or {}
        code = err.get("code") or ""
        message = err.get("message") or ""
        # Truncate and strip anything that looks like a key
        message = re.sub(r"sk_[A-Za-z0-9_-]+", "[redacted]", str(message))[:200]
        if code and message:
            return f"{code}: {message}"
        if message:
            return message
        if code:
            return str(code)
    except Exception:
        pass
    return f"HTTP {resp.status_code}"


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise SarvamResponseError("API response contained no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise SarvamResponseError("API response message has no content.")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)
    return str(content).strip()


def extract_json_object(text: str) -> Any:
    """Parse JSON from model output, tolerating fenced code blocks."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for opener, closer in (("[", "]"), ("{", "}")):
            start = cleaned.find(opener)
            end = cleaned.rfind(closer)
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise SarvamResponseError("Model output was not valid JSON.")


def chat_completions(
    messages: list[dict[str, str]],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT,
    response_format: Optional[dict[str, Any]] = None,
    reasoning_effort: Optional[str] = "medium",
    base_url: str = SARVAM_BASE_URL,
) -> dict[str, Any]:
    """
    Call Sarvam POST /v1/chat/completions per docs.sarvam.ai.

    Auth: api-subscription-key (preferred official header).
    Models: sarvam-105b | sarvam-105b-conversations
    """
    key = get_api_key(explicit=api_key)
    if not key:
        raise SarvamAuthError(
            "SARVAM_API_KEY is not configured. Add it in the sidebar, "
            ".env, or Streamlit secrets — or use Demo Mode. "
            f"See {DOCS_URL}"
        )

    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if response_format:
        body["response_format"] = response_format

    # Docs prefer api-subscription-key; Bearer is the OpenAI-compat alternative.
    headers = {
        "api-subscription-key": key,
        "Content-Type": "application/json",
    }
    url = base_url.rstrip("/") + SARVAM_CHAT_PATH

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=body)
            # Fallback chain for response_format / reasoning_effort incompatibilities
            if resp.status_code in (400, 422) and (
                response_format is not None or reasoning_effort is not None
            ):
                # 1) json_schema → json_object
                if response_format and response_format.get("type") == "json_schema":
                    body["response_format"] = {"type": "json_object"}
                    resp = client.post(url, headers=headers, json=body)
                # 2) drop response_format
                if resp.status_code in (400, 422) and "response_format" in body:
                    body.pop("response_format", None)
                    resp = client.post(url, headers=headers, json=body)
                # 3) drop reasoning_effort
                if resp.status_code in (400, 422) and "reasoning_effort" in body:
                    body.pop("reasoning_effort", None)
                    resp = client.post(url, headers=headers, json=body)
    except httpx.TimeoutException as exc:
        raise SarvamTimeoutError(
            f"Sarvam API timed out after {timeout}s. Try fewer chunks or raise timeout."
        ) from exc
    except httpx.HTTPError as exc:
        raise SarvamError(f"Sarvam API transport error: {exc}") from exc

    # Auth failures are 403 on Sarvam (not only 401) — per authentication docs
    if resp.status_code in (401, 403):
        detail = _safe_error_message(resp)
        raise SarvamAuthError(
            f"Sarvam authentication failed ({detail}). "
            "Check SARVAM_API_KEY / api-subscription-key."
        )
    if resp.status_code == 429:
        raise SarvamRateLimitError(
            f"Sarvam rate limit / quota exceeded ({_safe_error_message(resp)})."
        )
    if resp.status_code >= 400:
        raise SarvamResponseError(
            f"Sarvam API error ({_safe_error_message(resp)})."
        )

    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise SarvamResponseError("Sarvam API returned non-JSON body.") from exc


def generate_text(
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> str:
    payload = chat_completions(messages, **kwargs)
    return _extract_message_content(payload)


def ping_api(api_key: Optional[str] = None) -> tuple[bool, str]:
    """Lightweight connectivity check. Returns (ok, message) — never echoes the key."""
    try:
        text = generate_text(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            api_key=api_key,
            model=DEFAULT_MODEL,
            temperature=0,
            max_tokens=16,
            reasoning_effort=None,
            response_format=None,
            timeout=30.0,
        )
        return True, f"Connected to Sarvam ({DEFAULT_MODEL}). Response: {text[:40]}"
    except SarvamError as exc:
        return False, str(exc)
