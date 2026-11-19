"""Stage 4: prompt building + the single LLM call.

This is the ONLY file that knows which model provider we use. Swapping Sarvam
for another provider means editing this file and nothing else.

IMPORTANT: the URL, model name and request shape below are placeholders.
Confirm them in the official Sarvam API docs, then set them in .env:
    SARVAM_API_URL=...
    SARVAM_MODEL=...
Do not copy them from a blog post.
"""

from __future__ import annotations

import time

import requests

from src.config import (
    LLM_ENABLED,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    SARVAM_API_KEY,
    SARVAM_API_URL,
    SARVAM_MODEL,
)

SYSTEM_PROMPT = """You are LexAI, an assistant for Indian intellectual
property information.

Rules you must follow:
1. Answer ONLY using the numbered passages in the CONTEXT block.
2. After every claim, cite the passage number and its source, like
   [Passage 2 - Patents_Act_1970.pdf, p.14].
3. If the passages answer part of the question, answer that part and then state
   plainly, in your own sentence, which part they do not cover. Never append the
   refusal sentence to an answer you have already given.
4. If the passages cannot answer the question at all, reply with exactly:
   "The provided documents do not cover this. I can only answer from the
   ingested IP documents." Do not use outside knowledge.
5. Never give legal advice or predict an outcome. Explain what the documents say.
6. Be concise and use plain language a student or founder can follow."""


def build_context(hits: list[dict]) -> str:
    """Turn retrieved chunks into a numbered context block.

    The numbering is what makes citations checkable by the user.
    """
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Passage {number}] Source: {hit['source']}, page {hit['page']}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict], history: list[dict] | None = None) -> list[dict]:
    """System prompt + recent conversation + the grounded user turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Only the last few turns: old turns cost tokens and add no grounding.
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{build_context(hits)}\n\n"
                f"QUESTION: {question}\n\n"
                "Answer using only the passages above, with citations."
            ),
        }
    )
    return messages


def llm_available() -> bool:
    return bool(LLM_ENABLED and SARVAM_API_KEY and SARVAM_API_URL and SARVAM_MODEL)


def _extract_text(payload: dict) -> str:
    """Pull the answer text out of a response body.

    Written defensively because providers differ: some return
    {"choices": [{"message": {"content": ...}}]}, others a top-level "content".
    """
    choices = payload.get("choices")
    if choices:
        first = choices[0]
        message = first.get("message") or {}
        if isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(first.get("text"), str):
            return first["text"]
    for key in ("content", "output", "response", "answer"):
        if isinstance(payload.get(key), str):
            return payload[key]
    raise ValueError(f"Could not find answer text in response keys: {list(payload)}")


def call_llm(messages: list[dict], retries: int = 2) -> dict:
    """POST the messages to the provider. Returns {'text', 'latency_s', 'error'}."""
    started = time.perf_counter()
    payload = {
        "model": SARVAM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                SARVAM_API_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(2 ** attempt)          # 1s, 2s, 4s backoff
                continue
            response.raise_for_status()
            text = _extract_text(response.json()).strip()
            return {
                "text": text,
                "latency_s": round(time.perf_counter() - started, 2),
                "error": "",
            }
        except requests.Timeout:
            last_error = f"Request timed out after {LLM_TIMEOUT}s"
        except requests.RequestException as exc:
            last_error = f"Network error: {exc}"
        except ValueError as exc:
            last_error = str(exc)
        time.sleep(2 ** attempt)

    return {"text": "", "latency_s": round(time.perf_counter() - started, 2), "error": last_error}


def extractive_answer(question: str, hits: list[dict]) -> str:
    """No-LLM fallback: quote the best passages verbatim.

    Useful before you have an API key, after a quota error, and for demoing
    retrieval quality on its own. It cannot hallucinate: it only quotes.
    """
    if not hits:
        return "The provided documents do not cover this."
    lines = ["Here are the most relevant passages from the knowledge base:", ""]
    for number, hit in enumerate(hits[:3], start=1):
        snippet = " ".join(hit["text"].split())[:600]
        lines.append(f"[Passage {number} - {hit['source']}, p.{hit['page']}]")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines).strip()
