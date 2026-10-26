"""
rag/llm.py

Configurable Large Language Model (LLM) client for Temporal-Multimodal-RAG.
Supports:
  - OpenAI & OpenAI-compatible endpoints (Groq, Together, Ollama, OpenRouter, vLLM)
  - Google Gemini API
  - Graceful fallback with sanitized error reporting (never leaks API keys)
"""

import os
import re
import json
import logging
import requests

logger = logging.getLogger("rag.llm")

# Automatically load local .env if present
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    try:
        with open(_env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

# System prompt adhering strictly to specifications
SYSTEM_PROMPT = """You are a helpful video-understanding assistant. Answer the user's question using the supplied timestamped transcript evidence.

Your primary task is to explain the answer in clear, natural, complete sentences. Do not simply copy a random fragment from the transcript.

Requirements:

1. Start with a direct answer to the user's question.
2. Explain the relevant concept in simple, understandable language.
3. Use additional details or examples from the transcript when they help explain the answer.
4. Combine multiple relevant transcript segments when necessary.
5. Correct minor speech-transcription errors only when the intended meaning is clear.
6. Do not invent facts, examples, code, or explanations that are not supported by the evidence.
7. If the transcript provides only partial information, answer what can be supported and clearly state what remains unclear.
8. If the retrieved transcript is irrelevant to the question, explain that the available transcript does not provide enough information.
9. Never treat timestamps as the answer itself.
10. Return a concise but complete response, normally one to three paragraphs. For a complex question, provide a longer explanation or structured steps when justified.
11. Use bullet points or numbered steps when they improve readability.
12. Do not repeat the user's question unnecessarily.
13. Do not mention internal retrieval algorithms or implementation details.
14. Include supporting timestamps separately from the main explanation.
15. Never claim that visual content was analyzed if only transcript evidence was supplied.

The response must be useful to someone who wants to understand the video without manually searching through the transcript."""


def get_llm_config() -> dict:
    """
    Resolves active LLM configuration from environment variables.
    Auto-detects provider based on configured API keys if LLM_PROVIDER is not explicitly set.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()

    openai_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()

    openai_base = (os.environ.get("OPENAI_API_BASE") or os.environ.get("LLM_API_BASE") or "https://api.openai.com/v1").strip()
    openai_model = (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip()

    gemini_model = (os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite").strip()

    # Determine provider if not specified
    if not provider:
        if openai_key:
            provider = "openai"
        elif gemini_key:
            provider = "gemini"
        elif groq_key:
            provider = "groq"
        elif os.environ.get("OLLAMA_BASE_URL"):
            provider = "ollama"
        else:
            provider = "none"

    timeout = float(os.environ.get("LLM_TIMEOUT", "30").strip() or 30.0)
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2").strip() or 0.2)

    return {
        "provider": provider,
        "openai_key": openai_key,
        "openai_base": openai_base,
        "openai_model": openai_model,
        "gemini_key": gemini_key,
        "gemini_model": gemini_model,
        "groq_key": groq_key,
        "timeout": timeout,
        "temperature": temperature,
        "is_configured": (
            (provider == "openai" and bool(openai_key))
            or (provider == "gemini" and bool(gemini_key))
            or (provider == "groq" and bool(groq_key))
            or (provider == "ollama")
            or (provider in ("custom", "openai_compatible") and bool(openai_key))
        ),
    }


def call_openai_compatible(
    question: str,
    context_text: str,
    api_key: str,
    api_base: str,
    model: str,
    timeout: float = 30.0,
    temperature: float = 0.2,
) -> tuple[bool, str, str]:
    """
    Sends request to an OpenAI-compatible /chat/completions endpoint using requests.
    """
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "",
        "Content-Type": "application/json",
    }
    # Clean headers if no api key (e.g. local ollama without key)
    if not api_key:
        headers.pop("Authorization", None)

    user_prompt = (
        f"Question: {question}\n\n"
        f"Timestamped Transcript Evidence:\n{context_text}\n\n"
        f"Please provide a complete, clear, and grounded answer to the question based on the evidence above."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                content = choices[0]["message"].get("content", "").strip()
                if content:
                    return True, content, ""
            return False, "", "Empty response returned from LLM provider."

        # Non-200 response
        err_msg = f"LLM provider error (HTTP {resp.status_code})"
        try:
            err_json = resp.json()
            if "error" in err_json and isinstance(err_json["error"], dict):
                err_msg = f"LLM error: {err_json['error'].get('message', resp.text)}"
            elif "error" in err_json:
                err_msg = f"LLM error: {err_json.get('error')}"
        except Exception:
            pass
        logger.error("OpenAI-compatible API failure: %s", err_msg)
        return False, "", err_msg

    except requests.Timeout:
        logger.error("LLM request timed out after %s seconds.", timeout)
        return False, "", "LLM request timed out."
    except requests.RequestException as e:
        logger.error("LLM network request error: %s", str(e))
        return False, "", f"Network connection to LLM failed: {type(e).__name__}"
    except Exception as e:
        logger.error("Unexpected error invoking LLM: %s", str(e))
        return False, "", f"LLM invocation error: {type(e).__name__}"


def call_gemini(
    question: str,
    context_text: str,
    api_key: str,
    model: str = "gemini-1.5-flash",
    timeout: float = 30.0,
    temperature: float = 0.2,
) -> tuple[bool, str, str]:
    """
    Sends request to Google Gemini generateContent REST endpoint using requests.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    user_prompt = (
        f"Question: {question}\n\n"
        f"Timestamped Transcript Evidence:\n{context_text}\n\n"
        f"Please provide a complete, clear, and grounded answer to the question based on the evidence above."
    )

    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                content_obj = candidates[0].get("content", {})
                parts = content_obj.get("parts", [])
                if parts and "text" in parts[0]:
                    ans_text = parts[0]["text"].strip()
                    if ans_text:
                        return True, ans_text, ""
            return False, "", "Gemini returned empty candidate output."

        err_msg = f"Gemini API error (HTTP {resp.status_code})"
        try:
            err_json = resp.json()
            if "error" in err_json and isinstance(err_json["error"], dict):
                err_msg = f"Gemini error: {err_json['error'].get('message', resp.text)}"
        except Exception:
            pass
        logger.error("Gemini API call failed: %s", err_msg)
        return False, "", err_msg

    except requests.Timeout:
        logger.error("Gemini request timed out after %s seconds.", timeout)
        return False, "", "Gemini request timed out."
    except requests.RequestException as e:
        logger.error("Gemini network error: %s", str(e))
        return False, "", f"Network connection to Gemini failed: {type(e).__name__}"
    except Exception as e:
        logger.error("Unexpected Gemini error: %s", str(e))
        return False, "", f"Gemini invocation error: {type(e).__name__}"


def generate_llm_answer(question: str, context_text: str) -> tuple[bool, str, str]:
    """
    High-level entry point to generate an answer using the configured LLM.
    Returns:
      (success: bool, answer_or_empty: str, error_or_reason: str)
    """
    config = get_llm_config()

    if not config["is_configured"]:
        return (
            False,
            "",
            "No LLM provider configured. Set OPENAI_API_KEY or GEMINI_API_KEY in environment or .env file.",
        )

    provider = config["provider"]

    if provider in ("openai", "openai_compatible", "custom"):
        return call_openai_compatible(
            question=question,
            context_text=context_text,
            api_key=config["openai_key"],
            api_base=config["openai_base"],
            model=config["openai_model"],
            timeout=config["timeout"],
            temperature=config["temperature"],
        )

    elif provider == "groq":
        return call_openai_compatible(
            question=question,
            context_text=context_text,
            api_key=config["groq_key"],
            api_base="https://api.groq.com/openai/v1",
            model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            timeout=config["timeout"],
            temperature=config["temperature"],
        )

    elif provider == "ollama":
        ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return call_openai_compatible(
            question=question,
            context_text=context_text,
            api_key="ollama",
            api_base=ollama_base,
            model=os.environ.get("OLLAMA_MODEL", "llama3"),
            timeout=config["timeout"],
            temperature=config["temperature"],
        )

    elif provider == "gemini":
        return call_gemini(
            question=question,
            context_text=context_text,
            api_key=config["gemini_key"],
            model=config["gemini_model"],
            timeout=config["timeout"],
            temperature=config["temperature"],
        )

    return False, "", f"Unsupported LLM provider: {provider}"
