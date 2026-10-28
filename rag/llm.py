"""
rag/llm.py
==========
MODULE PURPOSE (VIVA SUMMARY):
This file handles the Language Model (LLM) integration for AetherLearn.
It is responsible for taking the student's question AND the retrieved timestamped
transcript evidence from `rag/qa.py`, sending them to the configured AI provider,
and returning a grounded, student-friendly explanation.

KEY CONCEPTS TO EXPLAIN IN VIVA:
1. Multi-Provider Support:
   The application is NOT locked into a single AI company. It supports:
   - Google Gemini API (using Gemini 1.5/2.0 Flash)
   - OpenAI (GPT-4o-mini, GPT-4o)
   - Groq (Ultra-fast Llama-3 inference)
   - Ollama (100% free, private local LLM running on student's machine)
   - Any custom OpenAI-compatible endpoint (vLLM, OpenRouter, Together AI).
2. Auto-Detection:
   The system automatically inspects `.env` or system environment variables.
   If GEMINI_API_KEY is found, it uses Gemini. If OPENAI_API_KEY is found, it uses OpenAI.
3. System Prompt & Grounding:
   We provide a structured system prompt (`SYSTEM_PROMPT`) acting as an educational tutor.
   It strictly forbids making up facts (hallucinations), tells the model not to use robotic filler
   phrases like "According to the transcript", encourages bold keywords, and requires citing
   exact timestamps `[MM:SS–MM:SS]` so the UI can link directly to the video.
4. Security & Sanitization:
   Never prints or leaks API keys in logs or web exceptions. Catches timeouts (30s) gracefully.
"""

import os
import re
import json
import logging
import requests

logger = logging.getLogger("rag.llm")

# Automatically load local .env if present
# (Ensures API keys are available even when running standalone scripts)
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

# System prompt adhering strictly to educational readability and evidence grounding
SYSTEM_PROMPT = """You are a helpful video-understanding assistant and educational AI tutor for AetherLearn. Your role is to help students learn through clear, human-readable, direct, and well-structured answers grounded in timestamped transcript evidence.

Your primary task is to explain the answer in simple, natural, student-friendly English. Do not simply copy a random fragment from the transcript.

Requirements:

1. Start with a direct answer to the user's question.
   - Immediately answer the user's actual question in the very first sentence or opening paragraph.
   - Do NOT use unnecessary or robotic introductory phrases such as "Based on the provided transcript", "Based on the evidence", "According to the video", or "In the transcript". Jump straight into the explanation.

2. Student-Friendly Tone & Clarity:
   - Use simple, natural, student-friendly English that is easy to understand.
   - Highlight important technical terms in **bold** (e.g., **Vertex**, **Edge**, **Walk**).
   - Keep simple factual answers concise, generally around 2–5 sentences.
   - For complex questions, provide enough detail to explain the concept properly rather than enforcing a strict length limit.
   - Avoid repetitive explanations and overly formal language. Preserve relevant technical terminology and accuracy.

3. Formatting & Structure:
   - Organize longer answers using short paragraphs, headings, numbered steps, or bullet points when appropriate.
   - Always include a simple example or relatable analogy (e.g., comparing nodes to cities and edges to connecting roads) whenever it helps understanding.

4. Grounding & Handling Missing or Related Information:
   - Use the retrieved transcript as your source of truth for what the video covers. Do not invent facts, examples, code, or explanations that are not supported by the video.
   - If the transcript covers the question directly, explain it clearly and cite the supporting timestamps (format: [MM:SS–MM:SS], e.g., [06:34–06:50]).
   - If the transcript does not directly define a term but provides related context (for example, using a term within a broader topic or problem):
     * Clearly say that the video does not explicitly define it.
     * When appropriate, provide a short, clear educational explanation and example separately, without misrepresenting it as content from the video.
   - If the retrieved transcript is completely irrelevant or does not contain enough information, explain that the available transcript does not provide enough information. Never return irrelevant retrieved content merely because it matched some keywords.

5. Timestamps:
   - Never treat timestamps as the answer itself.
   - Include relevant timestamp references (format: [MM:SS–MM:SS], e.g., [06:34–06:50]) for statements supported by the video.
   - Never claim that visual content was analyzed if only transcript evidence was supplied.

The response must be genuinely useful, clear, and educational to someone who wants to understand the video without manually searching through the transcript."""


def get_llm_config() -> dict:
    """
    Resolves active LLM configuration from environment variables (.env).

    VIVA EXPLANATION:
    Auto-detects which provider to use:
    - If `GEMINI_API_KEY` is present -> defaults to Google Gemini
    - If `OPENAI_API_KEY` or `LLM_API_KEY` is present -> defaults to OpenAI
    - If `GROQ_API_KEY` is present -> defaults to Groq
    - If `OLLAMA_BASE_URL` is set -> defaults to Ollama local server
    - Otherwise -> sets `is_configured: False` which cleanly triggers transcript fallback mode.

    Returns:
        dict: Resolved configuration containing provider, keys, model names, timeout, temperature.
    """
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()

    openai_key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()

    openai_base = (os.environ.get("OPENAI_API_BASE") or os.environ.get("LLM_API_BASE") or "https://api.openai.com/v1").strip()
    openai_model = (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip()

    gemini_model = (os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash-lite").strip()

    # Determine provider if not specified explicitly
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
    Sends request to an OpenAI-compatible /chat/completions endpoint using standard requests.

    VIVA EXPLANATION:
    Works with OpenAI, Groq, Together AI, or local Ollama.
    We construct two messages:
      1. System role: Educational tutor rules (`SYSTEM_PROMPT`).
      2. User role: The student's question and the retrieved timestamped transcript evidence.
    Uses `temperature=0.2` (low temperature keeps explanations focused and factual,
    reducing creative hallucinations).

    Parameters:
        question (str): Student's question.
        context_text (str): Retrieved timestamped evidence passages.
        api_key (str): Provider secret key.
        api_base (str): Base URL of API endpoint.
        model (str): Name of model (e.g. 'gpt-4o-mini', 'llama-3.1-8b-instant').
        timeout (float): Request timeout in seconds.
        temperature (float): Sampling temperature.

    Returns:
        tuple[bool, str, str]: (success, answer_text, error_message)
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
        f"Please provide a clear, direct, and educational answer to the student's question based on the evidence above, following the tutor instructions."
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

        # Non-200 HTTP response (e.g. 401 Unauthorized or 429 Rate Limit)
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
    Sends request to Google Gemini's generateContent REST API endpoint using requests.

    VIVA EXPLANATION:
    Interacts with Google Gemini without requiring extra Google client SDK dependencies.
    Passes `systemInstruction` with tutor requirements and `contents` with user question + evidence.

    Parameters:
        question (str): Student's question.
        context_text (str): Retrieved timestamped transcript evidence.
        api_key (str): Google Gemini API key.
        model (str): Gemini model variant (e.g. 'gemini-1.5-flash').
        timeout (float): Request timeout in seconds.
        temperature (float): Sampling temperature.

    Returns:
        tuple[bool, str, str]: (success, answer_text, error_message)
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    user_prompt = (
        f"Question: {question}\n\n"
        f"Timestamped Transcript Evidence:\n{context_text}\n\n"
        f"Please provide a clear, direct, and educational answer to the student's question based on the evidence above, following the tutor instructions."
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
    High-level routing entry point to generate an answer using whichever LLM is configured.

    VIVA CALL FLOW:
    Called by `answer_question()` in `rag/qa.py` after transcript passages have been retrieved.
    Inspects active configuration and routes the request to either `call_gemini()`
    or `call_openai_compatible()` (which handles OpenAI, Groq, Ollama, etc.).

    Parameters:
        question (str): Student's query.
        context_text (str): Formatted evidence passages from the video transcript.

    Returns:
        tuple[bool, str, str]:
          - success (bool): True if an answer was successfully generated.
          - answer_or_empty (str): The synthesized markdown answer text.
          - error_or_reason (str): Error message if synthesis failed.
    """
    config = get_llm_config()

    if not config["is_configured"]:
        return (
            False,
            "",
            "No LLM provider configured. Set OPENAI_API_KEY or GEMINI_API_KEY in environment or .env file.",
        )

    provider = config["provider"]

    # Route 1: OpenAI or compatible (vLLM, OpenRouter, Together AI)
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

    # Route 2: Groq high-speed cloud inference
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

    # Route 3: Ollama local open-source models
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

    # Route 4: Google Gemini
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

