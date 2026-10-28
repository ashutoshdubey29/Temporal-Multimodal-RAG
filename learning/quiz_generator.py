"""
learning/quiz_generator.py
==========================
VIVA / ARCHITECTURAL OVERVIEW:
This module powers the automated assessment engine of AetherLearn. It transforms
unstructured video transcripts or student topics into rigorous, educational
Multiple-Choice Questions (MCQs).

Key Concepts & Architecture for a Viva:
1. Dual-Path Quiz Generation:
   - Primary Path: Calls configured LLMs (Gemini, OpenAI, Groq, or Ollama) with a
     strict JSON system prompt requiring 4 options, 1 correct choice, educational
     explanations, topic tagging, and timestamp grounding.
   - Offline / Transparent Grounded Fallback: If no API key is set or the network is
     down, it uses regex heuristics to extract definitions ("X is Y", "we use X to Y")
     directly from transcript sentences, generating plausible distractors and linking
     them to real video timestamps.
2. Robust Parsing & Normalization:
   - LLMs often output markdown code blocks (```json ... ```) or conversational commentary.
   - `extract_json_payload` uses regex fences and bracket extractors to extract the clean payload.
   - `normalize_options` standardizes disparate formats into `{"A": "...", "B": "...", "C": "...", "D": "..."}`.
3. Server-Side Authoritative Scoring:
   - Answers are NEVER graded on the client side (prevents inspect-element cheating).
   - `score_quiz_submission()` calculates exact score, updates `QuizAttempt` and `QuizResponse` records,
     and immediately triggers downstream skill tracking and learning path adaptations.
"""

import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from models.database import (
    Course,
    Quiz,
    QuizAttempt,
    QuizQuestion,
    QuizResponse,
    UserSkill,
    VideoLearningSession,
    db,
)
from rag.llm import get_llm_config

logger = logging.getLogger("learning.quiz_generator")

# Allowed option identifiers in canonical MCQ format
VALID_OPTIONS = {"A", "B", "C", "D"}

SYSTEM_PROMPT_QUIZ = """You are an expert educational assessment creator.
Generate high-quality multiple-choice questions (MCQs) grounded strictly in the provided text.

Rules:
1. Every question must be directly answerable from the provided text. Do not invent external facts.
2. Provide exactly four options: "A", "B", "C", "D".
3. Exactly ONE option must be correct.
4. Distractors must be plausible but unambiguously incorrect.
5. Provide a clear, educational explanation for why the correct option is right.
6. Associate each question with a specific topic/skill (e.g., "Python Functions", "Data Structures").
7. Set difficulty to "easy", "medium", or "hard".
8. Include source start and end timestamps (in seconds) indicating where the concept was discussed.
9. Return ONLY a valid JSON array of question objects. No markdown formatting, no conversation.

JSON Format:
[
  {
    "question": "Question text here?",
    "options": {
      "A": "Option 1",
      "B": "Option 2",
      "C": "Option 3",
      "D": "Option 4"
    },
    "correct_option": "A",
    "explanation": "Explanation here.",
    "topic": "Python Loops",
    "difficulty": "medium",
    "source_start": 45.2,
    "source_end": 62.0
  }
]"""


# ----------------------------------------------------
# 1. Transcript Cleaning & Segment Selection
# ----------------------------------------------------

def prepare_transcript_context(
    segments: List[Dict[str, Any]],
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    max_tokens_approx: int = 3500
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Filters and formats transcript segments within the specified timestamp window.
    
    Viva Explanation:
    - Allows quizzes to target a specific portion of a long lecture (e.g., minutes 10 to 15)
      or the full video.
    - Caps output at `max_tokens_approx` (~3500 words) to avoid exceeding LLM context limits.
    - Formats each segment with readable time brackets: `[12.5s - 18.0s] Speech text...`
      so the LLM can associate generated questions with exact video timestamps.

    Returns:
        Tuple[str, List[Dict]]: (formatted_transcript_text, list_of_selected_segments)
    """
    if not segments:
        return "", []

    filtered = []
    for seg in segments:
        s_time = float(seg.get("start", 0.0))
        e_time = float(seg.get("end", s_time + 5.0))
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # Filter out segments outside the requested time interval
        if start_sec is not None and e_time < start_sec:
            continue
        if end_sec is not None and s_time > end_sec:
            continue

        filtered.append({
            "start": round(s_time, 1),
            "end": round(e_time, 1),
            "text": text
        })

    # Fallback to entire transcript if filter yielded nothing
    if not filtered:
        filtered = segments

    # Format transcript with timestamps
    lines = []
    total_words = 0
    selected_segments = []

    for seg in filtered:
        words = seg["text"].split()
        if total_words + len(words) > max_tokens_approx:
            break
        lines.append(f"[{seg['start']}s - {seg['end']}s] {seg['text']}")
        total_words += len(words)
        selected_segments.append(seg)

    return "\n".join(lines), selected_segments


# ----------------------------------------------------
# 2. Validation & Quality Checks
# ----------------------------------------------------

def extract_json_payload(raw_text: str) -> Any:
    """
    Safely extracts a JSON list or dict from raw text returned by an LLM.
    
    Viva Explanation:
    - Large Language Models frequently wrap JSON responses in markdown fences
      (e.g., ```json [...] ```) or add introductory text ("Here is your quiz:").
    - This function applies a 4-stage extraction pipeline:
      1. Direct JSON parse.
      2. Regex capture between markdown code fences (` ```(?:json)? ... ``` `).
      3. Regex capture of outer brackets `[...]`.
      4. Regex capture of outer braces `{...}`.
    - Returns Python parsed object (list or dict), or None if unparseable.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None
    raw_text = raw_text.strip()
    # 1. Direct parse
    try:
        return json.loads(raw_text)
    except Exception:
        pass

    # 2. Markdown fence: ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        fence_content = match.group(1).strip()
        try:
            return json.loads(fence_content)
        except Exception:
            pass

    # 3. Outer array
    array_match = re.search(r"(\[[\s\S]*\])", raw_text)
    if array_match:
        try:
            return json.loads(array_match.group(1).strip())
        except Exception:
            pass

    # 4. Outer object
    obj_match = re.search(r"(\{[\s\S]*\})", raw_text)
    if obj_match:
        try:
            return json.loads(obj_match.group(1).strip())
        except Exception:
            pass

    return None


def normalize_options(raw_options: Any) -> Optional[Dict[str, str]]:
    """
    Normalizes options from dict, list of dicts, or list of strings into:
    {"A": "...", "B": "...", "C": "...", "D": "..."}.
    
    Viva Explanation:
    - Different LLM models output choices in different structures:
      e.g. `{"A": "...", "B": "..."}`, `["A. text", "B. text"]`, or `[{"key": "A", "label": "..."}]`.
    - This function normalizes all variants into a canonical 4-key Python dict: {"A": ..., "B": ..., "C": ..., "D": ...}.
    - Rejects questions if:
      1. Exactly 4 choices are not found.
      2. Any choice text is empty.
      3. Any choices are duplicated (ensures 4 distinct, non-trivial options).

    Returns:
        Optional[Dict[str, str]]: Normalized choices dict or None if invalid.
    """
    if not raw_options:
        return None

    norm: Dict[str, str] = {}
    letters = ["A", "B", "C", "D"]

    if isinstance(raw_options, dict):
        for k, v in raw_options.items():
            k_clean = str(k).strip().upper()
            v_clean = str(v).strip()
            if k_clean in VALID_OPTIONS and v_clean:
                norm[k_clean] = v_clean
    elif isinstance(raw_options, (list, tuple)):
        for idx, item in enumerate(raw_options):
            if idx >= 4:
                break
            if isinstance(item, dict):
                val = item.get("value") or item.get("id") or item.get("key") or item.get("option") or letters[idx]
                val_str = str(val).strip().upper()
                lbl = item.get("label") or item.get("text") or item.get("content") or item.get("title") or ""
                lbl_str = str(lbl).strip()
                if val_str in VALID_OPTIONS and lbl_str:
                    norm[val_str] = lbl_str
            elif isinstance(item, str):
                item_clean = item.strip()
                match = re.match(r"^([A-Da-d])[\.\)\:\-]\s*(.*)$", item_clean)
                if match:
                    norm[match.group(1).upper()] = match.group(2).strip()
                else:
                    norm[letters[idx]] = item_clean

    if set(norm.keys()) != VALID_OPTIONS:
        return None

    if any(len(v) == 0 for v in norm.values()):
        return None

    # Check distinct options (case-insensitive deduplication)
    if len(set(v.lower() for v in norm.values())) < 4:
        return None

    return norm


def clean_single_question(q: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates and normalizes an individual generated question into canonical form.
    
    Viva Checks:
    1. Question text length >= 8 characters.
    2. Options structure: Exactly 4 distinct choices A, B, C, D.
    3. Correct option: Must resolve unambiguously to 'A', 'B', 'C', or 'D'.
    4. Explanation: Fallback educational explanation provided if empty.
    5. Difficulty & Topic: Normalized with defaults ("medium", "General").
    6. Timestamps: Floating-point seconds rounded to 1 decimal place.

    Returns:
        Tuple[bool, str, Optional[Dict]]: (is_valid, rejection_reason, canonical_question_dict)
    """
    if not isinstance(q, dict):
        return False, "Question must be a dictionary object.", None

    question_text = (
        q.get("question") or q.get("question_text") or q.get("prompt") or q.get("text") or ""
    ).strip()
    if len(question_text) < 8:
        return False, "Question text is too short or missing.", None

    raw_opts = q.get("options")
    if not raw_opts:
        # Check flat fields option_a, option_b, etc.
        if q.get("option_a") and q.get("option_b") and q.get("option_c") and q.get("option_d"):
            raw_opts = {
                "A": q["option_a"],
                "B": q["option_b"],
                "C": q["option_c"],
                "D": q["option_d"],
            }

    norm_opts = normalize_options(raw_opts)
    if not norm_opts:
        return False, "Options must contain four distinct, non-empty choices labeled A, B, C, D.", None

    # Resolve and validate correct option identifier
    correct_raw = q.get("correct_option") or q.get("correct_answer") or q.get("answer") or q.get("correct")
    correct = None
    if isinstance(correct_raw, int) and 0 <= correct_raw <= 3:
        correct = ["A", "B", "C", "D"][correct_raw]
    elif correct_raw is not None:
        c_str = str(correct_raw).strip().upper()
        if c_str in VALID_OPTIONS:
            correct = c_str
        else:
            # Check if correct_raw matches option text instead of letter
            matches = [k for k, v in norm_opts.items() if v.lower() == str(correct_raw).strip().lower()]
            if matches:
                correct = matches[0]

    if not correct or correct not in VALID_OPTIONS:
        return False, f"Correct option must be one of {sorted(VALID_OPTIONS)}.", None

    explanation = (
        q.get("explanation") or q.get("rationale") or q.get("reason") or ""
    ).strip()
    if len(explanation) < 3:
        explanation = f"Option {correct} is the correct answer based on the source material."

    topic = (q.get("topic") or "General").strip()
    difficulty = (q.get("difficulty") or "medium").strip().lower()
    if difficulty not in ("easy", "medium", "hard", "mixed"):
        difficulty = "medium"

    source_start = None
    source_end = None
    try:
        if q.get("source_start") is not None:
            source_start = round(float(q["source_start"]), 1)
        if q.get("source_end") is not None:
            source_end = round(float(q["source_end"]), 1)
    except (ValueError, TypeError):
        pass

    cleaned = {
        "question": question_text,
        "question_text": question_text,
        "options": norm_opts,
        "correct_option": correct,
        "explanation": explanation,
        "topic": topic,
        "difficulty": difficulty,
        "source_start": source_start,
        "source_end": source_end,
    }
    return True, "", cleaned


def validate_single_question(q: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates that a generated question adheres to strict quality and schema requirements."""
    ok, err, _ = clean_single_question(q)
    return ok, err


def validate_quiz_questions(raw_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates, filters, and deduplicates a batch of generated questions.
    
    Viva Explanation:
    - Filters out questions that fail schema checks (e.g. malformed options, missing answers).
    - Deduplicates semantically identical or repeated questions using a normalized alphanumeric hash
      (`re.sub(r"\\W+", "", text.lower())`).
    - Returns a list of clean, verified question dictionaries.
    """
    valid_list = []
    seen_texts = set()

    for item in raw_questions:
        ok, reason, cleaned = clean_single_question(item)
        if not ok or not cleaned:
            logger.debug("Discarding invalid question (%s): %s", reason, item)
            continue

        norm_text = re.sub(r"\W+", "", cleaned["question"].lower())
        if norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)
        valid_list.append(cleaned)

    return valid_list


# ----------------------------------------------------
# 3. LLM Question Generation
# ----------------------------------------------------

def generate_questions_with_llm(
    context_text: str,
    num_questions: int = 5,
    difficulty: str = "medium",
    topic_focus: Optional[str] = None
) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Calls the configured Large Language Model to generate structured MCQ JSON.
    
    Viva Flow:
    1. Checks `.env` configuration via `get_llm_config()`.
    2. Builds a structured prompt combining `SYSTEM_PROMPT_QUIZ` and the transcript evidence.
    3. Dispatches HTTP POST to the active provider:
       - OpenAI / Groq / Ollama / Custom: `/chat/completions`
       - Google Gemini: `:generateContent`
    4. Extracts JSON payload from LLM output using `extract_json_payload()`.
    5. Validates and deduplicates questions using `validate_quiz_questions()`.

    Returns:
        Tuple[bool, List[Dict], str]: (success, list_of_validated_questions, status_message)
    """
    conf = get_llm_config()
    if not conf.get("is_configured"):
        return False, [], "LLM provider is not configured in .env."

    provider = conf.get("provider")
    user_prompt = f"""Generate {num_questions} multiple choice questions (MCQs) based on the following text.
Target difficulty: {difficulty}.
{f'Topic focus: {topic_focus}.' if topic_focus else ''}

TEXT EVIDENCE:
{context_text}
"""

    raw_response = ""
    try:
        # Route 1: OpenAI-compatible REST endpoint (OpenAI, Groq, Ollama, vLLM)
        if provider in ("openai", "groq", "ollama", "custom", "openai_compatible"):
            headers = {"Content-Type": "application/json"}
            api_key = conf.get("openai_key") or conf.get("groq_key") or "ollama"
            if api_key and api_key != "ollama":
                headers["Authorization"] = f"Bearer {api_key}"

            url = f"{conf['openai_base'].rstrip('/')}/chat/completions"
            payload = {
                "model": conf["openai_model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_QUIZ},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": conf["temperature"],
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=conf["timeout"])
            resp.raise_for_status()
            data = resp.json()
            raw_response = data["choices"][0]["message"]["content"]

        # Route 2: Google Gemini REST endpoint
        elif provider == "gemini":
            gemini_key = conf["gemini_key"]
            model = conf["gemini_model"]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT_QUIZ}\n\n{user_prompt}"}]}],
                "generationConfig": {"temperature": conf["temperature"]}
            }
            resp = requests.post(url, json=payload, timeout=conf["timeout"])
            resp.raise_for_status()
            data = resp.json()
            raw_response = data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return False, [], f"Unsupported LLM provider: {provider}"

    except Exception as e:
        logger.warning("LLM API call failed during quiz generation: %s", e)
        return False, [], str(e)

    # Parse and validate JSON output from LLM
    try:
        parsed = extract_json_payload(raw_response)
        if isinstance(parsed, dict) and "questions" in parsed:
            parsed = parsed["questions"]
        elif isinstance(parsed, dict) and "quiz" in parsed:
            parsed = parsed["quiz"]
        elif isinstance(parsed, dict) and "items" in parsed:
            parsed = parsed["items"]

        if not isinstance(parsed, list):
            return False, [], f"LLM response was not a JSON list (raw: {raw_response[:100]}...)"

        valid_questions = validate_quiz_questions(parsed)
        if not valid_questions:
            return False, [], "No valid questions passed schema validation from LLM output."

        return True, valid_questions, "Success"

    except Exception as e:
        logger.warning("Failed to parse JSON response from LLM: %s. Raw: %s", e, raw_response[:200])
        return False, [], f"Invalid JSON format in LLM output: {e}"


# ----------------------------------------------------
# 4. Transparent Grounded Fallback Generator
# ----------------------------------------------------

def generate_questions_rule_based(
    segments: List[Dict[str, Any]],
    num_questions: int = 5,
    difficulty: str = "medium",
    topic_focus: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Transparent, grounded rule-based question generator when LLMs are unavailable.
    
    Viva Explanation:
    - This is an essential resilience feature for viva demonstrations where an internet
      connection or paid API quota might not be active.
    - How it works:
      1. Pattern Matching: Scans transcript text using regex patterns for definitional phrases:
         - `[Subject] is/means/refers to [Predicate]`
         - `we use [Subject] to/for [Predicate]`
         - `the key [Subject] is [Predicate]`
      2. Distractor Formation: Gathers other subject terms from the video or generic CS distractors.
      3. Option Permutation: Shuffles the correct answer across choices (A, B, C, D) using a
         deterministic modulo rotation so the answer isn't always 'A'.
      4. Grounded Citation: Includes exact start and end timestamps so clicking the question
         navigates the student to the exact video segment where the fact was spoken.

    Returns:
        List[Dict]: List of validated MCQ question dictionaries.
    """
    if not segments:
        return []

    # Patterns indicating definitions, processes, or key facts
    definition_patterns = [
        re.compile(r"([A-Z][a-zA-Z\s]{2,25})\s+(?:is|are|means|refers to)\s+([^.!?]{15,100})", re.IGNORECASE),
        re.compile(r"(?:we use|use)\s+([a-zA-Z\s]{2,25})\s+(?:to|for)\s+([^.!?]{15,100})", re.IGNORECASE),
        re.compile(r"(?:the main|primary|key)\s+([a-zA-Z\s]{2,25})\s+is\s+([^.!?]{15,100})", re.IGNORECASE),
    ]

    extracted_facts = []
    for seg in segments:
        text = seg.get("text", "")
        start = seg.get("start", 0.0)
        end = seg.get("end", start + 5.0)

        for pat in definition_patterns:
            matches = pat.findall(text)
            for subject, predicate in matches:
                subject = subject.strip()
                predicate = predicate.strip()
                if len(subject) > 2 and len(predicate) > 10:
                    extracted_facts.append({
                        "subject": subject,
                        "predicate": predicate,
                        "full_text": text,
                        "start": round(start, 1),
                        "end": round(end, 1)
                    })

    # If pattern matches were sparse, extract informative sentence chunks
    if len(extracted_facts) < num_questions:
        for seg in segments:
            text = seg.get("text", "")
            words = text.split()
            if len(words) >= 7:
                start = seg.get("start", 0.0)
                end = seg.get("end", start + 5.0)
                # Pick a key word
                candidates = [w.strip(",.?!") for w in words if len(w) > 4 and w.isalpha()]
                if candidates:
                    key_term = candidates[0]
                    rest = " ".join([w for w in words if w != key_term])
                    extracted_facts.append({
                        "subject": key_term,
                        "predicate": f"is associated with: {rest[:80]}",
                        "full_text": text,
                        "start": round(start, 1),
                        "end": round(end, 1)
                    })

    if not extracted_facts:
        return []

    # Distractor pool
    all_subjects = list(dict.fromkeys([f["subject"] for f in extracted_facts]))
    generic_distractors = [
        "Iterative linear sequence",
        "Deterministic hash function",
        "Hierarchical tree index",
        "Dynamic runtime buffer",
        "Static compilation unit",
        "Asynchronous event loop"
    ]

    questions = []
    for idx, fact in enumerate(extracted_facts[:num_questions]):
        correct_answer = fact["subject"].capitalize()
        # Find 3 distinct distractors
        distractors = [s.capitalize() for s in all_subjects if s.lower() != correct_answer.lower()]
        while len(distractors) < 3:
            for gd in generic_distractors:
                if gd not in distractors and gd.lower() != correct_answer.lower():
                    distractors.append(gd)
                if len(distractors) >= 3:
                    break

        distractors = distractors[:3]
        options_list = [correct_answer] + distractors
        # Deterministic arrangement based on index
        rot = idx % 4
        arranged = {
            "A": options_list[rot % 4],
            "B": options_list[(rot + 1) % 4],
            "C": options_list[(rot + 2) % 4],
            "D": options_list[(rot + 3) % 4],
        }
        correct_key = [k for k, v in arranged.items() if v == correct_answer][0]

        questions.append({
            "question": f"Based on the video explanation, what {fact['predicate']}?",
            "options": arranged,
            "correct_option": correct_key,
            "explanation": f"According to the video at [{fact['start']}s - {fact['end']}s]: \"{fact['full_text']}\"",
            "topic": topic_focus or "Video Concepts",
            "difficulty": difficulty,
            "source_start": fact["start"],
            "source_end": fact["end"]
        })

    return validate_quiz_questions(questions)


# Curated question bank for topic quizzes when no transcript and no LLM are configured
TOPIC_CONCEPT_BANKS = {
    "python": [
        {
            "question": "Which Python built-in function returns the number of items in an iterable?",
            "options": {"A": "count()", "B": "len()", "C": "size()", "D": "length()"},
            "correct_option": "B",
            "explanation": "len() is Python's standard built-in function to return the length or item count of an object.",
            "topic": "Python Fundamentals",
            "difficulty": "easy"
        },
        {
            "question": "Which data type in Python is mutable?",
            "options": {"A": "tuple", "B": "str", "C": "list", "D": "int"},
            "correct_option": "C",
            "explanation": "Lists can be modified in-place (append, remove, slice assignment), whereas tuples, strings, and integers are immutable.",
            "topic": "Python Data Structures",
            "difficulty": "easy"
        },
        {
            "question": "What is the primary purpose of Python's 'yield' keyword in a function?",
            "options": {"A": "To terminate the function immediately", "B": "To produce values lazily as a generator", "C": "To declare a global variable", "D": "To import an external module"},
            "correct_option": "B",
            "explanation": "The 'yield' statement turns a function into a generator, pausing execution and returning a value on each iteration.",
            "topic": "Python Functions",
            "difficulty": "medium"
        },
        {
            "question": "How are exceptions properly caught and handled in Python?",
            "options": {"A": "try / catch", "B": "try / except", "C": "catch / finally", "D": "begin / rescue"},
            "correct_option": "B",
            "explanation": "Python uses 'try' followed by one or more 'except' blocks for exception handling.",
            "topic": "Python Error Handling",
            "difficulty": "easy"
        },
        {
            "question": "What does Python's '__init__' method represent in a class definition?",
            "options": {"A": "Static destructor", "B": "Instance initializer / constructor", "C": "Module loader", "D": "Class inheritance specifier"},
            "correct_option": "B",
            "explanation": "__init__ is called when an instance of the class is created, allowing initialization of instance attributes.",
            "topic": "Python OOP",
            "difficulty": "medium"
        }
    ],
    "data structures": [
        {
            "question": "Which data structure follows the First-In, First-Out (FIFO) principle?",
            "options": {"A": "Stack", "B": "Queue", "C": "Binary Heap", "D": "Trie"},
            "correct_option": "B",
            "explanation": "A Queue dequeues elements in the exact order they were enqueued (FIFO).",
            "topic": "Data Structures",
            "difficulty": "easy"
        },
        {
            "question": "What is the average time complexity of searching for an element in a balanced Binary Search Tree (BST)?",
            "options": {"A": "O(1)", "B": "O(n)", "C": "O(log n)", "D": "O(n log n)"},
            "correct_option": "C",
            "explanation": "A balanced BST halves the search space at each step, yielding O(log n) average time complexity.",
            "topic": "Algorithms",
            "difficulty": "medium"
        },
        {
            "question": "Which data structure uses LIFO (Last-In, First-Out) ordering?",
            "options": {"A": "Stack", "B": "Queue", "C": "Graph", "D": "Linked List"},
            "correct_option": "A",
            "explanation": "A Stack pushes and pops items from the top, following Last-In, First-Out order.",
            "topic": "Data Structures",
            "difficulty": "easy"
        }
    ],
    "machine learning": [
        {
            "question": "What is the primary purpose of splitting data into training and test sets in Machine Learning?",
            "options": {"A": "To compress the dataset size", "B": "To evaluate model generalization on unseen data", "C": "To remove duplicate data rows", "D": "To speed up gradient descent"},
            "correct_option": "B",
            "explanation": "A test set provides an unbiased evaluation of how well the trained model generalizes to new, unseen examples.",
            "topic": "Machine Learning",
            "difficulty": "easy"
        },
        {
            "question": "Which metric is most suitable for evaluating an imbalanced binary classification dataset?",
            "options": {"A": "Overall Accuracy", "B": "F1-Score / PR-AUC", "C": "Mean Squared Error", "D": "Explained Variance"},
            "correct_option": "B",
            "explanation": "Accuracy can be deceptively high by predicting the majority class; Precision, Recall, and F1-score account for class imbalance.",
            "topic": "Model Evaluation",
            "difficulty": "medium"
        }
    ]
}


def get_curated_topic_questions(topic: str, num_questions: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves curated questions from the static offline question bank.
    
    Viva Explanation:
    - Provides instant questions for standard CS topics (Python, Data Structures, ML)
      if the student requested a topic quiz and no LLM API is available.
    - Ensures zero crashes or empty states during offline viva demonstrations.
    """
    topic_lower = topic.lower()
    selected_pool = []

    for key, bank in TOPIC_CONCEPT_BANKS.items():
        if key in topic_lower or topic_lower in key:
            selected_pool.extend(bank)

    if not selected_pool:
        # Fall back to general Python questions if specific topic is not found
        selected_pool = TOPIC_CONCEPT_BANKS["python"]

    return selected_pool[:num_questions]


# ----------------------------------------------------
# 5. Core Public Generation Service
# ----------------------------------------------------

def generate_and_save_quiz(
    user_id: int,
    title: str,
    video_session_id: Optional[int] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    num_questions: int = 5,
    difficulty: str = "medium",
    topic_focus: Optional[str] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None
) -> Tuple[bool, str, Optional[Quiz]]:
    """
    Orchestrates the complete quiz generation and database persistence pipeline.
    
    Step-by-Step Viva Explanation:
    1. Parameter Normalization: Clamps question counts (5, 10, 15, 20) and difficulty ("easy", "medium", "hard").
    2. Video-Based vs Topic-Based Branching:
       - If `segments` provided:
         a) Prepares transcript text window via `prepare_transcript_context()`.
         b) Tries LLM generation first (`generate_questions_with_llm`).
         c) If LLM fails/unconfigured, falls back gracefully to `generate_questions_rule_based()`.
       - If `segments` is None (Topic quiz):
         a) Tries LLM conceptual generation.
         b) If LLM fails/unconfigured, uses `get_curated_topic_questions()`.
    3. Database Storage:
       - Creates a parent `Quiz` record with metadata (user_id, source_type, difficulty, topic_tags).
       - Commits child `QuizQuestion` records with serialized JSON options (`CanonicalOptionsList` compatible)
         and source start/end timestamps.

    Returns:
        Tuple[bool, str, Optional[Quiz]]: (success, message, created_quiz_orm_object)
    """
    if num_questions not in (5, 10, 15, 20):
        num_questions = 5

    difficulty = (difficulty or "medium").lower()
    if difficulty not in ("easy", "medium", "hard", "mixed"):
        difficulty = "medium"

    source_type = "video" if segments else "topic"
    questions = []
    generation_method = "llm"

    if segments:
        context_text, selected_segs = prepare_transcript_context(segments, start_sec, end_sec)
        if not context_text:
            return False, "Supplied video transcript is empty.", None

        # 1. Try LLM first
        ok, llm_questions, llm_err = generate_questions_with_llm(
            context_text=context_text,
            num_questions=num_questions,
            difficulty=difficulty,
            topic_focus=topic_focus
        )

        if ok and llm_questions:
            questions = llm_questions
            generation_method = "llm"
        else:
            logger.info("LLM generation unavailable or failed (%s). Using rule-based transcript fallback.", llm_err)
            questions = generate_questions_rule_based(
                segments=selected_segs,
                num_questions=num_questions,
                difficulty=difficulty,
                topic_focus=topic_focus
            )
            generation_method = "rule_based_fallback"

    else:
        # Topic-only generation
        topic_name = topic_focus or title or "General Computer Science"
        conf = get_llm_config()
        if conf.get("is_configured"):
            prompt = f"Topic: {topic_name}. Generate conceptual multiple-choice questions for students."
            ok, llm_questions, _ = generate_questions_with_llm(
                context_text=prompt,
                num_questions=num_questions,
                difficulty=difficulty,
                topic_focus=topic_name
            )
            if ok and llm_questions:
                questions = llm_questions
                generation_method = "llm"

        if not questions:
            questions = get_curated_topic_questions(topic_name, num_questions=num_questions)
            generation_method = "curated_bank"

    if not questions:
        return False, "Could not generate valid quiz questions from the provided content.", None

    # Persist Quiz and child QuizQuestion rows in the database
    try:
        topic_tags_list = list(dict.fromkeys([q.get("topic", "General") for q in questions]))
        quiz = Quiz(
            user_id=user_id,
            video_session_id=video_session_id,
            title=title or f"Quiz: {topic_tags_list[0] if topic_tags_list else 'Video'}",
            source_type=source_type,
            difficulty=difficulty,
            topic_tags=",".join(topic_tags_list),
            question_count=len(questions),
        )
        db.session.add(quiz)
        db.session.flush()  # Generate quiz.id

        for q_data in questions:
            q_model = QuizQuestion(
                quiz_id=quiz.id,
                question_text=q_data["question"],
                options_json=json.dumps(q_data["options"]),
                correct_option=q_data["correct_option"],
                explanation=q_data.get("explanation", ""),
                topic=q_data.get("topic", "General"),
                difficulty=q_data.get("difficulty", difficulty),
                source_start=q_data.get("source_start"),
                source_end=q_data.get("source_end"),
            )
            db.session.add(q_model)

        db.session.commit()
        logger.info("Quiz %d successfully saved with %d questions (method: %s)",
                    quiz.id, len(questions), generation_method)
        return True, f"Quiz generated successfully ({generation_method}).", quiz

    except Exception as e:
        db.session.rollback()
        logger.exception("Failed to save generated quiz: %s", e)
        return False, f"Failed to save quiz: {e}", None


# ----------------------------------------------------
# 6. Authoritative Server-Side Quiz Scoring
# ----------------------------------------------------

def score_quiz_submission(
    user_id: int,
    quiz_id: int,
    user_answers: Dict[str, str]
) -> Tuple[bool, str, Optional[QuizAttempt]]:
    """
    Authoritatively evaluates and grades a student's quiz submission on the backend.
    
    Step-by-Step Viva Explanation:
    1. Validation: Verifies that the quiz and its questions exist in SQLite.
    2. Answer Comparison: Iterates through each `QuizQuestion`, comparing the student's
       selection against `question.correct_option` (case-insensitive letter 'A'..'D').
       - Tracks `correct_count`, `incorrect_count`, and `unanswered_count`.
    3. Score Computation:
       - Calculates percentage: `(correct_count / total) * 100.0`.
    4. Database Persistence:
       - Records a `QuizAttempt` row with final score and percentage.
       - Records individual `QuizResponse` rows for every question, preserving student answers
         and correctness for historical review.
    5. Automatic Downstream Feedback Loops:
       - Calls `learning.skill_tracker:update_skills_from_attempt()` to update the student's mastery.
       - Calls `learning.path_generator:adapt_learning_path_after_quiz()` to unlock or flag roadmap milestones.

    Returns:
        Tuple[bool, str, Optional[QuizAttempt]]: (success, message, attempt_record)
    """
    quiz = db.session.get(Quiz, quiz_id)
    if not quiz:
        return False, "Quiz not found.", None

    questions = quiz.questions
    if not questions:
        return False, "Quiz contains no questions.", None

    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0
    responses_to_save = []

    try:
        # Create attempt record
        attempt = QuizAttempt(
            user_id=user_id,
            quiz_id=quiz.id,
            total_questions=len(questions),
        )
        db.session.add(attempt)
        db.session.flush()  # Generates attempt.id

        # Grade each question against stored canonical answers
        for q in questions:
            raw_sel = user_answers.get(str(q.id)) or user_answers.get(q.id)
            selected = raw_sel.strip().upper() if raw_sel else None

            if not selected or selected not in VALID_OPTIONS:
                unanswered_count += 1
                is_correct = False
                selected = None
            elif selected == q.correct_option:
                correct_count += 1
                is_correct = True
            else:
                incorrect_count += 1
                is_correct = False

            response_record = QuizResponse(
                attempt_id=attempt.id,
                question_id=q.id,
                selected_option=selected,
                is_correct=is_correct,
            )
            responses_to_save.append(response_record)

        total = len(questions)
        pct = round((correct_count / total) * 100.0, 1) if total > 0 else 0.0

        attempt.score = float(correct_count)
        attempt.percentage = pct
        attempt.correct_count = correct_count
        attempt.incorrect_count = incorrect_count
        attempt.unanswered_count = unanswered_count

        for r in responses_to_save:
            db.session.add(r)

        db.session.commit()

        # Update skills automatically in downstream analytics
        try:
            from learning.skill_tracker import update_skills_from_attempt
            update_skills_from_attempt(user_id, attempt.id)
        except Exception as e:
            logger.warning("Could not update skills from attempt %d: %s", attempt.id, e)

        # Update learning path milestones automatically
        try:
            from learning.path_generator import adapt_learning_path_after_quiz
            adapt_learning_path_after_quiz(user_id, attempt.id)
        except Exception as e:
            logger.warning("Could not adapt learning path after quiz %d: %s", attempt.id, e)

        logger.info("Quiz %d scored for user %d: %d/%d (%.1f%%)",
                    quiz.id, user_id, correct_count, total, pct)
        return True, "Quiz scored successfully.", attempt

    except Exception as e:
        db.session.rollback()
        logger.exception("Failed to score quiz submission: %s", e)
        return False, f"Scoring error: {e}", None
