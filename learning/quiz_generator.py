"""
learning/quiz_generator.py

Automatic Multiple-Choice Question (MCQ) generation pipeline.
Supports:
  - Video transcript (full or selected timestamp range)
  - Topic-based generation
  - Multi-LLM provider integration (OpenAI, Gemini, Groq, Ollama)
  - Robust rule-based fallback grounded directly in transcript segments
  - Strict question structure validation and server-side authoritative scoring
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
    Truncates gracefully to avoid exceeding LLM context windows.
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

        if start_sec is not None and e_time < start_sec:
            continue
        if end_sec is not None and s_time > end_sec:
            continue

        filtered.append({
            "start": round(s_time, 1),
            "end": round(e_time, 1),
            "text": text
        })

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

def validate_single_question(q: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates that a generated question adheres to strict quality and schema requirements."""
    if not isinstance(q, dict):
        return False, "Question must be a dictionary object."

    question_text = (q.get("question") or "").strip()
    if len(question_text) < 10:
        return False, "Question text is too short."

    options = q.get("options")
    if not isinstance(options, dict) or set(options.keys()) != VALID_OPTIONS:
        return False, "Options must be a dict with keys 'A', 'B', 'C', 'D'."

    option_values = [str(v).strip() for v in options.values()]
    if any(len(v) == 0 for v in option_values):
        return False, "All four options must be non-empty."

    # Avoid duplicate options
    if len(set(v.lower() for v in option_values)) < 4:
        return False, "All options must be distinct."

    correct = str(q.get("correct_option") or "").strip().upper()
    if correct not in VALID_OPTIONS:
        return False, f"Correct option must be one of {sorted(VALID_OPTIONS)}."

    explanation = (q.get("explanation") or "").strip()
    if len(explanation) < 5:
        return False, "Explanation must be provided."

    return True, ""


def validate_quiz_questions(raw_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validates and filters a list of generated questions, deduplicating similar ones."""
    valid_list = []
    seen_texts = set()

    for item in raw_questions:
        ok, reason = validate_single_question(item)
        if not ok:
            logger.debug("Discarding invalid question (%s): %s", reason, item)
            continue

        norm_text = re.sub(r"\W+", "", item["question"].lower())
        if norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)

        cleaned = {
            "question": item["question"].strip(),
            "options": {k: str(item["options"][k]).strip() for k in ("A", "B", "C", "D")},
            "correct_option": item["correct_option"].strip().upper(),
            "explanation": item["explanation"].strip(),
            "topic": (item.get("topic") or "General").strip(),
            "difficulty": (item.get("difficulty") or "medium").strip().lower(),
            "source_start": float(item["source_start"]) if item.get("source_start") is not None else None,
            "source_end": float(item["source_end"]) if item.get("source_end") is not None else None,
        }
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
    """Calls configured LLM to generate structured MCQ JSON array."""
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

    # Parse JSON output from LLM
    try:
        # Strip potential markdown fences
        cleaned_json = re.sub(r"^```(?:json)?\s*", "", raw_response.strip(), flags=re.MULTILINE)
        cleaned_json = re.sub(r"\s*```$", "", cleaned_json, flags=re.MULTILINE)
        parsed = json.loads(cleaned_json)
        if isinstance(parsed, dict) and "questions" in parsed:
            parsed = parsed["questions"]
        if not isinstance(parsed, list):
            return False, [], "LLM response was not a JSON list."

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
    Transparent, grounded rule-based question generator when LLM is unavailable.
    Extracts key definition sentences and concepts directly from transcript segments.
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
    """Retrieves curated topic questions from the verified question bank."""
    topic_lower = topic.lower()
    selected_pool = []

    for key, bank in TOPIC_CONCEPT_BANKS.items():
        if key in topic_lower or topic_lower in key:
            selected_pool.extend(bank)

    if not selected_pool:
        # Fall back to general Python questions
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
    Orchestrates the entire quiz generation pipeline:
    1. Validates inputs & limits
    2. Gathers context (transcript or topic)
    3. Runs LLM or transparent grounded fallback
    4. Validates questions
    5. Persists Quiz and QuizQuestions to database
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

    # Persist in DB
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
        db.session.flush()

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
    Authoritatively scores a quiz submission:
    - user_answers: Dict mapping question_id (as str or int) to selected option ("A", "B", "C", "D")
    - Computes correct, incorrect, unanswered counts, percentage score
    - Saves QuizAttempt and QuizResponses
    - Triggers skill tracker and learning path adaptations
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
        attempt = QuizAttempt(
            user_id=user_id,
            quiz_id=quiz.id,
            total_questions=len(questions),
        )
        db.session.add(attempt)
        db.session.flush()  # Generates attempt.id

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

        # Update skills automatically
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
