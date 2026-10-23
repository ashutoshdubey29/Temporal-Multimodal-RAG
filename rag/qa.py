"""
rag/qa.py

Temporal-Aware RAG Question Answering pipeline for Temporal-Multimodal-RAG:
- Context-aware transcript retrieval with stop-word filtering & term weighting
- Detection of broad summary vs. specific factual questions
- Neighboring segment expansion and adjacent passage merging
- Timestamp preservation and deduplication
- Synthesis of human-readable, grounded answers via configurable LLM
- Clear, evidence-based fallback when LLM is unavailable or unconfigured
- Full backward compatibility for find_relevant_segments
"""

import re
import math
import logging
from typing import Optional

from rag.llm import generate_llm_answer, get_llm_config

logger = logging.getLogger("rag.qa")

# Common English stop words plus question / video meta terms
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
    # Question / video meta terms
    "video", "speaker", "discussed", "tell", "explain", "talk", "talks", "talking",
    "show", "give", "please", "can", "want", "know"
}

# Patterns indicating broad/summary inquiries about the video
BROAD_QUESTION_PATTERNS = [
    r"\b(summariz|summary|summaries)\b",
    r"\b(main concept|main idea|core concept|key concept|central theme)\b",
    r"\b(overall|overview|takeaway|takeaways|key points|important points)\b",
    r"\bwhat is this video about\b",
    r"\bwhat does the video discuss\b",
    r"\bwhat does the speaker talk about\b",
    r"\bwhat problem is the speaker solving\b",
    r"\b(whole video|entire video|throughout the video)\b",
    r"\bexplain me question discussed\b",
    r"\bquestion discussed in this video\b",
    r"\bwhat was discussed\b",
    r"\btopic of this video\b",
]


def tokenize(text: str) -> list[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def is_broad_query(question: str) -> bool:
    """Detects if a user question is asking for an overarching summary or broad topic."""
    q_lower = question.lower()
    for pattern in BROAD_QUESTION_PATTERNS:
        if re.search(pattern, q_lower):
            return True

    # If all tokens in question are stop words or meta terms, treat as broad
    tokens = tokenize(question)
    content_tokens = [t for t in tokens if t not in STOP_WORDS]
    if not content_tokens and len(tokens) >= 3:
        return True

    return False


def normalize_word(w: str) -> str:
    """Lightweight suffix normalization so inflections (transitions/transition) match."""
    w = w.lower()
    if len(w) > 4:
        if w.endswith("ies") and len(w) > 5:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 5:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        if w.endswith("ing") and len(w) > 6:
            return w[:-3]
        if w.endswith("ed") and len(w) > 5:
            return w[:-2]
    return w


def find_relevant_segments(question: str, segments: list[dict]) -> list[dict]:
    """
    Retrieves and ranks relevant transcript segments for a question.
    Improved implementation with:
      - Stop-word filtering for prioritized keyword matching
      - Suffix normalization for plural and verb inflection matching
      - Fallback to unfiltered matching if query contains only common words
      - Inverse segment frequency weighting so distinctive terms score higher
      - Returns top scored segments preserving original format
    """
    if not question or not segments:
        return []

    tokens = tokenize(question)
    if not tokens:
        return []

    content_words = [w for w in tokens if w not in STOP_WORDS]
    target_words = content_words if content_words else tokens

    # Map normalized stems to words
    target_stems = {normalize_word(w) for w in target_words}

    # Calculate document frequencies for target stems
    doc_freq = {s: 0 for s in target_stems}
    segment_stem_sets = []
    for seg in segments:
        stems = {normalize_word(w) for w in tokenize(seg.get("text", ""))}
        segment_stem_sets.append(stems)
        for s in target_stems:
            if s in stems:
                doc_freq[s] += 1

    total_segs = max(1, len(segments))
    idf = {
        s: math.log(1.0 + (total_segs - df + 0.5) / (df + 0.5))
        for s, df in doc_freq.items()
    }

    results = []
    for idx, (seg, stems) in enumerate(zip(segments, segment_stem_sets)):
        matched = target_stems.intersection(stems)
        if not matched:
            continue

        # Score based on IDF weights of matched words
        score = sum(idf.get(s, 1.0) for s in matched)

        # Exact phrase bonus if contiguous query appears in segment
        seg_text_lower = seg.get("text", "").lower()
        if len(target_words) >= 2:
            query_phrase = " ".join(target_words)
            if query_phrase in seg_text_lower:
                score += 5.0
            # Also check stems phrase
            stems_phrase = " ".join(normalize_word(w) for w in target_words)
            if stems_phrase in " ".join(normalize_word(w) for w in tokenize(seg_text_lower)):
                score += 3.0

        results.append({
            "score": round(score, 3),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", "").strip(),
            "segment_index": idx
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:8]


def merge_adjacent_segments(selected_indices: list[int], segments: list[dict], max_gap: float = 3.0) -> list[dict]:
    """
    Expands selected segment indices with immediate neighbors (+/- 1) to restore
    natural spoken context, merges continuous/overlapping ranges, and returns
    unified passage dicts with coherent start/end timestamps and joined text.
    """
    if not selected_indices or not segments:
        return []

    num_segs = len(segments)
    expanded = set()
    for idx in selected_indices:
        # Include previous, current, and next segment
        for offset in (-1, 0, 1):
            neighbor = idx + offset
            if 0 <= neighbor < num_segs:
                expanded.add(neighbor)

    sorted_indices = sorted(expanded)
    if not sorted_indices:
        return []

    merged_passages = []
    current_chunk = [sorted_indices[0]]

    for next_idx in sorted_indices[1:]:
        prev_idx = current_chunk[-1]
        prev_end = segments[prev_idx]["end"]
        next_start = segments[next_idx]["start"]

        # Merge only if contiguous AND within temporal gap tolerance, or if timestamps overlap
        temporal_gap = next_start - prev_end
        if (next_idx == prev_idx + 1 and temporal_gap <= max_gap) or temporal_gap <= 0.5:
            current_chunk.append(next_idx)
        else:
            # Finalize previous passage
            start_ts = segments[current_chunk[0]]["start"]
            end_ts = segments[current_chunk[-1]]["end"]
            passage_text = " ".join(segments[i]["text"].strip() for i in current_chunk if segments[i].get("text"))
            merged_passages.append({
                "start": start_ts,
                "end": end_ts,
                "text": passage_text,
                "indices": list(current_chunk)
            })
            current_chunk = [next_idx]

    if current_chunk:
        start_ts = segments[current_chunk[0]]["start"]
        end_ts = segments[current_chunk[-1]]["end"]
        passage_text = " ".join(segments[i]["text"].strip() for i in current_chunk if segments[i].get("text"))
        merged_passages.append({
            "start": start_ts,
            "end": end_ts,
            "text": passage_text,
            "indices": list(current_chunk)
        })

    return merged_passages


def retrieve_context_for_query(question: str, segments: list[dict], max_context_chars: int = 5000) -> tuple[list[dict], list[dict]]:
    """
    Retrieves the most relevant passages for a question.
    Returns:
      (sources: list[dict], timestamps: list[dict])
    """
    if not segments:
        return [], []

    # Check for broad / summary question
    if is_broad_query(question):
        # Distribute selection across the timeline (beginning, early, mid, late, conclusion)
        total = len(segments)
        if total <= 10:
            sample_indices = list(range(total))
        else:
            # Sample 6-8 evenly spaced key anchors across the video
            num_anchors = min(8, total)
            step = total / num_anchors
            sample_indices = [int(i * step) for i in range(num_anchors)]

        merged_passages = merge_adjacent_segments(sample_indices, segments)
    else:
        # Specific query retrieval
        ranked_segments = find_relevant_segments(question, segments)
        if not ranked_segments:
            return [], []

        # Take top scoring segment indices (up to 4 anchor points)
        top_indices = [s["segment_index"] for s in ranked_segments[:4] if "segment_index" in s]
        if not top_indices:
            # Fallback if segment_index missing
            top_indices = [
                idx for idx, s in enumerate(segments)
                if any(r["start"] == s["start"] for r in ranked_segments[:4])
            ]

        merged_passages = merge_adjacent_segments(top_indices, segments)

        # Rank merged passages by maximum relevance score of their constituent segments
        seg_score_map = {s.get("segment_index", -1): s.get("score", 1.0) for s in ranked_segments}
        for p in merged_passages:
            p["score"] = max((seg_score_map.get(i, 0.0) for i in p.get("indices", [])), default=0.0)
        merged_passages.sort(key=lambda p: p.get("score", 0.0), reverse=True)

    # Trim to max context budget if necessary
    total_chars = 0
    selected_passages = []
    for passage in merged_passages:
        p_len = len(passage["text"])
        if total_chars + p_len > max_context_chars and selected_passages:
            break
        selected_passages.append(passage)
        total_chars += p_len

    sources = [
        {
            "start": round(float(p["start"]), 2),
            "end": round(float(p["end"]), 2),
            "text": p["text"].strip()
        }
        for p in selected_passages
    ]

    timestamps = [
        {
            "start": s["start"],
            "end": s["end"]
        }
        for s in sources
    ]

    return sources, timestamps


def build_context_string(sources: list[dict]) -> str:
    """Formats retrieved sources into timestamped evidence lines for the LLM."""
    if not sources:
        return "No relevant transcript excerpts found."

    lines = []
    for s in sources:
        lines.append(f"[{s['start']:.2f}s - {s['end']:.2f}s]: {s['text']}")
    return "\n\n".join(lines)


def generate_fallback_answer(question: str, sources: list[dict], reason: str = "") -> str:
    """
    Builds a clear, transparent, and evidence-based fallback answer when
    AI synthesis is unavailable or fails. Never pretends fragments are AI answers.
    """
    if not sources:
        return "No relevant information was found in the video transcript for your question."

    reason_phrase = f" ({reason})" if reason else ""
    header = (
        f"AI answer generation is currently unavailable{reason_phrase}.\n\n"
        f"Here are the most relevant transcript excerpts found for your question:"
    )

    excerpts = []
    for idx, s in enumerate(sources[:3], 1):
        excerpts.append(
            f"{idx}. [{s['start']:.2f}s – {s['end']:.2f}s]: \"{s['text']}\""
        )

    return f"{header}\n\n" + "\n\n".join(excerpts)


def answer_question(question: str, segments: list[dict]) -> dict:
    """
    Main RAG Question Answering pipeline:
    1. Validates inputs
    2. Retrieves & ranks context-aware timestamped transcript segments
    3. Expands neighbors & merges continuous passages
    4. Calls configured LLM with prompt
    5. Returns structured answer, sources, timestamps, and metadata
    """
    clean_q = (question or "").strip()
    if not clean_q:
        return {
            "success": False,
            "answer": "Please enter a valid question.",
            "sources": [],
            "timestamps": [],
            "answer_mode": "error",
            "timestamp_str": "Not available",
            "primary_timestamp": None,
            "results": [],
            "error": "Empty question"
        }

    if not segments:
        return {
            "success": False,
            "answer": "Please upload or analyze a video before asking questions.",
            "sources": [],
            "timestamps": [],
            "answer_mode": "error",
            "timestamp_str": "Not available",
            "primary_timestamp": None,
            "results": [],
            "error": "No transcript available"
        }

    # Step 1: Retrieve context & merged passages
    sources, timestamps = retrieve_context_for_query(clean_q, segments)

    if not sources:
        return {
            "success": True,
            "answer": "No relevant information was found in the video transcript for your question.",
            "sources": [],
            "timestamps": [],
            "answer_mode": "no_match",
            "timestamp_str": "Not found",
            "primary_timestamp": None,
            "results": []
        }

    # Primary timestamp from the first / most relevant source
    primary_src = sources[0]
    primary_ts = {"start": primary_src["start"], "end": primary_src["end"]}
    timestamp_str = f"{primary_src['start']:.2f}s - {primary_src['end']:.2f}s"

    # Step 2: Format context for LLM
    context_text = build_context_string(sources)

    # Step 3: Check LLM configuration
    config = get_llm_config()

    if not config["is_configured"]:
        logger.info("LLM not configured. Returning transparent transcript fallback.")
        fallback_ans = generate_fallback_answer(
            clean_q,
            sources,
            reason="No LLM API key has been configured. Configure OPENAI_API_KEY or GEMINI_API_KEY in .env to enable AI synthesis"
        )
        return {
            "success": True,
            "answer": fallback_ans,
            "sources": sources,
            "timestamps": timestamps,
            "answer_mode": "fallback",
            "timestamp_str": timestamp_str,
            "primary_timestamp": primary_ts,
            "results": [{"score": 1.0, "start": s["start"], "end": s["end"], "text": s["text"]} for s in sources[:3]]
        }

    # Step 4: Invoke LLM
    llm_ok, llm_answer, err_msg = generate_llm_answer(clean_q, context_text)

    if llm_ok and llm_answer.strip():
        return {
            "success": True,
            "answer": llm_answer.strip(),
            "sources": sources,
            "timestamps": timestamps,
            "answer_mode": "llm",
            "timestamp_str": timestamp_str,
            "primary_timestamp": primary_ts,
            "results": [{"score": 1.0, "start": s["start"], "end": s["end"], "text": s["text"]} for s in sources[:3]]
        }

    # Step 5: Graceful fallback on LLM failure
    logger.warning("LLM generation failed: %s. Falling back to transcript excerpts.", err_msg)
    fallback_ans = generate_fallback_answer(
        clean_q,
        sources,
        reason="AI service temporarily unavailable"
    )

    return {
        "success": True,
        "answer": fallback_ans,
        "sources": sources,
        "timestamps": timestamps,
        "answer_mode": "fallback",
        "timestamp_str": timestamp_str,
        "primary_timestamp": primary_ts,
        "results": [{"score": 1.0, "start": s["start"], "end": s["end"], "text": s["text"]} for s in sources[:3]],
        "error": err_msg
    }