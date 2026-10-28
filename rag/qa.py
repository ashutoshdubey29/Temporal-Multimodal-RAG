"""
rag/qa.py
=========
MODULE PURPOSE (VIVA SUMMARY):
This file implements the core Temporal-Aware RAG (Retrieval-Augmented Generation)
Question Answering pipeline for AetherLearn.

WHAT IS RAG (RETRIEVAL-AUGMENTED GENERATION)?
Normally, asking a large language model (LLM) a question about a video leads to hallucinations
or generic guesses because the model has never seen the video.
With RAG:
  1. RETRIEVE: We first search the video's timestamped transcript segments to find the exact
     passages where the answer was discussed.
  2. AUGMENT: We pack those relevant timestamped excerpts into an educational prompt.
  3. GENERATE: We pass the evidence to the configured LLM (Gemini, GPT-4o-mini, Groq) to synthesize
     a clear, student-friendly explanation citing exact video timestamps (e.g. [06:34–06:50]).

IMPORTANT NOTE ON OUR ALGORITHM IMPLEMENTATION (CRITICAL FOR VIVA):
Our system does NOT require an external vector database or heavy embedding models.
Instead, it uses a lightweight, transparent Keyword + Inverse Document Frequency (IDF)
ranking engine with suffix normalization, query expansion, and temporal passage merging.
This makes it fast, deterministic, completely local, and easy to explain!

KEY ALGORITHMIC STEPS:
1. Stop-Word Filtering & Normalization:
   Common filler words ("the", "is", "video") are removed. Inflections ("transitions" -> "transition")
   are normalized using lightweight rule-based stemming.
2. IDF (Inverse Document Frequency) Term Weighting:
   Rare, distinctive words (e.g. "mitochondria", "backpropagation") get high scores,
   while common words get low scores:
     IDF(term) = ln(1 + (N - df + 0.5) / (df + 0.5))
3. Neighboring Segment Expansion:
   For every matching segment, we expand to include its immediate neighbors (+/- 1 segment)
   so that natural spoken sentences aren't cut in half.
4. Adjacent Passage Merging:
   Contiguous segments within a 3.0-second time gap are combined into a single unified passage.
5. Citation & Timestamp Alignment:
   If the LLM's response cites a specific timestamp, the primary video timestamp is aligned
   so the frontend video player jumps right to that moment.
6. Transparent Fallback:
   If no LLM API key is configured or the API times out, the system returns the exact
   transcript excerpts with timestamps rather than failing.
"""

import re
import math
import logging
from typing import Optional

from rag.llm import generate_llm_answer, get_llm_config

logger = logging.getLogger("rag.qa")

# Common English stop words plus question / video meta terms
# VIVA EXPLANATION: If a student asks "What did the speaker say about Python?", words like
# "what", "did", "the", "speaker", "say", "about" appear in every video. Removing them
# ensures our search focuses purely on content words like "Python".
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
# VIVA EXPLANATION: Broad questions like "Summarize this lecture" cannot be answered
# by a single keyword match. We detect them with regex patterns and sample across the video timeline.
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
]


def format_seconds_to_time(seconds: Optional[float]) -> str:
    """
    Formats seconds (e.g. 394.2) into human-readable MM:SS or HH:MM:SS format.

    VIVA EXAMPLE:
    394.2 seconds becomes "06:34" (6 minutes, 34 seconds).
    3665 seconds becomes "01:01:05" (1 hour, 1 minute, 5 seconds).

    Parameters:
        seconds (float, optional): Time in seconds from video start.

    Returns:
        str: Formatted time string (e.g., '06:34').
    """
    if seconds is None:
        return ""
    try:
        total_seconds = max(0, int(round(float(seconds))))
    except (ValueError, TypeError):
        return ""
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp_display(start: Optional[float], end: Optional[float] = None) -> str:
    """
    Formats a timestamp range like '06:34–06:50' or a single point '06:34'.

    Parameters:
        start (float): Beginning of segment in seconds.
        end (float, optional): End of segment in seconds.

    Returns:
        str: Range display like '06:34–06:50' or 'Not available'.
    """
    start_str = format_seconds_to_time(start)
    if not start_str:
        return "Not available"
    if end is not None and float(end) > float(start if start is not None else 0):
        end_str = format_seconds_to_time(end)
        return f"{start_str}–{end_str}"
    return start_str


def extract_cited_timestamps(text: str) -> list[float]:
    """
    Extracts timestamp citations from the LLM's generated response text.

    VIVA EXPLANATION:
    When the LLM writes an answer like:
    "The instructor defines a Graph at [04:12–04:30] as a set of vertices and edges..."
    This function parses '[04:12]' using regex and extracts 252.0 seconds.
    This allows the UI to automatically sync the video player with the LLM's citation!

    Supports:
      - [06:34–06:50] or [06:34]
      - 394.20s – 409.84s or 394s
      - (Timestamp: 06:34)

    Parameters:
        text (str): The markdown/plain text generated by the LLM.

    Returns:
        list[float]: Ordered list of start seconds found in the text.
    """
    if not text:
        return []
    found = []

    # 1. Match MM:SS or HH:MM:SS (e.g. 06:34 or 01:06:34)
    time_matches = re.finditer(r"\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\b", text)
    for m in time_matches:
        hrs, mins, secs = m.groups()
        total = int(mins) * 60 + int(secs)
        if hrs:
            total += int(hrs) * 3600
        found.append(float(total))

    # 2. Match explicit second notation: e.g. 394.20s or 394s
    sec_matches = re.finditer(r"\b(\d+(?:\.\d+)?)\s*s\b", text, re.IGNORECASE)
    for sm in sec_matches:
        try:
            found.append(float(sm.group(1)))
        except (ValueError, TypeError):
            pass

    return found


def tokenize(text: str) -> list[str]:
    """
    Converts raw text into clean, lowercase alphanumeric word tokens.

    Parameters:
        text (str): Input text string.

    Returns:
        list[str]: List of individual lowercase words.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def is_broad_query(question: str) -> bool:
    """
    Detects whether a user question is asking for an overarching summary or broad overview.

    VIVA EXPLANATION:
    If a student asks "What is this video about?", matching specific keywords won't work well
    because the whole video is relevant. When detected as broad, our retrieval algorithm
    samples key passages distributed evenly across the beginning, middle, and end of the video.

    Parameters:
        question (str): The student's question.

    Returns:
        bool: True if the question is an overview/summary question.
    """
    q_lower = question.lower()
    for pattern in BROAD_QUESTION_PATTERNS:
        if re.search(pattern, q_lower):
            return True

    # If all content words in the question were stop words, treat as broad
    tokens = tokenize(question)
    content_tokens = [t for t in tokens if t not in STOP_WORDS]
    if not content_tokens and len(tokens) >= 3:
        return True

    return False


def normalize_word(w: str) -> str:
    """
    Lightweight rule-based suffix stemming for English words.

    VIVA EXPLANATION:
    If a user searches for "transitions", we want it to match a transcript segment that says "transition".
    If they search for "running", we want it to match "run".
    This simple stemmer handles plurals (-s, -es, -ies) and verb endings (-ing, -ed).

    Parameters:
        w (str): A single word string.

    Returns:
        str: Normalized stem word.
    """
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
    Retrieves and ranks the most relevant transcript segments for a student's question.

    VIVA ALGORITHM EXPLANATION:
    1. Tokenize query & filter out common English stop words.
    2. Normalize words to their root stems (e.g. 'loops' -> 'loop').
    3. Calculate Inverse Document Frequency (IDF):
       Distinctive terms appearing in only 1 or 2 segments receive a high weight.
       Terms appearing in every segment receive a near-zero weight.
       Formula: IDF = ln(1 + (N - df + 0.5) / (df + 0.5))
    4. Compute relevance score:
       Sum of IDF weights for all matched terms in the segment.
       + Phrase bonus: +5.0 if exact consecutive phrase appears in segment.
    5. Sort descending by score and return top-8 matches.

    Parameters:
        question (str): The student's question.
        segments (list[dict]): All transcript segments from the video.

    Returns:
        list[dict]: Up to 8 top-scoring segment dictionaries with score, start, end, text.
    """
    if not question or not segments:
        return []

    tokens = tokenize(question)
    if not tokens:
        return []

    # Filter out stop words (e.g. "is", "the", "what")
    content_words = [w for w in tokens if w not in STOP_WORDS]
    target_words = content_words if content_words else tokens

    # Map normalized stems to words
    target_stems = {normalize_word(w) for w in target_words}

    # Calculate document frequencies (df): how many segments contain each stem
    doc_freq = {s: 0 for s in target_stems}
    segment_stem_sets = []
    for seg in segments:
        stems = {normalize_word(w) for w in tokenize(seg.get("text", ""))}
        segment_stem_sets.append(stems)
        for s in target_stems:
            if s in stems:
                doc_freq[s] += 1

    # Standard probabilistic BM25-style IDF formulation
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

        # Base score = sum of IDF weights of matched words
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

    # Sort descending by relevance score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:8]



def merge_adjacent_segments(selected_indices: list[int], segments: list[dict], max_gap: float = 3.0) -> list[dict]:
    """
    Expands selected segment indices with immediate neighbors (+/- 1) to restore
    natural spoken context, merges continuous/overlapping ranges, and returns
    unified passage dicts with coherent start/end timestamps and joined text.

    VIVA ALGORITHM EXPLANATION (WHY NEIGHBOR EXPANSION & PASSAGE MERGING?):
    1. Spoken sentences often span across 2 or 3 separate transcript chunks.
       If Whisper breaks a sentence like "Photosynthesis is the process..." [02:10-02:14]
       and "...by which plants convert sunlight into chemical energy" [02:14-02:18],
       retrieving ONLY the first segment would leave the answer incomplete.
    2. We expand every selected index `idx` to include `idx - 1`, `idx`, and `idx + 1`.
    3. We then check adjacent timestamps: if two segments are continuous (index difference == 1)
       AND the gap between them is <= 3.0 seconds, we merge them into a single coherent passage!

    Parameters:
        selected_indices (list[int]): Segment array indices that scored high in retrieval.
        segments (list[dict]): The full list of video transcript segments.
        max_gap (float): Maximum allowed gap in seconds to still consider two segments continuous (default 3.0s).

    Returns:
        list[dict]: Merged passages: [{'start': ..., 'end': ..., 'text': ..., 'indices': [...]}]
    """
    if not selected_indices or not segments:
        return []

    num_segs = len(segments)
    expanded = set()
    for idx in selected_indices:
        # Include previous (-1), current (0), and next (+1) neighbor
        for offset in (-1, 0, 1):
            neighbor = idx + offset
            if 0 <= neighbor < num_segs:
                expanded.add(neighbor)

    sorted_indices = sorted(expanded)
    if not sorted_indices:
        return []

    merged_passages = []
    current_chunk = [sorted_indices[0]]

    # Step through sorted indices and merge continuous chunks
    for next_idx in sorted_indices[1:]:
        prev_idx = current_chunk[-1]
        prev_end = segments[prev_idx]["end"]
        next_start = segments[next_idx]["start"]

        # Merge only if contiguous AND within temporal gap tolerance (<= 3.0s), or if timestamps overlap
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

    # Don't forget the last chunk
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
    High-level context retriever: orchestrates segment ranking, broad vs. specific branch,
    neighbor expansion, passage merging, and character budgeting.

    VIVA RETRIEVAL FLOW:
    1. Checks if question is broad ("Summarize this video"):
       If broad -> distributes 6-8 sample anchors evenly across the entire lecture timeline.
    2. If specific factual question:
       Runs `find_relevant_segments()` to find top-scoring keywords via IDF.
       Selects top 4 anchor segments and expands them with neighbors via `merge_adjacent_segments()`.
    3. Trims total text to `max_context_chars` (5000 chars) to prevent exceeding LLM prompt limits.
    4. Formats and returns sources (with timestamps and human-readable time strings).

    Parameters:
        question (str): Student's query.
        segments (list[dict]): Full transcript segments list.
        max_context_chars (int): Character budget for LLM context (default 5000).

    Returns:
        tuple[list[dict], list[dict]]: (sources, timestamps)
    """
    if not segments:
        return [], []

    # Branch A: Broad / Summary questions
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
        # Branch B: Specific factual query retrieval
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

        # Expand with neighbors and merge contiguous passages
        merged_passages = merge_adjacent_segments(top_indices, segments)

        # Rank merged passages by maximum relevance score of their constituent segments
        seg_score_map = {s.get("segment_index", -1): s.get("score", 1.0) for s in ranked_segments}
        for p in merged_passages:
            p["score"] = max((seg_score_map.get(i, 0.0) for i in p.get("indices", [])), default=0.0)
        merged_passages.sort(key=lambda p: p.get("score", 0.0), reverse=True)

    # Context window protection: Trim to max context budget (default 5000 chars)
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
            "text": p["text"].strip(),
            "timestamp_str": format_timestamp_display(round(float(p["start"]), 2), round(float(p["end"]), 2))
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
    """
    Formats retrieved sources into clean timestamped evidence lines for the LLM prompt.

    VIVA EXAMPLE OF GENERATED EVIDENCE STRING:
    [06:34–06:50] (394.20s - 409.84s): Now let's talk about Dijkstra's algorithm for finding the shortest path...

    Parameters:
        sources (list[dict]): Retrieved passage dictionaries.

    Returns:
        str: Formatted text block passed to the LLM system prompt.
    """
    if not sources:
        return "No relevant transcript excerpts found."

    lines = []
    for s in sources:
        time_range = format_timestamp_display(s["start"], s["end"])
        lines.append(f"[{time_range}] ({s['start']:.2f}s - {s['end']:.2f}s): {s['text']}")
    return "\n\n".join(lines)


def generate_fallback_answer(question: str, sources: list[dict], reason: str = "") -> str:
    """
    Builds a transparent, evidence-based fallback response when AI synthesis is unavailable.

    VIVA EXPLANATION:
    If a student runs the application without entering an API key in `.env`, or if their internet
    drops during the LLM call, this function ensures the application never crashes.
    Instead, it displays the top relevant transcript excerpts with their exact timestamps,
    so the student still gets their answer directly from the lecture!

    Parameters:
        question (str): Student question.
        sources (list[dict]): Retrieved transcript passages.
        reason (str, optional): Why LLM generation was bypassed (e.g. 'No API key configured').

    Returns:
        str: User-friendly markdown formatted fallback answer.
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
        time_range = format_timestamp_display(s["start"], s["end"])
        excerpts.append(
            f"{idx}. [{time_range}] ({s['start']:.2f}s – {s['end']:.2f}s): \"{s['text']}\""
        )

    return f"{header}\n\n" + "\n\n".join(excerpts)


def answer_question(question: str, segments: list[dict]) -> dict:
    """
    End-to-End RAG Question Answering Orchestrator.

    COMPLETE END-TO-END FLOW (EXPLAIN IN VIVA):
    Step 1: Input Validation - Ensures non-empty question and non-empty video transcript.
    Step 2: Retrieval - Calls `retrieve_context_for_query()` to find best passages using IDF scoring.
    Step 3: Prompt Construction - Formats evidence passages into timestamped lines.
    Step 4: LLM Check - If no API key is configured, immediately returns transparent fallback.
    Step 5: LLM Invocation - Sends prompt + evidence to LLM (Gemini / OpenAI / Groq / Ollama).
    Step 6: Timestamp Synchronization - If the LLM cited a timestamp in the text, syncs the
            primary timestamp so the frontend player jumps to the exact moment.
    Step 7: Structured Output - Returns full payload with answer text, sources, and start/end seconds.

    Parameters:
        question (str): Student's query.
        segments (list[dict]): Video transcript segments.

    Returns:
        dict: Standardized response payload for the frontend and API:
            {
                "success": bool,
                "answer": str,
                "sources": list[dict],
                "timestamps": list[dict],
                "answer_mode": 'llm' | 'fallback' | 'no_match' | 'error',
                "timestamp_str": str (e.g. '06:34–06:50'),
                "start": float,
                "end": float,
                "primary_timestamp": dict
            }
    """
    clean_q = (question or "").strip()
    if not clean_q:
        return {
            "success": False,
            "answer": "Please enter a valid question.",
            "sources": [],
            "timestamps": [],
            "answer_mode": "error",
            "timestamp": "Not available",
            "timestamp_str": "Not available",
            "primary_timestamp": None,
            "start": None,
            "end": None,
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
            "timestamp": "Not available",
            "timestamp_str": "Not available",
            "primary_timestamp": None,
            "start": None,
            "end": None,
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
            "timestamp": "Not found",
            "timestamp_str": "Not found",
            "primary_timestamp": None,
            "start": None,
            "end": None,
            "results": []
        }

    # Primary timestamp from the first / most relevant source
    primary_src = sources[0]
    primary_ts = {"start": primary_src["start"], "end": primary_src["end"]}
    timestamp_str = format_timestamp_display(primary_src["start"], primary_src["end"])

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
            "timestamp": timestamp_str,
            "timestamp_str": timestamp_str,
            "primary_timestamp": primary_ts,
            "start": primary_src["start"],
            "end": primary_src["end"],
            "results": [{"score": 1.0, "start": s["start"], "end": s["end"], "text": s["text"]} for s in sources[:3]]
        }

    # Step 4: Invoke LLM with grounded evidence prompt
    llm_ok, llm_answer, err_msg = generate_llm_answer(clean_q, context_text)

    if llm_ok and llm_answer.strip():
        ans_clean = llm_answer.strip()

        # Step 5: Synchronize citations with primary timestamp:
        # If the LLM cited a timestamp in the response, align primary timestamp with it
        cited_times = extract_cited_timestamps(ans_clean)
        if cited_times:
            best_idx = None
            best_diff = float("inf")
            target_t = cited_times[0]
            for idx, s in enumerate(sources):
                if s["start"] - 2.0 <= target_t <= s["end"] + 2.0:
                    best_idx = idx
                    break
                diff = abs(s["start"] - target_t)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            # If cited timestamp matches a retrieved source within 15 seconds, prioritize it
            if best_idx is not None and best_idx > 0 and best_diff <= 15.0:
                cited_source = sources.pop(best_idx)
                sources.insert(0, cited_source)
                primary_src = sources[0]
                primary_ts = {"start": primary_src["start"], "end": primary_src["end"]}
                timestamp_str = format_timestamp_display(primary_src["start"], primary_src["end"])
                timestamps = [{"start": s["start"], "end": s["end"]} for s in sources]

        return {
            "success": True,
            "answer": ans_clean,
            "sources": sources,
            "timestamps": timestamps,
            "answer_mode": "llm",
            "timestamp": timestamp_str,
            "timestamp_str": timestamp_str,
            "primary_timestamp": primary_ts,
            "start": primary_src["start"],
            "end": primary_src["end"],
            "results": [{"score": 1.0, "start": s["start"], "end": s["end"], "text": s["text"]} for s in sources[:3]]
        }

    # Step 6: Graceful fallback on LLM failure (e.g. rate limit, network timeout)
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
        "timestamp": timestamp_str,
        "timestamp_str": timestamp_str,
        "primary_timestamp": primary_ts,
        "start": primary_src["start"],
        "end": primary_src["end"],
        "results": [{"score": 1.0, "start": s["start"], "end": s["end"], "text": s["text"]} for s in sources[:3]],
        "error": err_msg
    }