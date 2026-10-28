"""
tests/test_rag_qa.py

Comprehensive test suite for the synthesized RAG Question Answering pipeline:
1. Multi-segment transcript synthesis
2. Broad summary questions
3. Specific factual questions
4. Irrelevant / out-of-domain questions
5. Empty / missing transcript data
6. LLM API failure (graceful fallback)
7. Missing API configuration (transparent fallback)
8. Complete answer with supporting timestamps
9. Response containing multiple paragraphs and bullet points
10. Correct HTML/frontend rendering of answer, sources, and timestamps
11. Preservation of local-video and YouTube workflows
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json
import requests

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, VIDEO_STORE
from rag.qa import (
    answer_question,
    find_relevant_segments,
    is_broad_query,
    retrieve_context_for_query,
    generate_fallback_answer,
    merge_adjacent_segments,
    format_seconds_to_time,
    format_timestamp_display,
    extract_cited_timestamps,
)
from rag.llm import (
    get_llm_config,
    call_openai_compatible,
    call_gemini,
    generate_llm_answer,
    SYSTEM_PROMPT,
)


class TestRAGQuestionAnswering(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        VIDEO_STORE.clear()

        # Sample realistic lecture transcript (segmented Whisper output)
        self.sample_segments = [
            {"start": 0.0, "end": 4.5, "text": "Welcome to today's lecture on state machine architectures."},
            {"start": 4.5, "end": 9.0, "text": "We will explore how complex workflows can be modeled with states."},
            {"start": 160.0, "end": 161.5, "text": "Now let us examine the main problem."},
            {"start": 161.52, "end": 164.64, "text": "because this is a state. In this, if I"},
            {"start": 164.64, "end": 168.0, "text": "transition from state A to state B, an event is emitted."},
            {"start": 168.0, "end": 172.5, "text": "The controller listens to this event and updates the database record."},
            {"start": 300.0, "end": 305.0, "text": "In summary, state machines prevent invalid state transitions reliably."},
            {"start": 305.0, "end": 310.0, "text": "Thank you for watching, and see you next time."},
        ]

    # =========================================================================
    # 1. Multi-Segment Transcript Synthesis
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_multi_segment_synthesis(self, mock_get_config, mock_llm):
        """Verify that segments fragmented across boundaries are merged and synthesized."""
        mock_get_config.return_value = {"is_configured": True, "provider": "openai"}
        mock_llm.return_value = (
            True,
            "The speaker explains that in this state machine architecture, when transitioning from state A to state B, an event is emitted. The controller catches this event and subsequently updates the database record.",
            ""
        )

        res = answer_question("Explain what happens when state transitions occur", self.sample_segments)

        self.assertTrue(res["success"])
        self.assertEqual(res["answer_mode"], "llm")
        self.assertIn("controller catches this event", res["answer"])
        # Verify supporting sources contain the merged passages
        self.assertGreaterEqual(len(res["sources"]), 1)
        merged_text = res["sources"][0]["text"]
        self.assertIn("transition from state A to state B", merged_text)
        self.assertIn("controller listens", merged_text)
        # Check start and end timestamps span the merged range
        self.assertLessEqual(res["sources"][0]["start"], 164.64)
        self.assertGreaterEqual(res["sources"][0]["end"], 168.0)

    # =========================================================================
    # 2. Broad Summary Question
    # =========================================================================
    def test_broad_summary_query_detection(self):
        """Verify is_broad_query accurately identifies overarching questions."""
        broad_queries = [
            "Explain the main concept discussed in this video",
            "Summarize the important points",
            "What problem is the speaker solving?",
            "What is this video about?",
            "Explain me question discussed in this video",
            "Give me an overview of the video",
        ]
        for q in broad_queries:
            self.assertTrue(is_broad_query(q), f"Failed to identify broad query: {q}")

        specific_queries = [
            "What happens at 160 seconds?",
            "What database record is updated?",
            "Why is state B emitted?",
        ]
        for q in specific_queries:
            self.assertFalse(is_broad_query(q), f"Incorrectly marked as broad: {q}")

    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_broad_summary_retrieval_and_answer(self, mock_get_config, mock_llm):
        """Verify broad summary retrieves timeline-spanning context and calls LLM."""
        mock_get_config.return_value = {"is_configured": True, "provider": "openai"}
        mock_llm.return_value = (
            True,
            "This video provides an overview of state machine architectures. It covers workflow modeling, event-driven state transitions, and how controllers update database records safely.",
            ""
        )

        res = answer_question("Explain the main concept discussed in this video", self.sample_segments)

        self.assertTrue(res["success"])
        self.assertEqual(res["answer_mode"], "llm")
        self.assertIn("state machine architectures", res["answer"])
        # Verify that multiple passages across timeline were retrieved
        self.assertGreaterEqual(len(res["sources"]), 2)
        # Check that beginning and concluding segments are represented
        all_text = " ".join(s["text"] for s in res["sources"])
        self.assertIn("Welcome", all_text)
        self.assertIn("state transitions", all_text)

    # =========================================================================
    # 3. Specific Factual Question
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_specific_factual_question(self, mock_get_config, mock_llm):
        """Verify a focused question retrieves the exact matching segment and neighbors."""
        mock_get_config.return_value = {"is_configured": True, "provider": "openai"}
        mock_llm.return_value = (
            True,
            "The controller listens to the transition event and updates the database record.",
            ""
        )

        res = answer_question("What updates the database record?", self.sample_segments)

        self.assertTrue(res["success"])
        self.assertIn("database record", res["answer"])
        # Source should target the ~168s range
        primary = res["sources"][0]
        self.assertIn("database", primary["text"].lower())
        self.assertAlmostEqual(primary["start"], 160.0, delta=15.0)

    # =========================================================================
    # 4. Irrelevant / Out-of-Domain Question
    # =========================================================================
    def test_irrelevant_question_returns_clean_no_match(self):
        """Verify asking completely unrelated questions returns no_match without hallucinations."""
        res = answer_question("What is the recipe for chocolate chip cookies?", self.sample_segments)

        self.assertTrue(res["success"])
        self.assertEqual(res["answer_mode"], "no_match")
        self.assertIn("No relevant information was found", res["answer"])
        self.assertEqual(res["sources"], [])
        self.assertEqual(res["timestamps"], [])
        self.assertEqual(res["timestamp_str"], "Not found")

    # =========================================================================
    # 5. Empty / Missing Transcript Data
    # =========================================================================
    def test_empty_transcript_handling(self):
        """Verify passing empty segments returns an informative error message."""
        res = answer_question("What is discussed?", [])
        self.assertFalse(res["success"])
        self.assertEqual(res["answer_mode"], "error")
        self.assertIn("upload or analyze a video", res["answer"])

    def test_empty_question_handling(self):
        """Verify passing empty question returns validation error."""
        res = answer_question("   ", self.sample_segments)
        self.assertFalse(res["success"])
        self.assertEqual(res["answer_mode"], "error")
        self.assertIn("valid question", res["answer"])

    # =========================================================================
    # 6. LLM API Failure (Graceful Fallback)
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_llm_api_failure_fallback(self, mock_get_config, mock_llm):
        """Verify when LLM throws an error or times out, a clean transparent fallback is returned."""
        mock_get_config.return_value = {"is_configured": True, "provider": "openai"}
        # Simulate LLM timeout / network failure
        mock_llm.return_value = (False, "", "Request timed out after 30 seconds")

        res = answer_question("What event is emitted?", self.sample_segments)

        self.assertTrue(res["success"])
        self.assertEqual(res["answer_mode"], "fallback")
        # Should NOT pretend to be an AI answer
        self.assertIn("AI answer generation is currently unavailable", res["answer"])
        self.assertIn("relevant transcript excerpts", res["answer"])
        # Should preserve the relevant excerpt and timestamps
        self.assertGreaterEqual(len(res["sources"]), 1)
        self.assertIn("event is emitted", res["answer"])
        self.assertIsNotNone(res["primary_timestamp"])

    # =========================================================================
    # 7. Missing API Configuration (Transparent Fallback)
    # =========================================================================
    @patch("rag.qa.get_llm_config")
    def test_missing_api_config_fallback(self, mock_get_config):
        """Verify that when no API key is configured, fallback is clearly labeled and accurate."""
        mock_get_config.return_value = {"is_configured": False, "provider": "none"}

        res = answer_question("How do state transitions work?", self.sample_segments)

        self.assertTrue(res["success"])
        self.assertIn("no llm api key has been configured", res["answer"].lower())
        self.assertIn("OPENAI_API_KEY or GEMINI_API_KEY", res["answer"])
        # Sources are provided transparently
        self.assertGreaterEqual(len(res["sources"]), 1)
        self.assertIn("transition from state A", res["sources"][0]["text"])

    # =========================================================================
    # 8. Complete Answer with Supporting Timestamps
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_complete_answer_with_supporting_timestamps(self, mock_get_config, mock_llm):
        """Verify separate answer, sources, and timestamps fields with valid numeric ranges."""
        mock_get_config.return_value = {"is_configured": True, "provider": "openai"}
        mock_llm.return_value = (
            True,
            "State machines ensure that only valid state transitions occur. When a transition happens, an event is emitted and processed by the controller.",
            ""
        )

        res = answer_question("Why use state machines?", self.sample_segments)

        self.assertTrue(res["success"])
        self.assertIsInstance(res["answer"], str)
        self.assertIsInstance(res["sources"], list)
        self.assertIsInstance(res["timestamps"], list)
        self.assertEqual(len(res["sources"]), len(res["timestamps"]))

        for ts in res["timestamps"]:
            self.assertIn("start", ts)
            self.assertIn("end", ts)
            self.assertGreaterEqual(ts["end"], ts["start"])

    # =========================================================================
    # 9. Response Containing Multiple Paragraphs and Bullet Points
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_multiparagraph_bullet_response(self, mock_get_config, mock_llm):
        """Verify structured multi-paragraph and bullet-point answers are preserved without truncation."""
        multi_part_answer = (
            "The state machine architecture operates through the following steps:\n\n"
            "• **State Transition**: The system moves from state A to state B based on valid business rules.\n"
            "• **Event Emission**: An event notification is generated immediately upon transition.\n"
            "• **Controller Handling**: The controller intercepts the event and updates relevant database tables.\n\n"
            "This pattern isolates state management logic from database persistence."
        )
        mock_get_config.return_value = {"is_configured": True, "provider": "openai"}
        mock_llm.return_value = (True, multi_part_answer, "")

        res = answer_question("Explain the steps in the state transition process", self.sample_segments)

        self.assertEqual(res["answer"], multi_part_answer)
        self.assertIn("• **State Transition**", res["answer"])
        self.assertIn("\n\n", res["answer"])

    # =========================================================================
    # 10. Frontend / HTML Template Rendering of Answer and Sources
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_html_rendering_of_answer_and_sources(self, mock_get_config, mock_llm):
        """Verify POST /ask HTML response renders the answer card, badge, and source cards."""
        mock_get_config.return_value = {"is_configured": True, "provider": "openai"}
        mock_llm.return_value = (
            True,
            "State transitions emit events that the controller handles.",
            ""
        )

        # Store test video in VIDEO_STORE
        v_id = "test_html_video"
        VIDEO_STORE[v_id] = {
            "id": v_id,
            "type": "local",
            "filename": "test_state.mp4",
            "segments": self.sample_segments
        }

        resp = self.client.post(
            "/ask",
            data={"question": "What happens on state transition?", "source_id": v_id},
            headers={"Accept": "text/html"}
        )

        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # Verify AI synthesized badge
        self.assertIn("✦ AI Synthesized", html)
        # Verify primary answer body
        self.assertIn('id="answerBody"', html)
        self.assertIn("State transitions emit events that the controller handles.", html)
        # Verify time-chip
        self.assertIn("class=\"time-chip\"", html)
        # Verify Jump to timestamp button
        self.assertIn("Jump to timestamp in video", html)
        # Verify Supporting Transcript Evidence section
        self.assertIn("Supporting Transcript Evidence", html)
        self.assertIn("source-card", html)

    # =========================================================================
    # 11. Preservation of Local-Video and YouTube Workflows
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_preservation_of_local_and_youtube_workflows(self, mock_get_config, mock_llm):
        """Verify JSON API and state isolation work for both local and YouTube videos."""
        mock_get_config.return_value = {"is_configured": True, "provider": "openai"}
        mock_llm.return_value = (
            True,
            "YouTube test explanation.",
            ""
        )

        yt_id = "yt_test_vid_11"
        VIDEO_STORE[yt_id] = {
            "id": yt_id,
            "type": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "title": "State Machines on YouTube",
            "segments": self.sample_segments
        }

        # Query YouTube video
        resp = self.client.post(
            "/ask",
            json={"question": "What event is emitted?", "source_id": yt_id}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["answer_mode"], "llm")
        self.assertIn("YouTube test explanation", data["answer"])
        self.assertIn("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=", data["youtube_url"])
    # =========================================================================
    # 12. Human-Readable Timestamps and Helper Utilities
    # =========================================================================
    def test_timestamp_formatting_helpers(self):
        """Verify format_seconds_to_time and format_timestamp_display convert seconds to MM:SS."""
        self.assertEqual(format_seconds_to_time(394.20), "06:34")
        self.assertEqual(format_seconds_to_time(409.84), "06:50")
        self.assertEqual(format_seconds_to_time(0.0), "00:00")
        self.assertEqual(format_seconds_to_time(3665.0), "01:01:05")
        self.assertEqual(format_seconds_to_time(None), "")

        # Range formatting
        self.assertEqual(format_timestamp_display(394.20, 409.84), "06:34–06:50")
        self.assertEqual(format_timestamp_display(10.0, None), "00:10")
        self.assertEqual(format_timestamp_display(None, None), "Not available")

    def test_extract_cited_timestamps(self):
        """Verify extract_cited_timestamps discovers timestamps from text in various formats."""
        text1 = "A walk is defined as an alternating sequence of vertices and edges [06:34–06:50]."
        cited1 = extract_cited_timestamps(text1)
        self.assertEqual(len(cited1), 2)
        self.assertAlmostEqual(cited1[0], 394.0, delta=1.0)
        self.assertAlmostEqual(cited1[1], 410.0, delta=1.0)

        text2 = "Refer to 394.20s – 409.84s in the video."
        cited2 = extract_cited_timestamps(text2)
        self.assertEqual(len(cited2), 2)
        self.assertAlmostEqual(cited2[0], 394.2, delta=0.1)

    # =========================================================================
    # 13. Timestamp Citation and Navigation Synchronization
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_timestamp_synchronization_with_cited_answer(self, mock_get_config, mock_llm):
        """Verify when LLM cites a timestamp range, primary_timestamp and navigation sync to that range."""
        mock_get_config.return_value = {"is_configured": True, "provider": "gemini"}
        # Simulate LLM referencing the 164s transition segment
        mock_llm.return_value = (
            True,
            "State transitions emit events that the controller handles [02:44–02:48].",
            ""
        )

        res = answer_question("Explain what happens when state transitions occur", self.sample_segments)

        self.assertTrue(res["success"])
        self.assertEqual(res["answer_mode"], "llm")
        # 164.64 seconds is ~02:45
        self.assertIn("02:4", res["timestamp_str"])
        self.assertAlmostEqual(res["primary_timestamp"]["start"], 164.64, delta=5.0)
        # Ensure the first source corresponds to the cited evidence
        self.assertIn("transition from state A", res["sources"][0]["text"])

    # =========================================================================
    # 14. Asynchronous JSON /ask Endpoint Response Consistency
    # =========================================================================
    @patch("rag.qa.generate_llm_answer")
    @patch("rag.qa.get_llm_config")
    def test_json_ask_endpoint_consistent_data(self, mock_get_config, mock_llm):
        """Verify POST /ask with JSON returns consistent timestamp, start, end, and sources."""
        mock_get_config.return_value = {"is_configured": True, "provider": "gemini"}
        mock_llm.return_value = (
            True,
            "In graph theory, a **vertex** is a node and an **edge** is a connection [02:44–02:48].",
            ""
        )

        v_id = "test_json_sync_video"
        VIDEO_STORE[v_id] = {
            "id": v_id,
            "type": "local",
            "filename": "graph_theory.mp4",
            "segments": [
                {
                    "start": 164.64,
                    "end": 168.96,
                    "text": "A walk in graph theory is an alternating sequence of vertices and edges."
                }
            ]
        }

        resp = self.client.post(
            "/ask",
            json={"question": "What are vertices and edges?", "source_id": v_id}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertTrue(data["success"])
        self.assertEqual(data["answer_mode"], "llm")
        self.assertIn("vertex", data["answer"].lower())
        self.assertIsNotNone(data["start"])
        self.assertIsNotNone(data["end"])
        self.assertIsInstance(data["timestamp"], str)
        self.assertIn("–", data["timestamp"])
        self.assertGreaterEqual(len(data["sources"]), 1)
        # Verify sources have timestamp_str
        self.assertIn("timestamp_str", data["sources"][0])

    # =========================================================================
    # 15. HTML Template Structure and Form Attributes
    # =========================================================================
    def test_html_template_elements_for_async_ask(self):
        """Verify the template includes askSubmitBtn, questionResultArea, and answerContainer without blocking overlays."""
        v_id = "test_template_video"
        VIDEO_STORE[v_id] = {
            "id": v_id,
            "type": "local",
            "filename": "test.mp4",
            "segments": self.sample_segments
        }

        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # Verify askForm does not have data-loading overlay trigger
        self.assertIn('id="askForm"', html)
        self.assertNotIn('id="askForm" action="/ask" method="POST" data-loading=', html)
        # Verify submit button ID
        self.assertIn('id="askSubmitBtn"', html)
        # Verify questionResultArea and answerContainer are in the DOM
        self.assertIn('id="questionResultArea"', html)
        self.assertIn('id="answerContainer"', html)


class TestLLMClientFunctions(unittest.TestCase):
    """Direct tests for rag/llm.py provider adapters."""

    def test_system_prompt_requirements(self):
        """Verify the system prompt contains all required instructions from the specification."""
        self.assertIn("video-understanding assistant", SYSTEM_PROMPT)
        self.assertIn("Start with a direct answer to the user's question.", SYSTEM_PROMPT)
        self.assertIn("Never treat timestamps as the answer itself.", SYSTEM_PROMPT)
        self.assertIn("Never claim that visual content was analyzed if only transcript evidence was supplied.", SYSTEM_PROMPT)

    @patch("rag.llm.requests.post")
    def test_call_openai_compatible_success(self, mock_post):
        """Verify successful response parsing from OpenAI-compatible endpoint."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": "This is a synthesized test answer."}}
            ]
        }
        mock_post.return_value = mock_resp

        ok, ans, err = call_openai_compatible(
            question="What is this?",
            context_text="[0s - 5s]: Test context",
            api_key="sk-testkey",
            api_base="https://api.openai.com/v1",
            model="gpt-4o-mini"
        )
        self.assertTrue(ok)
        self.assertEqual(ans, "This is a synthesized test answer.")
        self.assertEqual(err, "")

    @patch("rag.llm.requests.post")
    def test_call_openai_compatible_timeout(self, mock_post):
        """Verify timeout handling in OpenAI client."""
        mock_post.side_effect = requests.Timeout("Connection timed out")

        ok, ans, err = call_openai_compatible(
            question="What is this?",
            context_text="[0s - 5s]: Test context",
            api_key="sk-testkey",
            api_base="https://api.openai.com/v1",
            model="gpt-4o-mini"
        )
        self.assertFalse(ok)
        self.assertEqual(ans, "")
        self.assertIn("timed out", err.lower())

    @patch("rag.llm.requests.post")
    def test_call_gemini_success(self, mock_post):
        """Verify successful response parsing from Gemini REST endpoint."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini synthesized answer."}]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        ok, ans, err = call_gemini(
            question="What is this?",
            context_text="[0s - 5s]: Test context",
            api_key="gemini-key-123",
            model="gemini-1.5-flash"
        )
        self.assertTrue(ok)
        self.assertEqual(ans, "Gemini synthesized answer.")
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
