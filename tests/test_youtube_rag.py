import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing.youtube_processor import (
    extract_youtube_video_id,
    get_youtube_metadata,
    fetch_youtube_captions,
    process_youtube_video,
)
from rag.qa import find_relevant_segments
from app import app, VIDEO_STORE


class TestYouTubeRAG(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        VIDEO_STORE.clear()

    # ==========================================
    # 1. URL VALIDATION TESTS
    # ==========================================

    def test_valid_standard_youtube_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        valid, vid, err = extract_youtube_video_id(url)
        self.assertTrue(valid)
        self.assertEqual(vid, "dQw4w9WgXcQ")
        self.assertEqual(err, "")

    def test_valid_shortened_youtube_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        valid, vid, err = extract_youtube_video_id(url)
        self.assertTrue(valid)
        self.assertEqual(vid, "dQw4w9WgXcQ")

    def test_valid_shorts_youtube_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        valid, vid, err = extract_youtube_video_id(url)
        self.assertTrue(valid)
        self.assertEqual(vid, "dQw4w9WgXcQ")

    def test_valid_embed_youtube_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        valid, vid, err = extract_youtube_video_id(url)
        self.assertTrue(valid)
        self.assertEqual(vid, "dQw4w9WgXcQ")

    def test_valid_url_with_parameters(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&feature=share"
        valid, vid, err = extract_youtube_video_id(url)
        self.assertTrue(valid)
        self.assertEqual(vid, "dQw4w9WgXcQ")

    def test_invalid_urls(self):
        invalid_cases = [
            "",
            "   ",
            "not_a_url",
            "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
            None,
        ]
        for url in invalid_cases:
            valid, vid, err = extract_youtube_video_id(url)
            self.assertFalse(valid)
            self.assertEqual(vid, "")

    def test_unsupported_external_domain(self):
        unsupported = [
            "https://vimeo.com/12345678901",
            "https://dailymotion.com/video/x7tgad0",
            "https://evil.com/youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com.attacker.com/watch?v=dQw4w9WgXcQ",
        ]
        for url in unsupported:
            valid, vid, err = extract_youtube_video_id(url)
            self.assertFalse(valid)
            self.assertIn("Unsupported domain", err)

    def test_missing_video_id(self):
        url = "https://www.youtube.com/watch"
        valid, vid, err = extract_youtube_video_id(url)
        self.assertFalse(valid)
        self.assertIn("Could not find a valid YouTube video ID", err)

    def test_malformed_video_id(self):
        url = "https://www.youtube.com/watch?v=tooshort"
        valid, vid, err = extract_youtube_video_id(url)
        self.assertFalse(valid)
        self.assertIn("must be exactly 11 characters", err)

    # ==========================================
    # 2. METADATA RETRIEVAL TESTS
    # ==========================================

    @patch("requests.get")
    def test_metadata_retrieval_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "title": "Quantum Physics Lecture 1",
            "author_name": "MIT OpenCourseWare",
            "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        }
        mock_get.return_value = mock_resp

        meta = get_youtube_metadata("dQw4w9WgXcQ")
        self.assertTrue(meta["accessible"])
        self.assertEqual(meta["title"], "Quantum Physics Lecture 1")
        self.assertEqual(meta["author"], "MIT OpenCourseWare")

    @patch("requests.get")
    def test_metadata_retrieval_failure_private(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        meta = get_youtube_metadata("dQw4w9WgXcQ")
        self.assertFalse(meta["accessible"])
        self.assertIn("private or restricted", meta["error"])

    # ==========================================
    # 3. CAPTIONS & TIMESTAMP CONVERSION TESTS
    # ==========================================

    @patch("processing.youtube_processor.YouTubeTranscriptApi")
    def test_caption_fetching_and_timestamp_conversion(self, mock_api_cls):
        mock_instance = MagicMock()
        mock_api_cls.return_value = mock_instance

        mock_fetched = MagicMock()
        mock_fetched.to_raw_data.return_value = [
            {"text": "Hello &amp; welcome to the lecture.", "start": 0.5, "duration": 3.2},
            {"text": "Today we will discuss &quot;photons&quot;.", "start": 3.7, "duration": 4.1},
        ]
        mock_instance.fetch.return_value = mock_fetched

        segments = fetch_youtube_captions("dQw4w9WgXcQ")
        self.assertEqual(len(segments), 2)
        # Check start & end
        self.assertEqual(segments[0]["start"], 0.5)
        self.assertEqual(segments[0]["end"], 3.7)
        # Check HTML unescaping
        self.assertEqual(segments[0]["text"], "Hello & welcome to the lecture.")
        self.assertEqual(segments[1]["text"], 'Today we will discuss "photons".')

    @patch("processing.youtube_processor.YouTubeTranscriptApi")
    def test_empty_transcript(self, mock_api_cls):
        mock_instance = MagicMock()
        mock_api_cls.return_value = mock_instance
        mock_instance.fetch.side_effect = Exception("No captions found")
        mock_instance.list.side_effect = Exception("No captions list")

        segments = fetch_youtube_captions("dQw4w9WgXcQ")
        self.assertEqual(segments, [])

    # ==========================================
    # 4. AUDIO TRANSCRIPTION FALLBACK TEST
    # ==========================================

    @patch("processing.youtube_processor.fetch_youtube_captions")
    @patch("processing.youtube_processor.transcribe_youtube_audio")
    @patch("processing.youtube_processor.get_youtube_metadata")
    def test_authorized_audio_fallback(self, mock_meta, mock_audio, mock_captions):
        mock_meta.return_value = {
            "accessible": True,
            "title": "Fallback Video",
            "author": "Creator",
            "thumbnail": "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg"
        }
        # Captions unavailable
        mock_captions.return_value = []
        # Whisper audio fallback succeeds
        mock_audio.return_value = [
            {"start": 1.0, "end": 4.0, "text": "Speech recognized by Whisper."}
        ]

        result = process_youtube_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", enable_audio_fallback=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["transcript_source"], "whisper_audio")
        self.assertEqual(len(result["segments"]), 1)

    # ==========================================
    # 5. RAG RETRIEVAL WITH YOUTUBE TRANSCRIPT
    # ==========================================

    def test_qa_retrieval_with_youtube_transcript(self):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "In this video we introduce neural networks."},
            {"start": 5.0, "end": 12.5, "text": "Backpropagation is used to calculate the gradient."},
            {"start": 12.5, "end": 20.0, "text": "Gradient descent updates the weights of the network."},
        ]

        # Query with punctuation
        results = find_relevant_segments("How does backpropagation work?", segments)
        self.assertGreater(len(results), 0)
        self.assertIn("Backpropagation", results[0]["text"])
        self.assertEqual(results[0]["start"], 5.0)

        # Query with non-matching words
        no_results = find_relevant_segments("astrophysics galaxy collision", segments)
        self.assertEqual(len(no_results), 0)

    # ==========================================
    # 6. VIDEO SWITCHING & STATE ISOLATION
    # ==========================================

    def test_switching_videos_state_isolation(self):
        # Video A: Biology
        video_a_id = "yt_video_bio_01"
        VIDEO_STORE[video_a_id] = {
            "id": video_a_id,
            "type": "youtube",
            "title": "Cell Biology",
            "segments": [
                {"start": 0.0, "end": 10.0, "text": "Mitochondria is the powerhouse of the cell."}
            ]
        }

        # Video B: History
        video_b_id = "yt_video_hist02"
        VIDEO_STORE[video_b_id] = {
            "id": video_b_id,
            "type": "youtube",
            "title": "Roman History",
            "segments": [
                {"start": 0.0, "end": 15.0, "text": "Julius Caesar crossed the Rubicon."}
            ]
        }

        # Query Video A
        res_a = self.client.post("/ask", json={"question": "What is mitochondria?", "source_id": video_a_id})
        data_a = res_a.get_json()
        self.assertTrue(data_a["success"])
        self.assertIn("Mitochondria", data_a["answer"])

        # Query Video B
        res_b = self.client.post("/ask", json={"question": "Who was Caesar?", "source_id": video_b_id})
        data_b = res_b.get_json()
        self.assertTrue(data_b["success"])
        self.assertIn("Caesar", data_b["answer"])

        # Ensure cross-query fails cleanly without leaking
        res_leak = self.client.post("/ask", json={"question": "mitochondria", "source_id": video_b_id})
        data_leak = res_leak.get_json()
        self.assertEqual(data_leak["timestamp"], "Not found")

    # ==========================================
    # 7. FLASK ENDPOINTS
    # ==========================================

    @patch("app.process_youtube_video")
    def test_process_youtube_endpoint_html_renders_iframe(self, mock_process):
        mock_process.return_value = {
            "success": True,
            "video_id": "dQw4w9WgXcQ",
            "title": "Never Gonna Give You Up",
            "author": "Rick Astley",
            "thumbnail": "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
            "duration": 212.0,
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "transcript_source": "captions",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Never gonna give you up"}]
        }

        # Submit via HTML Form
        resp = self.client.post("/process_youtube", data={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
        self.assertEqual(resp.status_code, 200)
        html_text = resp.get_data(as_text=True)
        # Verify valid embed URL is present in iframe src
        self.assertIn("https://www.youtube.com/embed/dQw4w9WgXcQ", html_text)
        self.assertIn('id="ytPlayer"', html_text)
        self.assertIn("Never Gonna Give You Up", html_text)

    @patch("app.process_youtube_video")
    def test_process_youtube_endpoint_json(self, mock_process):
        mock_process.return_value = {
            "success": True,
            "video_id": "dQw4w9WgXcQ",
            "title": "Never Gonna Give You Up",
            "author": "Rick Astley",
            "thumbnail": "https://img.youtube.com/vi/dQw4w9WgXcQ/hqdefault.jpg",
            "duration": 212.0,
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "embed_url": "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "transcript_source": "captions",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Never gonna give you up"}]
        }

        resp = self.client.post("/process_youtube", json={"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["video"]["video_id"], "dQw4w9WgXcQ")

    def test_process_youtube_endpoint_invalid_url(self):
        resp = self.client.post("/process_youtube", json={"youtube_url": "https://vimeo.com/12345678901"})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Unsupported domain", data["error"])

    @patch("app.transcribe_video")
    def test_local_upload_preservation(self, mock_transcribe):
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 4.0, "text": "Local test transcription."}
        ]
        from io import BytesIO
        data = {
            "video": (BytesIO(b"dummy video content"), "test_lecture.mp4")
        }
        resp = self.client.post("/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        # Check that local video is in VIDEO_STORE
        self.assertIn("local_test_lecture.mp4", VIDEO_STORE)
        self.assertEqual(len(VIDEO_STORE["local_test_lecture.mp4"]["segments"]), 1)


if __name__ == "__main__":
    unittest.main()
