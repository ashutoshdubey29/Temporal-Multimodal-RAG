"""
tests/test_pipeline_diagnostics.py

Automated regression test suite validating:
- Valid local video transcription & stream inspection
- Missing and corrupt / 0-byte local video
- FFmpeg unavailable handling
- Whisper model initialization & transcription failure
- Empty transcription results
- YouTube captions available & multi-language fallback
- YouTube captions unavailable with Whisper audio fallback
- YouTube audio fallback failure with specific diagnostic error
- Rate limiting and age-restriction diagnostics
- Network timeouts
- Correct timestamp conversion
- Flask endpoints (/upload, /process_youtube, /ask)
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import tempfile
from io import BytesIO

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, VIDEO_STORE
from processing.transcription import (
    ensure_ffmpeg,
    get_ffmpeg_version,
    inspect_audio_file,
    transcribe_video,
    get_whisper_model,
)
from processing.youtube_processor import (
    extract_youtube_video_id,
    get_youtube_metadata,
    fetch_youtube_captions,
    transcribe_youtube_audio,
    process_youtube_video,
)
from youtube_transcript_api import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    VideoUnplayable,
    AgeRestricted,
    IpBlocked,
    RequestBlocked,
)


class TestPipelineDiagnostics(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        VIDEO_STORE.clear()

    # =========================================================================
    # 1. LOCAL VIDEO TRANSCRIPTION & VALIDATION TESTS
    # =========================================================================

    @patch("processing.transcription.inspect_audio_file")
    @patch("processing.transcription.get_whisper_model")
    @patch("processing.transcription.ensure_ffmpeg")
    def test_valid_local_video_transcription(self, mock_ffmpeg, mock_model_getter, mock_inspect):
        """Test valid local video produces structured timestamped segments."""
        mock_ffmpeg.return_value = (True, "C:\\ffmpeg.exe")
        mock_inspect.return_value = (True, "", {"audio_stream": "Stream #0:1: Audio: aac"})

        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 2.543, "text": " Welcome to the demonstration. "},
                {"start": 2.543, "end": 6.891, "text": " This is multimodal temporal RAG. "},
            ]
        }
        mock_model_getter.return_value = mock_whisper

        # Create temporary dummy file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(b"dummy video data")
            temp_path = tf.name

        try:
            segments = transcribe_video(temp_path)
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0]["start"], 0.0)
            self.assertEqual(segments[0]["end"], 2.54)
            self.assertEqual(segments[0]["text"], "Welcome to the demonstration.")
            self.assertEqual(segments[1]["start"], 2.54)
            self.assertEqual(segments[1]["end"], 6.89)
            self.assertEqual(segments[1]["text"], "This is multimodal temporal RAG.")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_missing_local_video_raises_filenotfound(self):
        """Test non-existent video path raises FileNotFoundError."""
        fake_path = os.path.join(tempfile.gettempdir(), "non_existent_file_98765.mp4")
        with self.assertRaises(FileNotFoundError):
            transcribe_video(fake_path)

    def test_zero_byte_local_video_fails_inspection(self):
        """Test 0-byte video is flagged by pre-flight stream inspection."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            temp_path = tf.name

        try:
            valid, err, meta = inspect_audio_file(temp_path)
            self.assertFalse(valid)
            self.assertIn("0 bytes", err)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch("processing.transcription.subprocess.run")
    @patch("processing.transcription.ensure_ffmpeg")
    def test_local_video_missing_audio_stream(self, mock_ffmpeg, mock_run):
        """Test video without audio streams fails inspection with actionable message."""
        mock_ffmpeg.return_value = (True, "ffmpeg.exe")
        mock_proc = MagicMock()
        mock_proc.stderr = "Input #0, mov,mp4:\n  Stream #0:0(und): Video: h264\n"
        mock_proc.stdout = ""
        mock_run.return_value = mock_proc

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(b"video without audio track")
            temp_path = tf.name

        try:
            valid, err, meta = inspect_audio_file(temp_path)
            self.assertFalse(valid)
            self.assertIn("does not contain an audio track", err)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch("processing.transcription.ensure_ffmpeg")
    @patch("processing.transcription.inspect_audio_file")
    def test_ffmpeg_unavailable_raises_runtime_error(self, mock_inspect, mock_ffmpeg):
        """Test FFmpeg unavailable raises clear RuntimeError."""
        mock_inspect.return_value = (True, "", {})
        mock_ffmpeg.return_value = (False, "FFmpeg binary missing from system")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(b"dummy")
            temp_path = tf.name

        try:
            with self.assertRaises(RuntimeError) as ctx:
                transcribe_video(temp_path)
            self.assertIn("FFmpeg is required", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch("processing.transcription.get_whisper_model")
    @patch("processing.transcription.ensure_ffmpeg")
    @patch("processing.transcription.inspect_audio_file")
    def test_whisper_model_transcription_failure(self, mock_inspect, mock_ffmpeg, mock_model_getter):
        """Test Whisper execution error is caught and converted to descriptive error."""
        mock_ffmpeg.return_value = (True, "ffmpeg.exe")
        mock_inspect.return_value = (True, "", {})
        mock_whisper = MagicMock()
        mock_whisper.transcribe.side_effect = RuntimeError("Torch internal CUDA/memory error")
        mock_model_getter.return_value = mock_whisper

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(b"dummy")
            temp_path = tf.name

        try:
            with self.assertRaises(RuntimeError) as ctx:
                transcribe_video(temp_path)
            self.assertIn("Speech transcription failed", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch("processing.transcription.inspect_audio_file")
    @patch("processing.transcription.get_whisper_model")
    @patch("processing.transcription.ensure_ffmpeg")
    def test_empty_transcription_result(self, mock_ffmpeg, mock_model_getter, mock_inspect):
        """Test video that yields zero recognized segments."""
        mock_ffmpeg.return_value = (True, "ffmpeg.exe")
        mock_inspect.return_value = (True, "", {})
        mock_whisper = MagicMock()
        mock_whisper.transcribe.return_value = {"segments": []}
        mock_model_getter.return_value = mock_whisper

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(b"dummy silent video")
            temp_path = tf.name

        try:
            segments = transcribe_video(temp_path)
            self.assertEqual(segments, [])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # =========================================================================
    # 2. YOUTUBE CAPTION RETRIEVAL & DIAGNOSTICS TESTS
    # =========================================================================

    @patch("processing.youtube_processor.YouTubeTranscriptApi")
    def test_youtube_captions_available_direct(self, mock_api_cls):
        """Test standard direct English caption retrieval."""
        mock_instance = MagicMock()
        mock_api_cls.return_value = mock_instance

        mock_fetched = MagicMock()
        mock_fetched.to_raw_data.return_value = [
            {"start": 1.25, "duration": 4.5, "text": "In this video we study state machines."}
        ]
        mock_instance.fetch.return_value = mock_fetched

        segments, diag = fetch_youtube_captions("abc12345678", return_diagnostics=True)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 1.25)
        self.assertEqual(segments[0]["end"], 5.75)
        self.assertEqual(diag["reason"], "success")

    @patch("processing.youtube_processor.YouTubeTranscriptApi")
    def test_youtube_captions_disabled_diagnostic(self, mock_api_cls):
        """Test when transcripts are explicitly disabled by the owner."""
        mock_instance = MagicMock()
        mock_api_cls.return_value = mock_instance
        mock_instance.fetch.side_effect = TranscriptsDisabled("abc12345678")
        mock_instance.list.side_effect = TranscriptsDisabled("abc12345678")

        segments, diag = fetch_youtube_captions("abc12345678", return_diagnostics=True)
        self.assertEqual(segments, [])
        self.assertEqual(diag["reason"], "disabled")
        self.assertIn("disabled by the video owner", diag["detail"])

    @patch("processing.youtube_processor.YouTubeTranscriptApi")
    def test_youtube_rate_limiting_diagnostic(self, mock_api_cls):
        """Test when YouTube blocks/rate-limits caption queries."""
        mock_instance = MagicMock()
        mock_api_cls.return_value = mock_instance
        mock_instance.fetch.side_effect = IpBlocked("IP blocked by YouTube")
        mock_instance.list.side_effect = IpBlocked("IP blocked by YouTube")

        segments, diag = fetch_youtube_captions("abc12345678", return_diagnostics=True)
        self.assertEqual(segments, [])
        self.assertEqual(diag["reason"], "rate_limited")
        self.assertIn("rate-limited", diag["detail"])

    @patch("processing.youtube_processor.YouTubeTranscriptApi")
    def test_youtube_age_restricted_diagnostic(self, mock_api_cls):
        """Test when YouTube flags video as age-restricted."""
        mock_instance = MagicMock()
        mock_api_cls.return_value = mock_instance
        mock_instance.fetch.side_effect = AgeRestricted("Video is age restricted")
        mock_instance.list.side_effect = AgeRestricted("Video is age restricted")

        segments, diag = fetch_youtube_captions("abc12345678", return_diagnostics=True)
        self.assertEqual(segments, [])
        self.assertEqual(diag["reason"], "age_restricted")

    # =========================================================================
    # 3. YOUTUBE AUDIO FALLBACK & FULL PROCESSOR TESTS
    # =========================================================================

    @patch("processing.youtube_processor.fetch_youtube_captions")
    @patch("processing.youtube_processor.transcribe_youtube_audio")
    @patch("processing.youtube_processor.get_youtube_metadata")
    def test_process_youtube_audio_fallback_success(self, mock_meta, mock_audio, mock_captions):
        """Test captions fail but Whisper audio fallback succeeds."""
        mock_meta.return_value = {
            "accessible": True,
            "title": "Audio Speech Lecture",
            "author": "Professor",
            "thumbnail": "https://img.youtube.com/thumb.jpg"
        }
        mock_captions.return_value = ([], {"reason": "not_found", "detail": "No subtitle tracks available."})
        mock_audio.return_value = (
            [{"start": 0.0, "end": 5.0, "text": "Recognized from audio."}],
            {"success": True, "reason": "success", "detail": "Whisper audio transcribed."}
        )

        res = process_youtube_video("https://www.youtube.com/watch?v=12345678901", enable_audio_fallback=True)
        self.assertTrue(res["success"])
        self.assertEqual(res["transcript_source"], "whisper_audio")
        self.assertEqual(len(res["segments"]), 1)
        self.assertEqual(res["segments"][0]["text"], "Recognized from audio.")

    @patch("processing.youtube_processor.fetch_youtube_captions")
    @patch("processing.youtube_processor.transcribe_youtube_audio")
    @patch("processing.youtube_processor.get_youtube_metadata")
    def test_process_youtube_captions_disabled_audio_fails(self, mock_meta, mock_audio, mock_captions):
        """Test both captions disabled and audio fallback fails with specific diagnostic."""
        mock_meta.return_value = {
            "accessible": True,
            "title": "Private/Restricted Video",
            "author": "Creator",
            "thumbnail": "https://img.youtube.com/thumb.jpg"
        }
        mock_captions.return_value = ([], {"reason": "disabled", "detail": "Captions disabled."})
        mock_audio.return_value = ([], {"reason": "restricted", "detail": "Sign in required."})

        res = process_youtube_video("https://www.youtube.com/watch?v=12345678901", enable_audio_fallback=True)
        self.assertFalse(res["success"])
        self.assertIn("age-restricted or private", res["error"])
        self.assertEqual(res["error_type"], "restricted")

    @patch("processing.youtube_processor.fetch_youtube_captions")
    @patch("processing.youtube_processor.transcribe_youtube_audio")
    @patch("processing.youtube_processor.get_youtube_metadata")
    def test_process_youtube_rate_limited_diagnostic(self, mock_meta, mock_audio, mock_captions):
        """Test rate limiting produces actionable instructions for user."""
        mock_meta.return_value = {"accessible": True, "title": "Rate Limited Video"}
        mock_captions.return_value = ([], {"reason": "rate_limited", "detail": "Rate limited."})
        mock_audio.return_value = ([], {"reason": "rate_limited", "detail": "Rate limited."})

        res = process_youtube_video("https://www.youtube.com/watch?v=12345678901", enable_audio_fallback=True)
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "rate_limited")
        self.assertIn("temporarily rate-limited", res["error"])

    @patch("requests.get")
    def test_metadata_network_timeout(self, mock_get):
        """Test oEmbed network timeout is handled gracefully."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        meta = get_youtube_metadata("12345678901")
        # Should gracefully return fallback metadata
        self.assertTrue(meta["accessible"])
        self.assertIn("12345678901", meta["title"])

    # =========================================================================
    # 4. FLASK ROUTE REGRESSION TESTS (/upload, /process_youtube, /ask)
    # =========================================================================

    def test_upload_missing_file(self):
        """Test upload without file in form data."""
        res = self.client.post("/upload", data={}, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Please select a video file", res.get_data(as_text=True))

    def test_upload_unsupported_extension_json(self):
        """Test uploading non-video extension via JSON request."""
        data = {
            "video": (BytesIO(b"malicious script"), "script.exe")
        }
        res = self.client.post("/upload", data=data, content_type="multipart/form-data", headers={"Accept": "application/json"})
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data["success"])
        self.assertEqual(json_data["error_type"], "unsupported_format")
        self.assertIn("Unsupported file format", json_data["error"])

    def test_upload_empty_zero_byte_file(self):
        """Test uploading a 0-byte file returns empty_file error."""
        data = {
            "video": (BytesIO(b""), "empty.mp4")
        }
        res = self.client.post("/upload", data=data, content_type="multipart/form-data", headers={"Accept": "application/json"})
        self.assertEqual(res.status_code, 400)
        json_data = res.get_json()
        self.assertFalse(json_data["success"])
        self.assertEqual(json_data["error_type"], "empty_file")

    @patch("app.transcribe_video")
    def test_upload_valid_file_json(self, mock_transcribe):
        """Test successful local video upload and transcription."""
        mock_transcribe.return_value = [
            {"start": 0.0, "end": 3.0, "text": "Testing upload route."}
        ]
        data = {
            "video": (BytesIO(b"dummy valid mp4 content"), "sample_test.mp4")
        }
        res = self.client.post("/upload", data=data, content_type="multipart/form-data", headers={"Accept": "application/json"})
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["segments_count"], 1)
        self.assertIn("local_sample_test.mp4", VIDEO_STORE)

    @patch("app.process_youtube_video")
    def test_process_youtube_failure_json(self, mock_process):
        """Test /process_youtube returns structured error json on failure."""
        mock_process.return_value = {
            "success": False,
            "error": "Captions are disabled and audio fallback failed.",
            "error_type": "audio_fallback_failed",
            "caption_diagnostics": {"reason": "disabled"},
            "audio_diagnostics": {"reason": "restricted"}
        }
        res = self.client.post("/process_youtube", json={"youtube_url": "https://www.youtube.com/watch?v=12345678901"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_type"], "audio_fallback_failed")
        self.assertIn("Captions are disabled", data["error"])

    def test_ask_without_analyzed_video(self):
        """Test /ask when no video has been processed yet."""
        res = self.client.post("/ask", json={"question": "What is discussed?"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Please upload or analyze a video first", data["error"])


if __name__ == "__main__":
    unittest.main()
