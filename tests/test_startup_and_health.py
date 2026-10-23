import unittest
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app, VIDEO_STORE, check_port_available, find_available_port, get_server_config
from processing.transcription import ensure_ffmpeg


class TestStartupAndHealth(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_health_endpoint_success(self):
        """Verify GET /health returns 200 OK and expected diagnostic schema."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("healthy"))
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("app"), "Temporal-Multimodal-RAG")
        self.assertIn("ffmpeg_available", data)
        self.assertIn("active_videos", data)
        self.assertEqual(data.get("model_status"), "ready")

    def test_dynamic_ffmpeg_discovery(self):
        """Verify ensure_ffmpeg resolves a valid FFmpeg binary."""
        available, path_or_msg = ensure_ffmpeg()
        self.assertTrue(available, f"FFmpeg could not be resolved: {path_or_msg}")
        self.assertTrue(os.path.exists(path_or_msg))

    def test_check_port_available(self):
        """Verify port availability checker functions properly."""
        # Port 0 or an arbitrary high port should be available
        is_free = check_port_available("127.0.0.1", 58999)
        self.assertTrue(is_free)

    def test_find_available_port(self):
        """Verify finding an available port returns a valid port integer."""
        port = find_available_port("127.0.0.1", 58000)
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 58000)

    def test_get_server_config_defaults(self):
        """Verify server configuration defaults."""
        # Clear env temporarily
        old_host = os.environ.pop("HOST", None)
        old_port = os.environ.pop("PORT", None)
        old_debug = os.environ.pop("FLASK_DEBUG", None)
        try:
            host, port, debug = get_server_config()
            self.assertEqual(host, "127.0.0.1")
            self.assertEqual(port, 5000)
            self.assertFalse(debug)
        finally:
            if old_host is not None:
                os.environ["HOST"] = old_host
            if old_port is not None:
                os.environ["PORT"] = old_port
            if old_debug is not None:
                os.environ["FLASK_DEBUG"] = old_debug

    def test_get_server_config_overrides(self):
        """Verify server configuration honors environment variables."""
        os.environ["HOST"] = "0.0.0.0"
        os.environ["PORT"] = "8080"
        os.environ["FLASK_DEBUG"] = "true"
        try:
            host, port, debug = get_server_config()
            self.assertEqual(host, "0.0.0.0")
            self.assertEqual(port, 8080)
            self.assertTrue(debug)
        finally:
            os.environ.pop("HOST", None)
            os.environ.pop("PORT", None)
            os.environ.pop("FLASK_DEBUG", None)

    def test_find_available_port_when_occupied(self):
        """Verify that when a port is occupied, find_available_port selects the next open port."""
        import socket
        base_port = 59100
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", base_port))
        sock.listen(1)
        try:
            is_occupied = not check_port_available("127.0.0.1", base_port)
            self.assertTrue(is_occupied)
            next_port = find_available_port("127.0.0.1", base_port)
            self.assertGreater(next_port, base_port)
        finally:
            sock.close()

    def test_run_coordinator_checks(self):
        """Verify run.py pre-flight check functions."""
        from run import check_python_environment, check_dependencies, check_youtube_integration
        py_ok, _ = check_python_environment()
        self.assertTrue(py_ok)
        deps_ok, missing = check_dependencies()
        self.assertTrue(deps_ok, f"Missing dependencies: {missing}")
        yt_ok, _ = check_youtube_integration()
        self.assertTrue(yt_ok)


if __name__ == "__main__":
    unittest.main()
