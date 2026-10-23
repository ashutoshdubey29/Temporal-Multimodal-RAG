"""
run.py

Startup coordinator for Temporal-Multimodal-RAG:
- Verifies Python environment and application dependencies
- Dynamically resolves FFmpeg availability
- Checks YouTube integration status
- Resolves available host and port without collisions
- Starts the Flask application
- Verifies server readiness via GET /health endpoint
- Displays startup status banner with the working URL
- Automatically opens the default browser
- Gracefully terminates on Ctrl+C
"""

import argparse
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from werkzeug.serving import make_server

# Ensure immediate console output flushing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
def load_env_file(env_path: str = ".env") -> None:
    """Loads simple KEY=VALUE configuration from a local .env file if present."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
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


def check_python_environment() -> tuple[bool, str]:
    """Ensures Python version is 3.10 or compatible."""
    v = sys.version_info
    if v.major == 3 and v.minor >= 10:
        return True, f"Python {v.major}.{v.minor}.{v.micro}"
    return False, f"Unsupported Python version: {sys.version}. Python 3.10+ is required."


def check_dependencies() -> tuple[bool, list[str]]:
    """Verifies that all required packages can be imported."""
    required = [
        ("flask", "Flask"),
        ("flask_sqlalchemy", "Flask-SQLAlchemy"),
        ("werkzeug", "Werkzeug"),
        ("whisper", "openai-whisper"),
        ("torch", "torch"),
        ("imageio_ffmpeg", "imageio-ffmpeg"),
        ("youtube_transcript_api", "youtube-transcript-api"),
        ("yt_dlp", "yt-dlp"),
        ("requests", "requests"),
    ]
    missing = []
    for module_name, pkg_name in required:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pkg_name)
    return len(missing) == 0, missing


def check_youtube_integration() -> tuple[bool, str]:
    """Tests YouTube processor imports and connectivity."""
    try:
        from processing.youtube_processor import extract_youtube_video_id
        valid, _, _ = extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        if not valid:
            return False, "Validation logic error"

        import requests
        try:
            resp = requests.get(
                "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ&format=json",
                timeout=2.0
            )
            if resp.status_code == 200:
                return True, "Online"
            return True, f"Restricted (status {resp.status_code})"
        except Exception:
            return True, "Offline fallback ready"
    except Exception as e:
        return False, str(e)


def wait_for_health(url: str, timeout_sec: float = 15.0) -> bool:
    """Polls the /health endpoint until the server is responsive or times out."""
    start_time = time.time()
    health_url = f"{url.rstrip('/')}/health"
    while time.time() - start_time < timeout_sec:
        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "TemporalRAG-StartupCheck/1.0"}
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return False


def main():
    parser = argparse.ArgumentParser(description="Start Temporal-Multimodal-RAG application.")
    parser.add_argument("--host", default=None, help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--debug", action="store_true", help="Run with Flask debug reloader")
    args = parser.parse_args()

    # Load local environment configuration
    load_env_file()

    # 1. Check Python environment
    py_ok, py_info = check_python_environment()
    if not py_ok:
        print("\n====================================================")
        print(" [ERROR] Python Environment Incompatible")
        print("====================================================")
        print(py_info)
        print("Please run 'setup.bat' to initialize Python 3.10 virtual environment.")
        sys.exit(1)

    # 2. Check dependencies
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print("\n====================================================")
        print(" [ERROR] Missing Application Dependencies")
        print("====================================================")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease run 'setup.bat' to install required packages.")
        sys.exit(1)

    # 3. Check Dynamic FFmpeg
    from processing.transcription import ensure_ffmpeg
    ffmpeg_ok, ffmpeg_info = ensure_ffmpeg()

    # 4. Check YouTube integration
    yt_ok, yt_msg = check_youtube_integration()

    # 5. Import Flask app and port resolution helpers
    from app import app, check_port_available, find_available_port

    host = args.host or os.environ.get("HOST", "127.0.0.1").strip()
    requested_port = args.port or int(os.environ.get("PORT", "5000").strip())
    debug_mode = args.debug or os.environ.get("FLASK_DEBUG", "false").strip().lower() in ("true", "1", "yes")

    # Resolve port conflicts
    actual_port = requested_port
    if not check_port_available(host, requested_port):
        actual_port = find_available_port(host, requested_port)
        print(f"[INFO] Port {requested_port} is in use. Selected open port {actual_port}.")

    app_url = f"http://{host}:{actual_port}"

    # 6. Start the server
    if debug_mode:
        # Debug mode with reloader
        print(f"Starting server in debug mode at {app_url}...")
        if not args.no_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
            # Schedule browser opening once server starts
            threading.Timer(1.2, lambda: webbrowser.open(app_url)).start()
        app.run(host=host, port=actual_port, debug=True)
        return

    # Standard clean server startup using make_server
    try:
        server = make_server(host, actual_port, app, threaded=True)
    except Exception as e:
        print(f"\n[ERROR] Could not start server on {host}:{actual_port}: {e}")
        sys.exit(1)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # 7. Wait until the server is actually responding on /health
    ready = wait_for_health(app_url, timeout_sec=15.0)
    if not ready:
        print(f"\n[ERROR] Server failed to respond to health check at {app_url}/health within 15 seconds.")
        server.shutdown()
        sys.exit(1)

    # 8. Print startup status banner
    print("====================================================")
    print("       TEMPORAL MULTIMODAL RAG")
    print("====================================================")
    print()
    print("[OK] Python environment ready")
    print("[OK] Dependencies verified")

    if ffmpeg_ok:
        print("[OK] FFmpeg available")
    else:
        print("[WARN] FFmpeg not found (Transcription will fail)")

    print("[OK] Flask server running")

    if yt_ok:
        print("[OK] YouTube integration initialized")
    else:
        print(f"[WARN] YouTube integration limited: {yt_msg}")

    from rag.llm import get_llm_config
    llm_conf = get_llm_config()
    if llm_conf.get("is_configured"):
        p_name = llm_conf.get("provider", "LLM").upper()
        p_model = llm_conf.get("openai_model") if "openai" in p_name.lower() or "groq" in p_name.lower() else llm_conf.get("gemini_model")
        print(f"[OK] AI QA Synthesis ready ({p_name} - {p_model})")
    else:
        print("[INFO] AI QA Synthesis: Fallback mode (Configure API key in .env for full LLM answers)")

    print()
    print("APPLICATION URL:")
    print(app_url)
    print()

    # 9. Automatically open default browser
    if not args.no_browser:
        print("Opening application in your default browser...")
        try:
            webbrowser.open(app_url)
        except Exception:
            pass
    print()
    print("Press Ctrl+C to stop the server.")
    print("====================================================")
    print()

    # 10. Run until Ctrl+C
    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Temporal-Multimodal-RAG server...")
        try:
            server.shutdown()
        except Exception:
            pass
        print("Server stopped cleanly. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
