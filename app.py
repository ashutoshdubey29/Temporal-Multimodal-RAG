"""
app.py
======
VIVA / ARCHITECTURAL OVERVIEW:
This is the core Flask application entry point for the Temporal Multimodal RAG platform (AetherLearn).

Key Architecture to Explain in a Viva:
1. Flask Web Application:
   - Configures application settings, session secrets, upload directories, and database URIs.
   - Registers modular Blueprints: `auth_bp` (authentication & onboarding) and `learning_bp` (dashboard, quizzes, paths).
   - Injects the currently authenticated student into Jinja templates using `@app.context_processor`.
2. Database Optimization:
   - Sets SQLite Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and a 30-second busy timeout
     (`PRAGMA busy_timeout=30000`) via SQLAlchemy event listeners to prevent database locking
     during concurrent reads and writes.
3. Multi-Video State Management:
   - Maintains an in-memory dictionary `VIDEO_STORE` mapping `source_id` to video metadata and transcript segments.
   - When a student is logged in, sessions are also persisted to the SQLite database
     (`VideoLearningSession`), guaranteeing multi-user isolation.
4. Core Video Pipeline Endpoints:
   - `POST /process_youtube`: Extracts captions via `youtube_transcript_api` or runs `yt-dlp` + Whisper fallback.
   - `POST /upload`: Saves uploaded media and transcribes speech using local OpenAI Whisper.
   - `POST /ask`: Runs the custom Temporal RAG algorithm (probabilistic IDF keyword scoring + neighbor expansion)
     and invokes the configured LLM (OpenAI, Gemini, Groq, or Ollama) to produce timestamped answers.
   - `GET /health`: Lightweight system readiness check for process monitors and health probes.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename

from processing.transcription import transcribe_video, ensure_ffmpeg
from processing.youtube_processor import process_youtube_video, extract_youtube_video_id
from rag.qa import answer_question, find_relevant_segments
from rag.llm import get_llm_config

import os
import secrets
import socket
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("app")

# Allowed video and audio formats for local file upload
ALLOWED_MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".m4a"}

# Load local environment configuration (.env) into os.environ
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
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

app = Flask(__name__)
# Cryptographically sign client-side session cookies
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "temporal-rag-session-secret-key-2026"

# =====================================
# PATH SETTINGS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =====================================
# DATABASE & BLUEPRINT CONFIGURATION
# =====================================

db_path = os.path.join(BASE_DIR, "learning_platform.db")
raw_db_url = (os.environ.get("DATABASE_URL") or "").strip()
if not raw_db_url or raw_db_url == "sqlite:///learning_platform.db":
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = raw_db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "connect_args": {"timeout": 30},
}

from models.database import init_db, db, VideoLearningSession
from sqlalchemy import event
init_db(app)

with app.app_context():
    try:
        # SQLite Write-Ahead Logging (WAL) concurrency listener
        @event.listens_for(db.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()
            except Exception:
                pass
    except Exception:
        pass

from routes.auth_routes import auth_bp
from routes.learning_routes import learning_bp

# Mount modular Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(learning_bp)

@app.context_processor
def inject_globals():
    """Injects current_user into every Jinja template context automatically."""
    from auth.auth_manager import get_current_user
    return {
        "current_user": get_current_user()
    }


# =====================================
# MULTI-VIDEO IN-MEMORY STATE STORE
# =====================================
# Maps source_id (e.g. 'yt_dQw4w9WgXcQ' or 'local_lecture.mp4') to video metadata & transcript segments
VIDEO_STORE = {}

# Legacy global fallback for backward compatibility
current_segments = []
current_filename = None


def get_active_video(source_id=None):
    """
    Resolves the currently active video session and its transcript segments.
    
    Step-by-Step Viva Explanation:
    1. Multi-User Database Priority: If a user is logged in, queries `VideoLearningSession`
       for that specific student (`user_id`). This ensures data isolation between different students.
    2. In-Memory Fallback: If not logged in (guest mode), checks `VIDEO_STORE` using `source_id`
       or `session['current_video_id']`.
    3. Most Recent Fallback: If no ID specified, returns the most recently analyzed video.

    Returns:
        dict | None: Dictionary with video metadata (id, type, title, url, segments) or None.
    """
    target_id = source_id or session.get("current_video_id")
    user_id = session.get("user_id")

    if user_id:
        try:
            query = VideoLearningSession.query.filter_by(user_id=user_id)
            if target_id:
                record = query.filter_by(source_id=target_id).first()
            else:
                record = query.order_by(VideoLearningSession.created_at.desc()).first()

            if record:
                yt_id = record.source_id.replace("yt_", "") if record.video_type == "youtube" else None
                return {
                    "id": record.source_id,
                    "db_id": record.id,
                    "type": record.video_type,
                    "video_id": yt_id,
                    "title": record.title,
                    "filename": record.title,
                    "video_url": record.video_url,
                    "thumbnail": record.thumbnail,
                    "duration": record.duration,
                    "transcript_source": record.transcript_source,
                    "segments": record.get_segments()
                }
        except Exception as e:
            logger.warning("Error querying VideoLearningSession: %s", e)

    if target_id and target_id in VIDEO_STORE:
        return VIDEO_STORE[target_id]
    if VIDEO_STORE:
        # Return most recently added video if available
        return next(reversed(VIDEO_STORE.values()))
    return None


# =====================================
# HEALTH CHECK ENDPOINT
# =====================================

@app.route("/health", methods=["GET"])
def health():
    """
    Lightweight health check endpoint.
    Returns JSON status, video count, and dynamic FFmpeg status.
    Does NOT reload the Whisper model on each check.
    """
    ffmpeg_ok, ffmpeg_info = ensure_ffmpeg()
    llm_conf = get_llm_config()
    return jsonify({
        "status": "healthy",
        "healthy": True,
        "ready": True,
        "app": "Temporal-Multimodal-RAG",
        "ffmpeg_available": ffmpeg_ok,
        "ffmpeg_path": ffmpeg_info if ffmpeg_ok else None,
        "active_videos": len(VIDEO_STORE),
        "model_status": "ready",
        "llm_configured": llm_conf.get("is_configured", False),
        "llm_provider": llm_conf.get("provider", "none")
    }), 200


# =====================================
# HOME
# =====================================

@app.route("/")
def home():
    active_video = get_active_video()
    if active_video:
        yt_id = active_video.get("video_id")
        embed_url = active_video.get("embed_url") or (f"https://www.youtube.com/embed/{yt_id}" if yt_id else None)
        return render_template(
            "index.html",
            video_type=active_video.get("type", "local"),
            source_id=active_video.get("id"),
            filename=active_video.get("filename") or active_video.get("title"),
            video_title=active_video.get("title"),
            video_url=active_video.get("video_url") or active_video.get("url"),
            youtube_id=yt_id,
            youtube_embed_url=embed_url,
            video_thumbnail=active_video.get("thumbnail"),
            video_duration=active_video.get("duration"),
            transcript=active_video.get("segments", [])
        )

    return render_template("index.html")


# =====================================
# PROCESS YOUTUBE VIDEO
# =====================================

@app.route("/process_youtube", methods=["POST"])
def process_youtube():
    global current_segments, current_filename

    # Read URL from either form submission or JSON payload
    youtube_url = ""
    is_json_request = request.is_json or request.headers.get("Accept") == "application/json"

    if request.is_json:
        data = request.get_json(silent=True) or {}
        youtube_url = data.get("youtube_url", "")
    else:
        youtube_url = request.form.get("youtube_url", "")

    youtube_url = (youtube_url or "").strip()

    if not youtube_url:
        error_msg = "Please enter a YouTube video URL."
        if is_json_request:
            return jsonify({"success": False, "error": error_msg}), 400
        return render_template("index.html", error=error_msg)

    # 1. Fast URL Validation
    is_valid, video_id, val_error = extract_youtube_video_id(youtube_url)
    if not is_valid:
        if is_json_request:
            return jsonify({"success": False, "error": val_error}), 400
        return render_template("index.html", error=val_error)

    print("\n================================")
    print("PROCESSING YOUTUBE VIDEO")
    print("================================")
    print("URL:", youtube_url)
    print("Video ID:", video_id)

    # 2. Process video (metadata + captions/audio transcription)
    result = process_youtube_video(youtube_url, enable_audio_fallback=True)

    if not result.get("success"):
        err = result.get("error") or "Failed to process YouTube video."
        err_type = result.get("error_type", "processing_failed")
        logger.warning("YouTube processing failed for %s (%s): %s", video_id, err_type, err)
        print(f"YouTube processing error ({err_type}): {err}")
        if is_json_request:
            return jsonify({
                "success": False,
                "error": err,
                "error_type": err_type,
                "video_id": video_id,
                "caption_diagnostics": result.get("caption_diagnostics"),
                "audio_diagnostics": result.get("audio_diagnostics")
            }), 400
        return render_template("index.html", error=err, youtube_url=youtube_url)

    # 3. Store in VIDEO_STORE with dedicated source_id
    source_id = f"yt_{video_id}"
    video_record = {
        "id": source_id,
        "type": "youtube",
        "video_id": video_id,
        "title": result["title"],
        "author": result["author"],
        "thumbnail": result["thumbnail"],
        "duration": result["duration"],
        "url": result["url"],
        "embed_url": result["embed_url"],
        "transcript_source": result["transcript_source"],
        "segments": result["segments"]
    }

    VIDEO_STORE[source_id] = video_record
    session["current_video_id"] = source_id

    # Persist for authenticated user
    if session.get("user_id"):
        try:
            import json as py_json
            user_session = VideoLearningSession.query.filter_by(
                user_id=session["user_id"],
                source_id=source_id
            ).first()
            if not user_session:
                user_session = VideoLearningSession(
                    user_id=session["user_id"],
                    source_id=source_id,
                    video_type="youtube",
                    title=result["title"],
                    video_url=result["url"],
                    thumbnail=result.get("thumbnail", ""),
                    duration=float(result.get("duration", 0.0) or 0.0),
                    transcript_source=result.get("transcript_source", "captions"),
                    transcript_json=py_json.dumps(result["segments"]),
                    processing_status="completed"
                )
                db.session.add(user_session)
            else:
                user_session.title = result["title"]
                user_session.transcript_json = py_json.dumps(result["segments"])
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.warning("Could not persist YouTube session for user %s: %s", session.get("user_id"), e)

    # Update global fallbacks
    current_segments = result["segments"]
    current_filename = result["title"]

    logger.info("YouTube processing complete for %s. Source: %s, Segments: %d",
                video_id, result["transcript_source"], len(result["segments"]))
    print("YOUTUBE TRANSCRIPTION COMPLETE!")
    print("Total segments:", len(result["segments"]))

    if is_json_request:
        return jsonify({
            "success": True,
            "source_id": source_id,
            "video": video_record,
            "segments_count": len(result["segments"]),
            "transcript_source": result["transcript_source"],
            "caption_diagnostics": result.get("caption_diagnostics"),
            "audio_diagnostics": result.get("audio_diagnostics")
        })

    return render_template(
        "index.html",
        video_type="youtube",
        source_id=source_id,
        youtube_id=video_id,
        youtube_embed_url=result.get("embed_url") or f"https://www.youtube.com/embed/{video_id}",
        video_title=result["title"],
        video_thumbnail=result["thumbnail"],
        video_url=result["url"],
        transcript=result["segments"],
        filename=result["title"]
    )


# =====================================
# UPLOAD LOCAL VIDEO
# =====================================

@app.route("/upload", methods=["POST"])
def upload():
    global current_segments, current_filename

    is_json_request = request.is_json or request.headers.get("Accept") == "application/json"

    if "video" not in request.files:
        err = "Please select a video file to upload."
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "missing_file"}), 400
        return render_template("index.html", error=err)

    video = request.files["video"]
    if not video or video.filename == "":
        err = "Please select a video file to upload."
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "empty_filename"}), 400
        return render_template("index.html", error=err)

    filename = secure_filename(video.filename)
    if not filename:
        err = "Invalid filename provided."
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "invalid_filename"}), 400
        return render_template("index.html", error=err)

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_MEDIA_EXTENSIONS:
        err = f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_MEDIA_EXTENSIONS))}."
        logger.warning("Unsupported file format upload attempted: %s", filename)
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "unsupported_format"}), 400
        return render_template("index.html", error=err)

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # Save video to disk
    video.save(video_path)

    # Check file size
    file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
    if file_size == 0:
        err = "Uploaded file is 0 bytes (empty file)."
        logger.warning("Empty file uploaded: %s", filename)
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "empty_file"}), 400
        return render_template("index.html", error=err)

    logger.info("Local media uploaded: %s (%d bytes)", filename, file_size)
    print("\n================================")
    print("LOCAL VIDEO UPLOADED")
    print("================================")
    print("Filename:", filename)
    print("Path:", video_path)
    print("Size:", file_size, "bytes")

    try:
        segments = transcribe_video(video_path)
    except ValueError as ve:
        # Pre-flight audio validation or format error
        err = str(ve)
        logger.warning("Audio validation failed for local video %s: %s", filename, err)
        print("\nAUDIO VALIDATION ERROR:", err)
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "no_audio_track"}), 400
        return render_template("index.html", error=err)
    except RuntimeError as re_err:
        err = str(re_err)
        logger.error("Transcription runtime error for %s: %s", filename, err)
        print("\nTRANSCRIPTION RUNTIME ERROR:", err)
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "transcription_error"}), 500
        return render_template("index.html", error=err)
    except Exception as e:
        logger.exception("Unexpected error transcribing %s: %s", filename, e)
        err = f"Unexpected transcription error: {type(e).__name__} - {e}"
        print("\nUNEXPECTED TRANSCRIPTION ERROR:", err)
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "internal_error"}), 500
        return render_template("index.html", error=err)

    if not segments:
        err = "No spoken words were detected in the audio track of this video."
        logger.warning("Transcription resulted in 0 segments for %s", filename)
        if is_json_request:
            return jsonify({"success": False, "error": err, "error_type": "empty_transcript"}), 400
        return render_template("index.html", error=err)

    source_id = f"local_{filename}"
    video_record = {
        "id": source_id,
        "type": "local",
        "filename": filename,
        "title": filename,
        "video_url": f"/uploads/{filename}",
        "segments": segments
    }

    VIDEO_STORE[source_id] = video_record
    session["current_video_id"] = source_id

    # Persist for authenticated user
    if session.get("user_id"):
        try:
            import json as py_json
            user_session = VideoLearningSession.query.filter_by(
                user_id=session["user_id"],
                source_id=source_id
            ).first()
            if not user_session:
                user_session = VideoLearningSession(
                    user_id=session["user_id"],
                    source_id=source_id,
                    video_type="local",
                    title=filename,
                    video_url=f"/uploads/{filename}",
                    thumbnail="",
                    duration=0.0,
                    transcript_source="whisper",
                    transcript_json=py_json.dumps(segments),
                    processing_status="completed"
                )
                db.session.add(user_session)
            else:
                user_session.title = filename
                user_session.transcript_json = py_json.dumps(segments)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.warning("Could not persist local upload session for user %s: %s", session.get("user_id"), e)

    # Update global fallbacks
    current_segments = segments
    current_filename = filename

    logger.info("Local video successfully transcribed: %s (%d segments)", filename, len(segments))

    if is_json_request:
        return jsonify({
            "success": True,
            "source_id": source_id,
            "video": video_record,
            "segments_count": len(segments)
        })

    return render_template(
        "index.html",
        video_type="local",
        source_id=source_id,
        video_url=f"/uploads/{filename}",
        filename=filename,
        video_title=filename,
        transcript=segments
    )


# =====================================
# SERVE LOCAL VIDEO
# =====================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# =====================================
# ASK AI
# =====================================

@app.route("/ask", methods=["POST"])
def ask():
    is_json_request = request.is_json or request.headers.get("Accept") == "application/json"

    # Support JSON and form submissions
    if request.is_json:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        source_id = data.get("source_id")
    else:
        question = (request.form.get("question") or "").strip()
        source_id = request.form.get("source_id")

    if not question:
        err = "Please enter a question."
        if is_json_request:
            return jsonify({"success": False, "error": err}), 400
        return render_template("index.html", error=err)

    # Resolve active video
    active_video = get_active_video(source_id)
    segments = active_video.get("segments", []) if active_video else current_segments

    if not segments:
        err = "Please upload or analyze a video first."
        if is_json_request:
            return jsonify({"success": False, "error": err}), 400
        return render_template(
            "index.html",
            question=question,
            answer=err,
            sources=[],
            timestamps=[],
            answer_mode="error",
            timestamp="Not available"
        )

    print("\n================================")
    print("USER QUESTION:", question)
    print("Target Video Source:", active_video.get("id") if active_video else "global")
    print("================================")

    # Execute full RAG QA pipeline
    rag_response = answer_question(question, segments)

    answer_text = rag_response.get("answer", "")
    sources = rag_response.get("sources", [])
    timestamps = rag_response.get("timestamps", [])
    answer_mode = rag_response.get("answer_mode", "llm")
    timestamp_str = rag_response.get("timestamp_str", "Not found")
    primary_ts = rag_response.get("primary_timestamp")
    results = rag_response.get("results", [])

    if not sources or answer_mode == "no_match":
        if is_json_request:
            return jsonify({
                "success": True,
                "question": question,
                "answer": answer_text,
                "sources": [],
                "timestamps": [],
                "answer_mode": "no_match",
                "timestamp": "Not found",
                "start": None,
                "end": None,
                "results": []
            })
        return render_template(
            "index.html",
            question=question,
            answer=answer_text,
            sources=[],
            timestamps=[],
            answer_mode="no_match",
            timestamp="Not found",
            transcript=segments,
            video_type=active_video.get("type", "local") if active_video else "local",
            source_id=active_video.get("id") if active_video else None,
            filename=active_video.get("filename") or active_video.get("title") if active_video else current_filename,
            video_title=active_video.get("title") if active_video else current_filename,
            video_url=active_video.get("video_url") or active_video.get("url") if active_video else None,
            youtube_id=active_video.get("video_id") if active_video else None,
            video_thumbnail=active_video.get("thumbnail") if active_video else None
        )

    start_sec = primary_ts["start"] if primary_ts else sources[0]["start"]
    end_sec = primary_ts["end"] if primary_ts else sources[0]["end"]

    youtube_url_with_timestamp = None
    if active_video and active_video.get("type") == "youtube" and start_sec is not None:
        youtube_url_with_timestamp = f"https://www.youtube.com/watch?v={active_video.get('video_id')}&t={int(start_sec)}s"

    if is_json_request:
        return jsonify({
            "success": True,
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "timestamps": timestamps,
            "answer_mode": answer_mode,
            "timestamp": timestamp_str,
            "start": start_sec,
            "end": end_sec,
            "score": 1.0,
            "youtube_url": youtube_url_with_timestamp,
            "results": results
        })

    yt_id = active_video.get("video_id") if active_video else None
    embed_url = active_video.get("embed_url") or (f"https://www.youtube.com/embed/{yt_id}" if yt_id else None)

    return render_template(
        "index.html",
        question=question,
        answer=answer_text,
        sources=sources,
        timestamps=timestamps,
        answer_mode=answer_mode,
        timestamp=timestamp_str,
        start=start_sec,
        end=end_sec,
        youtube_timestamp_url=youtube_url_with_timestamp,
        transcript=segments,
        video_type=active_video.get("type", "local") if active_video else "local",
        source_id=active_video.get("id") if active_video else None,
        filename=active_video.get("filename") or active_video.get("title") if active_video else current_filename,
        video_title=active_video.get("title") if active_video else current_filename,
        video_url=active_video.get("video_url") or active_video.get("url") if active_video else f"/uploads/{current_filename}",
        youtube_id=yt_id,
        youtube_embed_url=embed_url,
        video_thumbnail=active_video.get("thumbnail") if active_video else None
    )


# =====================================
# SERVER CONFIGURATION & ENTRY POINT
# =====================================

def check_port_available(host: str, port: int) -> bool:
    """Checks if a TCP port is currently free to bind on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(host: str, start_port: int, max_tries: int = 50) -> int:
    """Finds the next open TCP port starting from start_port."""
    for port in range(start_port, start_port + max_tries):
        if check_port_available(host, port):
            return port
    return start_port


def get_server_config():
    """Reads configurable host, port, and debug settings with safe defaults."""
    host = os.environ.get("HOST", "127.0.0.1").strip()
    port_str = os.environ.get("PORT", "5000").strip()
    try:
        port = int(port_str)
    except ValueError:
        port = 5000
    debug_val = os.environ.get("FLASK_DEBUG", "false").strip().lower()
    debug = debug_val in ("true", "1", "yes")
    return host, port, debug


# =====================================
# SYSTEM ERROR HANDLERS
# =====================================

@app.errorhandler(500)
def handle_internal_server_error(e):
    logger.exception("Internal server error caught: %s", e)
    if request.is_json or request.headers.get("Accept") == "application/json":
        return jsonify({
            "success": False,
            "error": "A system error occurred. Please try again or return to the dashboard.",
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500
    from flask import flash
    flash("A system error occurred while processing your request. Please try again.", "danger")
    return render_template("base.html"), 500


@app.errorhandler(404)
def handle_not_found(e):
    if request.is_json or request.headers.get("Accept") == "application/json":
        return jsonify({
            "success": False,
            "error": "Requested resource not found.",
            "error_code": "NOT_FOUND"
        }), 404
    return render_template(
        "base.html",
        page_title="Page Not Found",
        error="The requested page could not be found. Please check the address or return to the dashboard."
    ), 404


@app.errorhandler(500)
def handle_server_error(e):
    logger.exception("Internal Server Error: %s", e)
    if request.is_json or request.headers.get("Accept") == "application/json":
        return jsonify({
            "success": False,
            "error": "An internal server error occurred.",
            "error_code": "INTERNAL_SERVER_ERROR"
        }), 500
    return render_template(
        "base.html",
        page_title="Server Error",
        error="An unexpected server error occurred. Please return to the dashboard or try again."
    ), 500


if __name__ == "__main__":
    host, requested_port, debug = get_server_config()

    # Dynamic FFmpeg resolution
    ffmpeg_ok, ffmpeg_info = ensure_ffmpeg()

    # Check port availability
    actual_port = requested_port
    if not check_port_available(host, requested_port):
        actual_port = find_available_port(host, requested_port)
        print(f"\n[NOTICE] Port {requested_port} is already in use. Selected available port: {actual_port}")

    app_url = f"http://{host}:{actual_port}"

    print("\n================================")
    print("TEMPORAL-AWARE MULTIMODAL RAG")
    print("================================")
    print("Host:", host)
    print("Port:", actual_port)
    print("Application URL:", app_url)
    print("Upload folder:", UPLOAD_FOLDER)
    print(f"FFmpeg: {'Available' if ffmpeg_ok else 'Unavailable'}")
    print(f"Debug Mode: {'Enabled' if debug else 'Disabled'}")
    print("================================\n")

    app.run(
        debug=debug,
        host=host,
        port=actual_port
    )