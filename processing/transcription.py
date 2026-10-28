"""
processing/transcription.py
===========================
MODULE PURPOSE (VIVA SUMMARY):
This file handles audio extraction and speech-to-text transcription for local video files.
It ensures that speech in uploaded lecture videos is automatically converted into text
with precise time stamps (start second and end second).

KEY CONCEPTS TO EXPLAIN IN VIVA:
1. Whisper:
   OpenAI's open-source deep learning Automatic Speech Recognition (ASR) model.
   It listens to the audio track of a video and produces timestamped text segments.
2. FFmpeg:
   A standard open-source multimedia engine. Whisper and yt-dlp rely on FFmpeg
   under the hood to decode, convert, and extract raw audio from video containers.
3. Timestamped Segments:
   Whisper outputs a list of dictionaries: [{"start": 12.5, "end": 18.0, "text": "..."}].
   These timestamps are critical because they allow our RAG system and video player
   to jump directly to the exact second in the lecture where a topic was taught.

HOW THIS MODULE CONNECTS TO OTHER FILES:
- Called by `app.py` in the `/upload` route when a student uploads a local video (.mp4).
- Called by `processing/youtube_processor.py` as an audio fallback when a YouTube video
  has no subtitles/captions available.
- The output segments are saved to `VIDEO_STORE` in `app.py` and stored in the database
  table `video_learning_sessions` for question answering and quiz generation.
"""

import os
import shutil
import subprocess
import time
import logging
import whisper

logger = logging.getLogger("processing.transcription")

# Cached FFmpeg path and loaded Whisper models
# WHY CACHE? Loading Whisper from disk takes several seconds and significant RAM.
# By keeping loaded models in memory (_LOADED_WHISPER_MODELS), subsequent calls are instant.
_FFMPEG_RESOLVED_PATH = None
_LOADED_WHISPER_MODELS = {}


def ensure_ffmpeg() -> tuple[bool, str]:
    """
    Dynamically discovers and configures FFmpeg for Whisper and yt-dlp.

    VIVA EXPLANATION:
    On Windows systems, users often don't have FFmpeg permanently installed in their
    system PATH. This helper ensures the application never crashes due to a missing FFmpeg:
    1. Checks if 'ffmpeg' is already directly available in system PATH.
    2. Checks custom FFMPEG_PATH or FFMPEG_DIR environment variables from .env.
    3. Finds the bundled FFmpeg binary provided by the `imageio-ffmpeg` Python package.
    4. Automatically prepends FFmpeg's directory to `os.environ["PATH"]` for the current process,
       so external tools (Whisper, yt-dlp, subprocess) can invoke it immediately.

    Returns:
        tuple[bool, str]: (is_available, resolved_path_or_error_message)
    """
    global _FFMPEG_RESOLVED_PATH

    # If we already successfully resolved FFmpeg previously in this process, reuse it
    if _FFMPEG_RESOLVED_PATH and os.path.exists(_FFMPEG_RESOLVED_PATH):
        return True, _FFMPEG_RESOLVED_PATH

    # Step 1: Check custom environment variable overrides (e.g. from .env)
    env_ffmpeg_path = os.environ.get("FFMPEG_PATH")
    if env_ffmpeg_path and os.path.isfile(env_ffmpeg_path):
        bin_dir = os.path.dirname(env_ffmpeg_path)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        _FFMPEG_RESOLVED_PATH = env_ffmpeg_path
        return True, env_ffmpeg_path

    env_ffmpeg_dir = os.environ.get("FFMPEG_DIR")
    if env_ffmpeg_dir and os.path.isdir(env_ffmpeg_dir):
        os.environ["PATH"] = env_ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        which_custom = shutil.which("ffmpeg")
        if which_custom:
            _FFMPEG_RESOLVED_PATH = which_custom
            return True, which_custom

    # Step 2: Check if ffmpeg is already discoverable on standard system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        _FFMPEG_RESOLVED_PATH = system_ffmpeg
        return True, system_ffmpeg

    # Step 3: Dynamic discovery via imageio-ffmpeg package (bundled binary)
    try:
        import imageio_ffmpeg
        raw_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if raw_exe and os.path.exists(raw_exe):
            bin_dir = os.path.dirname(raw_exe)
            target_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            target_exe = os.path.join(bin_dir, target_name)

            # Ensure standard executable name exists so subprocess calls like 'ffmpeg' succeed
            if not os.path.exists(target_exe):
                try:
                    shutil.copyfile(raw_exe, target_exe)
                except Exception:
                    pass

            # Prepend binary directory to PATH for the running Python session
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

            resolved = shutil.which("ffmpeg")
            if resolved:
                _FFMPEG_RESOLVED_PATH = resolved
                return True, resolved
            if os.path.exists(target_exe):
                _FFMPEG_RESOLVED_PATH = target_exe
                return True, target_exe
            _FFMPEG_RESOLVED_PATH = raw_exe
            return True, raw_exe
    except Exception as e:
        logger.warning("Dynamic FFmpeg discovery failed: %s", e)

    return False, "FFmpeg executable not found in PATH or imageio-ffmpeg."


def get_ffmpeg_version() -> str:
    """
    Returns the version string of the active FFmpeg executable.

    VIVA EXPLANATION:
    Used for diagnostics and health check endpoint (`/health`) so administrators
    or examiners can see which FFmpeg build is active.

    Returns:
        str: First line of 'ffmpeg -version' output, or 'Not available'.
    """
    available, path = ensure_ffmpeg()
    if not available:
        return "Not available"
    try:
        # Run 'ffmpeg -version' and grab the top summary line
        res = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
        if res.stdout:
            return res.stdout.splitlines()[0].strip()
    except Exception as e:
        logger.debug("Failed to query FFmpeg version: %s", e)
    return "Unknown version"


# Resolve and verify FFmpeg immediately when this module is imported
ensure_ffmpeg()


def inspect_audio_file(file_path: str) -> tuple[bool, str, dict]:
    """
    Pre-flight media inspection: verifies that the file exists, is not empty,
    and contains an active audio stream before passing it to Whisper.

    WHY IS THIS NEEDED IN VIVA?
    If a user uploads a screen recording with no microphone enabled (silent video)
    or a 0-byte corrupt file, running Whisper directly would crash or hang.
    This function inspects the video container using `ffmpeg -i` in a lightweight way,
    catching issues upfront and returning clear, user-friendly error messages.

    Parameters:
        file_path (str): Full path to the video or audio file on disk.

    Returns:
        tuple[bool, str, dict]:
          - is_valid (bool): True if valid audio track is present.
          - error_msg (str): Friendly explanation if invalid.
          - metadata (dict): Stream details, file size, duration.
    """
    if not file_path or not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}", {}

    size = os.path.getsize(file_path)
    if size == 0:
        return False, "Uploaded file is 0 bytes (empty file).", {}

    available, ffmpeg = ensure_ffmpeg()
    if not available:
        return False, f"FFmpeg is not available: {ffmpeg}", {}

    metadata = {"size_bytes": size, "path": file_path}

    try:
        # Run 'ffmpeg -i <file>' without decoding to quickly inspect streams
        cmd = [ffmpeg, "-i", file_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = res.stderr or res.stdout or ""

        # Scan FFmpeg output for stream descriptors (e.g. "Stream #0:1: Audio: aac")
        has_audio = False
        duration_sec = 0.0

        for line in output.splitlines():
            line_str = line.strip()
            if "Stream #" in line_str and "Audio:" in line_str:
                has_audio = True
                metadata["audio_stream"] = line_str
            if "Duration:" in line_str:
                metadata["duration_info"] = line_str

        # If video has no audio stream, report back clearly
        if not has_audio:
            logger.warning("No audio stream found in file: %s", os.path.basename(file_path))
            return False, "The video file does not contain an audio track. Please upload a video with audible speech.", metadata

        return True, "", metadata

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg stream inspection timed out for %s", file_path)
        return False, "Inspection of the video file timed out.", metadata
    except Exception as e:
        logger.error("Error inspecting audio file %s: %s", file_path, e)
        # Non-blocking fallback if ffmpeg inspection encountered unexpected parsing
        return True, "", metadata


def get_whisper_model(model_name: str = None):
    """
    Loads or retrieves a cached Whisper speech recognition model instance.

    VIVA EXPLANATION:
    Whisper models ('tiny', 'base', 'small', etc.) are neural networks stored in PyTorch.
    Loading a model takes several seconds and disk I/O.
    We implement the In-Memory Cache (Singleton-like) pattern here:
    if the model is already in `_LOADED_WHISPER_MODELS`, we return it instantly.

    Parameters:
        model_name (str, optional): Model size (defaults to WHISPER_MODEL env var or 'base').

    Returns:
        whisper.model.Whisper: The loaded PyTorch Whisper model.
    """
    selected_model = model_name or os.environ.get("WHISPER_MODEL", "base")
    if selected_model in _LOADED_WHISPER_MODELS:
        return _LOADED_WHISPER_MODELS[selected_model]

    logger.info("Loading Whisper model '%s'...", selected_model)
    print(f"Loading Whisper model ({selected_model})...")
    start_t = time.time()
    model = whisper.load_model(selected_model)
    logger.info("Whisper model '%s' loaded in %.2fs", selected_model, time.time() - start_t)

    _LOADED_WHISPER_MODELS[selected_model] = model
    return model


def transcribe_video(video_path: str, model_name: str = None) -> list[dict]:
    """
    Transcribes spoken audio from a video file into timestamped text segments.

    STEP-BY-STEP FLOW (EXPLAIN IN VIVA):
    1. Pre-flight check: verifies file exists and has an audible audio track using `inspect_audio_file()`.
    2. FFmpeg check: verifies FFmpeg is ready to decode the media container.
    3. Model loading: gets the cached Whisper model (default 'base').
    4. Speech Recognition: executes `model.transcribe()` with `fp16=False` (ensures stable execution on CPU).
    5. Structuring: formats raw segments into standard clean dictionaries with rounded second timestamps.

    Parameters:
        video_path (str): File system path to the video/audio file.
        model_name (str, optional): Whisper model variant ('base', 'tiny', etc.).

    Returns:
        list[dict]: List of timestamped segments, where each item is:
            {
                "start": float (e.g. 14.2),
                "end": float (e.g. 19.5),
                "text": str (spoken words during this interval)
            }

    Raises:
        FileNotFoundError: If the file does not exist on disk.
        ValueError: If pre-flight validation fails (e.g., 0-byte or silent file).
        RuntimeError: If FFmpeg or transcription execution fails.
    """
    start_time = time.time()
    filename = os.path.basename(video_path) if video_path else "unknown"

    logger.info("Beginning transcription for: %s", filename)

    if not os.path.exists(video_path):
        err = f"Video file not found at path: {video_path}"
        logger.error(err)
        raise FileNotFoundError(err)

    # 1. Pre-flight audio stream check (catches silent / corrupt videos early)
    is_valid, err_msg, meta = inspect_audio_file(video_path)
    if not is_valid:
        logger.error("Pre-flight audio validation failed for %s: %s", filename, err_msg)
        raise ValueError(err_msg)

    # 2. Ensure FFmpeg is accessible for Whisper's internal audio decoding
    ffmpeg_ok, ffmpeg_info = ensure_ffmpeg()
    if not ffmpeg_ok:
        err = f"FFmpeg is required for Whisper transcription: {ffmpeg_info}"
        logger.error(err)
        raise RuntimeError(err)

    # 3. Load model (cached in memory)
    selected_model = model_name or os.environ.get("WHISPER_MODEL", "base")
    model = get_whisper_model(selected_model)

    # 4. Perform speech recognition using Whisper
    print(f"Transcribing video: {filename}...")
    try:
        # fp16=False prevents warnings and numerical issues on Windows CPUs without CUDA
        result = model.transcribe(
            video_path,
            verbose=False,
            fp16=False  # CPU safety on Windows
        )
    except Exception as e:
        logger.exception("Whisper transcription failed on %s: %s", filename, e)
        raise RuntimeError(f"Speech transcription failed: {type(e).__name__} - {e}")

    # 5. Extract and normalize transcript segments
    segments = []
    raw_segments = result.get("segments", [])

    for segment in raw_segments:
        text = segment.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "start": round(float(segment["start"]), 2),
            "end": round(float(segment["end"]), 2),
            "text": text
        })

    duration = time.time() - start_time
    logger.info(
        "Transcription complete for %s. Total segments: %d, Elapsed: %.2fs",
        filename, len(segments), duration
    )
    print(f"Transcription complete: {len(segments)} segments in {duration:.2f}s")

    return segments
