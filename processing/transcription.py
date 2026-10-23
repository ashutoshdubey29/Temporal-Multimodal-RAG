"""
processing/transcription.py

Provides robust dynamic FFmpeg discovery, audio stream validation,
diagnostic logging, and speech-to-text transcription via Whisper.
"""

import os
import shutil
import subprocess
import time
import logging
import whisper

logger = logging.getLogger("processing.transcription")

# Cached FFmpeg path and loaded Whisper models
_FFMPEG_RESOLVED_PATH = None
_LOADED_WHISPER_MODELS = {}


def ensure_ffmpeg() -> tuple[bool, str]:
    """
    Dynamically discovers and configures FFmpeg for Whisper and yt-dlp.
    1. Checks if 'ffmpeg' is already directly available in PATH.
    2. Checks FFMPEG_PATH / FFMPEG_DIR environment variables if set.
    3. Uses imageio-ffmpeg's bundled binary if available, ensuring a standard
       'ffmpeg.exe' alias exists so external tools (Whisper/yt-dlp) can invoke it.
    4. Updates os.environ['PATH'] so subprocess calls find it immediately.

    Returns:
      (is_available: bool, path_or_message: str)
    """
    global _FFMPEG_RESOLVED_PATH

    if _FFMPEG_RESOLVED_PATH and os.path.exists(_FFMPEG_RESOLVED_PATH):
        return True, _FFMPEG_RESOLVED_PATH

    # 1. Custom environment overrides
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

    # 2. Check if ffmpeg is already discoverable via PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        _FFMPEG_RESOLVED_PATH = system_ffmpeg
        return True, system_ffmpeg

    # 3. Dynamic discovery via imageio-ffmpeg package
    try:
        import imageio_ffmpeg
        raw_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if raw_exe and os.path.exists(raw_exe):
            bin_dir = os.path.dirname(raw_exe)
            target_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            target_exe = os.path.join(bin_dir, target_name)

            if not os.path.exists(target_exe):
                try:
                    shutil.copyfile(raw_exe, target_exe)
                except Exception:
                    pass

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
    """Returns the version string of the active FFmpeg executable."""
    available, path = ensure_ffmpeg()
    if not available:
        return "Not available"
    try:
        res = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
        if res.stdout:
            return res.stdout.splitlines()[0].strip()
    except Exception as e:
        logger.debug("Failed to query FFmpeg version: %s", e)
    return "Unknown version"


# Resolve FFmpeg on module import
ensure_ffmpeg()


def inspect_audio_file(file_path: str) -> tuple[bool, str, dict]:
    """
    Inspects media file container and verifies an audio stream exists before passing to Whisper.
    Returns:
      (is_valid: bool, error_msg: str, metadata: dict)
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
        # Run ffmpeg -i to inspect streams without decoding
        cmd = [ffmpeg, "-i", file_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = res.stderr or res.stdout or ""

        # Check for audio streams
        has_audio = False
        duration_sec = 0.0

        for line in output.splitlines():
            line_str = line.strip()
            if "Stream #" in line_str and "Audio:" in line_str:
                has_audio = True
                metadata["audio_stream"] = line_str
            if "Duration:" in line_str:
                metadata["duration_info"] = line_str

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
    Loads or returns cached Whisper model instance to avoid expensive disk reloads.
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
    Transcribes audio from a video file into timestamped segments using Whisper.
    Performs pre-flight stream inspection, model loading, and diagnostic recording.
    """
    start_time = time.time()
    filename = os.path.basename(video_path) if video_path else "unknown"

    logger.info("Beginning transcription for: %s", filename)

    if not os.path.exists(video_path):
        err = f"Video file not found at path: {video_path}"
        logger.error(err)
        raise FileNotFoundError(err)

    # 1. Pre-flight audio stream check
    is_valid, err_msg, meta = inspect_audio_file(video_path)
    if not is_valid:
        logger.error("Pre-flight audio validation failed for %s: %s", filename, err_msg)
        raise ValueError(err_msg)

    # 2. Ensure FFmpeg is accessible
    ffmpeg_ok, ffmpeg_info = ensure_ffmpeg()
    if not ffmpeg_ok:
        err = f"FFmpeg is required for Whisper transcription: {ffmpeg_info}"
        logger.error(err)
        raise RuntimeError(err)

    # 3. Load model (cached)
    selected_model = model_name or os.environ.get("WHISPER_MODEL", "base")
    model = get_whisper_model(selected_model)

    # 4. Transcribe
    print(f"Transcribing video: {filename}...")
    try:
        result = model.transcribe(
            video_path,
            verbose=False,
            fp16=False  # CPU safety on Windows
        )
    except Exception as e:
        logger.exception("Whisper transcription failed on %s: %s", filename, e)
        raise RuntimeError(f"Speech transcription failed: {type(e).__name__} - {e}")

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