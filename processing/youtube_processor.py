"""
processing/youtube_processor.py

Provides robust URL validation, metadata retrieval, multi-tiered caption extraction,
permitted Whisper audio fallback, and granular diagnostic error reporting for YouTube videos.
"""

from urllib.parse import urlparse, parse_qs
import html
import os
import re
import tempfile
import time
import logging
import requests

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    VideoUnplayable,
    AgeRestricted,
    IpBlocked,
    RequestBlocked,
    YouTubeRequestFailed,
    YouTubeTranscriptApiException,
)

logger = logging.getLogger("processing.youtube")

# Supported YouTube domains
YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

# Standard 11-character base64url YouTube video ID pattern
VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")


def extract_youtube_video_id(url: str) -> tuple[bool, str, str]:
    """
    Validates a YouTube URL and safely extracts the 11-character video ID.
    
    Supported formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - URLs with additional query parameters (e.g., &t=..., &feature=...)
    
    Returns:
      (is_valid: bool, video_id: str, error_message: str)
    """
    if not url or not isinstance(url, str):
        return False, "", "Please enter a valid YouTube URL."

    cleaned_url = url.strip()
    if not cleaned_url:
        return False, "", "Please enter a YouTube URL."

    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = "https://" + cleaned_url

    try:
        parsed = urlparse(cleaned_url)
    except Exception:
        return False, "", "Malformed URL provided."

    netloc = parsed.netloc.lower()
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    if netloc not in YOUTUBE_DOMAINS:
        return False, "", f"Unsupported domain '{netloc}'. Only YouTube links (youtube.com, youtu.be) are supported."

    video_id = ""
    path = parsed.path

    if netloc in ("youtu.be", "www.youtu.be"):
        parts = [p for p in path.split("/") if p]
        if parts:
            video_id = parts[0]
    else:
        if path.startswith("/watch"):
            qs = parse_qs(parsed.query)
            v_list = qs.get("v")
            if v_list and len(v_list) > 0:
                video_id = v_list[0]
        elif path.startswith(("/embed/", "/shorts/", "/v/")):
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2:
                video_id = parts[1]
        else:
            qs = parse_qs(parsed.query)
            v_list = qs.get("v")
            if v_list and len(v_list) > 0:
                video_id = v_list[0]

    if not video_id:
        return False, "", "Could not find a valid YouTube video ID in the provided URL."

    if not VIDEO_ID_REGEX.match(video_id):
        return False, "", f"Invalid YouTube video ID '{video_id}'. YouTube video IDs must be exactly 11 characters."

    return True, video_id, ""


def get_youtube_metadata(video_id: str) -> dict:
    """
    Retrieves video metadata (title, author, thumbnail) using YouTube oEmbed.
    Verifies if video is publicly accessible.
    """
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    fallback_thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

    metadata = {
        "accessible": True,
        "title": f"YouTube Video ({video_id})",
        "author": "YouTube Creator",
        "thumbnail": fallback_thumbnail,
        "error": None
    }

    try:
        resp = requests.get(oembed_url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            metadata["title"] = data.get("title") or metadata["title"]
            metadata["author"] = data.get("author_name") or metadata["author"]
            metadata["thumbnail"] = data.get("thumbnail_url") or metadata["thumbnail"]
            logger.info("Metadata fetched successfully for %s: '%s'", video_id, metadata["title"])
        elif resp.status_code in (401, 403):
            metadata["accessible"] = False
            metadata["error"] = "YouTube video is private or restricted, and requires authentication."
            logger.warning("Metadata 401/403 for %s: %s", video_id, metadata["error"])
        elif resp.status_code == 404:
            metadata["accessible"] = False
            metadata["error"] = "YouTube video not found. It may be deleted or the URL is incorrect."
            logger.warning("Metadata 404 for %s: %s", video_id, metadata["error"])
        else:
            logger.debug("oEmbed returned status %d for %s", resp.status_code, video_id)
    except requests.RequestException as e:
        logger.warning("Network issue querying oEmbed for %s: %s", video_id, e)

    return metadata


def fetch_youtube_captions(video_id: str, return_diagnostics: bool = False):
    """
    Fetches official or auto-generated captions for the given video ID.
    Implements a multi-tiered fallback strategy:
      1. Broad English variants (en, en-US, en-GB, en-CA, en-AU, en-IN)
      2. Manual transcripts in English
      3. Auto-generated transcripts in English
      4. Any translatable transcript translated to English
      5. Primary native transcript if translation is unsupported
    """
    api = YouTubeTranscriptApi()
    raw_snippets = None
    diagnostics = {
        "success": False,
        "reason": "unknown",
        "detail": "",
        "language": None,
        "is_generated": False
    }

    # Priority 1: Direct fetch in English variants
    try:
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IN"])
        raw_snippets = fetched.to_raw_data()
        diagnostics = {
            "success": True,
            "reason": "success",
            "detail": "Direct English captions retrieved.",
            "language": "en",
            "is_generated": False
        }
    except TranscriptsDisabled:
        diagnostics = {
            "success": False,
            "reason": "disabled",
            "detail": "Subtitles/captions are disabled by the video owner."
        }
        logger.info("Captions disabled for %s", video_id)
    except (VideoUnavailable, VideoUnplayable):
        diagnostics = {
            "success": False,
            "reason": "unavailable",
            "detail": "This YouTube video is unavailable or unplayable."
        }
        logger.info("Video unavailable/unplayable for %s", video_id)
    except AgeRestricted:
        diagnostics = {
            "success": False,
            "reason": "age_restricted",
            "detail": "This video is age-restricted and requires sign-in."
        }
        logger.info("Video age-restricted for %s", video_id)
    except (IpBlocked, RequestBlocked):
        diagnostics = {
            "success": False,
            "reason": "rate_limited",
            "detail": "YouTube caption requests are temporarily blocked/rate-limited from your network."
        }
        logger.warning("YouTube blocked/rate-limited caption fetch for %s", video_id)
    except Exception as first_exc:
        logger.debug("Direct English fetch failed for %s (%s). Inspecting transcript list...", video_id, first_exc)

        # Priority 2: Inspect available transcripts and select best fit
        try:
            transcript_list = api.list(video_id)
            selected_transcript = None

            # 2a. Search manual English
            try:
                selected_transcript = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IN"])
            except Exception:
                pass

            # 2b. Search auto-generated English
            if not selected_transcript:
                try:
                    selected_transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IN"])
                except Exception:
                    pass

            # 2c. Search any translatable transcript
            if not selected_transcript:
                for t in transcript_list:
                    if getattr(t, "is_translatable", False):
                        try:
                            selected_transcript = t.translate("en")
                            break
                        except Exception:
                            pass

            # 2d. Fallback: Take the first available native transcript
            if not selected_transcript:
                try:
                    selected_transcript = next(iter(transcript_list))
                except Exception:
                    selected_transcript = None

            if selected_transcript:
                raw_snippets = selected_transcript.fetch().to_raw_data()
                diagnostics = {
                    "success": True,
                    "reason": "success",
                    "detail": f"Retrieved {getattr(selected_transcript, 'language_code', 'native')} captions.",
                    "language": getattr(selected_transcript, "language_code", "unknown"),
                    "is_generated": getattr(selected_transcript, "is_generated", False)
                }
            else:
                diagnostics = {
                    "success": False,
                    "reason": "not_found",
                    "detail": "No subtitle/caption tracks are available for this video."
                }
        except TranscriptsDisabled:
            diagnostics = {
                "success": False,
                "reason": "disabled",
                "detail": "Subtitles/captions are disabled by the video owner."
            }
        except (IpBlocked, RequestBlocked):
            diagnostics = {
                "success": False,
                "reason": "rate_limited",
                "detail": "YouTube caption requests are temporarily blocked/rate-limited from your network."
            }
        except (VideoUnavailable, VideoUnplayable):
            diagnostics = {
                "success": False,
                "reason": "unavailable",
                "detail": "This YouTube video is unavailable or deleted."
            }
        except AgeRestricted:
            diagnostics = {
                "success": False,
                "reason": "age_restricted",
                "detail": "This video is age-restricted and requires sign-in."
            }
        except Exception as e:
            err_msg = str(e)
            if "blocked" in err_msg.lower() or "too many" in err_msg.lower() or "429" in err_msg:
                diagnostics = {
                    "success": False,
                    "reason": "rate_limited",
                    "detail": "YouTube caption requests are temporarily rate-limited."
                }
            else:
                diagnostics = {
                    "success": False,
                    "reason": "not_found",
                    "detail": f"Could not list captions: {type(e).__name__}"
                }

    if not raw_snippets:
        if return_diagnostics:
            return [], diagnostics
        return []

    segments = []
    for item in raw_snippets:
        raw_text = item.get("text", "")
        clean_text = html.unescape(raw_text)
        clean_text = re.sub(r"<[^>]+>", "", clean_text).replace("\n", " ").strip()
        if not clean_text:
            continue

        start = round(float(item.get("start", 0.0)), 2)
        dur = float(item.get("duration", 0.0))
        end = round(start + dur, 2) if dur > 0 else round(start + 2.0, 2)
        if end <= start:
            end = round(start + 1.0, 2)

        segments.append({
            "start": start,
            "end": end,
            "text": clean_text
        })

    logger.info("Successfully fetched %d caption segments for %s", len(segments), video_id)

    if return_diagnostics:
        return segments, diagnostics
    return segments


def transcribe_youtube_audio(video_id: str, return_diagnostics: bool = False):
    """
    Fallback method: Downloads permitted audio using yt-dlp to a temporary file
    and transcribes with the local Whisper pipeline.
    Configures explicit FFmpeg location and cleans up temp files immediately.
    """
    import yt_dlp
    from processing.transcription import transcribe_video, ensure_ffmpeg

    ffmpeg_ok, ffmpeg_path = ensure_ffmpeg()
    ffmpeg_dir = os.path.dirname(ffmpeg_path) if ffmpeg_ok else None

    diagnostics = {
        "success": False,
        "reason": "unknown",
        "detail": "",
        "download_time": 0.0,
        "transcribe_time": 0.0
    }

    temp_dir = tempfile.gettempdir()
    output_template = os.path.join(temp_dir, f"yt_audio_{video_id}_%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "max_filesize": 150 * 1024 * 1024,  # Generous 150MB limit
    }

    if ffmpeg_dir:
        ydl_opts["ffmpeg_location"] = ffmpeg_dir

    target_audio_path = None
    start_dl = time.time()

    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info("Downloading permitted audio for fallback transcription: %s", video_url)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            if info:
                base_path = ydl.prepare_filename(info)
                target_audio_path = os.path.splitext(base_path)[0] + ".mp3"
                if not os.path.exists(target_audio_path):
                    target_audio_path = base_path

        diagnostics["download_time"] = round(time.time() - start_dl, 2)

        if not target_audio_path or not os.path.exists(target_audio_path):
            diagnostics["reason"] = "audio_missing"
            diagnostics["detail"] = "Audio extraction did not produce a readable audio file."
            logger.error("Audio download produced no file for %s", video_id)
            if return_diagnostics:
                return [], diagnostics
            return []

        print(f"Running Whisper on downloaded audio: {target_audio_path}")
        start_tr = time.time()
        segments = transcribe_video(target_audio_path)
        diagnostics["transcribe_time"] = round(time.time() - start_tr, 2)

        diagnostics["success"] = True
        diagnostics["reason"] = "success"
        diagnostics["detail"] = f"Transcribed {len(segments)} segments using Whisper audio fallback."

        if return_diagnostics:
            return segments, diagnostics
        return segments

    except yt_dlp.utils.DownloadError as e:
        err_str = str(e)
        logger.warning("yt-dlp DownloadError for %s: %s", video_id, err_str)
        if "429" in err_str or "Too Many Requests" in err_str:
            diagnostics["reason"] = "rate_limited"
            diagnostics["detail"] = "YouTube is temporarily rate-limiting audio downloads from your IP."
        elif "Sign in" in err_str or "Private video" in err_str:
            diagnostics["reason"] = "restricted"
            diagnostics["detail"] = "The video is private, age-restricted, or requires sign-in."
        elif "unavailable" in err_str.lower():
            diagnostics["reason"] = "unavailable"
            diagnostics["detail"] = "The video audio source is unavailable on YouTube."
        elif "File is larger than max-filesize" in err_str:
            diagnostics["reason"] = "size_exceeded"
            diagnostics["detail"] = "The audio track exceeds the maximum permitted download size (150MB)."
        else:
            diagnostics["reason"] = "download_failed"
            diagnostics["detail"] = f"Audio download failed: {err_str.splitlines()[0] if err_str else 'Unknown error'}"

        if return_diagnostics:
            return [], diagnostics
        return []

    except Exception as e:
        logger.exception("Unexpected error in audio fallback for %s: %s", video_id, e)
        diagnostics["reason"] = "error"
        diagnostics["detail"] = f"Transcription fallback error: {type(e).__name__} - {e}"

        if return_diagnostics:
            return [], diagnostics
        return []

    finally:
        # Always clean up temporary audio files
        if target_audio_path and os.path.exists(target_audio_path):
            try:
                os.remove(target_audio_path)
                logger.debug("Cleaned up temporary audio: %s", target_audio_path)
            except OSError:
                pass


def process_youtube_video(url: str, enable_audio_fallback: bool = True) -> dict:
    """
    Main entrypoint for YouTube video processing:
      1. Validates URL and extracts video ID safely.
      2. Checks video accessibility and fetches metadata via oEmbed.
      3. Attempts fast caption/transcript retrieval with granular diagnostics.
      4. If captions unavailable, falls back to authorized Whisper audio transcription.
      5. Returns specific, user-friendly diagnostic errors rather than generic notices.
    """
    start_time = time.time()
    logger.info("Starting processing for YouTube URL: %s", url)

    # 1. URL validation
    is_valid, video_id, error_msg = extract_youtube_video_id(url)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
            "error_type": "invalid_url",
            "video_id": ""
        }

    # 2. Metadata & accessibility check
    meta = get_youtube_metadata(video_id)
    if not meta.get("accessible", True):
        return {
            "success": False,
            "error": meta.get("error") or "Video is not accessible on YouTube.",
            "error_type": "video_unavailable",
            "video_id": video_id,
            "title": meta.get("title", f"YouTube Video ({video_id})"),
            "thumbnail": meta.get("thumbnail"),
        }

    # 3. Attempt captions first
    caption_res = fetch_youtube_captions(video_id, return_diagnostics=True)
    if isinstance(caption_res, tuple) and len(caption_res) == 2:
        segments, caption_diag = caption_res
    else:
        segments = caption_res or []
        caption_diag = {
            "success": bool(segments),
            "reason": "success" if segments else "not_found",
            "detail": ""
        }
    transcript_source = "captions"

    # 4. If captions unavailable, attempt Whisper audio fallback
    audio_diag = {}
    if not segments and enable_audio_fallback:
        print(f"No captions found for {video_id} ({caption_diag.get('detail')}). Attempting Whisper audio fallback...")
        logger.info("Attempting audio fallback for %s because captions failed (%s)", video_id, caption_diag.get("reason"))
        audio_res = transcribe_youtube_audio(video_id, return_diagnostics=True)
        if isinstance(audio_res, tuple) and len(audio_res) == 2:
            segments, audio_diag = audio_res
        else:
            segments = audio_res or []
            audio_diag = {
                "success": bool(segments),
                "reason": "success" if segments else "none",
                "detail": ""
            }
        if segments:
            transcript_source = "whisper_audio"

    # 5. Synthesize specific diagnostic error if both failed
    if not segments:
        cap_reason = caption_diag.get("reason", "unknown")
        aud_reason = audio_diag.get("reason", "none")

        # Select the most accurate and actionable explanation
        if cap_reason == "rate_limited" or aud_reason == "rate_limited":
            user_error = (
                "YouTube access is temporarily rate-limited from your network. "
                "Please wait a few minutes before trying again, or upload a local video file."
            )
            error_type = "rate_limited"
        elif cap_reason == "age_restricted" or aud_reason == "restricted":
            user_error = (
                "This video is age-restricted or private, and cannot be accessed without authentication."
            )
            error_type = "restricted"
        elif cap_reason == "unavailable" or aud_reason == "unavailable":
            user_error = "The requested YouTube video is unavailable or has been removed."
            error_type = "video_unavailable"
        elif cap_reason == "disabled" and not enable_audio_fallback:
            user_error = "Captions are disabled by the video owner, and audio fallback is disabled."
            error_type = "captions_disabled"
        elif cap_reason == "disabled" and audio_diag.get("detail"):
            user_error = (
                f"Captions are disabled by the video owner. "
                f"Audio extraction fallback failed: {audio_diag.get('detail')}"
            )
            error_type = "audio_fallback_failed"
        elif cap_reason == "not_found" and audio_diag.get("detail"):
            user_error = (
                f"No subtitles/captions found for this video. "
                f"Audio extraction fallback failed: {audio_diag.get('detail')}"
            )
            error_type = "audio_fallback_failed"
        else:
            detail_msg = caption_diag.get("detail") or audio_diag.get("detail") or "Subtitles are disabled and audio extraction was not permitted."
            user_error = f"Could not process video transcript: {detail_msg}"
            error_type = "processing_failed"

        logger.error(
            "YouTube processing failed for %s. Caption reason: %s, Audio reason: %s, Error: %s",
            video_id, cap_reason, aud_reason, user_error
        )

        return {
            "success": False,
            "error": user_error,
            "error_type": error_type,
            "video_id": video_id,
            "title": meta.get("title", f"YouTube Video ({video_id})"),
            "thumbnail": meta.get("thumbnail"),
            "caption_diagnostics": caption_diag,
            "audio_diagnostics": audio_diag,
        }

    duration = segments[-1]["end"] if segments else 0.0
    elapsed = time.time() - start_time
    logger.info("YouTube processing complete for %s in %.2fs. Source: %s, Segments: %d",
                video_id, elapsed, transcript_source, len(segments))

    return {
        "success": True,
        "video_id": video_id,
        "title": meta.get("title", f"YouTube Video ({video_id})"),
        "author": meta.get("author", "YouTube Creator"),
        "thumbnail": meta.get("thumbnail"),
        "duration": duration,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "embed_url": f"https://www.youtube.com/embed/{video_id}",
        "transcript_source": transcript_source,
        "segments": segments,
        "caption_diagnostics": caption_diag,
        "audio_diagnostics": audio_diag,
        "error": None
    }
