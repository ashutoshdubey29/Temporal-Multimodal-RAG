import whisper
import os


# FFmpeg ka exact path
FFMPEG_PATH = r"C:\Users\ASHUTOSH DUBEY\Downloads\ffmpeg-9.0.2-essentials_build\ffmpeg-9.0.2-essentials_build\bin"

# FFmpeg ko PATH mein add karo
os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")


def transcribe_video(video_path):

    print("Loading Whisper model...")

    model = whisper.load_model("base")

    print("Transcribing video...")

    result = model.transcribe(
        video_path,
        verbose=False
    )

    segments = []

    for segment in result["segments"]:

        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })

    return segments