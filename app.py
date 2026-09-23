from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from processing.transcription import transcribe_video
from rag.qa import find_relevant_segments

import os


app = Flask(__name__)

# =====================================
# PATH SETTINGS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =====================================
# TEMPORARY SERVER MEMORY
# =====================================

current_segments = []
current_filename = None


# =====================================
# HOME
# =====================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =====================================
# UPLOAD VIDEO
# =====================================

@app.route("/upload", methods=["POST"])
def upload():

    global current_segments
    global current_filename


    if "video" not in request.files:

        return render_template(
            "index.html",
            error="Please select a video."
        )


    video = request.files["video"]


    if video.filename == "":

        return render_template(
            "index.html",
            error="Please select a video."
        )


    filename = secure_filename(
        video.filename
    )


    video_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # Save video
    video.save(video_path)


    print("\n================================")
    print("VIDEO UPLOADED")
    print("================================")

    print("Filename:", filename)
    print("Path:", video_path)
    print("Exists:", os.path.exists(video_path))


    # =================================
    # TRANSCRIPTION
    # =================================

    print("\nStarting transcription...")

    try:

        segments = transcribe_video(
            video_path
        )

    except Exception as e:

        print("\nTRANSCRIPTION ERROR:")
        print(e)

        return render_template(
            "index.html",
            error=f"Transcription error: {e}"
        )


    print("\nTRANSCRIPTION COMPLETED!")

    print("Total segments:", len(segments))


    # =================================
    # STORE ON SERVER
    # =================================

    current_segments = segments

    current_filename = filename


    # =================================
    # SHOW RESULT
    # =================================

    return render_template(

        "index.html",

        video_url=f"/uploads/{filename}",

        filename=filename,

        transcript=segments

    )


# =====================================
# SERVE VIDEO
# =====================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):

    return send_from_directory(

        app.config["UPLOAD_FOLDER"],

        filename

    )


# =====================================
# ASK AI
# =====================================

@app.route("/ask", methods=["POST"])
def ask():

    global current_segments
    global current_filename


    question = request.form.get(
        "question",
        ""
    ).strip()


    if not question:

        return render_template(
            "index.html",
            error="Please enter a question."
        )


    # =================================
    # CHECK TRANSCRIPT
    # =================================

    if not current_segments:

        return render_template(

            "index.html",

            question=question,

            answer="Please upload and analyze a video first.",

            timestamp="Not available"

        )


    print("\n================================")
    print("USER QUESTION")
    print("================================")

    print(question)


    # =================================
    # RAG RETRIEVAL
    # =================================

    results = find_relevant_segments(

        question,

        current_segments

    )


    # =================================
    # NO MATCH
    # =================================

    if not results:

        return render_template(

            "index.html",

            question=question,

            answer="No relevant information was found in the video transcript.",

            timestamp="Not found",

            transcript=current_segments,

            filename=current_filename,

            video_url=f"/uploads/{current_filename}"

        )


    # =================================
    # BEST RESULT
    # =================================

    best_result = results[0]


    answer = best_result["text"]


    timestamp = (

        f"{best_result['start']:.2f}s - "
        f"{best_result['end']:.2f}s"

    )


    print("\n================================")
    print("RAG RESULT")
    print("================================")

    print("Answer:", answer)
    print("Timestamp:", timestamp)


    # =================================
    # SHOW ANSWER
    # =================================

    return render_template(

        "index.html",

        question=question,

        answer=answer,

        timestamp=timestamp,

        transcript=current_segments,

        filename=current_filename,

        video_url=f"/uploads/{current_filename}"

    )


# =====================================
# RUN FLASK
# =====================================

if __name__ == "__main__":

    print("\n================================")
    print("TEMPORAL-AWARE MULTIMODAL RAG")
    print("================================")

    print("Upload folder:")
    print(UPLOAD_FOLDER)

    print("\nFlask starting...")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )