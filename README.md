# Temporal-Multimodal-RAG: AI-Powered Personalized Learning Platform

An integrated, temporal-aware multimodal video learning platform and AI tutor with automated speech transcription (Whisper), grounded video question answering, automatic transcript-based MCQ quiz generation, empirical skill diagnostics, explainable course recommendations, and adaptive personalized learning paths.

---

## Quickstart (One Command)

To start the application, open your terminal in the project root and run:

```bat
run.bat
```

or in PowerShell:

```powershell
.\run.bat
```

### What happens automatically:
1. Validates the Python 3.10 virtual environment (`.venv`).
2. Verifies application dependencies (`Flask`, `Flask-SQLAlchemy`, `Whisper`, `PyTorch`, `yt-dlp`, etc.).
3. Initializes the relational database (`learning_platform.db`) idempotently and seeds verified course resources.
4. Dynamically resolves **FFmpeg** without requiring global system installation.
5. Starts the Flask server on an available port (default `http://127.0.0.1:5000`).
6. Polls the `/health` endpoint until the server is accepting requests.
7. Displays the working application URL in the terminal and automatically opens the browser.
8. Runs until you press **Ctrl+C**, stopping cleanly.

---

## First-Time Setup & Environment Repair

If you are setting up the project or need to repair the virtual environment:

```bat
setup.bat
```

### Options:
- **Standard Setup**: `setup.bat` (verifies/creates `.venv`, installs requirements, checks dynamic FFmpeg, and runs the test suite).
- **Environment Repair**: `setup.bat --repair` (recreates the virtual environment cleanly from scratch).

---

## System Architecture & Features

### 1. User Registration, Authentication & Profiles
- **Secure Authentication**: Registration with server-side validation (email format regex, password complexity, duplicate rejection) and Scrypt password hashing.
- **Session Security**: Session cookies with configurable secret keys, session lifetime, and CSRF token protection.
- **Multi-Step Onboarding**: Collects education level, field of study, subject-specific skill levels (Beginner / Intermediate / Advanced), learning interests, career goals, and daily study availability.
- **Strict Multi-User Isolation**: User-specific database records ensure one learner cannot view or modify another's learning sessions, quizzes, skill profiles, or learning paths.

### 2. Temporal Multimodal Video RAG ("Learn with Video")
- **Dual Ingestion**:
  - **YouTube URLs**: Fetches official captions or falls back to authorized Whisper audio transcription.
  - **Local Video Files**: Upload `.mp4`, `.mov`, `.avi`, or `.mkv` files directly.
- **Context-Aware Temporal RAG**: Stop-word filtered token matching, passage merging, and interactive timestamp seeking.
- **Configurable LLM Synthesis**: Supports OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`), Groq, or local Ollama with transparent transcript-excerpt fallbacks when an LLM is not configured.
- **Session Persistence**: When logged in, processed videos are linked directly to the user's `VideoLearningSession`.

### 3. Automatic AI MCQ Quiz Generator
- **Grounded Question Generation**: Automatically extracts key concepts, definitions, and facts from video transcripts (full or section-specific) or custom topics.
- **Strict Schema Validation**: Produces 5, 10, 15, or 20 questions with 4 distinct options, exactly one correct option, detailed explanations, and source timestamps.
- **Authoritative Server-Side Scoring**: Hides answers from the browser until submission, calculates scores, tracks time, and records detailed question responses.
- **Reliable Fallback**: Uses a deterministic rule-based generator from transcript segments if the LLM provider is unavailable.

### 4. Empirical Skill Analytics
- **Topic-Level Tracking**: Tracks accuracy across fine-grained skills (e.g., `Python: Loops`, `DSA: Trees`, `ML: Evaluation`).
- **Confidence Scoring**: Computes low / medium / high confidence metrics based on question volume (e.g., <5 questions vs. 15+ questions).
- **Trend Detection**: Tracks recent performance trends (`improving`, `declining`, `steady`).
- **Post-Quiz Diagnostic Feedback**: Suggests targeted revision modules and links back to the specific video timestamps where concepts were taught.

### 5. Explainable Course Recommendation Engine
- **Multi-Factor Weighted Ranking**:
  - Goal Alignment: 30%
  - Skill Gap Match (targeting weak skills): 30%
  - Topic / Interest Match: 15%
  - Difficulty Level Match: 15%
  - Study Time Fit: 10%
- **Transparent Explanations**: Explains exactly why each course was suggested based on empirical quiz performance and profile goals.
- **Curated Catalogue**: Seeded with verified, free educational resources from Harvard CS50, MIT OpenCourseWare, freeCodeCamp, MDN, Python.org, and Kaggle.
- **Course Status Management**: Save/enroll in courses, mark as completed, or dismiss recommendations.

### 6. Adaptive Personalized Learning Roadmaps
- **Goal-Oriented Milestones**: Generates ordered milestones with estimated durations tailored to the user's daily study time budget.
- **Adaptive Milestones**: Quiz performance dynamically updates milestone states (e.g., inserting `needs_review` flags on poor prerequisite scores).
- **Interactive Progress Tracking**: Step-by-step milestone tracking, marking milestones as In Progress or Completed.

---

## Application Endpoints

### Core Video & RAG Routes
| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Main interactive learning dashboard (or landing page if guest) |
| `/video` | `GET` | Interactive video learning interface ("Learn with Video") |
| `/health` | `GET` | Lightweight health check, diagnostics, and LLM provider status |
| `/process_youtube` | `POST` | Ingest YouTube video URL (JSON or form) |
| `/upload` | `POST` | Upload and transcribe local video file |
| `/ask` | `POST` | Ask a question against the active video transcript |
| `/uploads/<filename>` | `GET` | Secure static video playback stream |

### Authentication Routes
| Endpoint | Method | Description |
|---|---|---|
| `/register` | `GET`, `POST` | User registration with validation and password hashing |
| `/login` | `GET`, `POST` | User login and session initiation |
| `/logout` | `GET` | User logout and session invalidation |
| `/onboarding` | `GET`, `POST` | Multi-step learning profile setup wizard |

### Learning Platform Routes
| Endpoint | Method | Description |
|---|---|---|
| `/dashboard` | `GET` | Personalized dashboard with real metrics, streaks, and recommendations |
| `/quizzes` | `GET` | User's quiz library and past attempts |
| `/quizzes/generate` | `GET`, `POST` | Generate an AI MCQ quiz from a video transcript or topic |
| `/quizzes/<id>/take` | `GET` | Interactive quiz-taking interface |
| `/quizzes/<id>/submit` | `POST` | Authoritative server-side scoring and response submission |
| `/quizzes/attempts/<id>` | `GET` | Detailed quiz review with explanations and source timestamps |
| `/skills` | `GET` | Skill diagnostic dashboard with proficiency, confidence, and trends |
| `/courses` | `GET` | Personalized course recommendations with transparent explanations |
| `/courses/<id>/enroll` | `POST` | Save / enroll in a course |
| `/courses/<id>/complete`| `POST` | Mark an enrolled course as completed |
| `/courses/<id>/dismiss` | `POST` | Dismiss a course recommendation |
| `/learning-path` | `GET` | Personalized learning roadmap and milestone schedule |
| `/learning-path/milestones/<id>/status` | `POST` | Update milestone completion status |
| `/history` | `GET` | Consolidated learning activity, video sessions, and quiz history |
| `/profile` | `GET`, `POST` | Edit profile, education, skills, interests, and study preferences |

---

## Database Schema (Relational SQLite / SQLAlchemy)

The database models are defined in `models/database.py`:
- `User`: Accounts with email, hashed passwords, creation dates.
- `UserProfile`: Educational background, interests, goals, daily study time, onboarding status.
- `VideoLearningSession`: Video source (local/YouTube), title, transcript, timestamps, and owner.
- `Quiz`: Generated quizzes with difficulty, question count, topic tags, and video session links.
- `QuizQuestion`: Question text, options (A, B, C, D), correct option, explanation, source timestamps.
- `QuizAttempt`: Completed attempts, calculated score, correct/incorrect/unanswered counts, duration.
- `QuizResponse`: Individual answer per question with correctness status.
- `UserSkill`: Tracked skill, rolling accuracy, question volume, confidence rating, recent trend.
- `Course`: Curated course catalog with difficulty, duration, skills taught, prerequisites, and URLs.
- `UserCourse`: User enrollment status (`saved`, `in_progress`, `completed`, `dismissed`).
- `LearningPath`: Tailored roadmap with target goal, estimated total hours, and completion percentage.
- `LearningMilestone`: Ordered roadmap milestones with status (`not_started`, `in_progress`, `completed`, `needs_review`).

---

## Configuration

Customize runtime settings by copying `.env.example` to `.env`:

```ini
# Network Binding
HOST=127.0.0.1
PORT=5000

# Flask Security
FLASK_DEBUG=false
FLASK_SECRET_KEY=your-custom-secret-key-change-in-production

# Database Configuration (Optional override)
# DATABASE_URL=sqlite:///learning_platform.db

# Whisper Model (tiny, base, small, medium, large)
WHISPER_MODEL=base

# ----------------------------------------------------
# LLM Provider Configuration (Optional)
# ----------------------------------------------------
# Option 1: OpenAI or OpenAI-Compatible (Groq, Together, Ollama, OpenRouter)
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_BASE=https://api.openai.com/v1

# Option 2: Google Gemini
# GEMINI_API_KEY=your-gemini-api-key
# GEMINI_MODEL=gemini-1.5-flash

# Option 3: Local Ollama (No API Key Required)
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_MODEL=llama3
```

---

## Running Automated Tests

Run the full automated test suite anytime with:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests
```

**77 comprehensive automated tests** verify both the original video RAG pipeline and the new AI learning platform:
- **Authentication & Security (11 tests)**: Registration validation, duplicate email rejection, password hashing, session management, route protection.
- **AI Quiz Generation & Scoring**: Transcript-grounded generation, schema validation, answer hiding, server-side scoring, attempt persistence.
- **Adaptive Skill Analytics**: Rolling accuracy calculation, sample-size confidence estimation, diagnostic revision recommendations.
- **Course Recommendation**: Weighted score calculation, skill-gap alignment, prerequisite handling, course status updates.
- **Learning Path Roadmaps**: Milestone generation, study-time pace planning, adaptive milestone updates (`needs_review`).
- **Multi-User Isolation**: Independent video sessions, quizzes, skill profiles, and learning paths across different users.
- **Backward Compatibility (66 tests preserved)**: Video ingestion, Whisper transcription, timestamp navigation, keyword RAG retrieval, and health checks.

