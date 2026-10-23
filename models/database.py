"""
models/database.py

Relational database models and initialization for the AI Personalized Learning Platform.
Uses SQLAlchemy with SQLite (or configurable DATABASE_URL).
"""

from datetime import datetime
import json
import logging
from flask_sqlalchemy import SQLAlchemy

logger = logging.getLogger("models.database")

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    account_status = db.Column(db.String(30), default="active")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    profile = db.relationship("UserProfile", backref="user", uselist=False, cascade="all, delete-orphan")
    video_sessions = db.relationship("VideoLearningSession", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    quizzes = db.relationship("Quiz", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    quiz_attempts = db.relationship("QuizAttempt", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    skills = db.relationship("UserSkill", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    courses = db.relationship("UserCourse", backref="user", lazy="dynamic", cascade="all, delete-orphan")
    learning_paths = db.relationship("LearningPath", backref="user", lazy="dynamic", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "account_status": self.account_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserProfile(db.Model):
    __tablename__ = "user_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False, index=True)
    
    education_level = db.Column(db.String(80), default="undergraduate")  # school, diploma, undergraduate, postgraduate, professional, other
    field_of_study = db.Column(db.String(150), default="")
    current_semester = db.Column(db.String(50), default="")
    
    # Serialized JSON lists / maps
    interests = db.Column(db.Text, default="[]")  # e.g. ["Python", "Machine Learning"]
    subject_skill_levels = db.Column(db.Text, default="{}")  # e.g. {"Python": "intermediate", "DSA": "beginner"}
    
    primary_goal = db.Column(db.String(150), default="Learn a new technology")
    target_skill = db.Column(db.String(120), default="Python")
    target_completion_date = db.Column(db.String(50), default="")
    
    daily_study_minutes = db.Column(db.Integer, default=45)
    weekly_study_hours = db.Column(db.Float, default=5.0)
    preferred_difficulty = db.Column(db.String(30), default="medium")  # easy, medium, hard, mixed
    preferred_format = db.Column(db.String(50), default="mixture")  # video, reading, practice questions, mixture
    career_goal = db.Column(db.String(150), default="")
    
    onboarding_completed = db.Column(db.Boolean, default=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_interests_list(self) -> list:
        try:
            return json.loads(self.interests) if self.interests else []
        except Exception:
            return []

    def set_interests_list(self, items: list):
        self.interests = json.dumps(items or [])

    def get_skill_levels_dict(self) -> dict:
        try:
            return json.loads(self.subject_skill_levels) if self.subject_skill_levels else {}
        except Exception:
            return {}

    def set_skill_levels_dict(self, levels: dict):
        self.subject_skill_levels = json.dumps(levels or {})

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "education_level": self.education_level,
            "field_of_study": self.field_of_study,
            "current_semester": self.current_semester,
            "interests": self.get_interests_list(),
            "subject_skill_levels": self.get_skill_levels_dict(),
            "primary_goal": self.primary_goal,
            "target_skill": self.target_skill,
            "target_completion_date": self.target_completion_date,
            "daily_study_minutes": self.daily_study_minutes,
            "weekly_study_hours": self.weekly_study_hours,
            "preferred_difficulty": self.preferred_difficulty,
            "preferred_format": self.preferred_format,
            "career_goal": self.career_goal,
            "onboarding_completed": self.onboarding_completed,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class VideoLearningSession(db.Model):
    __tablename__ = "video_learning_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    
    source_id = db.Column(db.String(120), nullable=False, index=True)  # e.g., 'yt_xyz' or 'local_lecture.mp4'
    video_type = db.Column(db.String(30), default="local")  # 'youtube' or 'local'
    title = db.Column(db.String(255), default="Untitled Video")
    video_url = db.Column(db.String(500), default="")
    thumbnail = db.Column(db.String(500), default="")
    duration = db.Column(db.Float, default=0.0)
    transcript_source = db.Column(db.String(50), default="whisper")
    transcript_json = db.Column(db.Text, default="[]")
    processing_status = db.Column(db.String(30), default="completed")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    quizzes = db.relationship("Quiz", backref="video_session", lazy="dynamic")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_segments(self) -> list:
        try:
            return json.loads(self.transcript_json) if self.transcript_json else []
        except Exception:
            return []

    def set_segments(self, segments: list):
        self.transcript_json = json.dumps(segments or [])

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source_id": self.source_id,
            "video_type": self.video_type,
            "title": self.title,
            "video_url": self.video_url,
            "thumbnail": self.thumbnail,
            "duration": self.duration,
            "transcript_source": self.transcript_source,
            "segments_count": len(self.get_segments()),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Quiz(db.Model):
    __tablename__ = "quizzes"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    video_session_id = db.Column(db.Integer, db.ForeignKey("video_learning_sessions.id"), nullable=True, index=True)
    
    title = db.Column(db.String(255), nullable=False)
    source_type = db.Column(db.String(50), default="video")  # 'video', 'topic', 'revision'
    difficulty = db.Column(db.String(30), default="medium")  # easy, medium, hard, mixed
    topic_tags = db.Column(db.String(255), default="")  # comma-separated
    question_count = db.Column(db.Integer, default=5)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    questions = db.relationship("QuizQuestion", backref="quiz", cascade="all, delete-orphan", lazy="joined")
    attempts = db.relationship("QuizAttempt", backref="quiz", cascade="all, delete-orphan", lazy="dynamic")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self, include_answers=False):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "video_session_id": self.video_session_id,
            "title": self.title,
            "source_type": self.source_type,
            "difficulty": self.difficulty,
            "topic_tags": [t.strip() for t in self.topic_tags.split(",") if t.strip()] if self.topic_tags else [],
            "question_count": self.question_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "questions": [q.to_dict(include_answer=include_answers) for q in self.questions]
        }


class QuizQuestion(db.Model):
    __tablename__ = "quiz_questions"

    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey("quizzes.id"), nullable=False, index=True)
    
    question_text = db.Column(db.Text, nullable=False)
    options_json = db.Column(db.Text, nullable=False)  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_option = db.Column(db.String(10), nullable=False)  # "A", "B", "C", "D"
    explanation = db.Column(db.Text, default="")
    topic = db.Column(db.String(100), default="General")
    difficulty = db.Column(db.String(30), default="medium")
    source_start = db.Column(db.Float, nullable=True)
    source_end = db.Column(db.Float, nullable=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_options(self) -> dict:
        try:
            return json.loads(self.options_json) if self.options_json else {}
        except Exception:
            return {}

    def set_options(self, opts: dict):
        self.options_json = json.dumps(opts or {})

    def to_dict(self, include_answer=False):
        data = {
            "id": self.id,
            "quiz_id": self.quiz_id,
            "question": self.question_text,
            "options": self.get_options(),
            "topic": self.topic,
            "difficulty": self.difficulty,
            "source_start": self.source_start,
            "source_end": self.source_end,
        }
        if include_answer:
            data["correct_option"] = self.correct_option
            data["explanation"] = self.explanation
        return data


class QuizAttempt(db.Model):
    __tablename__ = "quiz_attempts"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey("quizzes.id"), nullable=False, index=True)
    
    score = db.Column(db.Float, default=0.0)
    percentage = db.Column(db.Float, default=0.0)
    correct_count = db.Column(db.Integer, default=0)
    incorrect_count = db.Column(db.Integer, default=0)
    unanswered_count = db.Column(db.Integer, default=0)
    total_questions = db.Column(db.Integer, default=0)
    
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    responses = db.relationship("QuizResponse", backref="attempt", cascade="all, delete-orphan", lazy="joined")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "quiz_id": self.quiz_id,
            "quiz_title": self.quiz.title if self.quiz else "Quiz",
            "score": self.score,
            "percentage": self.percentage,
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "unanswered_count": self.unanswered_count,
            "total_questions": self.total_questions,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "responses": [r.to_dict() for r in self.responses]
        }


class QuizResponse(db.Model):
    __tablename__ = "quiz_responses"

    id = db.Column(db.Integer, primary_key=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey("quiz_attempts.id"), nullable=False, index=True)
    question_id = db.Column(db.Integer, db.ForeignKey("quiz_questions.id"), nullable=False, index=True)
    
    selected_option = db.Column(db.String(10), nullable=True)  # "A", "B", "C", "D" or None
    is_correct = db.Column(db.Boolean, default=False)

    # Relationships
    question = db.relationship("QuizQuestion", lazy="joined")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        return {
            "id": self.id,
            "attempt_id": self.attempt_id,
            "question_id": self.question_id,
            "question_text": self.question.question_text if self.question else "",
            "options": self.question.get_options() if self.question else {},
            "selected_option": self.selected_option,
            "correct_option": self.question.correct_option if self.question else "",
            "is_correct": self.is_correct,
            "explanation": self.question.explanation if self.question else "",
            "topic": self.question.topic if self.question else "",
            "source_start": self.question.source_start if self.question else None,
            "source_end": self.question.source_end if self.question else None,
        }


class UserSkill(db.Model):
    __tablename__ = "user_skills"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    skill_name = db.Column(db.String(100), nullable=False, index=True)
    
    estimated_proficiency = db.Column(db.Float, default=0.0)  # 0 to 100%
    confidence_score = db.Column(db.Float, default=0.0)  # 0.0 to 1.0
    attempted_count = db.Column(db.Integer, default=0)
    correct_count = db.Column(db.Integer, default=0)
    trend = db.Column(db.String(30), default="stable")  # 'improving', 'declining', 'stable'
    last_assessed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def confidence_level(self) -> str:
        if self.attempted_count < 5:
            return "low"
        elif self.attempted_count < 15:
            return "medium"
        return "high"

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "skill_name": self.skill_name,
            "estimated_proficiency": round(self.estimated_proficiency, 1),
            "confidence_score": round(self.confidence_score, 2),
            "confidence_level": self.confidence_level(),
            "attempted_count": self.attempted_count,
            "correct_count": self.correct_count,
            "trend": self.trend,
            "last_assessed_at": self.last_assessed_at.isoformat() if self.last_assessed_at else None,
        }


class Course(db.Model):
    __tablename__ = "courses"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    provider = db.Column(db.String(100), nullable=False)
    course_url = db.Column(db.String(500), nullable=False)
    difficulty = db.Column(db.String(30), default="beginner")  # beginner, intermediate, advanced
    duration_hours = db.Column(db.Float, default=10.0)
    skills_covered = db.Column(db.Text, default="")  # comma-separated or JSON
    prerequisites = db.Column(db.Text, default="")  # comma-separated or JSON
    resource_type = db.Column(db.String(50), default="course")  # course, interactive, video_series, reading
    is_ai_generated = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_skills_list(self) -> list:
        if not self.skills_covered:
            return []
        try:
            parsed = json.loads(self.skills_covered)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [s.strip() for s in self.skills_covered.split(",") if s.strip()]

    def get_prerequisites_list(self) -> list:
        if not self.prerequisites:
            return []
        try:
            parsed = json.loads(self.prerequisites)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [p.strip() for p in self.prerequisites.split(",") if p.strip()]

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "provider": self.provider,
            "course_url": self.course_url,
            "difficulty": self.difficulty,
            "duration_hours": self.duration_hours,
            "skills_covered": self.get_skills_list(),
            "prerequisites": self.get_prerequisites_list(),
            "resource_type": self.resource_type,
            "is_ai_generated": self.is_ai_generated,
            "is_active": self.is_active,
        }


class UserCourse(db.Model):
    __tablename__ = "user_courses"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    course_id = db.Column(db.Integer, db.ForeignKey("courses.id"), nullable=False, index=True)
    
    status = db.Column(db.String(30), default="enrolled")  # saved, enrolled, in_progress, completed, dismissed
    progress_percent = db.Column(db.Float, default=0.0)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    # Relationships
    course = db.relationship("Course", lazy="joined")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "course": self.course.to_dict() if self.course else None,
            "status": self.status,
            "progress_percent": round(self.progress_percent, 1),
            "enrolled_at": self.enrolled_at.isoformat() if self.enrolled_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class LearningPath(db.Model):
    __tablename__ = "learning_paths"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    
    title = db.Column(db.String(255), nullable=False)
    goal = db.Column(db.String(255), nullable=False)
    starting_skill_level = db.Column(db.String(30), default="beginner")
    target_skill_level = db.Column(db.String(30), default="advanced")
    status = db.Column(db.String(30), default="active")  # active, paused, completed
    overall_progress = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    milestones = db.relationship(
        "LearningMilestone",
        backref="learning_path",
        cascade="all, delete-orphan",
        order_by="LearningMilestone.order_index",
        lazy="joined"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_progress(self) -> float:
        if not self.milestones:
            return 0.0
        completed = sum(1 for m in self.milestones if m.status == "completed")
        self.overall_progress = round((completed / len(self.milestones)) * 100.0, 1)
        return self.overall_progress

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "goal": self.goal,
            "starting_skill_level": self.starting_skill_level,
            "target_skill_level": self.target_skill_level,
            "status": self.status,
            "overall_progress": self.overall_progress,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "milestones": [m.to_dict() for m in self.milestones]
        }


class LearningMilestone(db.Model):
    __tablename__ = "learning_milestones"

    id = db.Column(db.Integer, primary_key=True)
    learning_path_id = db.Column(db.Integer, db.ForeignKey("learning_paths.id"), nullable=False, index=True)
    
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, default="")
    order_index = db.Column(db.Integer, default=0)
    estimated_hours = db.Column(db.Float, default=5.0)
    status = db.Column(db.String(30), default="not_started")  # not_started, in_progress, completed, needs_review
    target_skill = db.Column(db.String(100), default="")
    prerequisites = db.Column(db.Text, default="")
    related_course_id = db.Column(db.Integer, db.ForeignKey("courses.id"), nullable=True)
    related_video_id = db.Column(db.Integer, db.ForeignKey("video_learning_sessions.id"), nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)

    # Relationships
    related_course = db.relationship("Course", lazy="joined")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        return {
            "id": self.id,
            "learning_path_id": self.learning_path_id,
            "title": self.title,
            "description": self.description,
            "order_index": self.order_index,
            "estimated_hours": self.estimated_hours,
            "status": self.status,
            "target_skill": self.target_skill,
            "prerequisites": [p.strip() for p in self.prerequisites.split(",") if p.strip()] if self.prerequisites else [],
            "related_course": self.related_course.to_dict() if self.related_course else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


def init_db(app):
    """
    Initializes database tables safely without dropping existing data.
    Seeds initial curated educational courses if catalogue is empty.
    """
    db.init_app(app)
    with app.app_context():
        db.create_all()
        logger.info("Database tables initialized successfully.")
        
        # Seed course catalogue if empty
        try:
            from models.seed_data import seed_courses_if_needed
            seed_courses_if_needed()
        except Exception as e:
            logger.warning("Could not run course seeding: %s", e)
