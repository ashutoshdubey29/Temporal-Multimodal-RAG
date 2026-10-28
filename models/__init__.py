"""
models/__init__.py
==================
VIVA / ARCHITECTURAL OVERVIEW:
This initialization file exposes the core database instance and all SQLAlchemy
ORM data models for the AetherLearn application.

By importing models here and re-exporting them:
1. Other modules (e.g. `routes/`, `auth/`, `learning/`, `app.py`) can cleanly
   import models directly from `models` (e.g., `from models import User, Course, db`).
2. Flask-SQLAlchemy and Alembic discover all model metadata in one central place
   before table creation or database migrations run.
"""
from models.database import (
    db,
    init_db,
    User,
    UserProfile,
    VideoLearningSession,
    Quiz,
    QuizQuestion,
    QuizAttempt,
    QuizResponse,
    UserSkill,
    Course,
    UserCourse,
    LearningPath,
    LearningMilestone,
)
