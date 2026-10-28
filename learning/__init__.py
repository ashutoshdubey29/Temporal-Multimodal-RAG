"""
learning/__init__.py
====================
VIVA / ARCHITECTURAL OVERVIEW:
This package contains the adaptive learning intelligence modules of AetherLearn:
1. `quiz_generator.py`: Generates MCQs from video transcripts/topics using LLMs or rule-based fallbacks.
2. `skill_tracker.py`: Tracks student mastery over time with sample-size confidence and rolling accuracy.
3. `recommendation_engine.py`: Personalized 5-factor explainable course recommendations.
4. `path_generator.py`: Dynamic study path roadmaps that adapt when students pass or fail quizzes.
"""
from learning.quiz_generator import (
    generate_and_save_quiz,
    score_quiz_submission,
    prepare_transcript_context,
    validate_quiz_questions,
)
from learning.skill_tracker import (
    update_skills_from_attempt,
    get_user_skills,
    get_strong_and_weak_skills,
    generate_post_quiz_recommendations,
)
from learning.recommendation_engine import (
    get_personalized_recommendations,
    save_user_course_status,
)
from learning.path_generator import (
    get_or_create_learning_path,
    create_learning_path_for_user,
    adapt_learning_path_after_quiz,
    calculate_study_schedule,
)
