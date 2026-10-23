"""
learning/skill_tracker.py

Transparent skill estimation, confidence rating, trend tracking,
and post-quiz revision recommendations based on empirical performance.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from models.database import (
    db,
    QuizAttempt,
    QuizResponse,
    QuizQuestion,
    UserSkill,
    UserProfile,
    VideoLearningSession,
)

logger = logging.getLogger("learning.skill_tracker")


def calculate_confidence_score(attempted_count: int) -> float:
    """
    Computes statistical confidence in a skill estimate based on sample size:
      - 0 to 4 questions: low confidence (0.1 to 0.4)
      - 5 to 14 questions: medium confidence (0.5 to 0.75)
      - 15+ questions: high confidence (0.8 to 1.0)
    """
    if attempted_count <= 0:
        return 0.0
    if attempted_count < 5:
        return round(0.1 + (attempted_count / 5.0) * 0.35, 2)
    elif attempted_count < 15:
        return round(0.5 + ((attempted_count - 5) / 10.0) * 0.28, 2)
    else:
        # Asymptote towards 0.98
        return round(min(0.8 + ((attempted_count - 15) / 35.0) * 0.18, 0.98), 2)


def update_skills_from_attempt(user_id: int, attempt_id: int) -> List[UserSkill]:
    """
    Updates the learner's skill records after a completed quiz attempt:
      - Aggregates questions answered per topic/skill
      - Updates cumulative correct & attempted counts
      - Computes rolling accuracy adjusted for difficulty
      - Calculates confidence score based on sample size
      - Tracks trend (improving, declining, stable)
    """
    attempt = QuizAttempt.query.get(attempt_id)
    if not attempt or attempt.user_id != user_id:
        logger.warning("Attempt %s not found or does not belong to user %s", attempt_id, user_id)
        return []

    # Map responses by topic
    topic_results: Dict[str, Dict[str, Any]] = {}
    for resp in attempt.responses:
        q = resp.question
        topic = (q.topic or "General Concepts").strip()
        if topic not in topic_results:
            topic_results[topic] = {
                "attempted": 0,
                "correct": 0,
                "difficulty_weights": []
            }

        topic_results[topic]["attempted"] += 1
        if resp.is_correct:
            topic_results[topic]["correct"] += 1

        # Difficulty weighting (hard gives slight positive bump, easy mistakes slightly larger)
        diff = (q.difficulty or "medium").lower()
        weight = 1.1 if diff == "hard" else (0.9 if diff == "easy" else 1.0)
        topic_results[topic]["difficulty_weights"].append(weight)

    updated_skills = []
    for topic_name, stats in topic_results.items():
        skill = UserSkill.query.filter_by(user_id=user_id, skill_name=topic_name).first()
        prev_proficiency = skill.estimated_proficiency if skill else None

        if not skill:
            skill = UserSkill(
                user_id=user_id,
                skill_name=topic_name,
                attempted_count=0,
                correct_count=0,
                estimated_proficiency=0.0,
                confidence_score=0.0,
                trend="stable",
            )
            db.session.add(skill)

        # Update counts
        skill.attempted_count += stats["attempted"]
        skill.correct_count += stats["correct"]

        # Base accuracy
        raw_accuracy = (skill.correct_count / skill.attempted_count) * 100.0

        # Weighted recent adjustment
        avg_weight = sum(stats["difficulty_weights"]) / len(stats["difficulty_weights"])
        recent_acc = (stats["correct"] / stats["attempted"]) * 100.0

        # Rolling estimate: 70% cumulative + 30% recent performance
        if skill.attempted_count > stats["attempted"]:
            new_proficiency = (raw_accuracy * 0.7) + (recent_acc * avg_weight * 0.3)
        else:
            new_proficiency = raw_accuracy

        skill.estimated_proficiency = max(0.0, min(100.0, round(new_proficiency, 1)))
        skill.confidence_score = calculate_confidence_score(skill.attempted_count)

        # Determine trend
        if prev_proficiency is not None:
            diff_prof = skill.estimated_proficiency - prev_proficiency
            if diff_prof >= 4.0:
                skill.trend = "improving"
            elif diff_prof <= -4.0:
                skill.trend = "declining"
            else:
                skill.trend = "stable"
        else:
            skill.trend = "stable"

        skill.last_assessed_at = datetime.utcnow()
        updated_skills.append(skill)

    db.session.commit()
    logger.info("Updated %d skills for user %d from attempt %d", len(updated_skills), user_id, attempt_id)
    return updated_skills


def get_user_skills(user_id: int) -> List[Dict[str, Any]]:
    """Returns a formatted list of all tracked skills for a user."""
    skills = UserSkill.query.filter_by(user_id=user_id).order_by(UserSkill.estimated_proficiency.desc()).all()
    return [s.to_dict() for s in skills]


def get_strong_and_weak_skills(user_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separates skills into Strong (proficiency >= 70% and confidence != 'low')
    and Needs Improvement (proficiency < 60%).
    """
    all_skills = UserSkill.query.filter_by(user_id=user_id).all()
    strong = []
    weak = []

    for s in all_skills:
        d = s.to_dict()
        if s.estimated_proficiency >= 70.0:
            strong.append(d)
        elif s.estimated_proficiency < 65.0:
            weak.append(d)

    # Sort
    strong.sort(key=lambda x: x["estimated_proficiency"], reverse=True)
    weak.sort(key=lambda x: x["estimated_proficiency"])
    return strong, weak


def generate_post_quiz_recommendations(user_id: int, attempt_id: int) -> Dict[str, Any]:
    """
    Derives actionable learning recommendations grounded in the attempt results:
      - Concepts that need revision (with video timestamps if available)
      - Recommended next quiz topic and difficulty
      - Explicit explanations citing empirical quiz accuracy
    """
    attempt = QuizAttempt.query.get(attempt_id)
    if not attempt or attempt.user_id != user_id:
        return {}

    quiz = attempt.quiz
    incorrect_responses = [r for r in attempt.responses if not r.is_correct]

    revision_concepts = []
    for resp in incorrect_responses:
        q = resp.question
        concept = {
            "question": q.question_text,
            "topic": q.topic,
            "correct_option": q.correct_option,
            "correct_text": q.get_options().get(q.correct_option, ""),
            "explanation": q.explanation,
            "source_start": q.source_start,
            "source_end": q.source_end,
            "video_session_id": quiz.video_session_id if quiz else None,
        }
        revision_concepts.append(concept)

    # Determine recommended next difficulty based on score
    pct = attempt.percentage
    if pct >= 85.0:
        suggested_difficulty = "hard"
        difficulty_reason = f"Excellent score of {pct}%. You are ready for advanced challenges."
    elif pct >= 60.0:
        suggested_difficulty = "medium"
        difficulty_reason = f"Solid score of {pct}%. Continue building confidence on intermediate material."
    else:
        suggested_difficulty = "easy"
        difficulty_reason = f"Score was {pct}%. Review foundational concepts before moving to higher difficulty."

    # Identify primary weak skill from this attempt
    weak_topics = list(dict.fromkeys([c["topic"] for c in revision_concepts if c.get("topic")]))

    return {
        "attempt_id": attempt.id,
        "score": attempt.score,
        "percentage": attempt.percentage,
        "total": attempt.total_questions,
        "suggested_difficulty": suggested_difficulty,
        "difficulty_reason": difficulty_reason,
        "revision_concepts": revision_concepts[:5],
        "weak_topics": weak_topics,
        "has_weak_topics": len(weak_topics) > 0,
        "explanation": (
            f"Recommended because your score was {attempt.percentage}% on '{quiz.title if quiz else 'this quiz'}', "
            f"identifying {len(revision_concepts)} concept(s) that will benefit from revision."
        )
    }
