"""
learning/recommendation_engine.py

Personalized course recommendation engine using explainable weighted ranking.
Considers:
  - User's primary learning/career goal
  - Identified weak skills from quiz attempts (skill gaps)
  - Selected learning interests
  - Preferred learning difficulty & duration fit
  - Exclusion of completed or dismissed courses
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from models.database import Course, User, UserCourse, UserProfile, UserSkill, db

logger = logging.getLogger("learning.recommendation_engine")

# Configurable weights for explainable ranking (Sum = 1.0)
WEIGHT_GOAL = 0.30
WEIGHT_SKILL_GAP = 0.30
WEIGHT_INTEREST = 0.15
WEIGHT_DIFFICULTY = 0.15
WEIGHT_TIME_FIT = 0.10


def calculate_course_match_score(
    course: Course,
    profile: Optional[UserProfile],
    weak_skill_names: List[str],
    user_interest_names: List[str],
    completed_course_ids: List[int]
) -> Tuple[float, List[str]]:
    """
    Computes a normalized relevance score [0.0, 1.0] and generates transparent explanations:
      - Goal Match: alignment with target_skill and primary_goal
      - Skill Gap Match: course covers skills where user scored < 65% on quizzes
      - Interest Match: course tags match profile interests
      - Difficulty Match: course difficulty matches learner's preference / level
      - Time Fit: course duration matches learner's weekly available study time
    """
    if course.id in completed_course_ids:
        return 0.0, ["Already completed"]

    reasons = []
    course_skills = [s.lower() for s in course.get_skills_list()]
    course_text = f"{course.title} {course.description} {' '.join(course_skills)}".lower()

    # 1. Goal Match (0.0 to 1.0)
    goal_score = 0.0
    if profile:
        target = (profile.target_skill or "").strip().lower()
        goal = (profile.primary_goal or "").strip().lower()
        if target and (target in course_text or any(target in s for s in course_skills)):
            goal_score = 1.0
            reasons.append(f"Directly teaches your target skill '{profile.target_skill}'")
        elif goal and any(g_part in course_text for g_part in goal.split()):
            goal_score = 0.6
            reasons.append(f"Aligns with your primary goal of '{profile.primary_goal}'")

    # 2. Skill Gap Match (0.0 to 1.0)
    skill_gap_score = 0.0
    matched_weak = []
    for ws in weak_skill_names:
        ws_lower = ws.lower()
        if ws_lower in course_text or any(ws_lower in s for s in course_skills):
            matched_weak.append(ws)

    if matched_weak:
        skill_gap_score = min(1.0, 0.5 + (0.25 * len(matched_weak)))
        reasons.append(f"Reinforces weak topic(s) from your quizzes: {', '.join(matched_weak[:3])}")

    # 3. Interest Match (0.0 to 1.0)
    interest_score = 0.0
    matched_interests = []
    for item in user_interest_names:
        i_lower = item.lower()
        if i_lower in course_text or any(i_lower in s for s in course_skills):
            matched_interests.append(item)

    if matched_interests:
        interest_score = min(1.0, 0.4 + (0.2 * len(matched_interests)))
        if not matched_weak:
            reasons.append(f"Matches your listed interests: {', '.join(matched_interests[:2])}")

    # 4. Difficulty Match (0.0 to 1.0)
    difficulty_score = 0.5
    user_diff = (profile.preferred_difficulty if profile else "medium").lower()
    c_diff = (course.difficulty or "beginner").lower()
    if user_diff == "mixed":
        difficulty_score = 0.9
    elif user_diff == c_diff:
        difficulty_score = 1.0
        reasons.append(f"Matches your preferred '{user_diff}' difficulty level")
    elif (user_diff == "beginner" and c_diff == "intermediate") or (user_diff == "intermediate" and c_diff == "advanced"):
        difficulty_score = 0.5
    else:
        difficulty_score = 0.3

    # 5. Study Time Fit (0.0 to 1.0)
    time_score = 0.7
    if profile and profile.weekly_study_hours:
        weekly_avail = profile.weekly_study_hours
        weeks_to_finish = course.duration_hours / max(weekly_avail, 1.0)
        if 1.0 <= weeks_to_finish <= 4.0:
            time_score = 1.0
            reasons.append(f"Achievable in ~{round(weeks_to_finish, 1)} weeks with your {weekly_avail}h/week study pace")
        else:
            time_score = 0.6

    # Composite weighted score
    composite_score = (
        (WEIGHT_GOAL * goal_score) +
        (WEIGHT_SKILL_GAP * skill_gap_score) +
        (WEIGHT_INTEREST * interest_score) +
        (WEIGHT_DIFFICULTY * difficulty_score) +
        (WEIGHT_TIME_FIT * time_score)
    )

    if not reasons:
        reasons.append("Covers fundamental concepts valuable for all learners")

    return round(composite_score, 3), reasons


def get_personalized_recommendations(user_id: int, limit: int = 6) -> List[Dict[str, Any]]:
    """
    Ranks courses for the authenticated user based on profile, skills, and quiz history.
    Filters out courses already completed or dismissed by the user.
    """
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    
    # Weak skills (proficiency < 65)
    weak_skills = (
        UserSkill.query.filter(UserSkill.user_id == user_id, UserSkill.estimated_proficiency < 65.0)
        .order_by(UserSkill.estimated_proficiency.asc())
        .all()
    )
    weak_skill_names = [s.skill_name for s in weak_skills]

    user_interests = profile.get_interests_list() if profile else []

    # Completed or dismissed courses
    user_courses = UserCourse.query.filter_by(user_id=user_id).all()
    completed_ids = {uc.course_id for uc in user_courses if uc.status in ("completed", "dismissed")}
    enrolled_ids = {uc.course_id for uc in user_courses if uc.status in ("enrolled", "saved", "in_progress")}

    active_courses = Course.query.filter_by(is_active=True).all()
    ranked = []

    for c in active_courses:
        if c.id in completed_ids:
            continue

        score, reasons = calculate_course_match_score(
            course=c,
            profile=profile,
            weak_skill_names=weak_skill_names,
            user_interest_names=user_interests,
            completed_course_ids=list(completed_ids)
        )

        ranked.append({
            "course": c.to_dict(),
            "score": score,
            "relevance_percent": int(score * 100),
            "reasons": reasons,
            "primary_reason": reasons[0] if reasons else "Recommended for your learning path.",
            "is_enrolled": c.id in enrolled_ids,
        })

    # Sort descending by composite score
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:limit]


def save_user_course_status(user_id: int, course_id: int, status: str) -> Tuple[bool, str]:
    """
    Saves or updates course status for a user ('saved', 'enrolled', 'in_progress', 'completed', 'dismissed').
    """
    valid_statuses = {"saved", "enrolled", "in_progress", "completed", "dismissed"}
    status = status.lower().strip()
    if status not in valid_statuses:
        return False, f"Invalid course status '{status}'."

    course = Course.query.get(course_id)
    if not course:
        return False, "Course not found."

    try:
        user_course = UserCourse.query.filter_by(user_id=user_id, course_id=course_id).first()
        if not user_course:
            user_course = UserCourse(
                user_id=user_id,
                course_id=course_id,
                status=status,
                progress_percent=100.0 if status == "completed" else 0.0,
            )
            db.session.add(user_course)
        else:
            user_course.status = status
            if status == "completed":
                user_course.progress_percent = 100.0
                user_course.completed_at = db.func.now()

        db.session.commit()
        logger.info("Updated course %d status to '%s' for user %d", course_id, status, user_id)
        return True, f"Course status updated to '{status}'."

    except Exception as e:
        db.session.rollback()
        logger.exception("Failed to update course status: %s", e)
        return False, str(e)
