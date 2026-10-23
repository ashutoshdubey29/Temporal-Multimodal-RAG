"""
routes/learning_routes.py

Main learning platform routes:
  - Dashboard with authentic progress metrics & streaks
  - AI Quiz Generator, taking, and results review
  - Skill tracker overview
  - Course recommendations & catalogue
  - Interactive learning paths & study planner
  - Learning history
"""

from datetime import datetime, timedelta
import json
import logging
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, abort

from auth.auth_manager import login_required, get_current_user, generate_csrf_token
from models.database import (
    db,
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
from learning.quiz_generator import generate_and_save_quiz, score_quiz_submission
from learning.skill_tracker import (
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
    calculate_study_schedule,
)

logger = logging.getLogger("routes.learning")

learning_bp = Blueprint("learning_routes", __name__)


def calculate_user_streak(user_id: int) -> int:
    """
    Computes genuine daily learning streak (consecutive calendar days with activity).
    Considers quiz submissions and video learning sessions.
    """
    attempt_dates = db.session.query(QuizAttempt.submitted_at).filter_by(user_id=user_id).all()
    session_dates = db.session.query(VideoLearningSession.created_at).filter_by(user_id=user_id).all()

    active_days = set()
    for (d,) in attempt_dates:
        if d:
            active_days.add(d.date())
    for (d,) in session_dates:
        if d:
            active_days.add(d.date())

    if not active_days:
        return 0

    today = datetime.utcnow().date()
    # Check if active today or yesterday
    check_date = today
    if check_date not in active_days:
        check_date = today - timedelta(days=1)
        if check_date not in active_days:
            return 0

    streak = 0
    while check_date in active_days:
        streak += 1
        check_date -= timedelta(days=1)

    return streak


# ----------------------------------------------------
# 1. Dashboard
# ----------------------------------------------------

@learning_bp.route("/dashboard")
@login_required
def dashboard():
    """Main SaaS-style personalized dashboard."""
    user = get_current_user()
    if not user:
        return redirect(url_for("auth_routes.login"))

    profile = user.profile or UserProfile(user_id=user.id)
    if not profile.onboarding_completed:
        return redirect(url_for("auth_routes.onboarding"))

    # Active learning path
    learning_path = get_or_create_learning_path(user.id)
    learning_path.calculate_progress()
    db.session.commit()

    # User quiz stats
    attempts = QuizAttempt.query.filter_by(user_id=user.id).order_by(QuizAttempt.submitted_at.desc()).all()
    quizzes_completed = len(attempts)
    avg_score = round(sum(a.percentage for a in attempts) / quizzes_completed, 1) if quizzes_completed > 0 else 0.0

    streak = calculate_user_streak(user.id)
    strong_skills, weak_skills = get_strong_and_weak_skills(user.id)

    # Next milestone
    upcoming_milestones = [m for m in learning_path.milestones if m.status in ("in_progress", "not_started")][:3]

    # Recommended next activity
    recommended_activity = None
    if weak_skills:
        weak_topic = weak_skills[0]["skill_name"]
        recommended_activity = {
            "type": "quiz",
            "title": f"Practice Quiz: {weak_topic}",
            "description": f"Targeted review to boost your {weak_topic} proficiency from {weak_skills[0]['estimated_proficiency']}%.",
            "action_url": url_for("learning_routes.quiz_generator", topic=weak_topic, difficulty="easy"),
            "action_label": "Start Practice Quiz"
        }
    elif upcoming_milestones:
        m = upcoming_milestones[0]
        recommended_activity = {
            "type": "milestone",
            "title": f"Next Milestone: {m.title}",
            "description": m.description,
            "action_url": url_for("learning_routes.learning_path_view"),
            "action_label": "Continue Milestone"
        }
    else:
        recommended_activity = {
            "type": "video",
            "title": "Analyze a New Learning Video",
            "description": "Upload a lecture or paste a YouTube URL to extract timestamped concepts and generate quizzes.",
            "action_url": url_for("learning_routes.learn_with_video"),
            "action_label": "Learn with Video"
        }

    # Top course recommendations
    course_recs = get_personalized_recommendations(user.id, limit=3)

    # Recent video sessions
    recent_videos = VideoLearningSession.query.filter_by(user_id=user.id).order_by(VideoLearningSession.created_at.desc()).limit(3).all()

    return render_template(
        "learning/dashboard.html",
        user=user,
        profile=profile,
        learning_path=learning_path,
        quizzes_completed=quizzes_completed,
        avg_score=avg_score,
        streak=streak,
        strong_skills=strong_skills[:3],
        weak_skills=weak_skills[:3],
        recommended_activity=recommended_activity,
        course_recs=course_recs,
        upcoming_milestones=upcoming_milestones,
        recent_videos=recent_videos,
        recent_attempts=attempts[:5],
    )


# ----------------------------------------------------
# 2. Learn with Video Interface
# ----------------------------------------------------

@learning_bp.route("/learn")
def learn_with_video():
    """Renders the video understanding & transcript interface."""
    return redirect(url_for("home"))


# ----------------------------------------------------
# 3. AI Quiz Generator & Quiz Taking
# ----------------------------------------------------

@learning_bp.route("/quiz/generator", methods=["GET", "POST"])
@login_required
def quiz_generator():
    """Generates a new quiz from a video session transcript or custom topic."""
    user = get_current_user()

    # Pre-populate video sessions for selection dropdown
    user_videos = VideoLearningSession.query.filter_by(user_id=user.id).order_by(VideoLearningSession.created_at.desc()).all()

    prefill_video_id = request.args.get("video_session_id", type=int)
    prefill_topic = request.args.get("topic", "").strip()
    prefill_diff = request.args.get("difficulty", "medium").strip()

    if request.method == "POST":
        is_json = request.is_json or request.headers.get("Accept") == "application/json"
        data = request.get_json(silent=True) if request.is_json else request.form

        title = data.get("title", "").strip()
        num_questions = int(data.get("num_questions", 5))
        difficulty = data.get("difficulty", "medium").strip()
        topic_focus = data.get("topic_focus", "").strip()
        video_session_id = data.get("video_session_id")

        start_sec = None
        end_sec = None
        scope = data.get("scope", "entire")
        if scope == "section":
            try:
                start_sec = float(data.get("start_sec", 0.0))
                end_sec = float(data.get("end_sec", 0.0))
            except (ValueError, TypeError):
                pass

        segments = None
        vid_session = None

        if video_session_id:
            try:
                vid_session = VideoLearningSession.query.filter_by(id=int(video_session_id), user_id=user.id).first()
                if vid_session:
                    segments = vid_session.get_segments()
                    if not title:
                        title = f"Quiz: {vid_session.title}"
            except Exception:
                pass

        # Also support active session segments from session store if unpersisted
        if not segments and session.get("current_video_id"):
            from app import VIDEO_STORE
            v_rec = VIDEO_STORE.get(session["current_video_id"])
            if v_rec:
                segments = v_rec.get("segments", [])
                if not title:
                    title = f"Quiz: {v_rec.get('title', 'Active Video')}"

        if not title:
            title = f"Quiz: {topic_focus or 'Knowledge Check'}"

        ok, msg, quiz = generate_and_save_quiz(
            user_id=user.id,
            title=title,
            video_session_id=vid_session.id if vid_session else None,
            segments=segments,
            num_questions=num_questions,
            difficulty=difficulty,
            topic_focus=topic_focus,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        if not ok:
            if is_json:
                return jsonify({"success": False, "error": msg}), 400
            flash(msg, "danger")
            return render_template(
                "learning/quiz_generator.html",
                user=user,
                user_videos=user_videos,
                error=msg,
                prefill_topic=topic_focus,
                csrf_token=generate_csrf_token()
            )

        if is_json:
            return jsonify({
                "success": True,
                "message": msg,
                "quiz_id": quiz.id,
                "redirect": url_for("learning_routes.take_quiz", quiz_id=quiz.id)
            })

        flash("Quiz generated successfully!", "success")
        return redirect(url_for("learning_routes.take_quiz", quiz_id=quiz.id))

    return render_template(
        "learning/quiz_generator.html",
        user=user,
        user_videos=user_videos,
        prefill_video_id=prefill_video_id,
        prefill_topic=prefill_topic,
        prefill_diff=prefill_diff,
        csrf_token=generate_csrf_token()
    )


@learning_bp.route("/quiz/<int:quiz_id>")
@login_required
def take_quiz(quiz_id: int):
    """Interactive quiz taking view. Does NOT leak correct answers to the browser."""
    user = get_current_user()
    quiz = Quiz.query.filter_by(id=quiz_id, user_id=user.id).first()
    if not quiz:
        abort(404, description="Quiz not found or unauthorized.")

    return render_template(
        "learning/quiz_take.html",
        user=user,
        quiz=quiz,
        questions=[q.to_dict(include_answer=False) for q in quiz.questions],
        csrf_token=generate_csrf_token()
    )


@learning_bp.route("/quiz/<int:quiz_id>/submit", methods=["POST"])
@login_required
def submit_quiz(quiz_id: int):
    """Authoritative server-side quiz submission scoring."""
    user = get_current_user()
    quiz = Quiz.query.filter_by(id=quiz_id, user_id=user.id).first()
    if not quiz:
        abort(404, description="Quiz not found or unauthorized.")

    is_json = request.is_json or request.headers.get("Accept") == "application/json"
    data = request.get_json(silent=True) if request.is_json else request.form

    answers_dict = {}
    if is_json:
        answers_dict = data.get("answers", {})
    else:
        for key, val in request.form.items():
            if key.startswith("q_"):
                q_id = key.replace("q_", "")
                answers_dict[q_id] = val

    ok, msg, attempt = score_quiz_submission(
        user_id=user.id,
        quiz_id=quiz.id,
        user_answers=answers_dict
    )

    if not ok:
        if is_json:
            return jsonify({"success": False, "error": msg}), 400
        flash(msg, "danger")
        return redirect(url_for("learning_routes.take_quiz", quiz_id=quiz.id))

    if is_json:
        return jsonify({
            "success": True,
            "message": msg,
            "attempt_id": attempt.id,
            "score": attempt.score,
            "percentage": attempt.percentage,
            "redirect": url_for("learning_routes.quiz_attempt_results", attempt_id=attempt.id)
        })

    return redirect(url_for("learning_routes.quiz_attempt_results", attempt_id=attempt.id))


@learning_bp.route("/quiz/attempt/<int:attempt_id>")
@login_required
def quiz_attempt_results(attempt_id: int):
    """Displays comprehensive results, explanations, timestamp links, and skill updates."""
    user = get_current_user()
    attempt = QuizAttempt.query.filter_by(id=attempt_id, user_id=user.id).first()
    if not attempt:
        abort(404, description="Quiz attempt not found or unauthorized.")

    quiz = attempt.quiz
    post_quiz_advice = generate_post_quiz_recommendations(user.id, attempt.id)

    # Fetch video session if linked
    video_session = None
    if quiz and quiz.video_session_id:
        video_session = VideoLearningSession.query.get(quiz.video_session_id)

    return render_template(
        "learning/quiz_results.html",
        user=user,
        quiz=quiz,
        attempt=attempt,
        responses=[r.to_dict() for r in attempt.responses],
        advice=post_quiz_advice,
        video_session=video_session,
    )


# ----------------------------------------------------
# 4. Skills Profile
# ----------------------------------------------------

@learning_bp.route("/skills")
@login_required
def skills_overview():
    """Displays estimated proficiencies, confidence levels, and trends across all topics."""
    user = get_current_user()
    all_skills = get_user_skills(user.id)
    strong_skills, weak_skills = get_strong_and_weak_skills(user.id)

    return render_template(
        "learning/skills.html",
        user=user,
        skills=all_skills,
        strong_skills=strong_skills,
        weak_skills=weak_skills,
    )


# ----------------------------------------------------
# 5. Course Recommendations & Catalogue
# ----------------------------------------------------

@learning_bp.route("/courses")
@login_required
def courses_view():
    """Personalized course recommendations and searchable course catalogue."""
    user = get_current_user()
    recommendations = get_personalized_recommendations(user.id, limit=6)

    # All active courses
    catalogue = Course.query.filter_by(is_active=True).all()
    user_courses = UserCourse.query.filter_by(user_id=user.id).all()
    status_by_course = {uc.course_id: uc.status for uc in user_courses}

    return render_template(
        "learning/courses.html",
        user=user,
        recommendations=recommendations,
        catalogue=[c.to_dict() for c in catalogue],
        status_by_course=status_by_course,
        csrf_token=generate_csrf_token()
    )


@learning_bp.route("/courses/<int:course_id>/status", methods=["POST"])
@login_required
def update_course_status(course_id: int):
    """Enrolls, completes, or dismisses a course for the user."""
    user = get_current_user()
    is_json = request.is_json or request.headers.get("Accept") == "application/json"
    data = request.get_json(silent=True) if request.is_json else request.form

    status = (data.get("status") or "enrolled").strip()
    ok, msg = save_user_course_status(user.id, course_id, status)

    if not ok:
        if is_json:
            return jsonify({"success": False, "error": msg}), 400
        flash(msg, "danger")
        return redirect(url_for("learning_routes.courses_view"))

    if is_json:
        return jsonify({"success": True, "message": msg, "course_id": course_id, "status": status})

    flash(msg, "success")
    return redirect(url_for("learning_routes.courses_view"))


# ----------------------------------------------------
# 6. Personalized Learning Path
# ----------------------------------------------------

@learning_bp.route("/learning-path")
@login_required
def learning_path_view():
    """Interactive learning path view with study scheduler and milestone tracking."""
    user = get_current_user()
    path = get_or_create_learning_path(user.id)
    path.calculate_progress()
    db.session.commit()

    schedule = calculate_study_schedule(user.id, path)

    return render_template(
        "learning/learning_path.html",
        user=user,
        path=path,
        schedule=schedule,
        csrf_token=generate_csrf_token()
    )


@learning_bp.route("/learning-path/milestone/<int:milestone_id>/status", methods=["POST"])
@login_required
def update_milestone_status(milestone_id: int):
    """Updates a milestone's status ('not_started', 'in_progress', 'completed', 'needs_review')."""
    user = get_current_user()
    milestone = LearningMilestone.query.get(milestone_id)
    if not milestone or milestone.learning_path.user_id != user.id:
        abort(404, description="Milestone not found or unauthorized.")

    is_json = request.is_json or request.headers.get("Accept") == "application/json"
    data = request.get_json(silent=True) if request.is_json else request.form

    new_status = data.get("status", "completed").strip().lower()
    if new_status in ("not_started", "in_progress", "completed", "needs_review"):
        milestone.status = new_status
        if new_status == "completed":
            milestone.completed_at = datetime.utcnow()
        milestone.learning_path.calculate_progress()
        db.session.commit()

        if is_json:
            return jsonify({
                "success": True,
                "milestone_id": milestone.id,
                "status": milestone.status,
                "overall_progress": milestone.learning_path.overall_progress
            })
        flash(f"Milestone '{milestone.title}' updated.", "success")
        return redirect(url_for("learning_routes.learning_path_view"))

    if is_json:
        return jsonify({"success": False, "error": "Invalid milestone status."}), 400
    flash("Invalid milestone status.", "danger")
    return redirect(url_for("learning_routes.learning_path_view"))


@learning_bp.route("/learning-path/reset", methods=["POST"])
@login_required
def reset_learning_path():
    """Regenerates a fresh learning path based on the user's latest profile."""
    user = get_current_user()
    profile = user.profile or UserProfile(user_id=user.id)

    path = create_learning_path_for_user(
        user_id=user.id,
        goal=profile.primary_goal or "Master Python",
        target_skill=profile.target_skill or "Python",
        starting_level=profile.get_skill_levels_dict().get(profile.target_skill, "beginner"),
        target_level="advanced"
    )

    flash("Your learning path has been refreshed based on your current profile.", "success")
    return redirect(url_for("learning_routes.learning_path_view"))


# ----------------------------------------------------
# 7. Learning History
# ----------------------------------------------------

@learning_bp.route("/history")
@login_required
def learning_history():
    """Chronological history of video sessions and quiz attempts."""
    user = get_current_user()
    videos = VideoLearningSession.query.filter_by(user_id=user.id).order_by(VideoLearningSession.created_at.desc()).all()
    attempts = QuizAttempt.query.filter_by(user_id=user.id).order_by(QuizAttempt.submitted_at.desc()).all()

    return render_template(
        "learning/history.html",
        user=user,
        videos=videos,
        attempts=attempts,
    )
