"""
routes/auth_routes.py
=====================
VIVA / ARCHITECTURAL OVERVIEW:
This module defines the Flask Blueprint for student identity, session management,
and learning onboarding.

Key Viva Flow Points:
1. Registration & Auto-Login:
   - When a student registers at `/register`, passwords are cryptographically salted & hashed.
   - The user is automatically logged into the session and redirected to `/onboarding`.
2. Onboarding Questionnaire:
   - Collects student academic context: education level, field of study, specific interests,
     target skill, and daily study minutes.
   - On submission, it automatically calls `create_learning_path_for_user()` to instantiate
     a personalized milestone roadmap before redirecting to the main dashboard.
3. Content Negotiation:
   - All routes support both traditional server-rendered HTML forms and modern JSON APIs
     (inspecting `request.is_json` or `Accept: application/json`).
4. Session Security:
   - Protected routes use the `@login_required` decorator.
   - Sensitive forms include CSRF tokens generated via `generate_csrf_token()`.
"""

import json
import logging
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify

from auth.auth_manager import (
    register_user,
    authenticate_user,
    login_required,
    get_current_user,
    generate_csrf_token,
    validate_csrf_token,
)
from models.database import db, User, UserProfile
from learning.path_generator import create_learning_path_for_user

logger = logging.getLogger("routes.auth")

auth_bp = Blueprint("auth_routes", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """
    Handles student registration.
    
    Viva Flow:
    - If already logged in, redirects immediately to `/dashboard`.
    - On POST:
      1. Extracts full_name, email, password, confirm_password from JSON or Form.
      2. Validates and hashes password using `register_user()`.
      3. Establishes user session keys (`user_id`, `user_name`, `user_email`).
      4. Redirects to `/auth/onboarding` to customize the learning experience.
    """
    if session.get("user_id"):
        return redirect(url_for("learning_routes.dashboard"))

    if request.method == "POST":
        is_json = request.is_json or request.headers.get("Accept") == "application/json"
        data = request.get_json(silent=True) if request.is_json else request.form

        full_name = data.get("full_name", "")
        email = data.get("email", "")
        password = data.get("password", "")
        confirm_password = data.get("confirm_password", "")

        ok, msg, user = register_user(full_name, email, password, confirm_password)
        if not ok:
            if is_json:
                return jsonify({"success": False, "error": msg}), 400
            return render_template("auth/register.html", error=msg, full_name=full_name, email=email)

        # Log in newly registered user immediately in the active session
        session["user_id"] = user.id
        session["user_name"] = user.full_name
        session["user_email"] = user.email

        if is_json:
            return jsonify({
                "success": True,
                "message": msg,
                "user": user.to_dict(),
                "redirect": url_for("auth_routes.onboarding")
            })

        flash("Account created successfully! Please complete your learning profile.", "success")
        return redirect(url_for("auth_routes.onboarding"))

    return render_template("auth/register.html", csrf_token=generate_csrf_token())


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    Handles student login and credential verification.
    
    Viva Flow:
    - If already logged in, redirects immediately to `/dashboard`.
    - On POST:
      1. Calls `authenticate_user(email, password)`.
      2. If valid, writes user ID to session and redirects to `next_url` (or `/dashboard`).
      3. If invalid, renders login page with an informative error message.
    """
    if session.get("user_id"):
        return redirect(url_for("learning_routes.dashboard"))

    next_url = request.args.get("next") or request.form.get("next") or url_for("learning_routes.dashboard")

    if request.method == "POST":
        is_json = request.is_json or request.headers.get("Accept") == "application/json"
        data = request.get_json(silent=True) if request.is_json else request.form

        email = data.get("email", "")
        password = data.get("password", "")

        ok, msg, user = authenticate_user(email, password)
        if not ok:
            if is_json:
                return jsonify({"success": False, "error": msg}), 401
            return render_template("auth/login.html", error=msg, email=email, next=next_url)

        if is_json:
            return jsonify({
                "success": True,
                "message": msg,
                "user": user.to_dict(),
                "redirect": next_url
            })

        flash(f"Welcome back, {user.full_name}!", "success")
        return redirect(next_url)

    return render_template("auth/login.html", next=next_url, csrf_token=generate_csrf_token())


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    """
    Terminates user session and redirects to the login screen.
    
    Viva Flow:
    - Calls `session.clear()` to invalidate the client cookie and clear in-memory credentials.
    - Flashes a logout notice and redirects to `/auth/login`.
    """
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("auth_routes.login"))


@auth_bp.route("/onboarding", methods=["GET", "POST"])
@login_required
def onboarding():
    """
    Multi-step onboarding wizard for personalized learning configuration.
    
    Viva Flow:
    - Captures academic background (education level, field of study, current semester).
    - Captures topic interests and initial self-assessed skill levels.
    - Captures target skill and daily study budget (minutes/day).
    - Automatically builds an initial personalized learning path using `create_learning_path_for_user()`.
    - Marks `profile.onboarding_completed = True` and redirects to `/dashboard`.
    """
    user = get_current_user()
    if not user:
        return redirect(url_for("auth_routes.login"))

    profile = user.profile or UserProfile(user_id=user.id)

    if request.method == "POST":
        is_json = request.is_json or request.headers.get("Accept") == "application/json"
        data = request.get_json(silent=True) if request.is_json else request.form

        # Step 1: Basic Academic Information
        profile.education_level = data.get("education_level", profile.education_level or "undergraduate")
        profile.field_of_study = data.get("field_of_study", profile.field_of_study or "").strip()
        profile.current_semester = data.get("current_semester", profile.current_semester or "").strip()

        # Step 2: Learning Interests & Self-Assessed Skill Levels
        if is_json:
            interests_list = data.get("interests", [])
            skill_levels_dict = data.get("subject_skill_levels", {})
        else:
            interests_list = request.form.getlist("interests")
            custom_interest = request.form.get("custom_interest", "").strip()
            if custom_interest and custom_interest not in interests_list:
                interests_list.append(custom_interest)

            skill_levels_dict = {}
            for item in interests_list:
                lvl = request.form.get(f"skill_level_{item}", "beginner")
                skill_levels_dict[item] = lvl

        profile.set_interests_list(interests_list)
        profile.set_skill_levels_dict(skill_levels_dict)

        # Step 3: Goals & Study Time Budget
        profile.primary_goal = data.get("primary_goal", profile.primary_goal or "Learn a new technology").strip()
        profile.target_skill = data.get("target_skill", profile.target_skill or "Python").strip()
        profile.target_completion_date = data.get("target_completion_date", profile.target_completion_date or "").strip()

        try:
            profile.daily_study_minutes = int(data.get("daily_study_minutes", 45))
        except (ValueError, TypeError):
            profile.daily_study_minutes = 45

        profile.weekly_study_hours = round((profile.daily_study_minutes * 7) / 60.0, 1)
        profile.preferred_difficulty = data.get("preferred_difficulty", "medium").strip().lower()
        profile.preferred_format = data.get("preferred_format", "mixture").strip()
        profile.career_goal = data.get("career_goal", "").strip()
        profile.onboarding_completed = True

        try:
            db.session.add(profile)
            db.session.commit()

            # Initialize personalized learning path tailored to onboarding responses
            create_learning_path_for_user(
                user_id=user.id,
                goal=profile.primary_goal,
                target_skill=profile.target_skill,
                starting_level=skill_levels_dict.get(profile.target_skill, "beginner"),
                target_level="advanced"
            )

            logger.info("Onboarding completed for user %d (%s)", user.id, user.email)

            if is_json:
                return jsonify({
                    "success": True,
                    "message": "Onboarding completed successfully.",
                    "redirect": url_for("learning_routes.dashboard")
                })

            flash("Profile setup complete! Welcome to your personalized dashboard.", "success")
            return redirect(url_for("learning_routes.dashboard"))

        except Exception as e:
            db.session.rollback()
            logger.exception("Failed to save onboarding profile: %s", e)
            if is_json:
                return jsonify({"success": False, "error": str(e)}), 500
            return render_template("learning/onboarding.html", user=user, profile=profile, error=str(e))

    return render_template(
        "learning/onboarding.html",
        user=user,
        profile=profile,
        interests=profile.get_interests_list(),
        skill_levels=profile.get_skill_levels_dict(),
        csrf_token=generate_csrf_token()
    )


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """
    Allows students to view and update their profile and learning preferences.
    
    Viva Flow:
    - Protected by `@login_required`.
    - Allows modifying study goals, target skill, daily minutes, and academic details.
    - Updates SQLite record and syncs `weekly_study_hours` accordingly.
    """
    user = get_current_user()
    profile = user.profile or UserProfile(user_id=user.id)

    if request.method == "POST":
        is_json = request.is_json or request.headers.get("Accept") == "application/json"
        data = request.get_json(silent=True) if request.is_json else request.form

        # Update full name
        new_name = data.get("full_name", "").strip()
        if new_name and len(new_name) >= 2:
            user.full_name = new_name

        profile.education_level = data.get("education_level", profile.education_level)
        profile.field_of_study = data.get("field_of_study", profile.field_of_study).strip()
        profile.current_semester = data.get("current_semester", profile.current_semester).strip()
        profile.primary_goal = data.get("primary_goal", profile.primary_goal).strip()
        profile.target_skill = data.get("target_skill", profile.target_skill).strip()
        profile.target_completion_date = data.get("target_completion_date", profile.target_completion_date).strip()

        try:
            profile.daily_study_minutes = int(data.get("daily_study_minutes", profile.daily_study_minutes))
            profile.weekly_study_hours = round((profile.daily_study_minutes * 7) / 60.0, 1)
        except (ValueError, TypeError):
            pass

        profile.preferred_difficulty = data.get("preferred_difficulty", profile.preferred_difficulty)
        profile.preferred_format = data.get("preferred_format", profile.preferred_format)
        profile.career_goal = data.get("career_goal", profile.career_goal).strip()

        try:
            db.session.commit()
            if is_json:
                return jsonify({"success": True, "message": "Profile updated successfully."})
            flash("Profile updated successfully.", "success")
            return redirect(url_for("auth_routes.profile"))
        except Exception as e:
            db.session.rollback()
            logger.exception("Failed to update profile: %s", e)
            if is_json:
                return jsonify({"success": False, "error": str(e)}), 500
            return render_template("learning/profile.html", user=user, profile=profile, error=str(e))

    return render_template(
        "learning/profile.html",
        user=user,
        profile=profile,
        interests=profile.get_interests_list(),
        csrf_token=generate_csrf_token()
    )
