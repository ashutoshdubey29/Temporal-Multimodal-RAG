"""
auth/auth_manager.py

Authentication, password hashing, session management, and authorization guards.
"""

import re
import secrets
from functools import wraps
import logging
from flask import session, request, redirect, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from models.database import db, User, UserProfile

logger = logging.getLogger("auth.manager")

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_registration_input(full_name: str, email: str, password: str, confirm_password: str) -> tuple[bool, str]:
    """Validates registration form fields with strict constraints."""
    full_name = (full_name or "").strip()
    email = (email or "").strip().lower()

    if not full_name or len(full_name) < 2:
        return False, "Full name must be at least 2 characters long."

    if not email or not EMAIL_REGEX.match(email):
        return False, "Please provide a valid email address (e.g., student@example.com)."

    if not password:
        return False, "Password cannot be empty."

    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one numeric digit."

    if not any(char.isalpha() for char in password):
        return False, "Password must contain at least one letter."

    if password != confirm_password:
        return False, "Passwords do not match."

    return True, ""


def register_user(full_name: str, email: str, password: str, confirm_password: str) -> tuple[bool, str, User | None]:
    """Registers a new user, hashes password, and creates an initial profile."""
    valid, err_msg = validate_registration_input(full_name, email, password, confirm_password)
    if not valid:
        return False, err_msg, None

    email = email.strip().lower()
    full_name = full_name.strip()

    # Check duplicate email
    existing = User.query.filter_by(email=email).first()
    if existing:
        return False, "An account with this email address already exists.", None

    try:
        pw_hash = generate_password_hash(password, method="scrypt")
        user = User(
            full_name=full_name,
            email=email,
            password_hash=pw_hash,
            account_status="active",
        )
        db.session.add(user)
        db.session.flush()  # Generates user.id

        # Create linked profile
        profile = UserProfile(
            user_id=user.id,
            education_level="undergraduate",
            field_of_study="",
            primary_goal="Learn a new technology",
            target_skill="Python",
            onboarding_completed=False,
        )
        db.session.add(profile)
        db.session.commit()

        logger.info("Successfully registered user %s (%s)", user.id, user.email)
        return True, "Registration successful.", user

    except Exception as e:
        db.session.rollback()
        logger.exception("Failed to register user: %s", e)
        return False, "Registration failed due to a system error. Please try again.", None


def authenticate_user(email: str, password: str) -> tuple[bool, str, User | None]:
    """Validates user credentials and returns the authenticated User instance."""
    email = (email or "").strip().lower()
    password = password or ""

    if not email or not password:
        return False, "Email and password are required.", None

    user = User.query.filter_by(email=email).first()
    if not user:
        return False, "Invalid email or password.", None

    if not check_password_hash(user.password_hash, password):
        return False, "Invalid email or password.", None

    if user.account_status != "active":
        return False, "This account is inactive. Please contact support.", None

    # Set session state if inside active request context
    try:
        session["user_id"] = user.id
        session["user_name"] = user.full_name
        session["user_email"] = user.email
    except RuntimeError:
        pass

    logger.info("User authenticated: ID %s (%s)", user.id, user.email)
    return True, "Login successful.", user


def get_current_user() -> User | None:
    """Returns the current logged-in User instance from DB, or None."""
    try:
        user_id = session.get("user_id")
    except RuntimeError:
        user_id = None
    if not user_id:
        return None
    return User.query.get(user_id)


def login_required(f):
    """Decorator ensuring that only authenticated users can access the endpoint."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id"):
            is_json = request.is_json or request.headers.get("Accept") == "application/json"
            if is_json:
                return jsonify({
                    "success": False,
                    "error": "Authentication required. Please log in.",
                    "error_code": "AUTH_REQUIRED"
                }), 401
            return redirect(url_for("auth_routes.login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def generate_csrf_token() -> str:
    """Generates and persists a CSRF token in the user's session."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(24)
    return session["csrf_token"]


def validate_csrf_token(token: str) -> bool:
    """Validates a provided CSRF token against the user's session token."""
    expected = session.get("csrf_token")
    if not expected or not token:
        return False
    return secrets.compare_digest(expected, token)
