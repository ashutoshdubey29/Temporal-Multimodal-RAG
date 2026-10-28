"""
auth/auth_manager.py
====================
VIVA / ARCHITECTURAL OVERVIEW:
This module manages all user authentication, authorization, session persistence,
and cybersecurity defenses for AetherLearn.

Key Concepts to Explain in a Viva:
1. Password Security (Scrypt / PBKDF2):
   - We NEVER store plain-text passwords in the database.
   - We use Werkzeug's `generate_password_hash()`, which automatically generates a
     unique random cryptographic salt per user and applies a modern key-derivation function
     (such as scrypt or PBKDF2-HMAC-SHA256).
   - This prevents dictionary attacks and rainbow table lookups even if the database is leaked.
2. Cross-Site Request Forgery (CSRF) Defense:
   - We generate random 24-byte hex tokens (`secrets.token_hex(24)`) stored in the session.
   - When submitting sensitive forms, the client sends this token back.
   - We compare them using `secrets.compare_digest()`, which prevents timing attacks by
     comparing characters in constant time.
3. Access Control Decorator (`@login_required`):
   - Uses Python decorators to inspect Flask's encrypted client-side session cookie.
   - Detects whether the request is from a browser navigation or an AJAX/fetch JSON call,
     redirecting browsers to the login page with a `next` query parameter, or returning
     a 401 Unauthorized JSON payload to single-page fetch requests.
4. User-Profile Lifecycle:
   - When a user registers, an associated `UserProfile` is created immediately in the same
     transaction using `db.session.flush()` to retrieve the auto-incremented primary key `user.id`.
"""

import re
import secrets
from functools import wraps
import logging
from flask import session, request, redirect, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from models.database import db, User, UserProfile

logger = logging.getLogger("auth.manager")

# Standard RFC-compliant email validation pattern
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_registration_input(full_name: str, email: str, password: str, confirm_password: str) -> tuple[bool, str]:
    """
    Validates user input during registration before touching the database.
    
    Checks enforced:
    1. Full Name: Must be at least 2 characters long.
    2. Email: Must match standard email regex format.
    3. Password: Must be at least 8 characters long, contain at least 1 letter, and 1 numeric digit.
    4. Password Confirmation: Must match the entered password identically.

    Returns:
        tuple[bool, str]: (is_valid, error_message_if_invalid)
    """
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
    """
    Registers a new student user in the database.
    
    Step-by-Step Viva Explanation:
    1. Input Validation: Calls `validate_registration_input()` to ensure syntax & security requirements.
    2. Duplicate Check: Queries `User` by lowercase email to prevent duplicate accounts.
    3. Password Hashing: Calls `generate_password_hash()` to turn the plain password into a salted hash.
    4. Flush ID: Calls `db.session.flush()` so the database generates `user.id` without committing yet.
    5. Linked Profile: Automatically instantiates a `UserProfile` for this user (with `onboarding_completed=False`).
    6. Commit / Rollback: Commits both records atomically. If an error occurs, rolls back cleanly.

    Returns:
        tuple[bool, str, User | None]: (success, message, created_user_or_none)
    """
    valid, err_msg = validate_registration_input(full_name, email, password, confirm_password)
    if not valid:
        return False, err_msg, None

    email = email.strip().lower()
    full_name = full_name.strip()

    # Check for existing account with the same email
    existing = User.query.filter_by(email=email).first()
    if existing:
        return False, "An account with this email address already exists.", None

    try:
        # Generate salted cryptographic hash of password
        pw_hash = generate_password_hash(password)
        user = User(
            full_name=full_name,
            email=email,
            password_hash=pw_hash,
            account_status="active",
        )
        db.session.add(user)
        db.session.flush()  # Generates user.id without committing the transaction yet

        # Create linked student profile initialized for the onboarding questionnaire
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
        db.session.rollback()  # Revert changes if database insertion fails
        logger.exception("Failed to register user: %s", e)
        return False, f"Registration failed: {str(e)}", None


def authenticate_user(email: str, password: str) -> tuple[bool, str, User | None]:
    """
    Verifies user credentials during login and establishes session state.
    
    Step-by-Step Viva Explanation:
    1. Looks up the user record by lowercase email.
    2. Calls `check_password_hash(user.password_hash, password)`.
       - Werkzeug extracts the salt and algorithm from the hash string and re-computes
         the hash to verify matching.
    3. Verifies account is active (`account_status == "active"`).
    4. Writes `user_id`, `user_name`, and `user_email` into Flask's cryptographically
       signed session cookie so subsequent requests identify the student.

    Returns:
        tuple[bool, str, User | None]: (success, message, user_instance_or_none)
    """
    email = (email or "").strip().lower()
    password = password or ""

    if not email or not password:
        return False, "Email and password are required.", None

    user = User.query.filter_by(email=email).first()
    if not user:
        return False, "Invalid email or password.", None

    # Verify password against stored hash using constant-time comparison
    if not check_password_hash(user.password_hash, password):
        return False, "Invalid email or password.", None

    if user.account_status != "active":
        return False, "This account is inactive. Please contact support.", None

    # Store user identity in the session cookie if running within an active Flask request context
    try:
        session["user_id"] = user.id
        session["user_name"] = user.full_name
        session["user_email"] = user.email
    except RuntimeError:
        pass

    logger.info("User authenticated: ID %s (%s)", user.id, user.email)
    return True, "Login successful.", user


def get_current_user() -> User | None:
    """
    Retrieves the currently authenticated User ORM model from the active session.
    
    Viva Explanation:
    - Reads `session.get("user_id")` from the encrypted cookie.
    - If found, fetches the `User` from SQLite using `db.session.get(User, user_id)`.
    - Returns None if not logged in or if the user no longer exists in the DB.
    """
    try:
        user_id = session.get("user_id")
    except RuntimeError:
        user_id = None
    if not user_id:
        return None
    try:
        return db.session.get(User, user_id)
    except Exception:
        return None


def login_required(f):
    """
    Route decorator ensuring only authenticated users can access the endpoint.
    
    Viva Explanation:
    - Wraps any route function using `@wraps(f)`.
    - Checks `get_current_user()`.
    - If unauthenticated:
      - Clears any stale session keys to prevent invalid state.
      - If the client sent a JSON request (e.g., fetch API), returns HTTP 401 with JSON error.
      - If regular browser navigation, redirects to `/auth/login?next=<current_path>` so the
        student is redirected back to their intended page after logging in.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            # Clear invalid/stale session credentials to prevent 500 errors on subsequent requests
            session.pop("user_id", None)
            session.pop("user_name", None)
            session.pop("user_email", None)
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
    """
    Generates a cryptographically strong, random CSRF token stored in the session.
    
    Viva Explanation:
    - Uses Python's `secrets` module (specifically designed for cryptographic security).
    - If a token already exists in the session, reuses it; otherwise creates a 48-character
      hex string (from 24 random bytes).
    """
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(24)
    return session["csrf_token"]


def validate_csrf_token(token: str) -> bool:
    """
    Validates a submitted CSRF token against the session-stored token.
    
    Viva Explanation:
    - Uses `secrets.compare_digest()` to compare strings in constant time.
    - Constant-time comparison prevents timing attacks, where an attacker measures
      microsecond differences in string matching to guess valid tokens.
    """
    expected = session.get("csrf_token")
    if not expected or not token:
        return False
    return secrets.compare_digest(expected, token)
