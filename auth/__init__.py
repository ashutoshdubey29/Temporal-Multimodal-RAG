"""
auth/__init__.py
================
VIVA / ARCHITECTURAL OVERVIEW:
This package initializer exports the core authentication and authorization utilities
used across AetherLearn routes.

Key Components Exported:
- `register_user`: Creates a user with hashed passwords and initializes their profile.
- `authenticate_user`: Validates credentials against stored cryptographic hashes.
- `login_required`: Route decorator protecting sensitive student endpoints.
- `get_current_user`: Resolves the active logged-in User ORM object from session state.
- `validate_registration_input`: Enforces security constraints on registration inputs.
- `generate_csrf_token` / `validate_csrf_token`: Prevents Cross-Site Request Forgery.
"""
from auth.auth_manager import (
    register_user,
    authenticate_user,
    login_required,
    get_current_user,
    validate_registration_input,
    generate_csrf_token,
    validate_csrf_token,
)
