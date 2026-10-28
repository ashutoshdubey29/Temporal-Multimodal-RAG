"""
routes/__init__.py
==================
VIVA / ARCHITECTURAL OVERVIEW:
This package exposes the modular Flask Blueprints that structure the web endpoints:
1. `auth_bp` (`routes/auth_routes.py`): Registration, login, logout, onboarding wizard, and student profile.
2. `learning_bp` (`routes/learning_routes.py`): Dashboard, quiz generation/taking/scoring, skill visualization,
   course catalog/recommendations, and dynamic learning paths.

Blueprints allow modular route organization, separating URL routing and template rendering
from core business logic.
"""
from routes.auth_routes import auth_bp
from routes.learning_routes import learning_bp
