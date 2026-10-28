"""
tests/test_learning_platform.py

Comprehensive test suite for the AI Personalized Learning Platform:
  1. Authentication, validation, and session security
  2. Automatic MCQ quiz generation & authoritative scoring
  3. Skill tracking & confidence calculation
  4. Course recommendations & explainable weighted ranking
  5. Personalized learning paths & adaptive milestones
  6. Multi-user data isolation
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os

# Ensure test suite runs in-memory and never alters the persistent development database
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app import app
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
from auth.auth_manager import (
    register_user,
    authenticate_user,
    validate_registration_input,
)
from learning.quiz_generator import (
    generate_and_save_quiz,
    score_quiz_submission,
    validate_quiz_questions,
    validate_single_question,
)
from learning.skill_tracker import (
    calculate_confidence_score,
    update_skills_from_attempt,
    get_strong_and_weak_skills,
)
from learning.recommendation_engine import (
    get_personalized_recommendations,
    calculate_course_match_score,
    save_user_course_status,
)
from learning.path_generator import (
    create_learning_path_for_user,
    adapt_learning_path_after_quiz,
    calculate_study_schedule,
)


class TestLearningPlatform(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        app.config["WTF_CSRF_ENABLED"] = False
        self.app = app
        self.client = app.test_client()
        self.ctx = app.app_context()
        self.ctx.push()
        db.create_all()

        # Seed sample curated courses for recommendation tests
        from models.seed_data import seed_courses_if_needed
        seed_courses_if_needed()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    # ====================================================
    # 1. AUTHENTICATION & VALIDATION TESTS
    # ====================================================

    def test_registration_validation(self):
        # Invalid email
        ok, msg = validate_registration_input("Alice", "not-an-email", "Pass1234", "Pass1234")
        self.assertFalse(ok)
        self.assertIn("valid email", msg.lower())

        # Short password
        ok, msg = validate_registration_input("Alice", "alice@example.com", "short1", "short1")
        self.assertFalse(ok)
        self.assertIn("8 characters", msg.lower())

        # Password without number
        ok, msg = validate_registration_input("Alice", "alice@example.com", "PasswordOnly", "PasswordOnly")
        self.assertFalse(ok)
        self.assertIn("numeric digit", msg.lower())

        # Password mismatch
        ok, msg = validate_registration_input("Alice", "alice@example.com", "Password123", "Mismatch123")
        self.assertFalse(ok)
        self.assertIn("do not match", msg.lower())

        # Valid input
        ok, msg = validate_registration_input("Alice Smith", "alice@example.com", "Password123", "Password123")
        self.assertTrue(ok)

    def test_user_registration_and_duplicate_rejection(self):
        ok, msg, user = register_user("Bob Jones", "bob@example.com", "SecurePass1", "SecurePass1")
        self.assertTrue(ok)
        self.assertIsNotNone(user)
        self.assertNotEqual(user.password_hash, "SecurePass1")  # Password hashed

        # Duplicate registration
        ok_dup, msg_dup, user_dup = register_user("Bob Duplicate", "bob@example.com", "SecurePass1", "SecurePass1")
        self.assertFalse(ok_dup)
        self.assertIn("already exists", msg_dup.lower())
        self.assertNil(user_dup) if hasattr(self, 'assertNil') else self.assertIsNone(user_dup)

    def test_user_authentication_flow(self):
        register_user("Charlie Brown", "charlie@example.com", "Secret999", "Secret999")

        # Wrong password
        ok, msg, user = authenticate_user("charlie@example.com", "WrongPassword")
        self.assertFalse(ok)
        self.assertIsNone(user)

        # Correct password
        ok, msg, user = authenticate_user("charlie@example.com", "Secret999")
        self.assertTrue(ok)
        self.assertEqual(user.email, "charlie@example.com")

    def test_protected_routes_require_login(self):
        # Accessing dashboard without session redirects to login
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers.get("Location", ""))

        # JSON request returns 401
        json_res = self.client.get("/dashboard", headers={"Accept": "application/json"})
        self.assertEqual(json_res.status_code, 401)

    # ====================================================
    # 2. QUIZ GENERATION & VALIDATION TESTS
    # ====================================================

    def test_question_schema_validation(self):
        valid_q = {
            "question": "What is the primary function of a CPU cache?",
            "options": {"A": "Slow storage", "B": "High speed buffer", "C": "Display output", "D": "Power supply"},
            "correct_option": "B",
            "explanation": "Cache provides low-latency memory close to execution cores.",
            "topic": "Computer Architecture",
            "difficulty": "medium",
            "source_start": 10.0,
            "source_end": 20.0,
        }
        ok, err = validate_single_question(valid_q)
        self.assertTrue(ok)

        # Invalid option key
        bad_q = valid_q.copy()
        bad_q["options"] = {"1": "A", "2": "B", "3": "C", "4": "D"}
        ok, err = validate_single_question(bad_q)
        self.assertFalse(ok)

        # Duplicate options
        dup_opt_q = valid_q.copy()
        dup_opt_q["options"] = {"A": "Same", "B": "Same", "C": "Diff", "D": "Other"}
        ok, err = validate_single_question(dup_opt_q)
        self.assertFalse(ok)

    def test_generate_quiz_from_transcript_grounded_fallback(self):
        ok, _, user = register_user("David Lee", "david@example.com", "Pass12345", "Pass12345")
        
        sample_segments = [
            {"start": 10.0, "end": 20.0, "text": "A binary tree is a hierarchical data structure with at most two children per node."},
            {"start": 21.0, "end": 35.0, "text": "A queue is used for breadth first search and follows first in first out order."},
            {"start": 36.0, "end": 50.0, "text": "Quick sort is an efficient sorting algorithm with average O(n log n) runtime."},
            {"start": 51.0, "end": 65.0, "text": "A stack is a linear container where elements are added and removed in LIFO order."},
            {"start": 66.0, "end": 80.0, "text": "A hash table provides expected O(1) time complexity for key value lookups."},
        ]

        ok, msg, quiz = generate_and_save_quiz(
            user_id=user.id,
            title="Data Structures Video Quiz",
            segments=sample_segments,
            num_questions=5,
            difficulty="medium"
        )
        self.assertTrue(ok)
        self.assertIsNotNone(quiz)
        self.assertEqual(len(quiz.questions), 5)

        # Check question structure and answer confidentiality before submission
        q0 = quiz.questions[0]
        q_dict_no_ans = q0.to_dict(include_answer=False)
        self.assertNotIn("correct_option", q_dict_no_ans)
        self.assertNotIn("explanation", q_dict_no_ans)

        # Check server data has answer and options
        self.assertIn(q0.correct_option, ["A", "B", "C", "D"])
        self.assertEqual(len(q0.get_options()), 4)

    # ====================================================
    # 3. AUTHORITATIVE QUIZ SCORING & ATTEMPTS
    # ====================================================

    def test_authoritative_scoring_and_persistence(self):
        ok, _, user = register_user("Elena Rostova", "elena@example.com", "Pass12345", "Pass12345")
        
        # Create a mock quiz with 2 questions
        quiz = Quiz(user_id=user.id, title="Algorithm Check", question_count=2)
        db.session.add(quiz)
        db.session.flush()

        q1 = QuizQuestion(
            quiz_id=quiz.id,
            question_text="What is time complexity of merge sort?",
            options_json=json.dumps({"A": "O(1)", "B": "O(n log n)", "C": "O(n^2)", "D": "O(2^n)"}),
            correct_option="B",
            explanation="Merge sort runs in O(n log n) in all cases.",
            topic="Sorting Algorithms",
            difficulty="medium",
        )
        q2 = QuizQuestion(
            quiz_id=quiz.id,
            question_text="Which algorithm searches a sorted array in O(log n)?",
            options_json=json.dumps({"A": "Linear Search", "B": "Bubble Search", "C": "Binary Search", "D": "Depth Search"}),
            correct_option="C",
            explanation="Binary search divides sorted search space in half.",
            topic="Searching Algorithms",
            difficulty="easy",
        )
        db.session.add_all([q1, q2])
        db.session.commit()

        # Submit: q1 correct ('B'), q2 incorrect ('A')
        user_answers = {str(q1.id): "B", str(q2.id): "A"}
        ok, msg, attempt = score_quiz_submission(user.id, quiz.id, user_answers)
        self.assertTrue(ok)
        self.assertEqual(attempt.correct_count, 1)
        self.assertEqual(attempt.incorrect_count, 1)
        self.assertEqual(attempt.unanswered_count, 0)
        self.assertEqual(attempt.percentage, 50.0)

        # Check response records
        self.assertEqual(len(attempt.responses), 2)
        r1 = [r for r in attempt.responses if r.question_id == q1.id][0]
        r2 = [r for r in attempt.responses if r.question_id == q2.id][0]
        self.assertTrue(r1.is_correct)
        self.assertFalse(r2.is_correct)

    # ====================================================
    # 4. SKILL TRACKING & CONFIDENCE RATINGS
    # ====================================================

    def test_skill_confidence_and_tracking(self):
        # Confidence score progression
        self.assertLess(calculate_confidence_score(2), 0.5)   # Low
        self.assertGreaterEqual(calculate_confidence_score(8), 0.5)  # Medium
        self.assertGreaterEqual(calculate_confidence_score(20), 0.8) # High

        ok, _, user = register_user("Frank Miller", "frank@example.com", "Pass12345", "Pass12345")
        quiz = Quiz(user_id=user.id, title="Python Diagnostic", question_count=1)
        db.session.add(quiz)
        db.session.flush()

        q = QuizQuestion(
            quiz_id=quiz.id,
            question_text="What does len() return?",
            options_json=json.dumps({"A": "type", "B": "length", "C": "sum", "D": "product"}),
            correct_option="B",
            topic="Python Fundamentals",
        )
        db.session.add(q)
        db.session.commit()

        # Correct answer
        score_quiz_submission(user.id, quiz.id, {str(q.id): "B"})

        skill = UserSkill.query.filter_by(user_id=user.id, skill_name="Python Fundamentals").first()
        self.assertIsNotNone(skill)
        self.assertEqual(skill.correct_count, 1)
        self.assertEqual(skill.attempted_count, 1)
        self.assertEqual(skill.estimated_proficiency, 100.0)
        self.assertEqual(skill.confidence_level(), "low")  # Only 1 question

    # ====================================================
    # 5. PERSONALIZED COURSE RECOMMENDATIONS
    # ====================================================

    def test_course_recommendations_weighted_ranking(self):
        ok, _, user = register_user("Grace Hopper", "grace@example.com", "Pass12345", "Pass12345")
        profile = user.profile
        profile.target_skill = "Python"
        profile.primary_goal = "Build practical projects"
        profile.weekly_study_hours = 6.0
        profile.preferred_difficulty = "beginner"
        db.session.commit()

        # Simulate weak skill in Python Functions
        weak_skill = UserSkill(
            user_id=user.id,
            skill_name="Python",
            estimated_proficiency=40.0,
            attempted_count=6,
            correct_count=2,
        )
        db.session.add(weak_skill)
        db.session.commit()

        recs = get_personalized_recommendations(user.id, limit=3)
        self.assertGreater(len(recs), 0)
        
        top_rec = recs[0]
        self.assertIn("Python", top_rec["course"]["title"] + " " + " ".join(top_rec["course"]["skills_covered"]))
        self.assertGreater(top_rec["relevance_percent"], 50)
        self.assertGreater(len(top_rec["reasons"]), 0)

        # Enrolling and dismissing
        course_id = top_rec["course"]["id"]
        ok, _ = save_user_course_status(user.id, course_id, "dismissed")
        self.assertTrue(ok)

        # Dismissed course must no longer appear in active recommendations
        new_recs = get_personalized_recommendations(user.id, limit=3)
        rec_ids = [r["course"]["id"] for r in new_recs]
        self.assertNotIn(course_id, rec_ids)

    # ====================================================
    # 6. ADAPTIVE LEARNING PATH
    # ====================================================

    def test_learning_path_creation_and_adaptive_update(self):
        ok, _, user = register_user("Henry Ford", "henry@example.com", "Pass12345", "Pass12345")
        
        path = create_learning_path_for_user(
            user_id=user.id,
            goal="Master Python Programming",
            target_skill="Python",
            starting_level="beginner"
        )
        self.assertIsNotNone(path)
        self.assertGreater(len(path.milestones), 5)
        self.assertEqual(path.milestones[0].status, "in_progress")

        # Test schedule calculation
        schedule = calculate_study_schedule(user.id, path)
        self.assertIn("remaining_estimated_hours", schedule)
        self.assertGreater(schedule["remaining_estimated_hours"], 0)

        # Simulate poor quiz attempt on milestone 1 skill
        quiz = Quiz(user_id=user.id, title="Python Functions Quiz", question_count=1)
        db.session.add(quiz)
        db.session.flush()

        q = QuizQuestion(
            quiz_id=quiz.id,
            question_text="What keyword defines a function?",
            options_json=json.dumps({"A": "def", "B": "func", "C": "fn", "D": "method"}),
            correct_option="A",
            topic="Python Functions",
        )
        db.session.add(q)
        db.session.commit()

        # Submit incorrect answer (0%)
        _, _, attempt = score_quiz_submission(user.id, quiz.id, {str(q.id): "B"})
        
        # Verify adaptation triggered
        adapt_learning_path_after_quiz(user.id, attempt.id)
        
        # Check that Functions milestone is flagged for review
        func_milestone = [m for m in path.milestones if "Function" in m.title][0]
        self.assertEqual(func_milestone.status, "needs_review")

    # ====================================================
    # 7. MULTI-USER DATA ISOLATION
    # ====================================================

    def test_multi_user_isolation(self):
        # User 1
        _, _, user1 = register_user("User One", "user1@example.com", "Pass12345", "Pass12345")
        quiz1 = Quiz(user_id=user1.id, title="User1 Secret Quiz", question_count=1)
        db.session.add(quiz1)
        db.session.commit()

        # User 2
        _, _, user2 = register_user("User Two", "user2@example.com", "Pass12345", "Pass12345")

        # Log in as User 2
        with self.client.session_transaction() as sess:
            sess["user_id"] = user2.id
            sess["user_name"] = user2.full_name
            sess["user_email"] = user2.email

        # User 2 tries to access User 1's quiz
        resp = self.client.get(f"/quiz/{quiz1.id}")
        self.assertEqual(resp.status_code, 404)  # Unauthorized / Not Found for other user

    # ====================================================
    # 8. QUIZ QUESTIONS & OPTIONS BLANK BUG REGRESSION TESTS
    # ====================================================

    def test_quiz_question_canonical_format_and_options_visibility(self):
        """Verifies QuizQuestion.to_dict() provides question_text, question, canonical options, and option_a-d."""
        ok, _, user = register_user("Alice Smith", "alice_quiz@example.com", "Pass12345", "Pass12345")
        quiz = Quiz(user_id=user.id, title="Data Architecture", question_count=1)
        db.session.add(quiz)
        db.session.flush()

        qq = QuizQuestion(
            quiz_id=quiz.id,
            question_text="What is the primary function of an inverted index?",
            options_json=json.dumps({
                "A": "Store documents sequentially on magnetic disk",
                "B": "Map terms to the locations where they occur",
                "C": "Compress images using discrete wavelet transforms",
                "D": "Authenticate network sockets with public keys"
            }),
            correct_option="B",
            explanation="An inverted index maps distinct terms to their containing documents.",
            topic="Information Retrieval",
            difficulty="medium",
            source_start=45.0,
            source_end=60.0
        )
        db.session.add(qq)
        db.session.commit()

        d = qq.to_dict(include_answer=False)
        # Verify question text is populated
        self.assertEqual(d["question"], "What is the primary function of an inverted index?")
        self.assertEqual(d["question_text"], "What is the primary function of an inverted index?")

        # Verify options list contains 4 objects with value and label
        self.assertEqual(len(d["options"]), 4)
        opt_values = [opt["value"] for opt in d["options"]]
        self.assertEqual(opt_values, ["A", "B", "C", "D"])
        opt_labels = [opt["label"] for opt in d["options"]]
        self.assertIn("Map terms to the locations where they occur", opt_labels)

        # Verify options list also supports dict .get() and ['A']
        self.assertEqual(d["options"].get("B"), "Map terms to the locations where they occur")
        self.assertEqual(d["options"]["B"], "Map terms to the locations where they occur")

        # Verify individual option fields
        self.assertEqual(d["option_a"], "Store documents sequentially on magnetic disk")
        self.assertEqual(d["option_b"], "Map terms to the locations where they occur")
        self.assertEqual(d["option_c"], "Compress images using discrete wavelet transforms")
        self.assertEqual(d["option_d"], "Authenticate network sockets with public keys")

    def test_quiz_take_template_renders_question_and_all_options(self):
        """Verifies quiz_take.html renders the full question text and four complete option labels."""
        ok, _, user = register_user("Bob Jones", "bob_quiz@example.com", "Pass12345", "Pass12345")
        quiz = Quiz(user_id=user.id, title="Network Protocols", question_count=1)
        db.session.add(quiz)
        db.session.flush()

        qq = QuizQuestion(
            quiz_id=quiz.id,
            question_text="Which transport protocol provides reliable byte stream transmission?",
            options_json=json.dumps({
                "A": "User Datagram Protocol (UDP)",
                "B": "Transmission Control Protocol (TCP)",
                "C": "Internet Control Message Protocol (ICMP)",
                "D": "Border Gateway Protocol (BGP)"
            }),
            correct_option="B",
            topic="Networking",
            difficulty="easy"
        )
        db.session.add(qq)
        db.session.commit()

        with self.client.session_transaction() as sess:
            sess["user_id"] = user.id
            sess["user_name"] = user.full_name
            sess["user_email"] = user.email

        resp = self.client.get(f"/quiz/{quiz.id}")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # Question text must be present in HTML
        self.assertIn("Which transport protocol provides reliable byte stream transmission?", html)
        # All 4 option labels must be present in HTML
        self.assertIn("User Datagram Protocol (UDP)", html)
        self.assertIn("Transmission Control Protocol (TCP)", html)
        self.assertIn("Internet Control Message Protocol (ICMP)", html)
        self.assertIn("Border Gateway Protocol (BGP)", html)
        # Option indicators A, B, C, D must be present
        self.assertIn("opt_label_", html)

    def test_object_and_array_form_options_normalized(self):
        """Verifies both dict-form and array-form options are normalized into canonical structure."""
        from learning.quiz_generator import clean_single_question

        # Array of objects format
        array_q = {
            "question": "What is the capital of France?",
            "options": [
                {"value": "A", "label": "Berlin"},
                {"value": "B", "label": "Madrid"},
                {"value": "C", "label": "Paris"},
                {"value": "D", "label": "Rome"}
            ],
            "correct_option": "C",
            "explanation": "Paris is the capital of France.",
            "topic": "Geography"
        }
        ok, err, cleaned = clean_single_question(array_q)
        self.assertTrue(ok)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned["options"]["C"], "Paris")
        self.assertEqual(cleaned["correct_option"], "C")

        # Object format
        obj_q = {
            "question": "What is the capital of Germany?",
            "options": {"A": "Berlin", "B": "Madrid", "C": "Paris", "D": "Rome"},
            "correct_option": "A",
            "explanation": "Berlin is the capital of Germany.",
            "topic": "Geography"
        }
        ok, err, cleaned = clean_single_question(obj_q)
        self.assertTrue(ok)
        self.assertEqual(cleaned["options"]["A"], "Berlin")

    def test_rejection_of_invalid_questions(self):
        """Verifies empty question text, empty options, and invalid answer keys are strictly rejected."""
        from learning.quiz_generator import validate_single_question

        # Empty question text
        bad_text = {
            "question": "",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "correct_option": "A"
        }
        ok, _ = validate_single_question(bad_text)
        self.assertFalse(ok)

        # Empty option label
        bad_opt = {
            "question": "Valid question prompt text here?",
            "options": {"A": "One", "B": "", "C": "Three", "D": "Four"},
            "correct_option": "A"
        }
        ok, _ = validate_single_question(bad_opt)
        self.assertFalse(ok)

        # Invalid correct_option (not in A, B, C, D)
        bad_key = {
            "question": "Valid question prompt text here?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "correct_option": "Z"
        }
        ok, _ = validate_single_question(bad_key)
        self.assertFalse(ok)

    def test_extract_json_payload_resilience(self):
        """Verifies extract_json_payload parses markdown code fences and conversational wrappers."""
        from learning.quiz_generator import extract_json_payload

        raw_fenced = '```json\n[{"question": "What is 2+2?", "options": {"A":"1","B":"2","C":"3","D":"4"}, "correct_option":"D"}]\n```'
        parsed = extract_json_payload(raw_fenced)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 1)

        raw_conversational = 'Here are your requested quiz questions:\n```\n[{"question": "Q1", "options": {"A":"1","B":"2","C":"3","D":"4"}, "correct_option":"A"}]\n```\nHope this helps!'
        parsed = extract_json_payload(raw_conversational)
        self.assertIsInstance(parsed, list)
        self.assertEqual(parsed[0]["question"], "Q1")

    def test_quiz_persistence_and_reload_identity(self):
        """Verifies that a valid quiz persists and reloads with identical content."""
        ok, _, user = register_user("Persistence User", "persist@example.com", "Pass12345", "Pass12345")
        quiz = Quiz(user_id=user.id, title="Persistence Check", question_count=1)
        db.session.add(quiz)
        db.session.flush()

        original_text = "What is the primary role of the CPU program counter?"
        original_opts = {"A": "Hold address of next instruction", "B": "Store ALU results", "C": "Manage page faults", "D": "Buffer disk blocks"}
        qq = QuizQuestion(
            quiz_id=quiz.id,
            question_text=original_text,
            options_json=json.dumps(original_opts),
            correct_option="A",
            explanation="The program counter stores the memory address of the next instruction.",
            topic="Computer Architecture",
            difficulty="medium",
            source_start=15.5,
            source_end=22.0
        )
        db.session.add(qq)
        db.session.commit()

        # Reload from database in fresh query
        reloaded_quiz = db.session.get(Quiz, quiz.id)
        self.assertIsNotNone(reloaded_quiz)
        self.assertEqual(len(reloaded_quiz.questions), 1)
        rq = reloaded_quiz.questions[0]
        self.assertEqual(rq.question_text, original_text)
        self.assertEqual(rq.get_options(), original_opts)
        self.assertEqual(rq.correct_option, "A")
        self.assertEqual(rq.source_start, 15.5)

    def test_quiz_json_api_response(self):
        """Verifies /quiz/<id> returns canonical quiz JSON structure when requested with Accept: application/json."""
        ok, _, user = register_user("API User", "api_quiz@example.com", "Pass12345", "Pass12345")
        quiz = Quiz(user_id=user.id, title="API Architecture", question_count=1)
        db.session.add(quiz)
        db.session.flush()

        qq = QuizQuestion(
            quiz_id=quiz.id,
            question_text="What HTTP status code represents resource created?",
            options_json=json.dumps({"A": "200", "B": "201", "C": "204", "D": "301"}),
            correct_option="B",
            topic="Web APIs"
        )
        db.session.add(qq)
        db.session.commit()

        with self.client.session_transaction() as sess:
            sess["user_id"] = user.id
            sess["user_name"] = user.full_name
            sess["user_email"] = user.email

        resp = self.client.get(f"/quiz/{quiz.id}", headers={"Accept": "application/json"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        quiz_data = data.get("quiz", {})
        self.assertEqual(quiz_data.get("title"), "API Architecture")
        questions = quiz_data.get("questions", [])
        self.assertEqual(len(questions), 1)
        q0 = questions[0]
        self.assertEqual(q0.get("question"), "What HTTP status code represents resource created?")
        self.assertEqual(q0.get("question_text"), "What HTTP status code represents resource created?")
        # Options must be canonical list of 4 items
        self.assertEqual(len(q0.get("options")), 4)
        self.assertEqual(q0["options"][0]["value"], "A")
        # Correct answer must NOT be exposed before submission
        self.assertNotIn("correct_option", q0)


if __name__ == "__main__":
    unittest.main()
