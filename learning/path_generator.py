"""
learning/path_generator.py
==========================
VIVA / ARCHITECTURAL OVERVIEW:
This module manages structured, goal-oriented student learning paths and implements
dynamic path adaptation based on empirical quiz outcomes.

Key Concepts to Explain in a Viva:
1. Goal-Driven Curriculum Generation:
   - When a student completes onboarding, their goal and target skill (e.g. Python, Web Dev,
     DSA, Machine Learning) are matched against curated milestone templates (`GOAL_TEMPLATES`).
   - A sequential `LearningPath` with ordered `LearningMilestone` objects is persisted in SQLite.
   - Milestones specify estimated hours, prerequisites, and link to curated catalog courses.
2. Dynamic Milestone Adaptation (Feedback Loop):
   - Unlike static syllabi, AetherLearn dynamically adapts to student mastery:
     - Quiz Score < 50%: The milestone is flagged as `needs_review`, signaling to the student
       that foundational concepts need reinforcement before advancing.
     - Quiz Score >= 80%: The milestone is marked `completed` with a completion timestamp,
       and the subsequent milestone is automatically unlocked from `not_started` to `in_progress`.
     - Overall path progress percentage is re-calculated: `(completed_milestones / total) * 100`.
3. Pacing & Schedule Estimation:
   - Translates remaining milestone hours into calendar weeks based on the student's daily study
     commitment (`daily_study_minutes`), providing a realistic, personalized pacing roadmap.
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from models.database import (
    Course,
    LearningMilestone,
    LearningPath,
    QuizAttempt,
    UserProfile,
    UserSkill,
    db,
)

logger = logging.getLogger("learning.path_generator")

# Curated milestone templates for major computer science career and learning goals.
# Tuple format: (title, description, estimated_hours, target_skill, prerequisites)
GOAL_TEMPLATES = {
    "python": [
        ("Python Fundamentals & Control Flow", "Master syntax, conditionals, for/while loops, and basic terminal input/output.", 4.0, "Python", "None"),
        ("Functions, Scope & Modules", "Write modular code, understand arguments, return types, lambda functions, and imports.", 5.0, "Python Functions", "Python Fundamentals"),
        ("Data Structures: Lists, Tuples, Dictionaries", "Perform indexing, slicing, dictionary lookups, and nested data manipulation.", 6.0, "Python Data Structures", "Functions"),
        ("Object-Oriented Programming (OOP)", "Understand classes, object instances, inheritance, and encapsulation.", 6.0, "Python OOP", "Data Structures"),
        ("File I/O & Exception Handling", "Work safely with files, context managers (with statement), and structured error handling.", 4.0, "Python Error Handling", "OOP"),
        ("NumPy & Vectorized Mathematics", "Work with multi-dimensional arrays, mathematical broadcasting, and vectorization.", 6.0, "NumPy", "Data Structures"),
        ("Pandas for Data Manipulation", "Clean, filter, aggregate, and merge dataframes for real-world tabular data.", 8.0, "Pandas", "NumPy"),
        ("Exploratory Data Analysis & Visualization", "Plot distributions and relationships with Matplotlib & Seaborn.", 6.0, "Data Visualization", "Pandas"),
        ("Supervised Machine Learning with Scikit-Learn", "Train classification and regression models; evaluate with cross-validation.", 10.0, "Machine Learning", "Pandas"),
        ("Capstone: End-to-End Predictive Application", "Build a complete data pipeline and export trained models for inference.", 12.0, "Machine Learning Project", "Supervised ML"),
    ],
    "web development": [
        ("HTML5 Semantic Structure & Forms", "Build modern, accessible page layouts with semantic elements and validated forms.", 4.0, "HTML5", "None"),
        ("Modern CSS3, Flexbox & Grid Layouts", "Craft responsive, beautiful interfaces using Flexbox and CSS Grid.", 6.0, "CSS3", "HTML5"),
        ("JavaScript Fundamentals & DOM Manipulation", "Manipulate elements, handle user events, and manage state in the browser.", 8.0, "JavaScript", "HTML5 & CSS3"),
        ("Asynchronous JavaScript & REST APIs", "Fetch data asynchronously using Promises, async/await, and JSON parsing.", 6.0, "Async JS", "JavaScript"),
        ("Backend Web Framework Fundamentals (Flask)", "Handle HTTP requests, routing, template rendering, and session state.", 8.0, "Backend / Flask", "JavaScript"),
        ("Database Modeling & SQL (Relational DBs)", "Design normalized schemas, perform queries, and handle migrations with ORM.", 8.0, "Databases", "Backend / Flask"),
        ("Authentication & API Security", "Implement secure session tokens, password hashing, and CSRF protection.", 6.0, "Web Security", "Databases"),
        ("Full-Stack Capstone Project", "Deploy a complete full-stack web application with responsive UI and database persistence.", 14.0, "Fullstack Project", "Auth & DBs"),
    ],
    "data structures & algorithms": [
        ("Time & Space Complexity (Big-O)", "Analyze runtime and memory footprints of iterative algorithms.", 4.0, "Big-O Analysis", "Programming Fundamentals"),
        ("Arrays & Two-Pointer Techniques", "Solve subarray, windowing, and two-pointer array problems efficiently.", 6.0, "Arrays", "Big-O"),
        ("Linked Lists & Node Manipulation", "Implement singly/doubly linked lists, cycle detection, and reversal.", 6.0, "Linked Lists", "Arrays"),
        ("Stacks & Queues (LIFO / FIFO)", "Understand monotonic stacks, expression parsing, and queue-based BFS traversal.", 6.0, "Stacks & Queues", "Linked Lists"),
        ("Binary Trees & Tree Traversals", "Pre-order, in-order, post-order, and level-order traversals on binary trees.", 8.0, "Binary Trees", "Stacks & Queues"),
        ("Binary Search Trees & Heap Priority Queues", "BST operations, balancing properties, and min/max heap construction.", 8.0, "BST & Heaps", "Binary Trees"),
        ("Graph Algorithms: BFS, DFS & Shortest Path", "Adjacency representations, cycle detection, Dijkstra's shortest path.", 10.0, "Graphs", "BST & Heaps"),
        ("Dynamic Programming Fundamentals", "Memoization vs tabulation, subproblem identification, and optimization.", 12.0, "Dynamic Programming", "Recursion & Graphs"),
    ],
    "machine learning": [
        ("Python & Numerical Computing Essentials", "NumPy, matrix operations, and vectorized computations.", 6.0, "Python / NumPy", "None"),
        ("Data Preparation & Feature Engineering", "Handling missing values, scaling, encoding, and feature extraction.", 6.0, "Feature Engineering", "NumPy"),
        ("Supervised Learning: Regression Models", "Linear and Ridge regression, loss functions, and gradient descent basics.", 8.0, "Regression", "Feature Engineering"),
        ("Supervised Learning: Classification Models", "Logistic regression, Decision Trees, and Random Forest ensembles.", 8.0, "Classification", "Regression"),
        ("Model Evaluation, Validation & Metrics", "ROC-AUC, Precision, Recall, F1-Score, and cross-validation techniques.", 6.0, "Model Evaluation", "Classification"),
        ("Deep Learning Fundamentals", "Neural network architectures, activation functions, and backpropagation.", 10.0, "Deep Learning", "Supervised ML"),
        ("Hands-on ML Deployment Pipeline", "Export models, build serving endpoints, and monitor inference results.", 10.0, "MLOps & Deployment", "Deep Learning"),
    ],
}


def get_or_create_learning_path(user_id: int) -> LearningPath:
    """
    Retrieves the user's active learning path, or automatically creates one.
    
    Viva Explanation:
    - Checks SQLite for an existing `LearningPath` with `status="active"`.
    - If none exists (e.g. first login after onboarding), reads `UserProfile` to discover
      the student's `primary_goal` and `target_skill`, then calls `create_learning_path_for_user()`.
    """
    path = LearningPath.query.filter_by(user_id=user_id, status="active").first()
    if path:
        return path

    profile = UserProfile.query.filter_by(user_id=user_id).first()
    target_skill = (profile.target_skill if profile else "Python").strip()
    primary_goal = (profile.primary_goal if profile else "Learn a new technology").strip()

    return create_learning_path_for_user(
        user_id=user_id,
        goal=primary_goal,
        target_skill=target_skill,
        starting_level="beginner",
        target_level="advanced"
    )


def create_learning_path_for_user(
    user_id: int,
    goal: str,
    target_skill: str,
    starting_level: str = "beginner",
    target_level: str = "advanced"
) -> LearningPath:
    """
    Constructs an ordered learning path with milestones customized to the user's goal.
    
    Step-by-Step Viva Explanation:
    1. Template Selection: Matches keywords in `goal` and `target_skill` to choose the closest
       curriculum template (Python, Web Development, Data Structures, or Machine Learning).
    2. Deactivates Prior Paths: Sets any existing active paths for this user to `status="paused"`.
    3. Path Instantiation: Creates a new `LearningPath` row in SQLite.
    4. Milestone Creation: Loops through each milestone in the template:
       - Sets milestone 0 to `status="in_progress"` and remaining milestones to `not_started`.
       - Matches relevant catalog courses by checking if milestone skills appear in course titles/tags.
    5. Commits to SQLite and returns the active `LearningPath` instance.
    """
    # Match template
    goal_key = "python"
    query_str = f"{goal} {target_skill}".lower()

    if "web" in query_str or "fullstack" in query_str or "frontend" in query_str:
        goal_key = "web development"
    elif "algo" in query_str or "dsa" in query_str or "placement" in query_str or "structure" in query_str:
        goal_key = "data structures & algorithms"
    elif "machine learning" in query_str or "ai" in query_str or "data science" in query_str:
        goal_key = "machine learning"
    else:
        goal_key = "python"

    template = GOAL_TEMPLATES.get(goal_key, GOAL_TEMPLATES["python"])
    title = f"Learning Path: {target_skill or goal}"

    # Deactivate existing active paths to ensure a single active roadmap
    existing_paths = LearningPath.query.filter_by(user_id=user_id, status="active").all()
    for p in existing_paths:
        p.status = "paused"

    path = LearningPath(
        user_id=user_id,
        title=title,
        goal=goal,
        starting_skill_level=starting_level,
        target_skill_level=target_level,
        status="active",
        overall_progress=0.0,
    )
    db.session.add(path)
    db.session.flush()

    # Find relevant catalog courses to link to milestones
    all_courses = Course.query.filter_by(is_active=True).all()

    for idx, (m_title, m_desc, m_hours, m_skill, m_prereq) in enumerate(template):
        # Link related course if available
        matched_course_id = None
        for c in all_courses:
            if m_skill.lower() in c.title.lower() or any(m_skill.lower() in s.lower() for s in c.get_skills_list()):
                matched_course_id = c.id
                break

        milestone = LearningMilestone(
            learning_path_id=path.id,
            title=m_title,
            description=m_desc,
            order_index=idx,
            estimated_hours=m_hours,
            status="in_progress" if idx == 0 else "not_started",
            target_skill=m_skill,
            prerequisites=m_prereq,
            related_course_id=matched_course_id,
        )
        db.session.add(milestone)

    db.session.commit()
    logger.info("Created learning path '%s' with %d milestones for user %d", title, len(template), user_id)
    return path


def adapt_learning_path_after_quiz(user_id: int, attempt_id: int) -> bool:
    """
    Dynamically adapts the active learning path based on quiz attempt performance.
    
    Viva Adaptive Rules:
    - Low Score (< 50%): If the student struggled on concepts covered by a milestone,
      the milestone status is flagged as `needs_review`. This signals to the student and
      the dashboard that revision is required before moving on.
    - High Score (>= 80%): Mastery is proven! The milestone is updated to `completed` and
      its completion timestamp is recorded.
    - Milestone Unlocking: Automatically finds the next `not_started` milestone and advances
      it to `in_progress`.
    - Path Progress Update: Re-calculates total completion percentage via `path.calculate_progress()`.
    """
    path = LearningPath.query.filter_by(user_id=user_id, status="active").first()
    if not path or not path.milestones:
        return False

    attempt = db.session.get(QuizAttempt, attempt_id)
    if not attempt or attempt.user_id != user_id:
        return False

    attempt_pct = attempt.percentage
    # Map topics from attempt
    attempted_topics = {r.question.topic.lower() for r in attempt.responses if r.question and r.question.topic}

    changed = False
    for m in path.milestones:
        target_lower = (m.target_skill or "").lower()
        if any(t in target_lower or target_lower in t for t in attempted_topics):
            if attempt_pct < 50.0:
                # Needs review
                if m.status != "completed":
                    m.status = "needs_review"
                    changed = True
                    logger.info("Milestone '%s' marked 'needs_review' due to quiz score %.1f%%", m.title, attempt_pct)
            elif attempt_pct >= 80.0:
                # Mastery achieved
                if m.status in ("in_progress", "needs_review", "not_started"):
                    m.status = "completed"
                    m.completed_at = datetime.utcnow()
                    changed = True
                    logger.info("Milestone '%s' marked 'completed' due to quiz score %.1f%%", m.title, attempt_pct)

    if changed:
        # Unlock next milestone if needed
        unlocked = False
        for m in path.milestones:
            if m.status == "completed":
                continue
            if not unlocked and m.status == "not_started":
                m.status = "in_progress"
                unlocked = True

        path.calculate_progress()
        db.session.commit()

    return changed


def calculate_study_schedule(user_id: int, learning_path: LearningPath) -> Dict[str, Any]:
    """
    Calculates estimated completion timeline and pacing based on available study time.
    
    Viva Explanation:
    - Reads student's `daily_study_minutes` from their profile (defaults to 45 min/day = 5.25 hrs/week).
    - Sums estimated hours across all milestones and remaining uncompleted milestones.
    - Divides remaining hours by weekly study hours to give an estimated calendar duration
      (e.g., "~4.5 weeks remaining").
    """
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    daily_mins = profile.daily_study_minutes if profile and profile.daily_study_minutes else 45
    weekly_hours = (daily_mins * 7) / 60.0

    total_hours = sum(m.estimated_hours for m in learning_path.milestones)
    remaining_hours = sum(m.estimated_hours for m in learning_path.milestones if m.status != "completed")

    estimated_weeks_total = round(total_hours / max(weekly_hours, 1.0), 1)
    estimated_weeks_remaining = round(remaining_hours / max(weekly_hours, 1.0), 1)

    return {
        "daily_study_minutes": daily_mins,
        "weekly_study_hours": round(weekly_hours, 1),
        "total_estimated_hours": total_hours,
        "remaining_estimated_hours": remaining_hours,
        "estimated_weeks_total": estimated_weeks_total,
        "estimated_weeks_remaining": estimated_weeks_remaining,
        "pacing_note": (
            f"Based on your commitment of {daily_mins} minutes per day (~{round(weekly_hours, 1)} hours/week), "
            f"you have approximately {estimated_weeks_remaining} weeks of structured study remaining."
        )
    }
