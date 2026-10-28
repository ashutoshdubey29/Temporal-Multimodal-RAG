"""
models/seed_data.py
===================
VIVA / ARCHITECTURAL OVERVIEW:
This module seeds the database with curated, legitimate educational courses when
the application is started for the first time.

Why Seeding is Important in a Viva:
- Cold Start Problem: When a new user registers on an empty platform, recommendation
  algorithms and course browsers would be completely empty.
- Curated External Content: Instead of hallucinated dummy links, we use real,
  accessible URLs from respected open educational platforms (Harvard CS50, MIT OCW,
  freeCodeCamp, MDN Web Docs, Python.org, Kaggle).
- Idempotent Seeding: The function checks `Course.query.count()`. If records already
  exist, it gracefully skips insertion to prevent duplicate records every time the
  server restarts.
"""

import logging
from models.database import db, Course

logger = logging.getLogger("models.seed_data")

# Curated catalog of open-access courses covering foundational CS, Web, ML, and Security.
# Each entry specifies difficulty, skills covered, prerequisites, and resource types.
CURATED_COURSES = [
    {
        "title": "CS50: Introduction to Computer Science",
        "description": "An introduction to the intellectual enterprises of computer science and the art of programming from Harvard University. Covers algorithmic thinking, data structures, and problem-solving.",
        "provider": "Harvard University / edX",
        "course_url": "https://cs50.harvard.edu/x/",
        "difficulty": "beginner",
        "duration_hours": 30.0,
        "skills_covered": "Computer Science, Algorithms, Data Structures, Memory Management, Problem Solving",
        "prerequisites": "None",
        "resource_type": "course",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "Scientific Computing with Python",
        "description": "Comprehensive hands-on Python training covering foundational syntax, control flow, functions, data structures, algorithms, and object-oriented programming.",
        "provider": "freeCodeCamp",
        "course_url": "https://www.freecodecamp.org/learn/scientific-computing-with-python/",
        "difficulty": "beginner",
        "duration_hours": 20.0,
        "skills_covered": "Python, Functions, Loops, Data Structures, OOP, File Handling",
        "prerequisites": "None",
        "resource_type": "interactive",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "MIT 6.0001: Introduction to Computer Science and Programming in Python",
        "description": "MIT's flagship undergraduate course aimed at teaching students with little or no programming experience how to design algorithms to solve computational problems.",
        "provider": "MIT OpenCourseWare",
        "course_url": "https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/",
        "difficulty": "beginner",
        "duration_hours": 25.0,
        "skills_covered": "Python, Computational Thinking, Algorithms, Recursion, Object-Oriented Programming",
        "prerequisites": "High school algebra",
        "resource_type": "course",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "The Official Python Tutorial",
        "description": "The authoritative Python tutorial written by Python core developers, detailing language syntax, modules, standard library, and best practices.",
        "provider": "Python Software Foundation",
        "course_url": "https://docs.python.org/3/tutorial/",
        "difficulty": "beginner",
        "duration_hours": 10.0,
        "skills_covered": "Python, Syntax, Modules, Standard Library, Exception Handling",
        "prerequisites": "None",
        "resource_type": "reading",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "Intro to Machine Learning",
        "description": "Learn the core ideas in machine learning and build your first models. Covers decision trees, random forests, model validation, and underfitting/overfitting.",
        "provider": "Kaggle Learn",
        "course_url": "https://www.kaggle.com/learn/intro-to-machine-learning",
        "difficulty": "intermediate",
        "duration_hours": 8.0,
        "skills_covered": "Machine Learning, Scikit-Learn, Model Evaluation, Decision Trees, Classification",
        "prerequisites": "Python, Pandas",
        "resource_type": "interactive",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "Data Analysis with Python and Pandas",
        "description": "Master data reading, filtering, aggregation, and manipulation using NumPy and Pandas for data science workflows.",
        "provider": "freeCodeCamp",
        "course_url": "https://www.freecodecamp.org/learn/data-analysis-with-python/",
        "difficulty": "intermediate",
        "duration_hours": 15.0,
        "skills_covered": "Data Science, Pandas, NumPy, Data Cleaning, Data Visualization",
        "prerequisites": "Python Fundamentals",
        "resource_type": "interactive",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "Data Structures and Algorithms in Python",
        "description": "Master fundamental algorithms (search, sorting, recursion) and data structures (arrays, linked lists, stacks, queues, trees, and graphs) for technical interviews.",
        "provider": "freeCodeCamp",
        "course_url": "https://www.freecodecamp.org/news/data-structures-and-algorithms-in-python/",
        "difficulty": "intermediate",
        "duration_hours": 18.0,
        "skills_covered": "Data Structures, Algorithms, Stacks, Queues, Binary Trees, Big-O Notation",
        "prerequisites": "Python Fundamentals",
        "resource_type": "video_series",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "MDN Web Development Curriculum: Frontend Fundamentals",
        "description": "The gold-standard web documentation and structured guide to modern HTML5, CSS3, responsive layout (Flexbox/Grid), and modern JavaScript.",
        "provider": "Mozilla Developer Network (MDN)",
        "course_url": "https://developer.mozilla.org/en-US/docs/Learn",
        "difficulty": "beginner",
        "duration_hours": 20.0,
        "skills_covered": "Web Development, HTML5, CSS3, JavaScript, DOM Manipulation",
        "prerequisites": "Basic computer literacy",
        "resource_type": "reading",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "Foundational Cybersecurity and Network Defense",
        "description": "Understand core cybersecurity principles, authentication security, secure protocols, cryptography basics, and threat modeling.",
        "provider": "freeCodeCamp / NIST Open Curriculum",
        "course_url": "https://www.freecodecamp.org/news/tag/cybersecurity/",
        "difficulty": "beginner",
        "duration_hours": 16.0,
        "skills_covered": "Cybersecurity, Network Security, Cryptography, Threat Modeling, Authentication",
        "prerequisites": "Basic Networking",
        "resource_type": "course",
        "is_ai_generated": False,
        "is_active": True,
    },
    {
        "title": "Cloud Computing Essentials & Architecture",
        "description": "Core concepts of cloud computing, virtualization, containerization, microservices, cloud storage, and serverless architectures.",
        "provider": "Coursera / Google Cloud Training",
        "course_url": "https://www.coursera.org/learn/gcp-fundamentals",
        "difficulty": "intermediate",
        "duration_hours": 12.0,
        "skills_covered": "Cloud Computing, Infrastructure, Virtualization, Containers, Cloud Storage",
        "prerequisites": "Operating Systems, Networking",
        "resource_type": "course",
        "is_ai_generated": False,
        "is_active": True,
    },
]


def seed_courses_if_needed():
    """
    Populates the Course table with curated educational courses if it is currently empty.
    
    Viva Explanation:
    - This function is called during application initialization (in `models/database.py:init_db()`).
    - It performs an idempotent check: `Course.query.count()`.
    - If 0 courses exist, it loops through `CURATED_COURSES`, instantiates `Course` ORM
      objects, and commits them in a single atomic database transaction.
    - If courses already exist, it does nothing, preserving any user modifications or new courses.
    """
    existing_count = Course.query.count()
    if existing_count > 0:
        logger.info("Course catalogue already seeded (%d courses found).", existing_count)
        return

    logger.info("Seeding initial course catalogue with %d curated courses...", len(CURATED_COURSES))
    for item in CURATED_COURSES:
        course = Course(
            title=item["title"],
            description=item["description"],
            provider=item["provider"],
            course_url=item["course_url"],
            difficulty=item["difficulty"],
            duration_hours=item["duration_hours"],
            skills_covered=item["skills_covered"],
            prerequisites=item["prerequisites"],
            resource_type=item["resource_type"],
            is_ai_generated=item["is_ai_generated"],
            is_active=item["is_active"],
        )
        db.session.add(course)

    # Commit all course inserts in a single transaction
    db.session.commit()
    logger.info("Course catalogue seeded successfully.")
