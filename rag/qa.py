def find_relevant_segments(question, segments):
    """
    Simple keyword-based retrieval.
    Later we can replace this with embeddings + FAISS.
    """

    question_words = set(
        question.lower().split()
    )

    results = []

    for segment in segments:

        text_words = set(
            segment["text"].lower().split()
        )

        score = len(
            question_words.intersection(text_words)
        )

        if score > 0:

            results.append({
                "score": score,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:5]