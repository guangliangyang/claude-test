# Survey Data Explanation:
# ----------------------
# This file contains survey data collected from search experts regarding gender reranking.
# Each entry (a survey case) consists of:
#   - "question": A string representing the input sequence (e.g. a sequence of photos) provided to the experts.
#   - "responses": A list of strings, each representing a reranked sequence (as provided by a respondent) that balances relevance and diversity.
# For example, the first case has an input sequence "M M M M M F F F F F" and a list of responses (each a reranked sequence) from the experts.
# ----------------------

SURVEY_DATA = [
    {
        "question": "M M M M M F F F F F",
        "responses": [
            "M F M F M F M F M F",  # Respondant A thinks an F should be elevated to 2nd position
            "M F M F M F M F M F",  # Respondant B thinks an F should be elevated to 2nd position
            "M F M F M F M F M F",  # Respondant C thinks an F should be elevated to 2nd position
            "M F M F M F M F M F",  # Respondant D thinks an F should be elevated to 2nd position
            "M M F F M M M F F F",  # Respondant E thinks an F should be elevated to 3rd position
            "M M F M F M F F M F",  # Respondant F thinks an F should be elevated to 3rd position
            "M M F M F M F M F F",  # Respondant G thinks an F should be elevated to 3rd position
            "M M F M F M F M F F",  # Respondant H thinks an F should be elevated to 3rd position
            "M M F M F M F M F F",  # Respondant I thinks an F should be elevated to 3rd position
            "M M F M M F F M F F",  # Respondant J thinks an F should be elevated to 3rd position
            "M M F M M F M F F F",  # Respondant K thinks an F should be elevated to 3rd position
            "M M F M M F M F F F",  # Respondant L thinks an F should be elevated to 3rd position
            "M M M F F F M M F F",  # Respondant M thinks an F should be elevated to 4th position
            "M M M F M F M F F F",  # Respondant N thinks an F should be elevated to 4th position
        ]
    }
]