import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from msr_graphrag.pipeline.answer_generator import _clean_answer


def test_clean_answer_strips_prefix_and_explanation():
    assert _clean_answer("Answer: Robert Zemeckis because the evidence says so.") == "Robert Zemeckis"


def test_clean_answer_yes_no_atomic():
    assert _clean_answer("Yes, both are American.") == "yes"
    assert _clean_answer("No because they differ.") == "no"


def test_clean_answer_first_line_only():
    assert _clean_answer("Warsaw\nExplanation: ...") == "Warsaw"
