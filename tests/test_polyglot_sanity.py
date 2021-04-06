"""Test polyglot install.

https://analyticsindiamag.com/hands-on-tutorial-on-polyglot-python-toolkit-for-multilingual-nlp-applications/
"""
#

from polyglot.text import Detector
from polyglot.text import Word
from polyglot.detect.base import logger as polyglot_logger

polyglot_logger.setLevel("ERROR")  # suppress warning

TEXT = """Analytics India Magazine chronicles technological progress
in the space of  analytics, artificial intelligence, data science &
big data by highlighting the innovations, players, and challenges
shaping the future of India through promotion and discussion of ideas
and thoughts by smart, ardent, action-oriented individuals who want to
change the world."""


def test_detector():
    """Tes detector."""
    _ = Detector(TEXT)
    assert _.language.code in ["en"]

    _ = Detector("a", quiet=True)
    assert _.language.code in ["en"]


def test_morphemes():
    """Test polyglot morphemes.

    !polyglot download morph2.en
        remove signal SIGPIPE in __main__.py
        replace path.sep by "/"

    refer to polyglot-fix in snippets.
    """
    assert Word("programming", language="en").morphemes == ["program", "ming"]
