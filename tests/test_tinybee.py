"""Test sanity."""
from tinybee import __version__


def test_version():
    """Test version."""
    assert __version__[:4] == "0.1.1"[:4]
