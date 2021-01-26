from tinybee import __version__


def test_version():
    assert __version__[:4] == "0.1.0"[:4]
