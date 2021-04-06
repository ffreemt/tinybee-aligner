"""Test embed_text."""
import numpy as np
from logzero import logger

from tinybee.embed_text import embed_text


def test_embed_text1():
    """Test embed_text1."""
    res = embed_text("Test 1", livepbar=False)
    arr = np.array(res)

    logger.info(arr.shape)
    assert arr.shape == (1, 512)

    assert arr.mean() > -0.1


def test_embed_text2():
    """Test embed_text."""
    res = embed_text(["Test 1", "test a"], livepbar=False)
    arr = np.array(res)

    logger.info(arr.shape)
    assert arr.shape == (2, 512)

    assert arr.mean() > -0.1
