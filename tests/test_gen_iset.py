"""Test gen_iset."""
import joblib
from tinybee.gen_iset import gen_iset


def test_gen_iset():
    """Test gen_iset."""
    cmat = joblib.load("data/cmat.lzma")

    res = gen_iset(cmat, verbose=False)
    # logger.debug("res: %s, %s", res, res[68])
    # logger.info("res: %s, %s", res, res[68])

    # logger.debug("res[68]: %s", res[68])
    # (68, 68)
    # assert res[68] == (68, 68)
    assert res[68] == (68, 48)
