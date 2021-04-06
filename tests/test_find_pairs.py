"""Test find_pairs."""
import joblib

from tinybee.find_pairs import find_pairs
from tinybee.cos_matrix2 import cos_matrix2


def test_find_pairs():
    """Test find_pairs."""
    shen600 = joblib.load("data/shen600.lzma")
    shzh500 = joblib.load("data/shzh600.lzma")[:500]

    cmat = cos_matrix2(shen600, shzh500)

    assert cmat.shape == (600, 500)

    pairs = find_pairs(cmat)

    # assert not pairs[-5:]
    # [(488, 189, 0.6136768), (490, 191, 0.5593815), (491, 192, 0.6223874), (492, 193, 0.37926865), (493, 194, 0.4905991)]
    assert pairs[-5][:2] == (488, 189)
