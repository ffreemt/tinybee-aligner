"""Test lrtrace_tset."""
import joblib

from tinybee.lrtrace_tset import lrtrace_tset
from tinybee.cmat2tset import cmat2tset
from tinybee.cos_matrix2 import cos_matrix2


def test_lrtrace_tset():
    """Test find_pairs."""
    shen600 = joblib.load("data/shen600.lzma")
    shzh500 = joblib.load("data/shzh600.lzma")[:500]

    cmat = cos_matrix2(shen600, shzh500)

    assert cmat.shape == (600, 500)

    tset = cmat2tset(cmat)
    lr_tset = lrtrace_tset(tset)
    _ = """
    lr_tset[-5:]
    Out[109]:
    array([[4.87000000e+02, 1.88000000e+02, 7.70491838e-01],
           [4.91000000e+02, 1.92000000e+02, 6.22387409e-01],
           [4.88000000e+02, 1.89000000e+02, 6.13676786e-01],
           [4.90000000e+02, 1.91000000e+02, 5.59381485e-01],
           [4.93000000e+02, 1.94000000e+02, 4.90599096e-01]])
    # """
    assert lr_tset[-2, :2].tolist() == [490.0, 191.0]
