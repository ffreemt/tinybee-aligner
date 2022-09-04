r"""Test tinybee/dbscan_pairs."""
import pickle
from tinybee.dbscan_pairs import dbscan_pairs

wuch1 = pickle.load(open("tests/nll_matrix_wuch1.pkl", "rb"))


def test_dbscan_pairs_wuch1():
    """Test dbscan_pairs nll_matrix_wuch1."""
    res = dbscan_pairs(wuch1, eps=3, min_samples=3)
    assert len(res) > 10  # 15
