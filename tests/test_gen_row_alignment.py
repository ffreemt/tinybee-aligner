"""Test gen_row_alignment."""
from typing import Tuple, Union

import pickle
import numpy as np

from tinybee.gen_row_alignment import gen_row_alignment
from tinybee.find_pairs import find_pairs  # lowess


def test_wuch2():
    """Test wuch2."""
    filename = r"tests/t_set99_wuch2.pkl"
    with open(filename, "rb") as fhandle:
        tset_ch2 = pickle.load(fhandle)

    resu = gen_row_alignment(tset_ch2, 99, 106)

    assert len(resu) >= 99, "should be larger than 99"

    _ = sum(resu[0])
    assert np.isclose(_, sum([0, 0, -0.0062533836]))

    entry = ["", 5, ""]
    idx = resu.index(entry)
    resu_ = resu[idx]
    assert all([elm == resu_[idx] for idx, elm in enumerate(entry)])

    entry = ("", 95, "")
    entry = ["", 95, ""]
    idx = resu.index(entry)
    resu_ = resu[idx]
    assert all([elm == resu_[idx] for idx, elm in enumerate(entry)])

    _ = """
    assert all(np.isclose(resu[-1], [98, 105, -3.1654365])), (
        np.array(resu).shape,
        resu[-1],
    )
    # """
    # assert resu[-1] == [98, 105, -3.1654365]
    assert round(sum(resu[-1]), 2) == round(sum([98, 105, -3.1654365]), 2)  # type: ignore

def test_wuch1_find_pairs():
    """Test wuch1."""
    filename = r"tests/nll_matrix_wuch1.pkl"
    with open(filename, "rb") as fhandle:
        nll_matrix_ch1 = pickle.load(fhandle)

    # old gen_nllmatrix
    nll_matrix_ch1 = nll_matrix_ch1.T

    assert nll_matrix_ch1.shape == (30, 33)

    tset_ch1 = find_pairs(nll_matrix_ch1)
    src_len, tgt_len = nll_matrix_ch1.shape
    resu_ch1 = gen_row_alignment(tset_ch1, src_len, tgt_len)

    assert len(resu_ch1) >= src_len, "should be larger than 99"

    # assert all(np.isclose(resu_ch1[0], [0, 0, -0.02035301]))
    assert resu_ch1[0] == [0, 0, -0.020353009924292564]

    # assert all(np.isclose(resu_ch1[-2], [28, 31, -0.020703452]))
    # assert resu_ch1[-2] == [28, 31, -0.020703452]
    assert resu_ch1[-2] ==  [28, '', '']

    # entry = [29, 32, ""]
    # entry = (29, '', '')
    entry = [29, '', '']
    idx = resu_ch1.index(entry)
    resu_ = resu_ch1[idx]
    assert all([elm == resu_[idx] for idx, elm in enumerate(entry)])

    # assert False, resu_ch1

    # entxt = 'wu_ch1_en.txt'
    # zhtxt = 'wu_ch1_zh.txt'
    # en = load_paras(r'data\\' + entxt)
    # zh = load_paras(r'data\\' + zhtxt)
    # for elm in resu_ch1:
    #     print('\n', en[0][elm[0]] if elm[0] else '')
    #     print(zh[0][elm[1]] if elm[1] else '', elm[2])


def test_wuch1a():
    """Test wuch1a find_aligned_pairs(nll_matrix_ch1, numb=30)."""
    filename = r"tests/nll_matrix_wuch1.pkl"
    with open(filename, "rb") as fhandle:
        nll_matrix_ch1 = pickle.load(fhandle)

    # old gen_nllmatrix
    nll_matrix_ch1 = nll_matrix_ch1.T

    assert nll_matrix_ch1.shape == (30, 33)

    src_len, tgt_len = nll_matrix_ch1.shape
    tset_ch1 = find_pairs(nll_matrix_ch1)
    resu_ch1 = gen_row_alignment(tset_ch1, src_len, tgt_len)

    assert len(resu_ch1) >= src_len, "should be larger than 99"

    assert all(np.isclose(resu_ch1[0], [0, 0, -0.02035301]))  # type: ignore
    # assert all(np.isclose(resu_ch1[-2], [28, 31, -0.020703452]))
    # assert resu_ch1[0] == [0, 0, -0.02035301]
    # assert resu_ch1[-2] == [28, 31, -0.020703452]

    entry = ["", 25, ""]
    idx = resu_ch1.index(entry)
    resu_ = resu_ch1[idx]
    assert all([elm == resu_[idx] for idx, elm in enumerate(entry)])

    assert resu_ch1[25] == ["", 25, ""]
    # assert False, resu_ch1

    # entxt = 'wu_ch1_en.txt'
    # zhtxt = 'wu_ch1_zh.txt'
    # en = load_paras(r'data\\' + entxt)
    # zh = load_paras(r'data\\' + zhtxt)
    # for elm in resu_ch1:
    #     print('\n', en[0][elm[0]] if elm[0] != '' else '')
    #     print(zh[0][elm[1]] if elm[1] != '' else '', elm[2])
