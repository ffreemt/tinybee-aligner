"""Test gen_aset with data/test_en.txt test_zh.txt.

Refer also to st_app.py's main()'s UFast in st_bumblebee_aligner.
"""
# from pathlib import Path

# rid of from 'collections.abc' is deprecated warning
# patsy\constraint.py:13
# from six.moves import collections_abc: doesnt work

import joblib
from more_itertools import ilen
from math import isclose

from tinybee.cmat2tset import cmat2tset
from tinybee.gen_row_align import gen_row_align
from tinybee.gen_aset import gen_aset
from tinybee.lrtrace_tset import lrtrace_tset

"""
sys.path.insert(0, "../fast-scores/")

from pathlib import Path
from fast_scores import fast_scores
from fast_scores.process_zh import process_zh
from fast_scores.process_en import process_en
from fast_scores.en2zh import en2zh

import joblib
cmat_fast_scores
"""


def split_text(text, sep="\n"):
    """Split text and remove blank lines."""
    if isinstance(text, bytes):
        text = text.decode("utf8")
    return [elm.strip() for elm in text.split(sep) if elm.strip()]


def test_gen_aset_0():
    """Test gen_aset test_zh.txt test_en.txt."""
    _ = """

    # generated in fast_scores
    src_file = Path("data/test_zh.txt").read_text("utf8")
    src_lines = split_text(src_file)

    tgt_file = Path("data/test_en.txt").read_text("utf8")
    tgt_lines = split_text(tgt_file)

    src_blocks, tgt_blocks = src_lines.copy(), tgt_lines.copy()

    # tr_blocks = pipe(tgt_blocks, *(process_en, en2zh, list))
    _ = process_en(tgt_blocks)
    tr_blocks = en2zh(_)
    _ = process_zh(src_blocks)
    cmat = fast_scores(tr_blocks, _)

    joblib.dump(cmat, "data/cmat_36x32(zh-en).lzma")
    joblib.dump(src_blocks, "data/src_blocks_zh.lzma")
    joblib.dump(tgt_blocks, "data/tgt_blocks_en.lzma")
    # """

    # shape len(_) x len(tr_blocks) 36x33 (zh-en)
    cmat = joblib.load("data/cmat_36x33(zh-en).lzma")

    tset = cmat2tset(cmat).tolist()

    # n_row, n_col = cmat.shape
    pset = gen_row_align(tset, *cmat.shape)

    aset = gen_aset(pset, *cmat.shape)

    col, row, metric = aset[8]
    _ = float(metric)

    # assert np.isclose(metric, 0.42, atol=0.01)
    assert isclose(_, 0.42, abs_tol=0.01)
    assert metric == cmat[int(row), int(col)]

    col0, row0 = 0, 0
    for elm in aset:
        if elm[0] not in [""]:
            col0 += 1
        if elm[1] not in [""]:
            row0 += 1

    assert row0, col0 == cmat.shape

    xset, yset, metrics = zip(*aset)
    assert ilen(filter(None, metrics)) > 24  # ==27
    assert isclose(sum(filter(None, metrics)), 8.04, abs_tol=0.1)  # type: ignore #  8.04 vs lr 7.5


def test_gen_aset12_34():
    """Test simple 12_3,4."""
    pset = [[1, 2]]
    cols, rows = zip(*gen_aset(pset, 3, 4))  # type: ignore

    # check length only
    assert len(set(rows) - {"",}) == 3
    assert len(set(cols) - {"",}) == 4

    # check order
    # filter ""
    rows0 = [*filter(lambda x: str(x), rows)]
    cols0 = [*filter(lambda x: str(x), cols)]

    # convert to int and compare
    def s2int(s) -> int:
        try:
            return int(s)
        except Exception:
            return -1
    assert [*map(s2int, rows0)] == [*range(3)]
    assert [*map(s2int, cols0)] == [*range(4)]


def test_gen_aset_empty_34():
    """Test simple 12_3,4."""
    pset = [[]]
    cols, rows = zip(*gen_aset(pset, 3, 4))  # type: ignore

    # check length only
    assert len(set(rows) - {"",}) == 3
    assert len(set(cols) - {"",}) == 4

    # check order
    # filter ""
    rows0 = [*filter(lambda x: str(x), rows)]
    cols0 = [*filter(lambda x: str(x), cols)]

    # convert to int and compare
    def s2int(s) -> int:
        try:
            return int(s)
        except Exception:
            return -1
    assert [*map(s2int, rows0)] == [*range(3)]
    assert [*map(s2int, cols0)] == [*range(4)]


def test_gen_aset_w_lr_tset():
    """Test gen_aset using lr_tset."""
    cmat = joblib.load("data/cmat_36x33(zh-en).lzma")

    tset = cmat2tset(cmat).tolist()
    lr_tset = lrtrace_tset(tset).tolist()
    pset = gen_row_align(lr_tset, *cmat.shape)
    aset = gen_aset(pset, *cmat.shape)

    xset, yset, metrics = zip(*aset)

    assert ilen(filter(None, metrics)) > 24  # ==26
    _ = filter(None, metrics)
    _ = map(float, _)
    assert isclose(sum(_), 7.5, abs_tol=0.1)
