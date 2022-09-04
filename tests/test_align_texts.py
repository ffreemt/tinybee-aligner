"""Test align_texts with an aset and src_blocks/tgt_blocks."""
import joblib
from math import isclose

from tinybee.align_texts import align_texts


def test_align_texts():
    """Test align_texts."""
    aset = joblib.load("data/aset.lzma")
    src_blocks = joblib.load("data/src_blocks_zh.lzma")
    tgt_blocks = joblib.load("data/tgt_blocks_en.lzma")

    # len(src_blocks), len(tgt_blocks)
    texts = align_texts(aset, src_blocks, tgt_blocks)

    # [*zip(*texts)][2][:][:10]: ('', 0.0, '', 0.0, '', '', '', 0.5, 0.42, 0.29)
    _ = filter(None, [*zip(*texts)][2])
    _ = map(float, _)
    assert isclose(sum(_), 8.0, abs_tol=0.1)

    assert isclose(len(texts), 38, abs_tol=1)
