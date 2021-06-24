"""Test gen_aset with data/test_en.txt test_zh.txt.

Refer also to st_app.py's main()'s UFast in st_bumblebee_aligner.
"""
from pathlib import Path

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


def split_text(text, sep='\n'):
    """Split text and remove blank lines."""
    if isinstance(text, bytes):
        text = text.decode("utf8")
    return [elm.strip() for elm in text.split(sep) if elm.strip()]


def test_gen_aset():
    """Test gen_aset test_zh.txt test_en.txt."""
    src_file = Path("data/test_zh.txt").read_text("utf8")
    src_lines = split_text(src_file)

    tgt_file = Path("data/test_en.txt").read_text("utf8")
    tgt_lines = split_text(tgt_file)

    src_blocks, tgt_blocks = src_lines.copy(), tgt_lines.copy()

    # tr_blocks = pipe(tgt_blocks, *(process_en, en2zh, list))
    _ = process_en(tgt_blocks)
    _ = map(en2zh, _)
    tr_blocks = list(_)
    _ = process_zh(src_blocks)

    cmat = fast_scores(tr_blocks, _)