"""Test gen_model."""
# pylint:disable=invalid-name
from pathlib import Path
from more_itertools import ilen

from tinybee.en2zh import en2zh
from tinybee.en2zh_tokens import en2zh_tokens
from tinybee.gen_model import gen_model
from tinybee.insert_spaces import insert_spaces
from tinybee.docterm_scores import docterm_scores

texten = [
    elm.strip()
    for elm in Path("data/test_en.txt").read_text("utf8").splitlines()
    if elm.strip()
]
textzh = [
    elm.strip()
    for elm in Path("data/test_zh.txt").read_text("utf8").splitlines()
    if elm.strip()
]

tokenized_docs = [insert_spaces(elm).split() for elm in textzh]
model = gen_model(tokenized_docs)


def test_gen_model():
    """Test gen_model."""
    assert ilen(iter(tokenized_docs)) > 30  # 36

    assert model.vocabulary_terms.__len__() > 520  # 523
    assert gen_model.doc_term_matrix.shape == (36, 523)


def test_en2zh():
    """Test en2zh_tokens."""
    en2zh3 = en2zh(texten[:3])

    assert isinstance(en2zh3[0], str)

    assert en2zh3[1]  # not empty: ----- not stripped in mdx_e2c
    assert len(en2zh3) == 3


def test_en2zh_tokens():
    """Test en2zh_tokens."""
    textenzh3 = en2zh_tokens(texten[:3])

    assert isinstance(textenzh3[0], list)

    assert textenzh3[1]  # not empty: ----- not stripped in mdx_e2c
    assert len(textenzh3) == 3


def test_docterm_scores():
    """Test docterm_scores."""
    textenzh_tokens = en2zh_tokens(texten)
    mat = docterm_scores(tokenized_docs, textenzh_tokens, model)

    assert mat.shape == (33, 36)
