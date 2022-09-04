"""Translate english to chinese via a dict."""
from typing import List, Union

from tinybee.en2zh import en2zh
from tinybee.insert_spaces import insert_spaces


# fmt: off
def en2zh_tokens(
        # text: Union[str, List[List[str]]],
        text: Union[str, List[str]],
        dedup: bool = True,
) -> List[List[str]]:
    # fmt: on
    """Translate english to chinese tokens via a dict.

    Args
        text: to translate, list of str
        dedup: if True, remove all duplicates
    Returns
        res: list of list of str/token/char
    """
    res = en2zh(text)

    if dedup:
        return [list(set(insert_spaces(elm).split())) for elm in res]

    return [insert_spaces(elm).split() for elm in res]
