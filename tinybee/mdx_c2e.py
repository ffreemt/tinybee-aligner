"""Load mdx_dict_c2e.

mdx_e2c = joblib.load("./mdx_dict_e2c.lzma")
mdx_c2e = joblib.load("./mdx_dict_e2c.lzma")
"""
from pathlib import Path
import joblib

c_dir = Path(__file__).parent

# lazy load in __init__.py like this?
# mdx_e2c = importlib.import_module("tinybee.mdx_e2c")
# mdx_e2c = mdx_e2c.mdx_e2c

# mdx_c2e = importlib.import_module("tinybee.mdx_c2e")
# mdx_c2e = mdx_c2e.mdx_c2e

mdx_dict_e2c = joblib.load(c_dir / "mdx_dict_c2e.lzma")
print("c2e lzma loaded")


def mdx_c2e(word: str) -> str:
    """Fetch definition for word.

    Args:
        word: word to look up
    Returns:
        c1e definition entry or word itself""
    >>> mdx_c2e("do")
    'do'
    >>> mdx_c2e("我").__len__()
    13
    """
    return mdx_dict_e2c.get(word, word)
