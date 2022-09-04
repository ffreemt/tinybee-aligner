"""Init.

Load mdx_e2c mdx_c2e lazily:

# from lazy_loader import LazyLoader
from tinybee import LazyLoader

mdx_e2c = LazyLoader("tinybee.mdx_e2c")
mdx_e2c = mdx_e2c.mdx_e2c

mdx_c2e = LazyLoader("tinybee.mdx_c2e")
mdx_c2e = mdx_e2c.mdx_c2e
"""
__version__ = "0.1.5"
# from .mdx_e2c import mdx_e2c
# from .mdx_c2e import mdx_c2e
from .lazy_loader import LazyLoader