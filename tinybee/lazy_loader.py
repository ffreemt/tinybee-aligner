r"""Lazy loader.

https://levelup.gitconnected.com/python-trick-lazy-module-loading-df9b9dc111af https://github.com/uTensor/utensor_cgen/blob/develop/utensor_cgen/utils.py#L23-L44
myapps\tkaliger

in essence:
if "lib_name" not in sys.modules:
    _mod = importlib.import_module("lib_name")
mod_name = getattr(_mod, "mod_name")
# globas()["mod_name"] = getattr(_mod, "mod_name")

Usage:
from tinybee import LazyLoader

mdx_e2c = LazyLoader("tinybee.mdx_e2c")
mdx_e2c = mdx_e2c.mdx_e2c

"""
import types
import importlib


class LazyLoader(types.ModuleType):
    """Load package lazily."""

    def __init__(self, module_name="numpy", submod_name=None):
        """Init."""
        self._module_name = "{}{}".format(
            module_name, submod_name and ".{}".format(submod_name) or ""
        )
        self._mod = None
        super(LazyLoader, self).__init__(self._module_name)

    def _load(self):
        if self._mod is None:
            self._mod = importlib.import_module(self._module_name)
        return self._mod

    def __getattr__(self, attrb):
        """Get."""
        return getattr(self._load(), attrb)

    def __dir__(self):
        """Dir."""
        return dir(self._load())
