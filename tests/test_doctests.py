"""Run the documentation examples that live in docstrings.

Every ``>>>`` example in these modules is executable documentation: if the
behaviour changes, the docs change with it (or the build breaks).
"""

from __future__ import annotations

import doctest
import importlib

import pytest

DOCTEST_MODULES = [
    "pyagents.types",
    "pyagents.utils",
    "pyagents.errors",
    "pyagents.memory",
    "pyagents.stores",
    "pyagents.parsers",
    "pyagents.embeddings",
    "pyagents.workflows",
    "pyagents.teams",
    "pyagents.tools",
    "pyagents.tools.base",
    "pyagents.tools.builtin",
    "pyagents.llm",
    "pyagents.llm.echo",
]


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_module_doctests(module_name):
    module = importlib.import_module(module_name)
    results = doctest.testmod(module, verbose=False, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0, f"{results.failed} failing doctest(s) in {module_name}"
