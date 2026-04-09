"""Tests for review_loop.registry."""

from review_loop.registry import import_from_path
import pytest


class TestImportFromPath:
    def test_import_builtin(self):
        result = import_from_path("json.loads")
        import json
        assert result is json.loads

    def test_no_dot_raises(self):
        with pytest.raises(ValueError, match="dotted path"):
            import_from_path("nodot")

    def test_bad_module_raises(self):
        with pytest.raises(ImportError, match="not found"):
            import_from_path("nonexistent_pkg.Foo")

    def test_bad_attr_raises(self):
        with pytest.raises(ImportError, match="no attribute"):
            import_from_path("json.NonExistentThing")
