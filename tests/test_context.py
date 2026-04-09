"""Tests for review_loop.context."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from review_loop.config import (
    ReviewConfig,
    AuthorConfig,
    ReviewerConfig,
    ModelConfig,
)
from review_loop.context import ContextManager


def _make_config() -> ReviewConfig:
    return ReviewConfig(
        max_rounds=3,
        model_config=ModelConfig(model="claude-opus-4.6-1m"),
        author=AuthorConfig(
            name="Author",
            system_prompt="Write.",
            receiving_review_prompt="Process.",
        ),
        reviewers=[ReviewerConfig(name="R1", system_prompt="Review.")],
        tools=[],
        context={"key": "value"},
    )


class TestBuildInitialContext:
    @pytest.mark.asyncio
    async def test_calls_context_builder(self):
        config = _make_config()
        mock_builder = AsyncMock(return_value="built context")
        mgr = ContextManager(config, context_builder=mock_builder)

        result = await mgr.build_initial_context()

        mock_builder.assert_called_once_with(config.context)
        assert result == "built context"

    @pytest.mark.asyncio
    async def test_no_builder_returns_empty(self):
        config = _make_config()
        mgr = ContextManager(config, context_builder=None)

        result = await mgr.build_initial_context()
        assert result == ""

    @pytest.mark.asyncio
    async def test_default_no_builder(self):
        config = _make_config()
        mgr = ContextManager(config)

        result = await mgr.build_initial_context()
        assert result == ""
