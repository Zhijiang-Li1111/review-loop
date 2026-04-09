"""ContextManager — builds initial shared context for the review loop."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from review_loop.config import ReviewConfig

logger = logging.getLogger(__name__)


class ContextManager:
    def __init__(
        self,
        config: ReviewConfig,
        context_builder: Callable[[dict], Awaitable[str]] | None = None,
    ):
        self._config = config
        self._context_builder = context_builder

    async def build_initial_context(self) -> str:
        if self._context_builder:
            return await self._context_builder(self._config.context)
        logger.warning(
            "No context builder registered. Starting review with empty context."
        )
        return ""
