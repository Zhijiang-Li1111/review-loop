"""Configuration loading for the review-loop framework."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any

import yaml
from agno.models.anthropic import Claude


def resolve_env(value: str | None) -> str | None:
    """Resolve a value that may use the ``env:VAR_NAME`` prefix."""
    if value is None:
        return None
    if value.startswith("env:"):
        var_name = value[4:]
        env_val = os.environ.get(var_name)
        if env_val is None:
            raise ValueError(
                f"Environment variable '{var_name}' is not set "
                f"(referenced by api_key: {value})"
            )
        return env_val
    return value


@dataclass
class ModelConfig:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    def to_safe_dict(self) -> dict:
        d = asdict(self)
        if d.get("api_key") is not None:
            d["api_key"] = "***"
        return d


def build_claude(model_config: ModelConfig) -> Claude:
    """Create an Agno Claude instance from a ModelConfig."""
    kwargs: dict[str, Any] = {"id": model_config.model}
    if model_config.api_key is not None:
        kwargs["api_key"] = model_config.api_key
    if model_config.temperature is not None:
        kwargs["temperature"] = model_config.temperature
    if model_config.max_tokens is not None:
        kwargs["max_tokens"] = model_config.max_tokens
    client_params: dict[str, Any] = {}
    if model_config.base_url is not None:
        client_params["base_url"] = model_config.base_url
    client_params["timeout"] = 600.0
    if client_params:
        kwargs["client_params"] = client_params
    return Claude(**kwargs)


@dataclass
class ToolConfig:
    path: str


@dataclass
class AuthorConfig:
    name: str
    system_prompt: str
    receiving_review_prompt: str


@dataclass
class ReviewerConfig:
    name: str
    system_prompt: str


@dataclass
class ReviewConfig:
    max_rounds: int
    model_config: ModelConfig
    author: AuthorConfig
    reviewers: list[ReviewerConfig]
    tools: list[ToolConfig]
    context: dict
    context_builder: str | None = None


_REQUIRED_TOP_KEYS = ("review", "author", "reviewers")


class ConfigLoader:
    """Load and validate a review-loop YAML configuration file."""

    @staticmethod
    def load(path: str) -> ReviewConfig:
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        for key in _REQUIRED_TOP_KEYS:
            if key not in raw:
                raise ValueError(f"Missing required configuration key: '{key}'")

        # --- review block ---
        rev = raw.get("review", {}) or {}
        if "model" not in rev:
            raise ValueError(
                "Missing required configuration key: 'review.model'"
            )

        max_rounds: int = rev.get("max_rounds", 10)

        model_config = ModelConfig(
            model=rev["model"],
            api_key=resolve_env(rev.get("api_key")),
            base_url=resolve_env(rev.get("base_url")),
            temperature=rev.get("temperature"),
            max_tokens=rev.get("max_tokens"),
        )

        # --- author ---
        author_raw = raw["author"]
        author = AuthorConfig(
            name=author_raw["name"],
            system_prompt=author_raw["system_prompt"],
            receiving_review_prompt=author_raw.get("receiving_review_prompt", ""),
        )

        # --- reviewers ---
        reviewers = []
        for r in raw["reviewers"]:
            reviewers.append(
                ReviewerConfig(name=r["name"], system_prompt=r["system_prompt"])
            )
        if not reviewers:
            raise ValueError("'reviewers' list must not be empty")

        # --- tools (optional) ---
        raw_tools = raw.get("tools") or []
        tools: list[ToolConfig] = []
        for i, entry in enumerate(raw_tools):
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError(
                    f"tools[{i}]: each tool must be a dict with a 'path' key, "
                    f"got {entry!r}"
                )
            tools.append(ToolConfig(path=entry["path"]))

        # --- context (optional) ---
        context: dict = raw.get("context", {}) or {}

        # --- context_builder (optional) ---
        context_builder: str | None = raw.get("context_builder")

        return ReviewConfig(
            max_rounds=max_rounds,
            model_config=model_config,
            author=author,
            reviewers=reviewers,
            tools=tools,
            context=context,
            context_builder=context_builder,
        )
