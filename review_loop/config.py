"""Configuration loading for the review-loop framework."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Any

import yaml
from agno.models.anthropic import Claude


# ---------------------------------------------------------------------------
# Template variable resolution
# ---------------------------------------------------------------------------

_TEMPLATE_RE = re.compile(r"\{\{(\w+)\}\}")

# Keys consumed by the framework — never treated as template variables.
_RESERVED_KEYS = frozenset({
    "review", "author", "reviewers", "tools", "context", "context_builder",
    "skills",
})


def _resolve_template_vars(raw: dict[str, Any]) -> None:
    """Replace ``{{key}}`` placeholders in-place using top-level string values.

    Only top-level keys whose values are plain strings (and that are not
    reserved framework keys) are available as template variables.
    """
    # 1. Collect template variables from top-level string fields.
    template_vars: dict[str, str] = {}
    for key, value in raw.items():
        if key not in _RESERVED_KEYS and isinstance(value, str):
            template_vars[key] = value

    if not template_vars:
        return

    def _replace(s: str) -> str:
        def _sub(m: re.Match) -> str:
            name = m.group(1)
            return template_vars.get(name, m.group(0))
        return _TEMPLATE_RE.sub(_sub, s)

    def _walk(obj: Any) -> Any:
        if isinstance(obj, str):
            return _replace(obj)
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        return obj

    # 2. Walk and replace in all framework-consumed keys (not the var defs).
    for key in list(raw.keys()):
        if key in _RESERVED_KEYS:
            raw[key] = _walk(raw[key])


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
class SkillConfig:
    """Reference to a skill directory path."""
    path: str


_DEFAULT_INITIAL_PROMPT = "请基于上述背景资料，生成初始内容。"


@dataclass
class AuthorConfig:
    name: str
    system_prompt: str
    receiving_review_prompt: str
    initial_prompt: str = _DEFAULT_INITIAL_PROMPT
    skills: list[SkillConfig] | None = None


@dataclass
class ReviewerConfig:
    name: str
    system_prompt: str
    tools: list[ToolConfig] | None = None
    skills: list[SkillConfig] | None = None


@dataclass
class ReviewConfig:
    max_rounds: int
    model_config: ModelConfig
    author: AuthorConfig
    reviewers: list[ReviewerConfig]
    tools: list[ToolConfig]
    context: dict
    context_builder: str | None = None
    skills: list[SkillConfig] | None = None


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

        # --- resolve {{var}} template variables from top-level strings ---
        _resolve_template_vars(raw)

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
        author_skills: list[SkillConfig] | None = None
        raw_author_skills = author_raw.get("skills")
        if raw_author_skills:
            author_skills = []
            for i, s in enumerate(raw_author_skills):
                if not isinstance(s, dict) or "path" not in s:
                    raise ValueError(
                        f"author.skills[{i}]: each skill must be a dict with "
                        f"a 'path' key, got {s!r}"
                    )
                author_skills.append(SkillConfig(path=s["path"]))
        author = AuthorConfig(
            name=author_raw["name"],
            system_prompt=author_raw["system_prompt"],
            receiving_review_prompt=author_raw.get("receiving_review_prompt", ""),
            initial_prompt=author_raw.get("initial_prompt", _DEFAULT_INITIAL_PROMPT),
            skills=author_skills,
        )

        # --- reviewers ---
        reviewers = []
        for r in raw["reviewers"]:
            reviewer_tools: list[ToolConfig] | None = None
            raw_reviewer_tools = r.get("tools")
            if raw_reviewer_tools:
                reviewer_tools = []
                for j, rt in enumerate(raw_reviewer_tools):
                    if not isinstance(rt, dict) or "path" not in rt:
                        raise ValueError(
                            f"reviewers['{r.get('name', '?')}'].tools[{j}]: "
                            f"each tool must be a dict with a 'path' key, got {rt!r}"
                        )
                    reviewer_tools.append(ToolConfig(path=rt["path"]))
            reviewer_skills: list[SkillConfig] | None = None
            raw_reviewer_skills = r.get("skills")
            if raw_reviewer_skills:
                reviewer_skills = []
                for j, rs in enumerate(raw_reviewer_skills):
                    if not isinstance(rs, dict) or "path" not in rs:
                        raise ValueError(
                            f"reviewers['{r.get('name', '?')}'].skills[{j}]: "
                            f"each skill must be a dict with a 'path' key, got {rs!r}"
                        )
                    reviewer_skills.append(SkillConfig(path=rs["path"]))
            reviewers.append(
                ReviewerConfig(
                    name=r["name"],
                    system_prompt=r["system_prompt"],
                    tools=reviewer_tools,
                    skills=reviewer_skills,
                )
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

        # --- skills (optional, global skills shared by all agents) ---
        raw_skills = raw.get("skills") or []
        skills: list[SkillConfig] = []
        for i, entry in enumerate(raw_skills):
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError(
                    f"skills[{i}]: each skill must be a dict with a 'path' key, "
                    f"got {entry!r}"
                )
            skills.append(SkillConfig(path=entry["path"]))

        return ReviewConfig(
            max_rounds=max_rounds,
            model_config=model_config,
            author=author,
            reviewers=reviewers,
            tools=tools,
            context=context,
            context_builder=context_builder,
            skills=skills if skills else None,
        )
