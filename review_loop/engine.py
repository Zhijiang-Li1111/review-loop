"""ReviewEngine — orchestrates the write-review loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict
from typing import Optional

from pydantic import BaseModel, Field

from agno.agent import Agent

from review_loop.config import ReviewConfig, build_claude
from review_loop.context import ContextManager
from review_loop.models import (
    AuthorResponse,
    AuthorVerdictItem,
    ReviewerFeedback,
    ReviewIssue,
    ReviewResult,
)
from review_loop.persistence import Archiver
from review_loop.registry import import_from_path


class ReviewIssueOutput(BaseModel):
    """Structured output model for a single review issue."""
    severity: str = Field(description="Issue severity: critical, major, or minor")
    content: str = Field(description="Description of the issue found")


class ReviewerOutput(BaseModel):
    """Structured output model for reviewer feedback."""
    issues: list[ReviewIssueOutput] = Field(default_factory=list, description="List of issues found. Empty list means no issues.")

logger = logging.getLogger(__name__)


class AllReviewersFailedError(Exception):
    """Raised when every reviewer fails during a review step."""


class ReviewEngine:
    """Run a structured write-review loop to convergence or max rounds."""

    def __init__(self, config: ReviewConfig):
        self._config = config
        self._archiver = Archiver()

        # Import tool classes for Author
        tool_instances = []
        for tc in config.tools:
            cls = import_from_path(tc.path)
            tool_instances.append(cls(context=config.context))

        # Load context builder
        context_builder = None
        if config.context_builder:
            context_builder = import_from_path(config.context_builder)

        self._context_mgr = ContextManager(config, context_builder=context_builder)

        model = build_claude(config.model_config)

        # Create Author agent (with tools)
        self._author = Agent(
            name=config.author.name,
            model=model,
            system_message=config.author.system_prompt,
            tools=tool_instances if tool_instances else None,
            store_tool_messages=False,
            add_history_to_context=False,
        )

        # Expand template variables in reviewer system prompts
        template_vars = {
            "{{author.system_prompt}}": config.author.system_prompt,
        }
        for rc in config.reviewers:
            for placeholder, value in template_vars.items():
                if placeholder in rc.system_prompt:
                    rc.system_prompt = rc.system_prompt.replace(placeholder, value)

        # Create Reviewer agents (with optional per-reviewer tools)
        self._reviewers: list[Agent] = []
        for rc in config.reviewers:
            reviewer_tools = None
            if rc.tools:
                reviewer_tools = []
                for tc in rc.tools:
                    tool_cls = import_from_path(tc.path)
                    reviewer_tools.append(tool_cls(context=config.context))
            reviewer = Agent(
                name=rc.name,
                model=model,
                system_message=rc.system_prompt,
                tools=reviewer_tools if reviewer_tools else None,
                output_schema=ReviewerOutput,
                store_tool_messages=False,
                add_history_to_context=False,
            )
            self._reviewers.append(reviewer)

    # ------------------------------------------------------------------
    # Agent call with retry
    # ------------------------------------------------------------------

    async def _safe_agent_call(self, agent: Agent, prompt: str) -> str | None:
        for attempt in range(2):
            try:
                result = await agent.arun(input=prompt, stream=False)
                if result.content:
                    return result.content
                if attempt == 0:
                    continue
                return None
            except Exception:
                if attempt == 0:
                    continue
                logger.warning("Agent '%s' failed after 2 attempts", agent.name, exc_info=True)
                return None
        return None

    # ------------------------------------------------------------------
    # Author: Generate initial content
    # ------------------------------------------------------------------

    async def _author_generate(self, context: str) -> str:
        prompt = (
            f"{context}\n\n"
            f"请基于上述背景资料，生成初始内容。"
        )
        content = await self._safe_agent_call(self._author, prompt)
        if content is None:
            raise AllReviewersFailedError("Author failed to generate initial content")
        return content

    # ------------------------------------------------------------------
    # Reviewers: Parallel review
    # ------------------------------------------------------------------

    async def _review(
        self,
        content: str,
        per_reviewer_ctx: dict[str, str],
    ) -> list[ReviewerFeedback]:
        """All reviewers audit content in parallel."""

        async def call_reviewer(reviewer: Agent) -> ReviewerFeedback | None:
            prev_ctx = per_reviewer_ctx.get(reviewer.name, "")
            if prev_ctx:
                prompt = (
                    f"{prev_ctx}\n\n"
                    f"以下是修改后的内容：\n\n{content}\n\n"
                    f"请审核修改后的内容。对于 Author 接受并修改的 issue，检查修改是否真正解决了问题。"
                    f"对于 Author 反驳的 issue，评估反驳是否成立。可以提出新发现的 issue。"
                )
            else:
                prompt = (
                    f"以下是需要审核的内容：\n\n{content}\n\n"
                    f"请仔细审核上述内容，找出其中的问题。"
                )

            raw = await self._safe_agent_call(reviewer, prompt)
            if raw is None:
                return None

            # With output_model, raw might be a ReviewerOutput pydantic object serialized to string
            # Try to parse as structured output first
            return self._parse_reviewer_output(reviewer.name, raw)

        results = await asyncio.gather(*[call_reviewer(r) for r in self._reviewers])
        feedbacks = [r for r in results if r is not None]

        if not feedbacks:
            raise AllReviewersFailedError("All reviewers failed during review step")

        return feedbacks

    def _parse_reviewer_output(self, name: str, raw) -> ReviewerFeedback:
        """Parse reviewer output into ReviewerFeedback.
        
        Handles both structured output (ReviewerOutput pydantic object) and
        plain text/JSON string fallback.
        """
        # Handle structured output (Pydantic model from output_model)
        if isinstance(raw, ReviewerOutput):
            issues = [
                ReviewIssue(severity=i.severity, content=i.content)
                for i in raw.issues
            ]
            return ReviewerFeedback(reviewer_name=name, issues=issues)

        # Fallback: parse as string
        raw = str(raw)
        try:
            # Try to find JSON with "issues" key - use greedy match to handle nested braces
            match = re.search(r'\{[^{]*"issues"\s*:\s*\[.*\]\s*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                # Try extracting from markdown code block
                code_match = re.search(r'```(?:json)?\s*(\{.*?"issues".*?\})\s*```', raw, re.DOTALL)
                if code_match:
                    data = json.loads(code_match.group(1))
                else:
                    data = json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Reviewer '%s' returned non-JSON output, treating as no issues", name)
            return ReviewerFeedback(reviewer_name=name, issues=[])

        issues = []
        for item in data.get("issues", []):
            issues.append(
                ReviewIssue(
                    severity=item.get("severity", "minor"),
                    content=item.get("content", ""),
                )
            )
        return ReviewerFeedback(reviewer_name=name, issues=issues)

    # ------------------------------------------------------------------
    # Author: Process feedback
    # ------------------------------------------------------------------

    async def _author_process_feedback(
        self,
        content: str,
        feedbacks: list[ReviewerFeedback],
    ) -> AuthorResponse:
        """Author reviews all issues and outputs verdicts + updated content."""
        issues_text = self._format_issues_for_author(feedbacks)
        prompt = (
            f"{self._config.author.receiving_review_prompt}\n\n"
            f"当前内容：\n\n{content}\n\n"
            f"审核员反馈：\n\n{issues_text}\n\n"
            f"请对每个 issue 做出判断，然后输出修改后的完整内容。\n\n"
            f"请返回 JSON 格式：\n"
            f'{{"responses": [{{"reviewer": "审核员名", "issue_index": 0, '
            f'"verdict": "accept|reject|unclear", "reason": "理由"}}], '
            f'"updated_content": "修改后的完整内容"}}'
        )

        raw = await self._safe_agent_call(self._author, prompt)
        if raw is None:
            # Fallback: keep content unchanged
            return AuthorResponse(responses=[], updated_content=content)

        return self._parse_author_response(raw, content)

    def _format_issues_for_author(self, feedbacks: list[ReviewerFeedback]) -> str:
        parts: list[str] = []
        for fb in feedbacks:
            if not fb.issues:
                continue
            parts.append(f"[{fb.reviewer_name}]")
            for i, issue in enumerate(fb.issues):
                parts.append(f"  issue {i} ({issue.severity}): {issue.content}")
        return "\n".join(parts)

    def _parse_author_response(self, raw: str, fallback_content: str) -> AuthorResponse:
        """Parse author JSON output into AuthorResponse."""
        try:
            # Try to find JSON block
            match = re.search(r"\{.*\"responses\".*\"updated_content\".*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Author returned non-JSON output, keeping content unchanged")
            return AuthorResponse(responses=[], updated_content=fallback_content)

        responses = []
        for item in data.get("responses", []):
            responses.append(
                AuthorVerdictItem(
                    reviewer=item.get("reviewer", ""),
                    issue_index=item.get("issue_index", 0),
                    verdict=item.get("verdict", "unclear"),
                    reason=item.get("reason", ""),
                )
            )

        updated_content = data.get("updated_content", fallback_content)
        return AuthorResponse(responses=responses, updated_content=updated_content)

    # ------------------------------------------------------------------
    # Build per-reviewer context for next round
    # ------------------------------------------------------------------

    def _build_reviewer_context(
        self,
        feedbacks: list[ReviewerFeedback],
        author_response: AuthorResponse,
    ) -> dict[str, str]:
        """Build context for each reviewer: their issues + Author's responses."""
        # Index author responses by (reviewer, issue_index)
        verdict_map: dict[tuple[str, int], AuthorVerdictItem] = {}
        for item in author_response.responses:
            verdict_map[(item.reviewer, item.issue_index)] = item

        ctx: dict[str, str] = {}
        for fb in feedbacks:
            if not fb.issues:
                continue
            parts: list[str] = [f"[上一轮你提出的 issues 及 Author 的回应]"]
            for i, issue in enumerate(fb.issues):
                parts.append(f"\nissue {i} ({issue.severity}): {issue.content}")
                verdict_item = verdict_map.get((fb.reviewer_name, i))
                if verdict_item:
                    tag = verdict_item.verdict.upper()
                    parts.append(f"Author 回应: [{tag}] {verdict_item.reason}")
                else:
                    parts.append(f"Author 回应: [未回应]")
            ctx[fb.reviewer_name] = "\n".join(parts)

        return ctx

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        initial_content: str | None = None,
        context: str | None = None,
    ) -> ReviewResult:
        """Run the full write-review loop."""
        session_path = self._archiver.start_session(self._config)

        # Build or load context
        if context is not None:
            ctx = context
        else:
            ctx = await self._context_mgr.build_initial_context()
        self._archiver.save_context(ctx)

        rounds_completed: int = 0
        per_reviewer_ctx: dict[str, str] = {}

        try:
            # Get initial content
            if initial_content is not None:
                content = initial_content
            else:
                content = await self._author_generate(ctx)

            self._archiver.save_author_content(1, content)

            for round_num in range(1, self._config.max_rounds + 1):
                # Reviewers audit in parallel
                feedbacks = await self._review(content, per_reviewer_ctx)

                for fb in feedbacks:
                    self._archiver.save_reviewer_feedback(
                        round_num, fb.reviewer_name,
                        {"issues": [asdict(i) for i in fb.issues]},
                    )

                # Check convergence — all issues empty = done
                total_issues = sum(len(fb.issues) for fb in feedbacks)
                if total_issues == 0:
                    self._archiver.save_final(content)
                    return ReviewResult(
                        converged=True,
                        rounds_completed=round_num,
                        archive_path=session_path,
                        final_content=content,
                        unresolved_issues=[],
                    )

                # Author processes feedback
                author_response = await self._author_process_feedback(content, feedbacks)

                self._archiver.save_author_response(
                    round_num,
                    {
                        "responses": [asdict(r) for r in author_response.responses],
                        "updated_content": author_response.updated_content,
                    },
                )

                rounds_completed = round_num

                # Update content and prepare next round
                content = author_response.updated_content
                self._archiver.save_author_content(round_num + 1, content)

                # Build per-reviewer context for next round
                per_reviewer_ctx = self._build_reviewer_context(feedbacks, author_response)

            # Max rounds reached
            # Collect last round's unresolved issues
            last_feedbacks = feedbacks if feedbacks else []
            unresolved = [fb for fb in last_feedbacks if fb.issues]

            self._archiver.save_final(content)
            self._archiver.save_unresolved(
                [asdict(fb) for fb in unresolved]
            )

            return ReviewResult(
                converged=False,
                rounds_completed=self._config.max_rounds,
                archive_path=session_path,
                final_content=content,
                unresolved_issues=unresolved,
            )

        except AllReviewersFailedError as exc:
            self._archiver.save_error_log(str(exc))
            return ReviewResult(
                converged=False,
                rounds_completed=rounds_completed,
                archive_path=session_path,
                final_content=None,
                unresolved_issues=[],
                terminated_by_error=True,
            )
