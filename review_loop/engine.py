"""ReviewEngine — orchestrates the write-review loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time as _time
from collections.abc import Callable
from dataclasses import asdict
from pydantic import BaseModel, Field

from pathlib import Path

from agno.agent import Agent
from agno.skills import Skills
from agno.skills.loaders.local import LocalSkills
from agno.tools.file import FileTools

from review_loop.config import ReviewConfig, SkillConfig, build_claude
from review_loop.context import ContextManager
from review_loop.models import (
    AuthorResponse,
    AuthorVerdictItem,
    ReviewerFeedback,
    ReviewIssue,
    ReviewResult,
)
from review_loop.audit import AuditLogger, generate_usage_summary
from review_loop.file_protocol import parse_feedback_file, parse_verdict_file
from review_loop.persistence import Archiver
from review_loop.registry import import_from_path


# Keep these for backward compatibility (tests may import them)
class ReviewIssueOutput(BaseModel):
    """Structured output model for a single review issue."""
    severity: str = Field(description="Issue severity: critical, major, or minor")
    content: str = Field(description="Description of the issue found")
    why: str = Field(default="", description="Why this is a problem: what principle it violates or what consequence it causes. Helps the author understand the root cause and fix similar issues proactively.")
    pattern: str = Field(default="", description="Similar pattern hint: suggest the author check the entire text for similar occurrences of this issue type.")


class ReviewerOutput(BaseModel):
    """Structured output model for reviewer feedback."""
    issues: list[ReviewIssueOutput] = Field(default_factory=list, description="List of issues found. Empty list means no issues.")

logger = logging.getLogger(__name__)


def _build_skills(
    global_skills: list[SkillConfig] | None,
    local_skills: list[SkillConfig] | None,
) -> Skills | None:
    """Build an agno Skills object from global + per-agent skill configs.

    Returns None if no skills are configured.
    """
    all_configs: list[SkillConfig] = []
    if global_skills:
        all_configs.extend(global_skills)
    if local_skills:
        all_configs.extend(local_skills)
    if not all_configs:
        return None
    loaders = [LocalSkills(path=sc.path, validate=False) for sc in all_configs]
    return Skills(loaders=loaders)

# Instruction appended to every reviewer's system prompt to ensure
# they write a structured feedback file.
_SUBMIT_REVIEW_INSTRUCTION = (
    "\n\n审核完成后，请将你的发现写入 feedback 文件（使用 save_file 工具）。"
    "文件格式如下：\n"
    "每个 issue 用 ## Issue N 开头，字段用 - key: value 格式：\n"
    "## Issue 1\n"
    "- severity: critical/major/minor/suggestion\n"
    "- content: 问题描述\n"
    "- why: 为什么这是个问题\n"
    "- pattern: 建议检查全文是否有类似问题\n\n"
    "如果没有发现问题，请写：\n"
    "## No Issues\n"
    "审核通过，未发现问题。\n"
)

# Instruction appended to the Author's prompt when evaluating feedback,
# ensuring the Author writes a verdict file.
_SUBMIT_VERDICT_INSTRUCTION = (
    "\n\n评估完每个审核意见后，请将你的裁定写入 verdict 文件（使用 save_file 工具）。"
    "文件格式如下：\n"
    "每个裁定用 ## Issue N (审核员名) 开头：\n"
    "## Issue 0 (审核员名)\n"
    "- verdict: accept/reject/unclear\n"
    "- reason: 裁定原因\n\n"
    "对于接受的 issue，请在 reason 中详细描述计划的修改"
    "（改什么、在哪里、怎么改），使得独立的修订 agent 能够执行。"
    "对于拒绝的 issue，请引用正文中的具体证据。"
)

# Instruction appended to the Author's prompt when applying changes,
# ensuring the Author saves the revised content to draft.md.
_SUBMIT_REVISION_INSTRUCTION = (
    "\n\n修改完成后，请将完整的修改后正文保存到 draft.md（使用 save_file 工具）。"
    "内容必须是完整的——它将直接替换上一版本。"
)

# Instruction for Author initial generation, telling them to write to draft.md
_FILE_BASED_AUTHOR_GENERATE_INSTRUCTION = (
    "\n\n请将你生成的完整正文保存到 draft.md 文件（使用 save_file 工具）。"
    "这样审核员可以直接读取你的稿件。"
)

# Instruction for reviewers to read draft.md and write feedback files
_FILE_BASED_REVIEWER_INSTRUCTION = (
    "\n\n请先用 read_file('draft.md') 读取当前正文，再进行审核。"
    "审核完成后，请将你的详细反馈写入 feedback_R{round}_{name}.md（使用 save_file 工具）。"
    "请严格按照以下格式写反馈文件：\n"
    "## Issue 1\n"
    "- severity: critical/major/minor/suggestion\n"
    "- content: 问题描述\n"
    "- why: 为什么这是个问题\n"
    "- pattern: 建议检查全文是否有类似问题\n\n"
    "如果没有问题，请写：\n"
    "## No Issues\n"
    "审核通过，未发现问题。\n"
)




def _verdict_hint(verdict: str | None) -> str:
    """Return a short verification hint based on verdict type.

    Used by both _build_reviewer_context and _rebuild_reviewer_ctx_from_history
    to produce consistent hints.
    """
    if verdict == "accept":
        return "→ 请在当前内容中定位相关段落，核实此修改是否到位且合理。"
    elif verdict == "reject":
        return "→ Author 拒绝修改。请在当前内容中核实 Author 的理由是否成立。"
    elif verdict == "unclear":
        return "→ Author 回应不明确，请以当前内容为准独立判断。"
    else:
        # None or missing — Author didn't respond
        return "→ Author 未回应此 issue，请在当前内容中检查是否仍然存在。"


class ReviewEngine:
    """Run a structured write-review loop to convergence or max rounds."""

    @staticmethod
    def _issue_from_dict(d: dict) -> ReviewIssue:
        """Construct a ReviewIssue from a raw dict with safe defaults."""
        return ReviewIssue(
            severity=d.get("severity", "minor"),
            content=d.get("content", ""),
            why=d.get("why", ""),
            pattern=d.get("pattern", ""),
        )

    @staticmethod
    def _append_why_pattern(parts: list[str], issue: ReviewIssue) -> None:
        """Append why/pattern lines to parts list if non-empty."""
        if issue.why:
            parts.append(f"    why: {issue.why}")
        if issue.pattern:
            parts.append(f"    pattern: {issue.pattern}")

    def __init__(
        self,
        config: ReviewConfig,
        workspace_dir: Path | None = None,
        error_callback: Callable[[str, dict], None] | None = None,
    ):
        self._config = config
        self._archiver = Archiver()
        self._audit: AuditLogger | None = None
        self._workspace_dir: Path | None = workspace_dir
        self._file_tools_injected = False
        self._error_callback = error_callback

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

        # Load skills for author
        author_skills = _build_skills(config.skills, config.author.skills)

        # Create Author verdict agent (external tools only)
        verdict_tools = list(tool_instances) if tool_instances else []

        self._author_verdict = Agent(
            name=config.author.name,
            model=model,
            system_message=config.author.system_prompt,
            tools=verdict_tools,
            skills=author_skills,
            store_tool_messages=False,
            add_history_to_context=False,
        )

        # Create Author revision agent (external tools only)
        revision_tools = list(tool_instances) if tool_instances else []

        self._author_revision = Agent(
            name=config.author.name,
            model=model,
            system_message=config.author.system_prompt,
            tools=revision_tools,
            skills=author_skills,
            store_tool_messages=False,
            add_history_to_context=False,
        )

        # Create Reviewer agents (file-based feedback, no submit_review tool)
        self._reviewers: list[Agent] = []
        for rc in config.reviewers:
            reviewer_tools: list = []
            if rc.tools:
                for tc in rc.tools:
                    tool_cls = import_from_path(tc.path)
                    reviewer_tools.append(tool_cls(context=config.context))

            # Append file-based feedback instruction to system prompt
            reviewer_system_prompt = rc.system_prompt + _SUBMIT_REVIEW_INSTRUCTION

            # Load skills for this reviewer (global + per-reviewer)
            reviewer_skills = _build_skills(config.skills, rc.skills)

            reviewer = Agent(
                name=rc.name,
                model=model,
                system_message=reviewer_system_prompt,
                tools=reviewer_tools,
                skills=reviewer_skills,
                store_tool_messages=False,
                add_history_to_context=False,
            )
            self._reviewers.append(reviewer)

    # ------------------------------------------------------------------
    # Agent call with retry
    # ------------------------------------------------------------------

    async def _safe_agent_call(self, agent: Agent, prompt: str) -> str | None:
        if self._audit:
            start_extras = AuditLogger.extract_call_start_extras(agent)
            self._audit.log_call_start(agent.name, prompt, **start_extras)
        t0 = _time.monotonic()
        for attempt in range(2):
            try:
                result = await agent.arun(input=prompt, stream=False)
                if result.content:
                    elapsed = (_time.monotonic() - t0) * 1000
                    if self._audit:
                        self._audit.log_from_run_output(agent.name, result)
                        end_extras = AuditLogger.extract_call_end_extras(result)
                        self._audit.log_call_end(agent.name, elapsed, result.content, "end_turn", **end_extras)
                    return result.content
                if attempt == 0:
                    continue
                elapsed = (_time.monotonic() - t0) * 1000
                if self._audit:
                    self._audit.log_call_end(agent.name, elapsed, None, "empty_content")
                return None
            except Exception as exc:
                if attempt == 0:
                    continue
                elapsed = (_time.monotonic() - t0) * 1000
                logger.warning("Agent '%s' failed after 2 attempts", agent.name, exc_info=True)
                if self._audit:
                    self._audit.log_error(agent.name, str(exc), elapsed)
                return None
        elapsed = (_time.monotonic() - t0) * 1000
        if self._audit:
            self._audit.log_call_end(agent.name, elapsed, None, "exhausted_retries")
        return None

    # ------------------------------------------------------------------
    # File-based workspace helpers
    # ------------------------------------------------------------------

    def _setup_file_tools(self) -> None:
        """Inject FileTools (scoped to workspace/) into all agents.

        Called once in run() after workspace_dir is known. Safe to call
        multiple times — skips if already injected.
        """
        if self._file_tools_injected or self._workspace_dir is None:
            return

        file_tools = FileTools(
            base_dir=self._workspace_dir,
            enable_save_file=True,
            enable_read_file=True,
            enable_list_files=True,
            enable_delete_file=False,
        )

        for agent in [self._author_verdict, self._author_revision] + self._reviewers:
            agent.tools.append(file_tools)

        self._file_tools_injected = True

    def _read_draft_from_workspace(self) -> str | None:
        """Read draft.md from the workspace directory.

        Returns the content string, or None if the file doesn't exist.
        """
        if self._workspace_dir is None:
            return None
        draft_path = self._workspace_dir / "draft.md"
        if draft_path.exists():
            return draft_path.read_text()
        return None

    def _write_draft_to_workspace(self, content: str) -> None:
        """Write content to workspace/draft.md so agents can read it."""
        if self._workspace_dir is None:
            return
        draft_path = self._workspace_dir / "draft.md"
        draft_path.write_text(content)

    def _handle_runtime_error(
        self, error_msg: str, context: dict | None = None,
    ) -> None:
        """Log error, notify via callback, and raise RuntimeError."""
        logger.error("Runtime error: %s", error_msg)
        if self._audit:
            self._audit.log_error("RUNTIME", error_msg, 0)
        if self._archiver._session_dir is not None:
            self._archiver.save_error_log(error_msg)
        if self._error_callback:
            try:
                self._error_callback(error_msg, context or {})
            except Exception:
                logger.warning("Error callback failed", exc_info=True)
        raise RuntimeError(error_msg)

    # ------------------------------------------------------------------
    # Author: Generate initial content
    # ------------------------------------------------------------------

    async def _author_generate(self, context: str) -> str:
        prompt = (
            f"{context}\n\n"
            f"{self._config.author.initial_prompt}"
            f"{_FILE_BASED_AUTHOR_GENERATE_INSTRUCTION}"
        )
        content = await self._safe_agent_call(self._author_revision, prompt)

        # Check if Author wrote draft.md via FileTools
        draft = self._read_draft_from_workspace()
        if draft is not None and len(draft.strip()) > 500:
            return draft

        # Fallback: use the text returned by the agent
        if content is None:
            self._handle_runtime_error("Author 初始生成失败：未能生成正文")
        # Also write to workspace so reviewers can read
        self._write_draft_to_workspace(content)
        return content

    # ------------------------------------------------------------------
    # Reviewers: Parallel review
    # ------------------------------------------------------------------

    async def _review(
        self,
        content: str,
        per_reviewer_ctx: dict[str, str],
        guidance: str | None = None,
        round_num: int = 0,
    ) -> list[ReviewerFeedback]:
        """All reviewers audit content in parallel."""

        # Ensure draft.md exists before sending reviewers to read it
        if self._workspace_dir is not None:
            draft_path = self._workspace_dir / "draft.md"
            if not draft_path.exists():
                self._handle_runtime_error(
                    "draft.md 未找到，Author 可能未成功写入正文",
                    {"workspace_dir": str(self._workspace_dir)},
                )

        async def call_reviewer(reviewer: Agent) -> ReviewerFeedback | None:
            prev_ctx = per_reviewer_ctx.get(reviewer.name, "")
            file_based_hint = _FILE_BASED_REVIEWER_INSTRUCTION.format(
                round=round_num, name=reviewer.name,
            )
            if prev_ctx:
                prompt = (
                    f"{prev_ctx}\n\n"
                    f"---\n⚠️ 请先用 read_file('draft.md') 读取【当前内容】，这是本轮审核的唯一判断依据。上方引用的原文片段仅供定位参考，issue 列表和 Author 回应仍需重点关注。\n---\n\n"
                    f"## 你这轮的首要任务：全文重审\n\n"
                    f"Author 在修复上一轮 issue 时，几乎一定会引入新的问题——改了A段可能破坏B段的连贯性，"
                    f"补了一个数据可能口径和前文不一致，删了一句话可能让论证链断裂。"
                    f"这些'修改引入的新错误'比旧 issue 没修好危险得多，因为没有人专门盯过它们。\n\n"
                    f"所以：先用 read_file('draft.md') 通读完整正文，把它当作你第一次看到的文章来审。"
                    f"找出所有问题（包括新引入的），然后再回头检查上一轮的 issue：\n"
                    f"- Author 接受并修改的 issue → 修改是否真正解决了问题？\n"
                    f"- Author 反驳的 issue → 反驳理由是否成立？\n\n"
                    f"注意：以 read_file('draft.md') 读取的【当前内容】为唯一判断依据。"
                    f"{file_based_hint}"
                )
            else:
                prompt = (
                    f"请先用 read_file('draft.md') 读取需要审核的内容，然后仔细审核，找出其中的问题。"
                    f"{file_based_hint}"
                )
            if guidance:
                prompt = f"📋 主编指导意见（供审核参考）：{guidance}\n\n" + prompt

            # Call reviewer agent
            raw = await self._safe_agent_call(reviewer, prompt)
            if raw is None:
                return None

            # Parse feedback from file written by reviewer
            feedback_filename = f"feedback_R{round_num}_{reviewer.name}.md"
            if self._workspace_dir is not None:
                feedback_path = self._workspace_dir / feedback_filename
                if feedback_path.exists():
                    file_content = feedback_path.read_text()
                    issues = parse_feedback_file(file_content)
                    return ReviewerFeedback(reviewer_name=reviewer.name, issues=issues)

            # Fallback: parse content as string (backward compatibility)
            logger.warning(
                "Reviewer '%s' did not write feedback file, falling back to text parsing",
                reviewer.name,
            )
            return self._parse_reviewer_output(reviewer.name, raw)

        results = await asyncio.gather(*[call_reviewer(r) for r in self._reviewers])
        feedbacks = [r for r in results if r is not None]

        if not feedbacks:
            self._handle_runtime_error("所有审核员均失败")

        return feedbacks

    def _parse_reviewer_output(self, name: str, raw) -> ReviewerFeedback:
        """Parse reviewer output into ReviewerFeedback.

        Handles both structured output (ReviewerOutput pydantic object) and
        plain text/JSON string fallback.
        """
        # Handle structured output (Pydantic model from output_model) — kept for backward compat
        if isinstance(raw, ReviewerOutput):
            issues = [
                ReviewIssue(
                    severity=i.severity,
                    content=i.content,
                    why=i.why,
                    pattern=i.pattern,
                )
                for i in raw.issues
            ]
            return ReviewerFeedback(reviewer_name=name, issues=issues)

        # Fallback: parse as string
        raw = str(raw)
        try:
            # Try to find JSON with "issues" key (no nested objects before the key)
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
            issues.append(self._issue_from_dict(item))
        return ReviewerFeedback(reviewer_name=name, issues=issues)

    # ------------------------------------------------------------------
    # Author: Evaluate feedback (verdict only)
    # ------------------------------------------------------------------

    async def _author_evaluate_feedback(
        self,
        content: str,
        feedbacks: list[ReviewerFeedback],
        guidance: str | None = None,
        round_num: int = 0,
    ) -> list[AuthorVerdictItem]:
        """Author evaluates each issue and writes a verdict file.

        The Author writes verdict_R{round}.md with structured verdicts.
        If no verdict file is found, we fall back to JSON parsing from text.
        """
        issues_text = self._format_issues_for_author(feedbacks)
        guidance_prefix = ""
        if guidance:
            guidance_prefix = (
                f"⚠️ 主编指导意见：{guidance}\n"
                f"请在本轮修改中优先响应以上指导意见。\n\n"
            )
        verdict_filename = f"verdict_R{round_num}.md"
        prompt = (
            f"{guidance_prefix}"
            f"{self._config.author.receiving_review_prompt}\n\n"
            f"⚠️ 请先调用 read_file(\"draft.md\") 获取当前完整正文——draft.md 是所有审核和修改的唯一基准，其他审核轮可能已经修改过内容，凭记忆操作会基于过时版本。\n\n"
            f"审核员反馈：\n\n{issues_text}"
            f"{_SUBMIT_VERDICT_INSTRUCTION}"
            f"\n\n请将裁定写入 {verdict_filename}。"
        )

        raw = await self._safe_agent_call(self._author_verdict, prompt)
        if raw is None:
            return []

        # Strategy 1: Parse from verdict file
        if self._workspace_dir is not None:
            verdict_path = self._workspace_dir / verdict_filename
            if verdict_path.exists():
                file_content = verdict_path.read_text()
                verdicts = parse_verdict_file(file_content)
                if verdicts:
                    return verdicts

        # Strategy 2: Fall back to JSON parsing from text content
        return self._parse_verdict_response(raw)

    # ------------------------------------------------------------------
    # Author: Apply changes (revision only)
    # ------------------------------------------------------------------

    async def _author_apply_changes(
        self,
        content: str,
        verdicts: list[AuthorVerdictItem],
        feedbacks: list[ReviewerFeedback],
        round_num: int = 0,
    ) -> str:
        """Author applies accepted changes and saves to draft.md.

        The Author is expected to save the revised content to draft.md using
        save_file. If draft.md is not updated, we fall back to text response.
        """
        # Format verdicts for the Author prompt
        verdict_text = self._format_verdicts_for_author(verdicts, feedbacks)
        # Build concrete feedback file list for the Author
        feedback_files = [
            f"feedback_R{round_num}_{fb.reviewer_name}.md"
            for fb in feedbacks
            if fb.issues
        ]
        if feedback_files:
            file_list_hint = (
                "\n\n本轮审核员的反馈文件："
                + ", ".join(feedback_files)
                + "\n你可以用 read_file 读取这些文件获取详细反馈。"
            )
        else:
            file_list_hint = ""
        prompt = (
            f"⚠️ 请先调用 read_file(\"draft.md\") 获取当前完整正文——draft.md 是所有审核和修改的唯一基准，其他审核轮可能已经修改过内容，凭记忆操作会基于过时版本。\n\n"
            f"以下是你对审核意见的裁定：\n\n{verdict_text}\n\n"
            f"请根据你接受（accept）的意见修改内容，输出完整的修改后内容。"
            f"{_SUBMIT_REVISION_INSTRUCTION}"
            f"{file_list_hint}"
        )

        # Record draft.md mtime before agent call for stale-check
        draft_mtime_before: float | None = None
        if self._workspace_dir is not None:
            draft_path = self._workspace_dir / "draft.md"
            if draft_path.exists():
                draft_mtime_before = os.path.getmtime(draft_path)

        raw = await self._safe_agent_call(self._author_revision, prompt)
        if raw is None:
            return content

        # Strategy 1: Check if Author wrote a NEW draft.md via FileTools (mtime changed)
        if self._workspace_dir is not None:
            draft_path = self._workspace_dir / "draft.md"
            if draft_path.exists():
                draft_mtime_after = os.path.getmtime(draft_path)
                if draft_mtime_before is None or draft_mtime_after > draft_mtime_before:
                    draft = draft_path.read_text()
                    if len(draft.strip()) > 500:
                        return draft

        # Strategy 2: Fall back to text content
        self._write_draft_to_workspace(raw)
        return raw

    def _format_issues_for_author(self, feedbacks: list[ReviewerFeedback]) -> str:
        parts: list[str] = []
        for fb in feedbacks:
            if not fb.issues:
                continue
            parts.append(f"[{fb.reviewer_name}]")
            for i, issue in enumerate(fb.issues):
                parts.append(f"  issue {i} ({issue.severity}): {issue.content}")
                self._append_why_pattern(parts, issue)
        return "\n".join(parts)

    def _format_verdicts_for_author(
        self,
        verdicts: list[AuthorVerdictItem],
        feedbacks: list[ReviewerFeedback],
    ) -> str:
        """Format verdicts alongside original issues for the revision prompt."""
        verdict_map: dict[tuple[str, int], AuthorVerdictItem] = {}
        for item in verdicts:
            verdict_map[(item.reviewer, item.issue_index)] = item

        parts: list[str] = []
        for fb in feedbacks:
            if not fb.issues:
                continue
            parts.append(f"[{fb.reviewer_name}]")
            for i, issue in enumerate(fb.issues):
                parts.append(f"  issue {i} ({issue.severity}): {issue.content}")
                self._append_why_pattern(parts, issue)
                v = verdict_map.get((fb.reviewer_name, i))
                if v:
                    parts.append(f"  -> 裁定: [{v.verdict.upper()}] {v.reason}")
                else:
                    parts.append(f"  -> 裁定: [未回应]")
        return "\n".join(parts)

    def _parse_verdict_response(self, raw: str) -> list[AuthorVerdictItem]:
        """Parse author verdict JSON output into list of AuthorVerdictItem."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Author returned non-JSON verdict output")
            return []

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("verdicts", data.get("responses", []))
        else:
            return []

        responses = []
        for item in items:
            if isinstance(item, dict):
                responses.append(
                    AuthorVerdictItem(
                        reviewer=item.get("reviewer", ""),
                        issue_index=item.get("issue_index", 0),
                        verdict=item.get("verdict", "unclear"),
                        reason=item.get("reason", ""),
                    )
                )
        return responses

    def _parse_author_response(self, raw: str, fallback_content: str) -> AuthorResponse:
        """Parse author JSON output into AuthorResponse (kept for backward compat)."""
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
        verdicts: list[AuthorVerdictItem],
    ) -> dict[str, str]:
        """Build context for each reviewer: their issues + Author's responses."""
        # Index author verdicts by (reviewer, issue_index)
        verdict_map: dict[tuple[str, int], AuthorVerdictItem] = {}
        for item in verdicts:
            verdict_map[(item.reviewer, item.issue_index)] = item

        ctx: dict[str, str] = {}
        for fb in feedbacks:
            if not fb.issues:
                continue
            parts: list[str] = [f"[上一轮你提出的 issues 及 Author 的回应]"]
            for i, issue in enumerate(fb.issues):
                parts.append(f"\nissue {i} ({issue.severity}): {issue.content}")
                self._append_why_pattern(parts, issue)
                verdict_item = verdict_map.get((fb.reviewer_name, i))
                if verdict_item:
                    tag = verdict_item.verdict.upper()
                    parts.append(f"Author 回应: [{tag}] {verdict_item.reason}")
                    parts.append(_verdict_hint(verdict_item.verdict))
                else:
                    parts.append(f"Author 回应: [未回应]")
                    parts.append(_verdict_hint(None))
            ctx[fb.reviewer_name] = "\n".join(parts)

        return ctx

    # ------------------------------------------------------------------
    # Rebuild reviewer context from archived history
    # ------------------------------------------------------------------

    def _rebuild_reviewer_ctx_from_history(
        self,
        last_round: dict,
    ) -> dict[str, str]:
        """Rebuild per-reviewer context from the last archived round.

        Converts raw archived dicts back into the same context format that
        _build_reviewer_context produces, so reviewers see their prior
        issues and the Author's verdicts on resume.
        """
        feedbacks_raw = last_round.get("reviewer_feedbacks", {})
        verdict_raw = last_round.get("verdict") or []

        # Build verdict map: (reviewer, issue_index) -> verdict dict
        verdict_map: dict[tuple[str, int], dict] = {}
        for v in verdict_raw:
            if isinstance(v, dict):
                key = (v.get("reviewer", ""), v.get("issue_index", 0))
                verdict_map[key] = v

        ctx: dict[str, str] = {}
        for reviewer_name, fb_data in feedbacks_raw.items():
            issues = fb_data.get("issues", []) if isinstance(fb_data, dict) else []
            if not issues:
                continue
            parts: list[str] = ["[上一轮你提出的 issues 及 Author 的回应]"]
            for i, issue in enumerate(issues):
                sev = issue.get("severity", "minor") if isinstance(issue, dict) else "minor"
                content = issue.get("content", "") if isinstance(issue, dict) else ""
                parts.append(f"\nissue {i} ({sev}): {content}")
                if isinstance(issue, dict):
                    if issue.get("why"):
                        parts.append(f"    why: {issue['why']}")
                    if issue.get("pattern"):
                        parts.append(f"    pattern: {issue['pattern']}")
                v = verdict_map.get((reviewer_name, i))
                if v:
                    tag = v.get("verdict", "unclear").upper()
                    parts.append(f"Author 回应: [{tag}] {v.get('reason', '')}")
                    parts.append(_verdict_hint(v.get("verdict", "unclear")))
                else:
                    parts.append("Author 回应: [未回应]")
                    parts.append(_verdict_hint(None))
            ctx[reviewer_name] = "\n".join(parts)

        return ctx

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        initial_content: str | None = None,
        context: str | None = None,
        resume_path: str | None = None,
        extra_rounds: int | None = None,
        guidance: str | None = None,
    ) -> ReviewResult:
        """Run the full write-review loop."""

        # Resume mode: reload existing session
        if resume_path is not None:
            if not extra_rounds or extra_rounds < 1:
                raise ValueError(
                    "extra_rounds must be a positive integer when resuming"
                )
            session_path = self._archiver.resume_session(resume_path)
            self._audit = AuditLogger(session_path)
            self._workspace_dir = self._archiver.workspace_dir
            self._setup_file_tools()
            history = self._archiver.load_history()
            if not history:
                raise ValueError(
                    "No rounds found in archive; nothing to resume from"
                )
            start_round = len(history) + 1
            max_rounds = len(history) + extra_rounds

            # Rebuild state: latest author content
            last = history[-1]
            content = last["author_content"]
            # If last round had a response with updated_content, use that
            if last.get("response") and last["response"].get("updated_content"):
                content = last["response"]["updated_content"]

            # Rebuild per_reviewer_ctx from last round's feedbacks and verdicts
            per_reviewer_ctx = self._rebuild_reviewer_ctx_from_history(last)

            # Write current content to workspace/draft.md for file-based agents
            self._write_draft_to_workspace(content)

        else:
            # Fresh start
            session_path = self._archiver.start_session(self._config)
            self._audit = AuditLogger(session_path)
            self._workspace_dir = self._archiver.workspace_dir
            self._setup_file_tools()

            # Build or load context
            if context is not None:
                ctx = context
            else:
                ctx = await self._context_mgr.build_initial_context()
            self._archiver.save_context(ctx)

            start_round = 1
            max_rounds = self._config.max_rounds

            # Get initial content
            if initial_content is not None:
                content = initial_content
                self._write_draft_to_workspace(content)
            else:
                content = await self._author_generate(ctx)

            self._archiver.save_author_content(1, content)

            per_reviewer_ctx: dict[str, str] = {}

        rounds_completed: int = 0

        try:
            for round_num in range(start_round, max_rounds + 1):
                # Reviewers audit in parallel
                feedbacks = await self._review(content, per_reviewer_ctx, guidance=guidance, round_num=round_num)

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

                # Author evaluates feedback (verdict only)
                verdicts = await self._author_evaluate_feedback(content, feedbacks, guidance=guidance, round_num=round_num)

                self._archiver.save_author_verdict(
                    round_num,
                    [asdict(v) for v in verdicts],
                )

                # Author applies changes (revision only)
                updated_content = await self._author_apply_changes(content, verdicts, feedbacks, round_num=round_num)

                self._archiver.save_author_response(
                    round_num,
                    {"updated_content": updated_content},
                )

                rounds_completed = round_num

                # Update content and prepare next round
                content = updated_content
                self._write_draft_to_workspace(content)
                self._archiver.save_author_content(round_num + 1, content)

                # Build per-reviewer context for next round
                per_reviewer_ctx = self._build_reviewer_context(feedbacks, verdicts)

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
                rounds_completed=max_rounds,
                archive_path=session_path,
                final_content=content,
                unresolved_issues=unresolved,
            )

        except RuntimeError as exc:
            # error_callback already invoked by _handle_runtime_error if applicable
            if self._archiver._session_dir is not None:
                self._archiver.save_error_log(str(exc))
            return ReviewResult(
                converged=False,
                rounds_completed=rounds_completed,
                archive_path=session_path,
                final_content=None,
                unresolved_issues=[],
                terminated_by_error=True,
            )

        finally:
            if self._audit:
                self._audit.close()
            # Generate usage summary after all rounds complete
            try:
                generate_usage_summary(
                    session_path,
                    model_name=self._config.model_config.model,
                    total_rounds=rounds_completed,
                )
            except Exception:
                logger.warning("Failed to generate usage summary", exc_info=True)
