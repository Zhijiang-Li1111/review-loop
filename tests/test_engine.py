"""Tests for ReviewEngine in review_loop.engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from review_loop.config import (
    AuthorConfig,
    ModelConfig,
    ReviewConfig,
    ReviewerConfig,
    ToolConfig,
)


class MockRunOutput:
    """Simulates agno RunOutput with content, tools, and messages."""
    def __init__(self, content: str | None = None, tools=None, messages=None):
        self.content = content
        self.tools = tools or []
        self.messages = messages or []


def _make_config(
    max_rounds: int = 3,
    num_reviewers: int = 2,
    tools: list[ToolConfig] | None = None,
    context_builder: str | None = None,
) -> ReviewConfig:
    reviewers = []
    for i in range(num_reviewers):
        reviewers.append(
            ReviewerConfig(
                name=f"Reviewer-{chr(65 + i)}",
                system_prompt=f"You are reviewer {chr(65 + i)}.",
            )
        )
    return ReviewConfig(
        max_rounds=max_rounds,
        model_config=ModelConfig(model="claude-opus-4.6-1m"),
        author=AuthorConfig(
            name="Author",
            system_prompt="You are an author.",
            receiving_review_prompt="Process feedback carefully.",
        ),
        reviewers=reviewers,
        tools=tools or [],
        context={},
        context_builder=context_builder,
    )


# ---------------------------------------------------------------------------
# Agent Creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_creates_author_and_reviewers(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=3)
        engine = ReviewEngine(config)

        # 2 Author (verdict + revision) + 3 Reviewers = 5 Agent calls
        assert MockAgent.call_count == 5
        calls = MockAgent.call_args_list
        assert calls[0].kwargs["name"] == "Author"  # verdict agent
        assert calls[1].kwargs["name"] == "Author"  # revision agent
        reviewer_names = [c.kwargs["name"] for c in calls[2:]]
        assert reviewer_names == ["Reviewer-A", "Reviewer-B", "Reviewer-C"]

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_reviewers_have_no_submit_review_tool(self, MockAgent, MockCtxMgr, mock_import):
        """Reviewers should NOT have submit_review tool (file-based now)."""
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=2)
        engine = ReviewEngine(config)

        # Author verdict agent (first call) should have no tools (no external tools configured)
        author_verdict_call = MockAgent.call_args_list[0]
        verdict_tools = author_verdict_call.kwargs.get("tools", [])
        assert len(verdict_tools) == 0

        # Author revision agent (second call) should have no tools
        author_revision_call = MockAgent.call_args_list[1]
        revision_tools = author_revision_call.kwargs.get("tools", [])
        assert len(revision_tools) == 0

        # Reviewers should have no tools (no per-reviewer tools configured)
        for call in MockAgent.call_args_list[2:]:
            tools = call.kwargs.get("tools", [])
            assert len(tools) == 0

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_reviewer_system_prompt_has_feedback_instruction(self, MockAgent, MockCtxMgr, mock_import):
        """Reviewer system prompt should include file-based feedback instruction."""
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        reviewer_call = MockAgent.call_args_list[2]
        system_msg = reviewer_call.kwargs["system_message"]
        assert "feedback" in system_msg.lower() or "save_file" in system_msg
        assert "why" in system_msg
        assert "pattern" in system_msg

    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_author_gets_external_tools(self, MockAgent, MockCtxMgr):
        from review_loop.engine import ReviewEngine

        class FakeTool:
            def __init__(self, context=None):
                pass

        with patch("review_loop.engine.import_from_path", return_value=FakeTool):
            config = _make_config(tools=[ToolConfig(path="pkg.FakeTool")])
            engine = ReviewEngine(config)

        # Verdict agent (first call) should have only FakeTool (no submit_verdict)
        verdict_call = MockAgent.call_args_list[0]
        assert verdict_call.kwargs.get("tools") is not None
        assert len(verdict_call.kwargs["tools"]) == 1  # FakeTool only

        # Revision agent (second call) should have only FakeTool (no submit_revision)
        revision_call = MockAgent.call_args_list[1]
        assert revision_call.kwargs.get("tools") is not None
        assert len(revision_call.kwargs["tools"]) == 1  # FakeTool only

    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_reviewer_gets_per_reviewer_tools(self, MockAgent, MockCtxMgr):
        """Per-reviewer tools work without submit_review."""
        from review_loop.engine import ReviewEngine

        class FakeAuthorTool:
            def __init__(self, context=None):
                self.tag = "author"

        class FakeReviewerTool:
            def __init__(self, context=None):
                self.tag = "reviewer"

        def fake_import(path):
            if path == "pkg.AuthorTool":
                return FakeAuthorTool
            return FakeReviewerTool

        reviewers = [
            ReviewerConfig(
                name="Reviewer-A",
                system_prompt="You are reviewer A.",
                tools=[ToolConfig(path="pkg.ReviewerTool")],
            ),
            ReviewerConfig(
                name="Reviewer-B",
                system_prompt="You are reviewer B.",
            ),
        ]
        config = ReviewConfig(
            max_rounds=3,
            model_config=ModelConfig(model="claude-opus-4.6-1m"),
            author=AuthorConfig(
                name="Author",
                system_prompt="You are an author.",
                receiving_review_prompt="Process feedback.",
            ),
            reviewers=reviewers,
            tools=[ToolConfig(path="pkg.AuthorTool")],
            context={},
        )

        with patch("review_loop.engine.import_from_path", side_effect=fake_import):
            engine = ReviewEngine(config)

        # Agent calls: Author-verdict, Author-revision, Reviewer-A, Reviewer-B
        assert MockAgent.call_count == 4

        # Reviewer-A should have only per-reviewer tool (no submit_review)
        reviewer_a_call = MockAgent.call_args_list[2]
        assert reviewer_a_call.kwargs["name"] == "Reviewer-A"
        tools_a = reviewer_a_call.kwargs["tools"]
        assert len(tools_a) == 1  # FakeReviewerTool only
        assert tools_a[0].tag == "reviewer"

        # Reviewer-B should have no tools
        reviewer_b_call = MockAgent.call_args_list[3]
        assert reviewer_b_call.kwargs["name"] == "Reviewer-B"
        tools_b = reviewer_b_call.kwargs["tools"]
        assert len(tools_b) == 0


# ---------------------------------------------------------------------------
# Review Phase (file-based)
# ---------------------------------------------------------------------------


class TestReviewPhase:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_reviewers_write_feedback_files(self, MockCtxMgr, mock_import):
        """Reviewers submit feedback via files parsed by engine."""
        from review_loop.engine import ReviewEngine

        mock_reviewers = [MagicMock(), MagicMock()]
        mock_reviewers[0].name = "Reviewer-A"
        mock_reviewers[1].name = "Reviewer-B"
        mock_author_verdict = MagicMock(name="Author-verdict")
        mock_author_verdict.name = "Author"
        mock_author_revision = MagicMock(name="Author-revision")
        mock_author_revision.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author_verdict, mock_author_revision] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            engine._workspace_dir = workspace
            (workspace / "draft.md").write_text("content here")

            async def mock_safe_call(agent, prompt):
                if agent.name == "Reviewer-A":
                    (workspace / "feedback_R1_Reviewer-A.md").write_text(
                        "## Issue 1\n- severity: critical\n- content: Logic gap\n- why: Missing step\n- pattern: Check all logic\n"
                    )
                else:
                    (workspace / "feedback_R1_Reviewer-B.md").write_text(
                        "## No Issues\n审核通过。\n"
                    )
                return "done"

            with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
                feedbacks = await engine._review("Content v1", {}, round_num=1)

        assert len(feedbacks) == 2
        fb_a = next(f for f in feedbacks if f.reviewer_name == "Reviewer-A")
        fb_b = next(f for f in feedbacks if f.reviewer_name == "Reviewer-B")
        assert len(fb_a.issues) == 1
        assert fb_a.issues[0].severity == "critical"
        assert fb_a.issues[0].content == "Logic gap"
        assert len(fb_b.issues) == 0

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_review_fallback_to_string_parsing(self, MockCtxMgr, mock_import):
        """When no feedback file, fall back to string JSON parsing."""
        from review_loop.engine import ReviewEngine

        mock_reviewers = [MagicMock()]
        mock_reviewers[0].name = "Reviewer-A"
        mock_author_verdict = MagicMock(name="Author-verdict")
        mock_author_verdict.name = "Author"
        mock_author_revision = MagicMock(name="Author-revision")
        mock_author_revision.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author_verdict, mock_author_revision] + mock_reviewers):
            config = _make_config(num_reviewers=1)
            engine = ReviewEngine(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            engine._workspace_dir = workspace
            (workspace / "draft.md").write_text("content here")

            # No feedback file written — just returns JSON text
            async def mock_safe_call(agent, prompt):
                return '{"issues": [{"severity": "major", "content": "Bad data"}]}'

            with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
                feedbacks = await engine._review("Content v1", {}, round_num=1)

        assert len(feedbacks) == 1
        assert len(feedbacks[0].issues) == 1
        assert feedbacks[0].issues[0].severity == "major"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_reviewer_rebuttal_only_sees_own_issues(self, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        mock_reviewers = [MagicMock(), MagicMock()]
        mock_reviewers[0].name = "Reviewer-A"
        mock_reviewers[1].name = "Reviewer-B"
        mock_author_verdict = MagicMock(name="Author-verdict")
        mock_author_verdict.name = "Author"
        mock_author_revision = MagicMock(name="Author-revision")
        mock_author_revision.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author_verdict, mock_author_revision] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            engine._workspace_dir = workspace
            (workspace / "draft.md").write_text("content here")

            captured_prompts = {}

            async def mock_safe_call(agent, prompt):
                captured_prompts[agent.name] = prompt
                (workspace / f"feedback_R1_{agent.name}.md").write_text(
                    "## No Issues\n审核通过。\n"
                )
                return "done"

            per_reviewer_ctx = {
                "Reviewer-A": "Issue A context only",
                "Reviewer-B": "Issue B context only",
            }

            with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
                await engine._review("Content v2", per_reviewer_ctx, round_num=1)

        assert "Issue A context only" in captured_prompts["Reviewer-A"]
        assert "Issue B context only" not in captured_prompts["Reviewer-A"]
        assert "Issue B context only" in captured_prompts["Reviewer-B"]
        assert "Issue A context only" not in captured_prompts["Reviewer-B"]


# ---------------------------------------------------------------------------
# Author Feedback Processing
# ---------------------------------------------------------------------------


class TestAuthorFeedback:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_evaluate_feedback_via_file(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        from review_loop.models import ReviewerFeedback, ReviewIssue

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="critical", content="Missing step")],
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            engine._workspace_dir = workspace

            async def mock_safe_call(agent, prompt):
                (workspace / "verdict_R1.md").write_text(
                    "## Issue 0 (Reviewer-A)\n- verdict: accept\n- reason: Fixed in v2\n"
                )
                return "done"

            with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
                verdicts = await engine._author_evaluate_feedback("Content v1", feedbacks, round_num=1)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == "accept"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_apply_changes_via_draft(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        from review_loop.models import AuthorVerdictItem, ReviewerFeedback, ReviewIssue

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="critical", content="Missing step")],
            )
        ]
        verdicts = [
            AuthorVerdictItem(
                reviewer="Reviewer-A", issue_index=0,
                verdict="accept", reason="Fixed in v2",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            engine._workspace_dir = workspace
            new_draft = "Content v2 with fix applied " * 20

            async def mock_safe_call(agent, prompt):
                (workspace / "draft.md").write_text(new_draft)
                return "done"

            with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
                updated = await engine._author_apply_changes("Content v1", verdicts, feedbacks)

        assert "Content v2 with fix applied" in updated

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_evaluate_prompt_includes_receiving_review(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        from review_loop.models import ReviewerFeedback, ReviewIssue

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="minor", content="Typo")],
            )
        ]

        captured_prompts = []

        async def mock_safe_call(agent, prompt):
            captured_prompts.append(prompt)
            return "done"

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            await engine._author_evaluate_feedback("v1", feedbacks, round_num=1)

        assert "Process feedback carefully." in captured_prompts[0]
        assert "verdict" in captured_prompts[0].lower()


# ---------------------------------------------------------------------------
# Build Per-Reviewer Context
# ---------------------------------------------------------------------------


class TestBuildReviewerContext:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_builds_per_reviewer_context(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        from review_loop.models import (
            AuthorVerdictItem,
            ReviewerFeedback,
            ReviewIssue,
        )

        config = _make_config(num_reviewers=2)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="critical", content="Gap in logic")],
            ),
            ReviewerFeedback(
                reviewer_name="Reviewer-B",
                issues=[ReviewIssue(severity="minor", content="Typo")],
            ),
        ]

        verdicts = [
            AuthorVerdictItem(
                reviewer="Reviewer-A", issue_index=0,
                verdict="reject", reason="Logic is sound because X",
            ),
            AuthorVerdictItem(
                reviewer="Reviewer-B", issue_index=0,
                verdict="accept", reason="Fixed typo",
            ),
        ]

        ctx = engine._build_reviewer_context(feedbacks, verdicts)

        assert "Reviewer-A" in ctx
        assert "Reviewer-B" in ctx
        # Each reviewer should only see their own issues
        assert "Gap in logic" in ctx["Reviewer-A"]
        assert "Typo" not in ctx["Reviewer-A"]
        assert "Typo" in ctx["Reviewer-B"]
        assert "Gap in logic" not in ctx["Reviewer-B"]
        # Author rebuttals should be included
        assert "REJECT" in ctx["Reviewer-A"]
        assert "ACCEPT" in ctx["Reviewer-B"]


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_single_reviewer_failure_non_fatal(self, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        mock_reviewers = [MagicMock(), MagicMock()]
        mock_reviewers[0].name = "Reviewer-A"
        mock_reviewers[1].name = "Reviewer-B"
        mock_author_verdict = MagicMock(name="Author-verdict")
        mock_author_verdict.name = "Author"
        mock_author_revision = MagicMock(name="Author-revision")
        mock_author_revision.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author_verdict, mock_author_revision] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            engine._workspace_dir = workspace
            (workspace / "draft.md").write_text("content")

            async def mock_safe_call(agent, prompt):
                if agent.name == "Reviewer-A":
                    return None  # failure
                (workspace / "feedback_R0_Reviewer-B.md").write_text(
                    "## No Issues\n审核通过。\n"
                )
                return "done"

            with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
                feedbacks = await engine._review("Content", {})

        assert len(feedbacks) == 1
        assert feedbacks[0].reviewer_name == "Reviewer-B"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_all_reviewers_fail_raises(self, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        mock_reviewers = [MagicMock(), MagicMock()]
        mock_reviewers[0].name = "Reviewer-A"
        mock_reviewers[1].name = "Reviewer-B"
        mock_author_verdict = MagicMock(name="Author-verdict")
        mock_author_verdict.name = "Author"
        mock_author_revision = MagicMock(name="Author-revision")
        mock_author_revision.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author_verdict, mock_author_revision] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        async def mock_safe_call(agent, prompt):
            return None

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            with pytest.raises(RuntimeError, match="所有审核员均失败"):
                await engine._review("Content", {})


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------


class TestMainLoop:
    def _setup_engine(self, config):
        """Create engine with all external deps mocked."""
        from review_loop.engine import ReviewEngine

        with (
            patch("review_loop.engine.import_from_path"),
            patch("review_loop.engine.ContextManager"),
            patch("review_loop.engine.Agent"),
        ):
            engine = ReviewEngine(config)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="Initial context")
        return engine

    @pytest.mark.asyncio
    async def test_converges_when_all_issues_empty(self):
        from review_loop.engine import ReviewEngine
        from review_loop.models import ReviewerFeedback

        config = _make_config(max_rounds=10, num_reviewers=2)
        engine = self._setup_engine(config)

        # Author generates v1
        engine._author_generate = AsyncMock(return_value="Generated content")

        # Reviewers return no issues on first round
        engine._review = AsyncMock(return_value=[
            ReviewerFeedback(reviewer_name="Reviewer-A", issues=[]),
            ReviewerFeedback(reviewer_name="Reviewer-B", issues=[]),
        ])

        result = await engine.run()

        assert result.converged is True
        assert result.rounds_completed == 1
        assert result.final_content == "Generated content"

    @pytest.mark.asyncio
    async def test_converges_after_feedback_round(self):
        from review_loop.engine import ReviewEngine
        from review_loop.models import (
            AuthorVerdictItem,
            ReviewerFeedback,
            ReviewIssue,
        )

        config = _make_config(max_rounds=10, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="v1 content")

        round_counter = {"n": 0}

        async def mock_review(content, per_reviewer_ctx, **kwargs):
            round_counter["n"] += 1
            if round_counter["n"] == 1:
                return [
                    ReviewerFeedback(
                        reviewer_name="Reviewer-A",
                        issues=[ReviewIssue(severity="critical", content="Gap")],
                    )
                ]
            return [ReviewerFeedback(reviewer_name="Reviewer-A", issues=[])]

        engine._review = AsyncMock(side_effect=mock_review)
        engine._author_evaluate_feedback = AsyncMock(
            return_value=[
                AuthorVerdictItem(
                    reviewer="Reviewer-A", issue_index=0,
                    verdict="accept", reason="Fixed",
                )
            ]
        )
        engine._author_apply_changes = AsyncMock(return_value="v2 content")

        result = await engine.run()

        assert result.converged is True
        assert result.rounds_completed == 2
        assert result.final_content == "v2 content"

    @pytest.mark.asyncio
    async def test_max_rounds_enforced(self):
        from review_loop.engine import ReviewEngine
        from review_loop.models import (
            ReviewerFeedback,
            ReviewIssue,
        )

        config = _make_config(max_rounds=2, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="v1")

        # Always returns issues (never converges)
        engine._review = AsyncMock(return_value=[
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="critical", content="Still bad")],
            )
        ])
        engine._author_evaluate_feedback = AsyncMock(return_value=[])
        engine._author_apply_changes = AsyncMock(return_value="still trying")

        result = await engine.run()

        assert result.converged is False
        assert result.rounds_completed == 2
        assert len(result.unresolved_issues) > 0

    @pytest.mark.asyncio
    async def test_uses_initial_content_when_provided(self):
        from review_loop.engine import ReviewEngine
        from review_loop.models import ReviewerFeedback

        config = _make_config(max_rounds=10)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock()
        engine._review = AsyncMock(return_value=[
            ReviewerFeedback(reviewer_name="Reviewer-A", issues=[]),
            ReviewerFeedback(reviewer_name="Reviewer-B", issues=[]),
        ])

        result = await engine.run(initial_content="Pre-written draft")

        engine._author_generate.assert_not_called()
        assert result.final_content == "Pre-written draft"

    @pytest.mark.asyncio
    async def test_uses_context_file_when_provided(self):
        from review_loop.engine import ReviewEngine
        from review_loop.models import ReviewerFeedback

        config = _make_config(max_rounds=10)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="Generated")
        engine._review = AsyncMock(return_value=[
            ReviewerFeedback(reviewer_name="Reviewer-A", issues=[]),
            ReviewerFeedback(reviewer_name="Reviewer-B", issues=[]),
        ])

        result = await engine.run(context="Direct context string")

        # Should use provided context, not build from context_builder
        engine._context_mgr.build_initial_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_termination(self):
        from review_loop.engine import ReviewEngine

        config = _make_config(max_rounds=10)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="v1")
        engine._review = AsyncMock(side_effect=RuntimeError("All failed"))

        result = await engine.run()

        assert result.terminated_by_error is True
        assert result.converged is False
        engine._archiver.save_error_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_persistence_calls(self):
        from review_loop.engine import ReviewEngine
        from review_loop.models import ReviewerFeedback, ReviewIssue

        config = _make_config(max_rounds=10, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="v1 content")

        # Round 1: issue found, round 2: no issues
        round_counter = {"n": 0}

        async def mock_review(content, ctx, **kwargs):
            round_counter["n"] += 1
            if round_counter["n"] == 1:
                return [ReviewerFeedback(
                    reviewer_name="Reviewer-A",
                    issues=[ReviewIssue(severity="minor", content="Tweak")],
                )]
            return [ReviewerFeedback(reviewer_name="Reviewer-A", issues=[])]

        engine._review = AsyncMock(side_effect=mock_review)

        from review_loop.models import AuthorVerdictItem
        engine._author_evaluate_feedback = AsyncMock(
            return_value=[AuthorVerdictItem(
                reviewer="Reviewer-A", issue_index=0,
                verdict="accept", reason="Fixed",
            )]
        )
        engine._author_apply_changes = AsyncMock(return_value="v2 content")

        result = await engine.run()

        # Check persistence was called
        engine._archiver.save_context.assert_called_once()
        assert engine._archiver.save_author_content.call_count >= 1
        assert engine._archiver.save_reviewer_feedback.call_count >= 1
        assert engine._archiver.save_author_verdict.call_count >= 1
        engine._archiver.save_final.assert_called_once()


# ---------------------------------------------------------------------------
# Parsing (LLM output handling - backward compatibility)
# ---------------------------------------------------------------------------


class TestParsing:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_reviewer_valid_json(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config()
        engine = ReviewEngine(config)

        raw = '{"issues": [{"severity": "critical", "content": "Logic gap"}]}'
        fb = engine._parse_reviewer_output("R1", raw)
        assert len(fb.issues) == 1
        assert fb.issues[0].severity == "critical"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_reviewer_empty_issues(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config()
        engine = ReviewEngine(config)

        raw = '{"issues": []}'
        fb = engine._parse_reviewer_output("R1", raw)
        assert fb.issues == []

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_reviewer_malformed_returns_empty(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config()
        engine = ReviewEngine(config)

        raw = "This is not JSON at all"
        fb = engine._parse_reviewer_output("R1", raw)
        assert fb.issues == []

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_reviewer_json_in_markdown(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config()
        engine = ReviewEngine(config)

        raw = '```json\n{"issues": [{"severity": "minor", "content": "Typo"}]}\n```'
        fb = engine._parse_reviewer_output("R1", raw)
        assert len(fb.issues) == 1

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_author_response_valid(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config()
        engine = ReviewEngine(config)

        raw = json.dumps({
            "responses": [{"reviewer": "R1", "issue_index": 0, "verdict": "accept", "reason": "Fixed"}],
            "updated_content": "New content",
        })
        resp = engine._parse_author_response(raw, "fallback")
        assert resp.updated_content == "New content"
        assert len(resp.responses) == 1

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_author_response_malformed_returns_fallback(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config()
        engine = ReviewEngine(config)

        raw = "Not JSON"
        resp = engine._parse_author_response(raw, "fallback content")
        assert resp.updated_content == "fallback content"
        assert resp.responses == []

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_reviewer_missing_severity_defaults_minor(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config()
        engine = ReviewEngine(config)

        raw = '{"issues": [{"content": "No severity field"}]}'
        fb = engine._parse_reviewer_output("R1", raw)
        assert len(fb.issues) == 1
        assert fb.issues[0].severity == "minor"


# ---------------------------------------------------------------------------
# Structured Output (Pydantic model from output_schema - backward compat)
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    """Tests for ReviewerOutput (Pydantic model) handling — kept for backward compat."""

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_reviewer_output_pydantic_instance(self, MockAgent, MockCtxMgr, mock_import):
        """String parsing still handles ReviewerOutput Pydantic models."""
        from review_loop.engine import ReviewEngine, ReviewerOutput, ReviewIssueOutput

        config = _make_config()
        engine = ReviewEngine(config)

        pydantic_output = ReviewerOutput(issues=[
            ReviewIssueOutput(severity="critical", content="Missing validation"),
            ReviewIssueOutput(severity="minor", content="Style issue"),
        ])
        fb = engine._parse_reviewer_output("R1", pydantic_output)
        assert len(fb.issues) == 2
        assert fb.issues[0].severity == "critical"
        assert fb.issues[0].content == "Missing validation"
        assert fb.issues[1].severity == "minor"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_reviewer_output_pydantic_empty_issues(self, MockAgent, MockCtxMgr, mock_import):
        """Pydantic model with no issues -> empty issues list."""
        from review_loop.engine import ReviewEngine, ReviewerOutput

        config = _make_config()
        engine = ReviewEngine(config)

        pydantic_output = ReviewerOutput(issues=[])
        fb = engine._parse_reviewer_output("R1", pydantic_output)
        assert fb.issues == []
        assert fb.reviewer_name == "R1"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_parse_reviewer_output_fallback_string(self, MockAgent, MockCtxMgr, mock_import):
        """When output_schema fails (model returns string), fallback JSON parsing works."""
        from review_loop.engine import ReviewEngine

        config = _make_config()
        engine = ReviewEngine(config)

        # String fallback still works
        raw = '{"issues": [{"severity": "major", "content": "Bad logic"}]}'
        fb = engine._parse_reviewer_output("R1", raw)
        assert len(fb.issues) == 1
        assert fb.issues[0].severity == "major"


class TestReviewerPromptTemplateExpansion:
    """Template variables in reviewer system_prompt get expanded at init."""

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_author_system_prompt_injected_into_reviewer(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = ReviewConfig(
            max_rounds=3,
            model_config=ModelConfig(model="claude-opus-4.6-1m"),
            author=AuthorConfig(
                name="Author",
                system_prompt="Author rules: write hooks, use frameworks.",
                receiving_review_prompt="Process feedback.",
            ),
            reviewers=[
                ReviewerConfig(
                    name="Checker",
                    system_prompt="Check against: {{author.system_prompt}}",
                ),
                ReviewerConfig(
                    name="Editor",
                    system_prompt="You are an editor.",
                ),
            ],
            tools=[],
            context={},
        )
        ReviewEngine(config)

        # Agent is called 4 times: 2 author (verdict + revision) + 2 reviewers
        assert MockAgent.call_count == 4

        # The Checker reviewer should have the author prompt expanded
        checker_call = MockAgent.call_args_list[2]
        assert checker_call.kwargs["name"] == "Checker"
        assert "Author rules: write hooks, use frameworks." in checker_call.kwargs["system_message"]
        assert "{{author.system_prompt}}" not in checker_call.kwargs["system_message"]

        # The Editor reviewer should have base prompt + feedback instruction
        editor_call = MockAgent.call_args_list[3]
        assert editor_call.kwargs["name"] == "Editor"
        assert "You are an editor." in editor_call.kwargs["system_message"]

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_no_template_variable_leaves_prompt_base_unchanged(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=2)
        ReviewEngine(config)

        # Default reviewers have no template variables — base prompts should be present
        reviewer_a_call = MockAgent.call_args_list[2]
        assert "You are reviewer A." in reviewer_a_call.kwargs["system_message"]
        reviewer_b_call = MockAgent.call_args_list[3]
        assert "You are reviewer B." in reviewer_b_call.kwargs["system_message"]


# ==================================================================
# Tests for why and pattern fields
# ==================================================================


class TestReviewIssueOutputSchema:
    """Verify ReviewIssueOutput Pydantic model includes why and pattern."""

    def test_schema_contains_why_and_pattern(self):
        from review_loop.engine import ReviewIssueOutput

        schema = ReviewIssueOutput.model_json_schema()
        props = schema["properties"]
        assert "why" in props
        assert "pattern" in props

    def test_defaults(self):
        from review_loop.engine import ReviewIssueOutput

        issue = ReviewIssueOutput(severity="minor", content="Typo")
        assert issue.why == ""
        assert issue.pattern == ""

    def test_explicit_values(self):
        from review_loop.engine import ReviewIssueOutput

        issue = ReviewIssueOutput(
            severity="major",
            content="Missing source",
            why="Violates evidence principle",
            pattern="Check all statistics for source",
        )
        assert issue.why == "Violates evidence principle"
        assert issue.pattern == "Check all statistics for source"


class TestWhyPatternSerialization:
    """Test that why and pattern survive JSON serialization (dataclasses.asdict)."""

    def test_serialization_includes_why_pattern(self):
        import dataclasses

        from review_loop.models import ReviewerFeedback, ReviewIssue

        fb = ReviewerFeedback(
            reviewer_name="R1",
            issues=[
                ReviewIssue(
                    severity="critical",
                    content="Logic gap",
                    why="Conclusion doesn't follow from premise",
                    pattern="Check all if-then arguments",
                )
            ],
        )
        data = dataclasses.asdict(fb)
        issue_dict = data["issues"][0]
        assert issue_dict["why"] == "Conclusion doesn't follow from premise"
        assert issue_dict["pattern"] == "Check all if-then arguments"

    def test_serialization_empty_defaults(self):
        import dataclasses

        from review_loop.models import ReviewIssue

        issue = ReviewIssue(severity="minor", content="Typo")
        d = dataclasses.asdict(issue)
        assert d["why"] == ""
        assert d["pattern"] == ""


class TestWhyPatternInFormatting:
    """Test that Author sees why and pattern in formatted feedback."""

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_format_issues_includes_why_pattern(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        from review_loop.models import ReviewerFeedback, ReviewIssue

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="R1",
                issues=[
                    ReviewIssue(
                        severity="major",
                        content="Missing source",
                        why="Readers cannot verify",
                        pattern="Check all claims",
                    )
                ],
            )
        ]

        formatted = engine._format_issues_for_author(feedbacks)
        assert "why: Readers cannot verify" in formatted
        assert "pattern: Check all claims" in formatted

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_format_issues_omits_empty_why_pattern(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        from review_loop.models import ReviewerFeedback, ReviewIssue

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="R1",
                issues=[
                    ReviewIssue(severity="minor", content="Typo")
                ],
            )
        ]

        formatted = engine._format_issues_for_author(feedbacks)
        assert "why:" not in formatted
        assert "pattern:" not in formatted

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_format_verdicts_includes_why_pattern(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        from review_loop.models import (
            AuthorVerdictItem,
            ReviewerFeedback,
            ReviewIssue,
        )

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="R1",
                issues=[
                    ReviewIssue(
                        severity="major",
                        content="Bad logic",
                        why="Premise unsupported",
                        pattern="Check causal chains",
                    )
                ],
            )
        ]
        verdicts = [
            AuthorVerdictItem(
                reviewer="R1", issue_index=0,
                verdict="accept", reason="Will fix",
            )
        ]

        formatted = engine._format_verdicts_for_author(verdicts, feedbacks)
        assert "why: Premise unsupported" in formatted
        assert "pattern: Check causal chains" in formatted

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_build_reviewer_context_includes_why_pattern(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        from review_loop.models import (
            AuthorVerdictItem,
            ReviewerFeedback,
            ReviewIssue,
        )

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[
                    ReviewIssue(
                        severity="critical",
                        content="Flaw",
                        why="Violates X",
                        pattern="Check Y",
                    )
                ],
            )
        ]
        verdicts = [
            AuthorVerdictItem(
                reviewer="Reviewer-A", issue_index=0,
                verdict="reject", reason="Disagree",
            )
        ]

        ctx = engine._build_reviewer_context(feedbacks, verdicts)
        assert "Reviewer-A" in ctx
        assert "why: Violates X" in ctx["Reviewer-A"]
        assert "pattern: Check Y" in ctx["Reviewer-A"]


class TestSubmitReviewInstructionWhyPattern:
    """Test that reviewer instruction mentions why and pattern."""

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_instruction_mentions_why_and_pattern(self, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=1)
        ReviewEngine(config)

        # The reviewer agent's system prompt should contain why/pattern mention
        reviewer_call = MockAgent.call_args_list[-1]
        system_msg = reviewer_call.kwargs.get("system_message", "")
        assert "why" in system_msg
        assert "pattern" in system_msg


# ---------------------------------------------------------------------------
# File-based Workspace Tests
# ---------------------------------------------------------------------------


class TestFileBasedWorkspace:
    """Tests for the file-based workspace (draft.md) features."""

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_file_tools_injected_into_agents(self, MockAgent, MockCtxMgr, mock_import):
        """FileTools should be injected into all agents when workspace_dir is set."""
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            engine = ReviewEngine(config, workspace_dir=workspace)

            engine._workspace_dir = workspace
            engine._file_tools_injected = False
            engine._author_verdict = MagicMock()
            engine._author_verdict.tools = []
            engine._author_revision = MagicMock()
            engine._author_revision.tools = []
            r1 = MagicMock()
            r1.tools = []
            engine._reviewers = [r1]

            engine._setup_file_tools()

            assert len(engine._author_verdict.tools) == 1
            assert len(engine._author_revision.tools) == 1
            assert len(r1.tools) == 1
            assert engine._file_tools_injected is True

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_setup_file_tools_idempotent(self, MockAgent, MockCtxMgr, mock_import):
        """Calling _setup_file_tools twice should only inject once."""
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            engine = ReviewEngine(config, workspace_dir=workspace)
            engine._author_verdict = MagicMock()
            engine._author_verdict.tools = []
            engine._author_revision = MagicMock()
            engine._author_revision.tools = []
            r1 = MagicMock()
            r1.tools = []
            engine._reviewers = [r1]

            engine._setup_file_tools()
            engine._setup_file_tools()  # second call should be no-op

            assert len(engine._author_verdict.tools) == 1

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_read_draft_from_workspace(self, MockAgent, MockCtxMgr, mock_import):
        """_read_draft_from_workspace should read draft.md from workspace."""
        from review_loop.engine import ReviewEngine

        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            engine = ReviewEngine(config, workspace_dir=workspace)

            # No draft.md yet
            assert engine._read_draft_from_workspace() is None

            # Write a draft
            (workspace / "draft.md").write_text("Hello world")
            assert engine._read_draft_from_workspace() == "Hello world"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_write_draft_to_workspace(self, MockAgent, MockCtxMgr, mock_import):
        """_write_draft_to_workspace should write draft.md."""
        from review_loop.engine import ReviewEngine

        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            engine = ReviewEngine(config, workspace_dir=workspace)

            engine._write_draft_to_workspace("Test content")
            assert (workspace / "draft.md").read_text() == "Test content"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_author_generate_reads_draft_from_file(self, MockAgent, MockCtxMgr, mock_import):
        """_author_generate should prefer draft.md written by agent over text content."""
        from review_loop.engine import ReviewEngine

        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            engine = ReviewEngine(config, workspace_dir=workspace)

            # Mock agent returns short text but writes long draft.md
            long_draft = "A" * 600
            async def fake_arun(**kwargs):
                # Simulate agent writing draft.md via FileTools
                (workspace / "draft.md").write_text(long_draft)
                return MockRunOutput(content="short summary")

            engine._author_revision = MagicMock()
            engine._author_revision.arun = fake_arun
            engine._author_revision.name = "Author"

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                engine._author_generate("context")
            )
            assert result == long_draft

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_author_generate_fallback_to_text(self, MockAgent, MockCtxMgr, mock_import):
        """_author_generate should fallback to text when no draft.md is written."""
        from review_loop.engine import ReviewEngine

        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            engine = ReviewEngine(config, workspace_dir=workspace)

            text_content = "B" * 600
            async def fake_arun(**kwargs):
                return MockRunOutput(content=text_content)

            engine._author_revision = MagicMock()
            engine._author_revision.arun = fake_arun
            engine._author_revision.name = "Author"

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                engine._author_generate("context")
            )
            assert result == text_content
            # Should also write to workspace
            assert (workspace / "draft.md").read_text() == text_content

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_author_apply_changes_reads_draft_from_file(self, MockAgent, MockCtxMgr, mock_import):
        """_author_apply_changes should prefer draft.md written via FileTools."""
        from review_loop.engine import ReviewEngine
        from review_loop.models import AuthorVerdictItem, ReviewerFeedback, ReviewIssue

        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            engine = ReviewEngine(config, workspace_dir=workspace)

            new_draft = "C" * 600
            async def fake_arun(**kwargs):
                (workspace / "draft.md").write_text(new_draft)
                return MockRunOutput(content="done")

            engine._author_revision = MagicMock()
            engine._author_revision.arun = fake_arun
            engine._author_revision.name = "Author"

            feedbacks = [ReviewerFeedback(
                reviewer_name="R",
                issues=[ReviewIssue(severity="minor", content="fix")]
            )]
            verdicts = [AuthorVerdictItem(
                reviewer="R", issue_index=0, verdict="accept", reason="ok"
            )]

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                engine._author_apply_changes("old content", verdicts, feedbacks)
            )
            assert result == new_draft

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_review_prompt_uses_file_based_instructions(self, MockAgent, MockCtxMgr, mock_import):
        """Reviewer prompts should instruct to use read_file('draft.md')."""
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            workspace.mkdir()
            engine = ReviewEngine(config, workspace_dir=workspace)
            # draft.md must exist for _review() to proceed
            (workspace / "draft.md").write_text("content here")

            # Capture the prompt passed to reviewer
            captured_prompt = None
            async def mock_safe_call(agent, prompt):
                nonlocal captured_prompt
                captured_prompt = prompt
                (workspace / "feedback_R1_Reviewer-A.md").write_text(
                    "## No Issues\n审核通过。\n"
                )
                return "done"

            engine._safe_agent_call = mock_safe_call

            reviewer = MagicMock()
            reviewer.name = "Reviewer-A"
            engine._reviewers = [reviewer]

            import asyncio
            asyncio.get_event_loop().run_until_complete(
                engine._review("content here", {}, round_num=1)
            )

            assert "read_file('draft.md')" in captured_prompt


# ---------------------------------------------------------------------------
# Error Callback
# ---------------------------------------------------------------------------


class TestErrorCallback:
    def _setup_engine(self, config, error_callback=None):
        """Create engine with all external deps mocked."""
        from review_loop.engine import ReviewEngine

        with (
            patch("review_loop.engine.import_from_path"),
            patch("review_loop.engine.ContextManager"),
            patch("review_loop.engine.Agent"),
        ):
            engine = ReviewEngine(config, error_callback=error_callback)

        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._archiver._session_dir = "/tmp/session"
        engine._context_mgr.build_initial_context = AsyncMock(return_value="ctx")
        return engine

    @pytest.mark.asyncio
    async def test_error_callback_called_on_all_reviewers_fail(self):
        """error_callback should be invoked when all reviewers fail."""
        from review_loop.engine import ReviewEngine

        callback = MagicMock()
        config = _make_config(max_rounds=10)
        engine = self._setup_engine(config, error_callback=callback)

        engine._author_generate = AsyncMock(return_value="v1")

        async def mock_safe_call(agent, prompt):
            return None

        engine._safe_agent_call = AsyncMock(side_effect=mock_safe_call)
        # Need real reviewers list for _review to iterate
        r1 = MagicMock()
        r1.name = "Reviewer-A"
        r2 = MagicMock()
        r2.name = "Reviewer-B"
        engine._reviewers = [r1, r2]

        result = await engine.run()

        assert result.terminated_by_error is True
        callback.assert_called_once()
        assert "所有审核员均失败" in callback.call_args[0][0]

    @pytest.mark.asyncio
    async def test_no_callback_still_terminates(self):
        """Without error_callback, errors still terminate and log."""
        from review_loop.engine import ReviewEngine

        config = _make_config(max_rounds=10)
        engine = self._setup_engine(config, error_callback=None)

        engine._author_generate = AsyncMock(return_value="v1")

        engine._safe_agent_call = AsyncMock(return_value=None)
        r1 = MagicMock()
        r1.name = "Reviewer-A"
        engine._reviewers = [r1]

        result = await engine.run()

        assert result.terminated_by_error is True
        engine._archiver.save_error_log.assert_called()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_block(self):
        """If error_callback raises, _handle_runtime_error still raises RuntimeError."""
        from review_loop.engine import ReviewEngine

        def bad_callback(msg, ctx):
            raise ValueError("callback broke")

        config = _make_config(max_rounds=10)
        engine = self._setup_engine(config, error_callback=bad_callback)

        engine._author_generate = AsyncMock(return_value="v1")
        engine._safe_agent_call = AsyncMock(return_value=None)
        r1 = MagicMock()
        r1.name = "Reviewer-A"
        engine._reviewers = [r1]

        result = await engine.run()

        assert result.terminated_by_error is True

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_handle_runtime_error_logs_and_raises(self, MockAgent, MockCtxMgr, mock_import):
        """_handle_runtime_error should log, save error, call callback, and raise."""
        from review_loop.engine import ReviewEngine

        callback = MagicMock()
        config = _make_config()
        engine = ReviewEngine(config, error_callback=callback)
        engine._audit = MagicMock()
        engine._archiver = MagicMock()
        engine._archiver._session_dir = "/tmp/session"

        with pytest.raises(RuntimeError, match="test error"):
            engine._handle_runtime_error("test error", {"key": "val"})

        engine._audit.log_error.assert_called_once_with("RUNTIME", "test error", 0)
        engine._archiver.save_error_log.assert_called_once_with("test error")
        callback.assert_called_once_with("test error", {"key": "val"})
