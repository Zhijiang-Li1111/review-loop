"""Tests for ReviewEngine in review_loop.engine."""

from __future__ import annotations

import json
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
    def __init__(self, content: str | None):
        self.content = content


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

        # 1 Author + 3 Reviewers = 4 Agent calls
        assert MockAgent.call_count == 4
        calls = MockAgent.call_args_list
        assert calls[0].kwargs["name"] == "Author"
        reviewer_names = [c.kwargs["name"] for c in calls[1:]]
        assert reviewer_names == ["Reviewer-A", "Reviewer-B", "Reviewer-C"]

    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_author_gets_tools(self, MockAgent, MockCtxMgr):
        from review_loop.engine import ReviewEngine

        class FakeTool:
            def __init__(self, context=None):
                pass

        with patch("review_loop.engine.import_from_path", return_value=FakeTool):
            config = _make_config(tools=[ToolConfig(path="pkg.FakeTool")])
            engine = ReviewEngine(config)

        author_call = MockAgent.call_args_list[0]
        assert author_call.kwargs.get("tools") is not None
        assert len(author_call.kwargs["tools"]) == 1

    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_reviewers_have_no_tools(self, MockAgent, MockCtxMgr):
        from review_loop.engine import ReviewEngine

        class FakeTool:
            def __init__(self, context=None):
                pass

        with patch("review_loop.engine.import_from_path", return_value=FakeTool):
            config = _make_config(tools=[ToolConfig(path="pkg.FakeTool")])
            engine = ReviewEngine(config)

        for call in MockAgent.call_args_list[1:]:
            tools_arg = call.kwargs.get("tools")
            assert tools_arg is None


# ---------------------------------------------------------------------------
# Review Phase
# ---------------------------------------------------------------------------


class TestReviewPhase:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_reviewers_audit_in_parallel(self, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        mock_reviewers = [MagicMock(), MagicMock()]
        mock_reviewers[0].name = "Reviewer-A"
        mock_reviewers[1].name = "Reviewer-B"
        mock_author = MagicMock(name="Author")
        mock_author.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        issues_a = json.dumps({"issues": [{"severity": "critical", "content": "Logic gap"}]})
        issues_b = json.dumps({"issues": []})

        async def mock_safe_call(agent, prompt):
            if agent.name == "Reviewer-A":
                return issues_a
            return issues_b

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            feedbacks = await engine._review("Content v1", {})

        assert len(feedbacks) == 2
        fb_a = next(f for f in feedbacks if f.reviewer_name == "Reviewer-A")
        fb_b = next(f for f in feedbacks if f.reviewer_name == "Reviewer-B")
        assert len(fb_a.issues) == 1
        assert fb_a.issues[0].severity == "critical"
        assert len(fb_b.issues) == 0

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_reviewer_rebuttal_only_sees_own_issues(self, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        mock_reviewers = [MagicMock(), MagicMock()]
        mock_reviewers[0].name = "Reviewer-A"
        mock_reviewers[1].name = "Reviewer-B"
        mock_author = MagicMock(name="Author")
        mock_author.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        captured_prompts = {}

        async def mock_safe_call(agent, prompt):
            captured_prompts[agent.name] = prompt
            return json.dumps({"issues": []})

        # Build per-reviewer context
        per_reviewer_ctx = {
            "Reviewer-A": "Issue A context only",
            "Reviewer-B": "Issue B context only",
        }

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            await engine._review("Content v2", per_reviewer_ctx)

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
    async def test_author_processes_feedback(self, MockAgent, MockCtxMgr, mock_import):
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

        author_json = json.dumps({
            "responses": [{
                "reviewer": "Reviewer-A",
                "issue_index": 0,
                "verdict": "accept",
                "reason": "Fixed in v2",
            }],
            "updated_content": "Content v2 with fix",
        })

        async def mock_safe_call(agent, prompt):
            return author_json

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            response = await engine._author_process_feedback("Content v1", feedbacks)

        assert response.updated_content == "Content v2 with fix"
        assert len(response.responses) == 1
        assert response.responses[0].verdict == "accept"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_prompt_includes_receiving_review(self, MockAgent, MockCtxMgr, mock_import):
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
            return json.dumps({
                "responses": [],
                "updated_content": "v2",
            })

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            await engine._author_process_feedback("v1", feedbacks)

        assert "Process feedback carefully." in captured_prompts[0]


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
            AuthorResponse,
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

        author_response = AuthorResponse(
            responses=[
                AuthorVerdictItem(
                    reviewer="Reviewer-A", issue_index=0,
                    verdict="reject", reason="Logic is sound because X",
                ),
                AuthorVerdictItem(
                    reviewer="Reviewer-B", issue_index=0,
                    verdict="accept", reason="Fixed typo",
                ),
            ],
            updated_content="v2",
        )

        ctx = engine._build_reviewer_context(feedbacks, author_response)

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
        mock_author = MagicMock(name="Author")
        mock_author.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        async def mock_safe_call(agent, prompt):
            if agent.name == "Reviewer-A":
                return None  # failure
            return json.dumps({"issues": []})

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            feedbacks = await engine._review("Content", {})

        assert len(feedbacks) == 1
        assert feedbacks[0].reviewer_name == "Reviewer-B"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_all_reviewers_fail_raises(self, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine, AllReviewersFailedError

        mock_reviewers = [MagicMock(), MagicMock()]
        mock_reviewers[0].name = "Reviewer-A"
        mock_reviewers[1].name = "Reviewer-B"
        mock_author = MagicMock(name="Author")
        mock_author.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        async def mock_safe_call(agent, prompt):
            return None

        with patch.object(engine, "_safe_agent_call", side_effect=mock_safe_call):
            with pytest.raises(AllReviewersFailedError):
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
            AuthorResponse,
            AuthorVerdictItem,
            ReviewerFeedback,
            ReviewIssue,
        )

        config = _make_config(max_rounds=10, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="v1 content")

        round_counter = {"n": 0}

        async def mock_review(content, per_reviewer_ctx):
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
        engine._author_process_feedback = AsyncMock(
            return_value=AuthorResponse(
                responses=[
                    AuthorVerdictItem(
                        reviewer="Reviewer-A", issue_index=0,
                        verdict="accept", reason="Fixed",
                    )
                ],
                updated_content="v2 content",
            )
        )

        result = await engine.run()

        assert result.converged is True
        assert result.rounds_completed == 2
        assert result.final_content == "v2 content"

    @pytest.mark.asyncio
    async def test_max_rounds_enforced(self):
        from review_loop.engine import ReviewEngine
        from review_loop.models import (
            AuthorResponse,
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
        engine._author_process_feedback = AsyncMock(
            return_value=AuthorResponse(
                responses=[], updated_content="still trying",
            )
        )

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
        from review_loop.engine import ReviewEngine, AllReviewersFailedError

        config = _make_config(max_rounds=10)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="v1")
        engine._review = AsyncMock(side_effect=AllReviewersFailedError("All failed"))

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

        async def mock_review(content, ctx):
            round_counter["n"] += 1
            if round_counter["n"] == 1:
                return [ReviewerFeedback(
                    reviewer_name="Reviewer-A",
                    issues=[ReviewIssue(severity="minor", content="Tweak")],
                )]
            return [ReviewerFeedback(reviewer_name="Reviewer-A", issues=[])]

        engine._review = AsyncMock(side_effect=mock_review)

        from review_loop.models import AuthorResponse, AuthorVerdictItem
        engine._author_process_feedback = AsyncMock(
            return_value=AuthorResponse(
                responses=[AuthorVerdictItem(
                    reviewer="Reviewer-A", issue_index=0,
                    verdict="accept", reason="Fixed",
                )],
                updated_content="v2 content",
            )
        )

        result = await engine.run()

        # Check persistence was called
        engine._archiver.save_context.assert_called_once()
        assert engine._archiver.save_author_content.call_count >= 1
        assert engine._archiver.save_reviewer_feedback.call_count >= 1
        engine._archiver.save_final.assert_called_once()


# ---------------------------------------------------------------------------
# Parsing (LLM output handling)
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
