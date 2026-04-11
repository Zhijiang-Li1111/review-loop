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
    """Simulates agno RunOutput with content, tools, and messages."""
    def __init__(self, content: str | None = None, tools=None, messages=None):
        self.content = content
        self.tools = tools or []
        self.messages = messages or []


class MockToolExecution:
    """Simulates agno ToolExecution."""
    def __init__(self, tool_name: str, tool_args: dict | None = None, result: str | None = None):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.result = result
        self.tool_call_id = None
        self.tool_call_error = None


class MockMessage:
    """Simulates agno Message with tool call fields."""
    def __init__(
        self,
        role: str = "assistant",
        content: str | None = None,
        tool_name: str | None = None,
        tool_args=None,
        tool_calls=None,
    ):
        self.role = role
        self.content = content
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_calls = tool_calls


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
    def test_reviewers_have_submit_review_tool(self, MockAgent, MockCtxMgr, mock_import):
        """Reviewers should get submit_review as a tool (not output_schema)."""
        from review_loop.engine import ReviewEngine
        from review_loop.tools import submit_review, submit_revision, submit_verdict

        config = _make_config(num_reviewers=2)
        engine = ReviewEngine(config)

        # Author verdict agent (first call) should have only submit_verdict
        author_verdict_call = MockAgent.call_args_list[0]
        assert "output_schema" not in author_verdict_call.kwargs
        verdict_tools = author_verdict_call.kwargs.get("tools", [])
        assert submit_verdict in verdict_tools
        assert submit_revision not in verdict_tools

        # Author revision agent (second call) should have only submit_revision
        author_revision_call = MockAgent.call_args_list[1]
        assert "output_schema" not in author_revision_call.kwargs
        revision_tools = author_revision_call.kwargs.get("tools", [])
        assert submit_revision in revision_tools
        assert submit_verdict not in revision_tools

        # Reviewers should NOT have output_schema
        for call in MockAgent.call_args_list[2:]:
            assert "output_schema" not in call.kwargs
            # Should have submit_review in tools
            tools = call.kwargs.get("tools", [])
            assert submit_review in tools

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_reviewer_system_prompt_has_submit_instruction(self, MockAgent, MockCtxMgr, mock_import):
        """Reviewer system prompt should include submit_review instruction."""
        from review_loop.engine import ReviewEngine

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        reviewer_call = MockAgent.call_args_list[2]
        system_msg = reviewer_call.kwargs["system_message"]
        assert "submit_review" in system_msg
        assert "call submit_review" in system_msg

    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_author_gets_tools(self, MockAgent, MockCtxMgr):
        from review_loop.engine import ReviewEngine
        from review_loop.tools import submit_revision, submit_verdict

        class FakeTool:
            def __init__(self, context=None):
                pass

        with patch("review_loop.engine.import_from_path", return_value=FakeTool):
            config = _make_config(tools=[ToolConfig(path="pkg.FakeTool")])
            engine = ReviewEngine(config)

        # Verdict agent (first call) should have FakeTool + submit_verdict
        verdict_call = MockAgent.call_args_list[0]
        assert verdict_call.kwargs.get("tools") is not None
        assert len(verdict_call.kwargs["tools"]) == 2  # FakeTool + submit_verdict
        assert submit_verdict in verdict_call.kwargs["tools"]

        # Revision agent (second call) should have FakeTool + submit_revision
        revision_call = MockAgent.call_args_list[1]
        assert revision_call.kwargs.get("tools") is not None
        assert len(revision_call.kwargs["tools"]) == 2  # FakeTool + submit_revision
        assert submit_revision in revision_call.kwargs["tools"]

    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_reviewers_get_submit_review_even_without_per_reviewer_tools(self, MockAgent, MockCtxMgr):
        """Reviewers always get submit_review, even when no per-reviewer tools are configured."""
        from review_loop.engine import ReviewEngine
        from review_loop.tools import submit_review

        class FakeTool:
            def __init__(self, context=None):
                pass

        with patch("review_loop.engine.import_from_path", return_value=FakeTool):
            config = _make_config(tools=[ToolConfig(path="pkg.FakeTool")])
            engine = ReviewEngine(config)

        for call in MockAgent.call_args_list[2:]:
            tools_arg = call.kwargs.get("tools")
            assert tools_arg is not None
            assert submit_review in tools_arg

    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def test_reviewer_gets_per_reviewer_tools_plus_submit_review(self, MockAgent, MockCtxMgr):
        """Per-reviewer tools coexist with submit_review."""
        from review_loop.engine import ReviewEngine
        from review_loop.tools import submit_review

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

        # Reviewer-A should have submit_review + per-reviewer tool
        reviewer_a_call = MockAgent.call_args_list[2]
        assert reviewer_a_call.kwargs["name"] == "Reviewer-A"
        tools_a = reviewer_a_call.kwargs["tools"]
        assert tools_a is not None
        assert submit_review in tools_a
        assert len(tools_a) == 2  # submit_review + FakeReviewerTool
        # Check that per-reviewer tool is also there
        per_reviewer_tools = [t for t in tools_a if t is not submit_review]
        assert len(per_reviewer_tools) == 1
        assert per_reviewer_tools[0].tag == "reviewer"

        # Reviewer-B should have only submit_review
        reviewer_b_call = MockAgent.call_args_list[3]
        assert reviewer_b_call.kwargs["name"] == "Reviewer-B"
        tools_b = reviewer_b_call.kwargs["tools"]
        assert tools_b is not None
        assert submit_review in tools_b
        assert len(tools_b) == 1


# ---------------------------------------------------------------------------
# Tool Call Extraction
# ---------------------------------------------------------------------------


class TestToolCallExtraction:
    """Tests for _extract_tool_call_issues static method."""

    def test_extract_from_tool_execution(self):
        """Extract issues from RunOutput.tools (ToolExecution objects)."""
        from review_loop.engine import ReviewEngine

        issues_json = json.dumps([
            {"severity": "critical", "content": "Missing validation"},
            {"severity": "minor", "content": "Style issue"},
        ])
        run_output = MockRunOutput(
            tools=[MockToolExecution(
                tool_name="submit_review",
                tool_args={"issues": issues_json},
            )]
        )

        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is not None
        assert len(result) == 2
        assert result[0]["severity"] == "critical"
        assert result[1]["content"] == "Style issue"

    def test_extract_empty_issues_from_tool_execution(self):
        """Empty issues list from tool call."""
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput(
            tools=[MockToolExecution(
                tool_name="submit_review",
                tool_args={"issues": "[]"},
            )]
        )

        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is not None
        assert result == []

    def test_extract_from_message_tool_name(self):
        """Extract from message where tool_name=submit_review and tool_args is set."""
        from review_loop.engine import ReviewEngine

        issues_json = json.dumps([{"severity": "major", "content": "Bad logic"}])
        run_output = MockRunOutput(
            messages=[MockMessage(
                role="tool",
                tool_name="submit_review",
                tool_args={"issues": issues_json},
            )]
        )

        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is not None
        assert len(result) == 1
        assert result[0]["severity"] == "major"

    def test_extract_from_message_tool_calls(self):
        """Extract from assistant message with tool_calls list."""
        from review_loop.engine import ReviewEngine

        issues_data = [{"severity": "minor", "content": "Typo"}]
        args = json.dumps({"issues": json.dumps(issues_data)})
        run_output = MockRunOutput(
            messages=[MockMessage(
                role="assistant",
                tool_calls=[{
                    "function": {
                        "name": "submit_review",
                        "arguments": args,
                    }
                }],
            )]
        )

        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "Typo"

    def test_no_submit_review_returns_none(self):
        """No submit_review tool call -> returns None."""
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput(
            content="Just some text",
            tools=[MockToolExecution(
                tool_name="search_research",
                tool_args={"query": "something"},
            )],
        )

        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is None

    def test_empty_run_output_returns_none(self):
        """Empty RunOutput -> returns None."""
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput()
        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is None

    def test_ignores_other_tools(self):
        """Only extracts from submit_review, ignores other tool calls."""
        from review_loop.engine import ReviewEngine

        issues_json = json.dumps([{"severity": "critical", "content": "Issue found"}])
        run_output = MockRunOutput(
            tools=[
                MockToolExecution(tool_name="search_research", tool_args={"query": "foo"}),
                MockToolExecution(tool_name="submit_review", tool_args={"issues": issues_json}),
            ]
        )

        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is not None
        assert len(result) == 1
        assert result[0]["severity"] == "critical"


# ---------------------------------------------------------------------------
# Review Phase
# ---------------------------------------------------------------------------


class TestReviewPhase:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @pytest.mark.asyncio
    async def test_reviewers_audit_in_parallel_via_tool_call(self, MockCtxMgr, mock_import):
        """Reviewers submit structured output via submit_review tool call."""
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

        issues_a = json.dumps([{"severity": "critical", "content": "Logic gap"}])
        issues_b = json.dumps([])

        async def mock_safe_call_full(agent, prompt):
            if agent.name == "Reviewer-A":
                return MockRunOutput(
                    content="Review complete",
                    tools=[MockToolExecution(
                        tool_name="submit_review",
                        tool_args={"issues": issues_a},
                    )],
                )
            return MockRunOutput(
                content="No issues found",
                tools=[MockToolExecution(
                    tool_name="submit_review",
                    tool_args={"issues": issues_b},
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
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
    async def test_review_fallback_to_string_parsing(self, MockCtxMgr, mock_import):
        """When no tool call present, fall back to string JSON parsing."""
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

        # No tool calls — just content with JSON
        async def mock_safe_call_full(agent, prompt):
            return MockRunOutput(
                content='{"issues": [{"severity": "major", "content": "Bad data"}]}',
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
            feedbacks = await engine._review("Content v1", {})

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

        captured_prompts = {}

        async def mock_safe_call_full(agent, prompt):
            captured_prompts[agent.name] = prompt
            return MockRunOutput(
                tools=[MockToolExecution(
                    tool_name="submit_review",
                    tool_args={"issues": "[]"},
                )],
            )

        # Build per-reviewer context
        per_reviewer_ctx = {
            "Reviewer-A": "Issue A context only",
            "Reviewer-B": "Issue B context only",
        }

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
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
    async def test_author_evaluate_feedback(self, MockAgent, MockCtxMgr, mock_import):
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

        verdicts_json = json.dumps([{
            "reviewer": "Reviewer-A",
            "issue_index": 0,
            "verdict": "accept",
            "reason": "Fixed in v2",
        }])

        async def mock_safe_call_full(agent, prompt):
            return MockRunOutput(
                tools=[MockToolExecution(
                    tool_name="submit_verdict",
                    tool_args={"verdicts": verdicts_json},
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
            verdicts = await engine._author_evaluate_feedback("Content v1", feedbacks)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == "accept"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_apply_changes(self, MockAgent, MockCtxMgr, mock_import):
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

        async def mock_safe_call_full(agent, prompt):
            return MockRunOutput(
                tools=[MockToolExecution(
                    tool_name="submit_revision",
                    tool_args={
                        "updated_content": "Content v2 with fix applied " * 10,
                    },
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
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

        async def mock_safe_call_full(agent, prompt):
            captured_prompts.append(prompt)
            return MockRunOutput(
                tools=[MockToolExecution(
                    tool_name="submit_verdict",
                    tool_args={"verdicts": "[]"},
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
            await engine._author_evaluate_feedback("v1", feedbacks)

        assert "Process feedback carefully." in captured_prompts[0]
        assert "call submit_verdict" in captured_prompts[0]


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

        async def mock_safe_call_full(agent, prompt):
            if agent.name == "Reviewer-A":
                return None  # failure
            return MockRunOutput(
                tools=[MockToolExecution(
                    tool_name="submit_review",
                    tool_args={"issues": "[]"},
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
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
        mock_author_verdict = MagicMock(name="Author-verdict")
        mock_author_verdict.name = "Author"
        mock_author_revision = MagicMock(name="Author-revision")
        mock_author_revision.name = "Author"

        with patch("review_loop.engine.Agent", side_effect=[mock_author_verdict, mock_author_revision] + mock_reviewers):
            config = _make_config(num_reviewers=2)
            engine = ReviewEngine(config)

        async def mock_safe_call_full(agent, prompt):
            return None

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
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
        # (plus submit_review instruction appended)
        checker_call = MockAgent.call_args_list[2]
        assert checker_call.kwargs["name"] == "Checker"
        assert "Author rules: write hooks, use frameworks." in checker_call.kwargs["system_message"]
        assert "{{author.system_prompt}}" not in checker_call.kwargs["system_message"]

        # The Editor reviewer should have base prompt + submit_review instruction
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


# ---------------------------------------------------------------------------
# submit_review Tool Tests
# ---------------------------------------------------------------------------


class TestSubmitReviewTool:
    """Tests for the submit_review tool function itself."""

    def test_valid_issues_json(self):
        from review_loop.tools import submit_review

        result = submit_review('[{"severity": "critical", "content": "Missing validation"}]')
        parsed = json.loads(result)
        assert parsed["status"] == "submitted"
        assert parsed["issue_count"] == 1

    def test_empty_issues(self):
        from review_loop.tools import submit_review

        result = submit_review("[]")
        parsed = json.loads(result)
        assert parsed["status"] == "submitted"
        assert parsed["issue_count"] == 0

    def test_multiple_issues(self):
        from review_loop.tools import submit_review

        issues = json.dumps([
            {"severity": "critical", "content": "Issue 1"},
            {"severity": "minor", "content": "Issue 2"},
            {"severity": "major", "content": "Issue 3"},
        ])
        result = submit_review(issues)
        parsed = json.loads(result)
        assert parsed["issue_count"] == 3

    def test_invalid_json(self):
        from review_loop.tools import submit_review

        result = submit_review("not json")
        assert "Error" in result

    def test_non_array_json(self):
        from review_loop.tools import submit_review

        result = submit_review('{"not": "array"}')
        assert "Error" in result

    def test_missing_severity(self):
        from review_loop.tools import submit_review

        result = submit_review('[{"content": "no severity"}]')
        assert "Error" in result
        assert "severity" in result

    def test_missing_content(self):
        from review_loop.tools import submit_review

        result = submit_review('[{"severity": "minor"}]')
        assert "Error" in result
        assert "content" in result


# ---------------------------------------------------------------------------
# submit_verdict Tool Tests
# ---------------------------------------------------------------------------


class TestSubmitVerdictTool:
    """Tests for the submit_verdict tool function itself."""

    def test_valid_verdicts(self):
        from review_loop.tools import submit_verdict

        verdicts = json.dumps([{
            "reviewer": "Reviewer-A",
            "issue_index": 0,
            "verdict": "accept",
            "reason": "Fixed",
        }])
        result = submit_verdict(verdicts)
        parsed = json.loads(result)
        assert parsed["status"] == "submitted"
        assert parsed["verdict_counts"]["accept"] == 1

    def test_multiple_verdicts(self):
        from review_loop.tools import submit_verdict

        verdicts = json.dumps([
            {"reviewer": "R-A", "issue_index": 0, "verdict": "accept", "reason": "Fixed"},
            {"reviewer": "R-A", "issue_index": 1, "verdict": "reject", "reason": "Disagree"},
            {"reviewer": "R-B", "issue_index": 0, "verdict": "unclear", "reason": "Need info"},
        ])
        result = submit_verdict(verdicts)
        parsed = json.loads(result)
        assert parsed["verdict_counts"]["accept"] == 1
        assert parsed["verdict_counts"]["reject"] == 1
        assert parsed["verdict_counts"]["unclear"] == 1

    def test_empty_verdicts(self):
        from review_loop.tools import submit_verdict

        result = submit_verdict("[]")
        parsed = json.loads(result)
        assert parsed["status"] == "submitted"
        assert parsed["verdict_counts"]["accept"] == 0

    def test_rejects_invalid_json(self):
        from review_loop.tools import submit_verdict

        result = submit_verdict("not json")
        assert "Error" in result

    def test_rejects_non_array(self):
        from review_loop.tools import submit_verdict

        result = submit_verdict('{"not": "array"}')
        assert "Error" in result

    def test_rejects_missing_fields(self):
        from review_loop.tools import submit_verdict

        verdicts = json.dumps([{"reviewer": "R-A"}])  # missing fields
        result = submit_verdict(verdicts)
        assert "Error" in result

    def test_rejects_invalid_verdict_value(self):
        from review_loop.tools import submit_verdict

        verdicts = json.dumps([{
            "reviewer": "R-A",
            "issue_index": 0,
            "verdict": "maybe",
            "reason": "Not sure",
        }])
        result = submit_verdict(verdicts)
        assert "Error" in result
        assert "maybe" in result


class TestSubmitRevisionTool:
    """Tests for the submit_revision tool function itself."""

    def test_valid_revision(self):
        from review_loop.tools import submit_revision

        content = "A" * 200  # Must be >= 100 chars
        result = submit_revision(content)
        parsed = json.loads(result)
        assert parsed["status"] == "submitted"
        assert parsed["content_length"] == 200

    def test_rejects_short_content(self):
        from review_loop.tools import submit_revision

        result = submit_revision("too short")
        assert "Error" in result
        assert "only" in result.lower() or "100" in result

    def test_rejects_empty_content(self):
        from review_loop.tools import submit_revision

        result = submit_revision("")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Author submit_revision Tool Call Extraction
# ---------------------------------------------------------------------------


class TestExtractSubmitRevision:
    """Tests for _extract_submit_revision static method."""

    def test_extract_from_tool_execution(self):
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput(
            tools=[MockToolExecution(
                tool_name="submit_revision",
                tool_args={
                    "updated_content": "Full revised content here " * 10,
                },
            )]
        )

        result = ReviewEngine._extract_submit_revision(run_output)
        assert result is not None
        assert "Full revised content here" in result

    def test_extract_from_message_tool_name(self):
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput(
            messages=[MockMessage(
                role="tool",
                tool_name="submit_revision",
                tool_args={
                    "updated_content": "Updated content from message " * 10,
                },
            )]
        )

        result = ReviewEngine._extract_submit_revision(run_output)
        assert result is not None
        assert "Updated content from message" in result

    def test_extract_from_message_tool_calls(self):
        from review_loop.engine import ReviewEngine

        args = json.dumps({
            "updated_content": "Content from tool_calls list " * 10,
        })
        run_output = MockRunOutput(
            messages=[MockMessage(
                role="assistant",
                tool_calls=[{
                    "function": {
                        "name": "submit_revision",
                        "arguments": args,
                    }
                }],
            )]
        )

        result = ReviewEngine._extract_submit_revision(run_output)
        assert result is not None
        assert "Content from tool_calls list" in result

    def test_no_submit_revision_returns_none(self):
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput(
            content="Just text",
            tools=[MockToolExecution(
                tool_name="submit_review",
                tool_args={"issues": "[]"},
            )],
        )

        result = ReviewEngine._extract_submit_revision(run_output)
        assert result is None

    def test_empty_run_output_returns_none(self):
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput()
        result = ReviewEngine._extract_submit_revision(run_output)
        assert result is None

    def test_missing_updated_content_returns_none(self):
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput(
            tools=[MockToolExecution(
                tool_name="submit_revision",
                tool_args={},  # no updated_content
            )]
        )

        result = ReviewEngine._extract_submit_revision(run_output)
        assert result is None


# ---------------------------------------------------------------------------
# Author submit_verdict Tool Call Extraction
# ---------------------------------------------------------------------------


class TestExtractSubmitVerdict:
    """Tests for _extract_submit_verdict static method."""

    def test_extract_from_tool_execution(self):
        from review_loop.engine import ReviewEngine

        verdicts_json = json.dumps([
            {"reviewer": "R-A", "issue_index": 0, "verdict": "accept", "reason": "Fixed"},
        ])
        run_output = MockRunOutput(
            tools=[MockToolExecution(
                tool_name="submit_verdict",
                tool_args={"verdicts": verdicts_json},
            )]
        )

        result = ReviewEngine._extract_submit_verdict(run_output)
        assert result is not None
        assert len(result) == 1
        assert result[0].verdict == "accept"
        assert result[0].reviewer == "R-A"

    def test_extract_from_message_tool_name(self):
        from review_loop.engine import ReviewEngine

        verdicts_json = json.dumps([
            {"reviewer": "R-B", "issue_index": 0, "verdict": "reject", "reason": "Disagree"},
        ])
        run_output = MockRunOutput(
            messages=[MockMessage(
                role="tool",
                tool_name="submit_verdict",
                tool_args={"verdicts": verdicts_json},
            )]
        )

        result = ReviewEngine._extract_submit_verdict(run_output)
        assert result is not None
        assert len(result) == 1
        assert result[0].verdict == "reject"

    def test_extract_from_message_tool_calls(self):
        from review_loop.engine import ReviewEngine

        verdicts = [{"reviewer": "R-A", "issue_index": 0, "verdict": "accept", "reason": "Done"}]
        args = json.dumps({"verdicts": json.dumps(verdicts)})
        run_output = MockRunOutput(
            messages=[MockMessage(
                role="assistant",
                tool_calls=[{
                    "function": {
                        "name": "submit_verdict",
                        "arguments": args,
                    }
                }],
            )]
        )

        result = ReviewEngine._extract_submit_verdict(run_output)
        assert result is not None
        assert len(result) == 1
        assert result[0].verdict == "accept"

    def test_empty_verdicts(self):
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput(
            tools=[MockToolExecution(
                tool_name="submit_verdict",
                tool_args={"verdicts": "[]"},
            )]
        )

        result = ReviewEngine._extract_submit_verdict(run_output)
        assert result is not None
        assert result == []

    def test_no_submit_verdict_returns_none(self):
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput(
            content="Just text",
            tools=[MockToolExecution(
                tool_name="submit_review",
                tool_args={"issues": "[]"},
            )],
        )

        result = ReviewEngine._extract_submit_verdict(run_output)
        assert result is None

    def test_empty_run_output_returns_none(self):
        from review_loop.engine import ReviewEngine

        run_output = MockRunOutput()
        result = ReviewEngine._extract_submit_verdict(run_output)
        assert result is None


# ---------------------------------------------------------------------------
# Author Tool-Call Integration
# ---------------------------------------------------------------------------


class TestAuthorToolCallIntegration:
    """Tests for _author_evaluate_feedback and _author_apply_changes using tool calls."""

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_uses_submit_verdict_tool(self, MockAgent, MockCtxMgr, mock_import):
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

        verdicts_json = json.dumps([{
            "reviewer": "Reviewer-A",
            "issue_index": 0,
            "verdict": "accept",
            "reason": "Fixed in v2",
        }])

        async def mock_safe_call_full(agent, prompt):
            return MockRunOutput(
                content="I've evaluated all feedback.",
                tools=[MockToolExecution(
                    tool_name="submit_verdict",
                    tool_args={"verdicts": verdicts_json},
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
            verdicts = await engine._author_evaluate_feedback("Content v1", feedbacks)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == "accept"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_uses_submit_revision_tool(self, MockAgent, MockCtxMgr, mock_import):
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

        async def mock_safe_call_full(agent, prompt):
            return MockRunOutput(
                content="I've applied all changes.",
                tools=[MockToolExecution(
                    tool_name="submit_revision",
                    tool_args={
                        "updated_content": "Content v2 with fix applied " * 10,
                    },
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
            updated = await engine._author_apply_changes("Content v1", verdicts, feedbacks)

        assert "Content v2 with fix applied" in updated

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_evaluate_fallback_to_json_parsing(self, MockAgent, MockCtxMgr, mock_import):
        """When no submit_verdict tool call, fall back to JSON parsing."""
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

        verdict_json = json.dumps([{"reviewer": "Reviewer-A", "issue_index": 0,
                                    "verdict": "accept", "reason": "Fixed"}])

        async def mock_safe_call_full(agent, prompt):
            return MockRunOutput(content=verdict_json)  # No tool calls

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
            verdicts = await engine._author_evaluate_feedback("Content v1", feedbacks)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == "accept"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_evaluate_prompt_includes_submit_verdict_instruction(self, MockAgent, MockCtxMgr, mock_import):
        """The Author evaluate prompt should include submit_verdict instruction."""
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

        async def mock_safe_call_full(agent, prompt):
            captured_prompts.append(prompt)
            return MockRunOutput(
                tools=[MockToolExecution(
                    tool_name="submit_verdict",
                    tool_args={"verdicts": "[]"},
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
            await engine._author_evaluate_feedback("v1", feedbacks)

        assert len(captured_prompts) == 1
        assert "call submit_verdict" in captured_prompts[0]

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_author_apply_prompt_includes_submit_revision_instruction(self, MockAgent, MockCtxMgr, mock_import):
        """The Author apply prompt should include submit_revision instruction."""
        from review_loop.engine import ReviewEngine
        from review_loop.models import AuthorVerdictItem, ReviewerFeedback, ReviewIssue

        config = _make_config(num_reviewers=1)
        engine = ReviewEngine(config)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="minor", content="Typo")],
            )
        ]
        verdicts = [
            AuthorVerdictItem(
                reviewer="Reviewer-A", issue_index=0,
                verdict="accept", reason="Fixed",
            )
        ]

        captured_prompts = []

        async def mock_safe_call_full(agent, prompt):
            captured_prompts.append(prompt)
            return MockRunOutput(
                tools=[MockToolExecution(
                    tool_name="submit_revision",
                    tool_args={"updated_content": "A" * 200},
                )],
            )

        with patch.object(engine, "_safe_agent_call_full", side_effect=mock_safe_call_full):
            await engine._author_apply_changes("v1", verdicts, feedbacks)

        assert len(captured_prompts) == 1
        assert "call submit_revision" in captured_prompts[0]


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


class TestWhyPatternInToolExtraction:
    """Test that why and pattern are preserved through tool call extraction."""

    def test_extract_with_why_pattern(self):
        from review_loop.engine import ReviewEngine

        issues_json = json.dumps([
            {
                "severity": "major",
                "content": "Missing data",
                "why": "No source cited",
                "pattern": "Check all data claims",
            }
        ])
        run_output = MockRunOutput(
            tools=[MockToolExecution(
                tool_name="submit_review",
                tool_args={"issues": issues_json},
            )],
        )
        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is not None
        assert len(result) == 1
        assert result[0]["why"] == "No source cited"
        assert result[0]["pattern"] == "Check all data claims"

    def test_extract_without_why_pattern_defaults_to_empty(self):
        """Backward compat: old-style issues without why/pattern."""
        from review_loop.engine import ReviewEngine

        issues_json = json.dumps([
            {"severity": "minor", "content": "Typo"}
        ])
        run_output = MockRunOutput(
            tools=[MockToolExecution(
                tool_name="submit_review",
                tool_args={"issues": issues_json},
            )],
        )
        result = ReviewEngine._extract_tool_call_issues(run_output)
        assert result is not None
        assert "why" not in result[0]  # raw dict doesn't have it
        # But when constructed into ReviewIssue, defaults kick in


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
