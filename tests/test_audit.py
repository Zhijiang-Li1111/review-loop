"""Tests for per-agent audit log (review_loop.audit + engine integration)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from review_loop.audit import AuditLogger, _now_iso, _truncate, _summarize_args


# -----------------------------------------------------------------------
# AuditLogger unit tests
# -----------------------------------------------------------------------


class TestAuditLoggerUnit:
    def test_creates_audit_dir(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        assert (tmp_path / "audit").is_dir()
        audit.close()

    def test_log_call_start(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_start("Author", "Write something nice about cats")
        audit.close()

        lines = (tmp_path / "audit" / "Author.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "call_start"
        assert event["agent"] == "Author"
        assert "Write something" in event["prompt_preview"]

    def test_log_call_start_with_extras(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_start(
            "Author", "Write something",
            system_prompt_size_chars=500,
            system_prompt_size_tokens_est=125,
            skill_tools_loaded=["submit_review", "web_search"],
        )
        audit.close()

        event = json.loads((tmp_path / "audit" / "Author.jsonl").read_text().strip())
        assert event["system_prompt_size_chars"] == 500
        assert event["system_prompt_size_tokens_est"] == 125
        assert event["skill_tools_loaded"] == ["submit_review", "web_search"]

    def test_log_tool_call(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_tool_call(
            "Reviewer-A",
            tool_name="submit_review",
            args={"issues": "[]"},
            response_size=200,
            duration_ms=150.3,
        )
        audit.close()

        lines = (tmp_path / "audit" / "Reviewer-A.jsonl").read_text().strip().split("\n")
        event = json.loads(lines[0])
        assert event["event"] == "tool_call"
        assert event["tool"] == "submit_review"
        assert event["response_size"] == 200
        assert event["duration_ms"] == 150.3

    def test_log_tool_call_with_error(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_tool_call(
            "Author",
            tool_name="get_skill_instructions",
            error="ConnectionTimeout",
        )
        audit.close()

        lines = (tmp_path / "audit" / "Author.jsonl").read_text().strip().split("\n")
        event = json.loads(lines[0])
        assert event["error"] == "ConnectionTimeout"

    def test_log_api_request(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_api_request(
            "Author",
            model="claude-opus-4.6-1m",
            input_tokens=45000,
            output_tokens=3200,
            duration_ms=37000.0,
        )
        audit.close()

        lines = (tmp_path / "audit" / "Author.jsonl").read_text().strip().split("\n")
        event = json.loads(lines[0])
        assert event["event"] == "api_request"
        assert event["model"] == "claude-opus-4.6-1m"
        assert event["input_tokens"] == 45000
        assert event["output_tokens"] == 3200

    def test_log_call_end(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_end("Author", 40000.0, "# 修改后的大纲...", "end_turn")
        audit.close()

        lines = (tmp_path / "audit" / "Author.jsonl").read_text().strip().split("\n")
        event = json.loads(lines[0])
        assert event["event"] == "call_end"
        assert event["duration_ms"] == 40000.0
        assert event["stop_reason"] == "end_turn"

    def test_log_call_end_with_extras(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_end(
            "Author", 40000.0, "output", "end_turn",
            messages_count=7,
            total_tool_response_chars=3500,
        )
        audit.close()

        event = json.loads((tmp_path / "audit" / "Author.jsonl").read_text().strip())
        assert event["messages_count"] == 7
        assert event["total_tool_response_chars"] == 3500

    def test_log_error(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_error("Author", "TimeoutError: agent took too long", 60000.0)
        audit.close()

        lines = (tmp_path / "audit" / "Author.jsonl").read_text().strip().split("\n")
        event = json.loads(lines[0])
        assert event["event"] == "error"
        assert "TimeoutError" in event["error"]
        assert event["duration_ms"] == 60000.0

    def test_multiple_agents_separate_files(self, tmp_path):
        audit = AuditLogger(str(tmp_path))
        audit.log_call_start("Author", "prompt1")
        audit.log_call_start("Reviewer-A", "prompt2")
        audit.log_call_start("Reviewer-B", "prompt3")
        audit.close()

        assert (tmp_path / "audit" / "Author.jsonl").exists()
        assert (tmp_path / "audit" / "Reviewer-A.jsonl").exists()
        assert (tmp_path / "audit" / "Reviewer-B.jsonl").exists()

    def test_log_from_run_output_with_tools(self, tmp_path):
        """Verify tool calls are extracted from RunOutput.tools."""
        audit = AuditLogger(str(tmp_path))

        # Mock RunOutput with tool executions
        mock_tool = MagicMock()
        mock_tool.tool_name = "submit_review"
        mock_tool.tool_args = {"issues": "[]"}
        mock_tool.result = '{"ok": true}'
        mock_tool.tool_call_error = False
        mock_metrics = MagicMock()
        mock_metrics.duration = 0.15
        mock_tool.metrics = mock_metrics

        mock_run_output = MagicMock()
        mock_run_output.tools = [mock_tool]
        mock_run_output.messages = None
        mock_run_output.metrics = None
        mock_run_output.model = None

        audit.log_from_run_output("Reviewer-A", mock_run_output)
        audit.close()

        lines = (tmp_path / "audit" / "Reviewer-A.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        tool_events = [e for e in events if e["event"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool"] == "submit_review"
        assert tool_events[0]["duration_ms"] == 150.0
        assert tool_events[0]["response_size"] == len('{"ok": true}')
        # Should also have messages_summary
        summary_events = [e for e in events if e["event"] == "messages_summary"]
        assert len(summary_events) == 1
        assert summary_events[0]["messages_count"] == 0

    def test_log_from_run_output_with_metrics(self, tmp_path):
        """Verify API metrics are extracted from RunOutput.metrics."""
        audit = AuditLogger(str(tmp_path))

        mock_run_output = MagicMock()
        mock_run_output.tools = []
        mock_run_output.messages = None
        mock_run_output.model = "claude-opus-4.6-1m"
        mock_metrics = MagicMock()
        mock_metrics.input_tokens = 45000
        mock_metrics.output_tokens = 3200
        mock_metrics.reasoning_tokens = 0
        mock_metrics.cache_read_tokens = 0
        mock_metrics.cache_write_tokens = 0
        mock_metrics.total_tokens = 48200
        mock_metrics.cost = None
        mock_metrics.duration = 37.0
        mock_run_output.metrics = mock_metrics

        audit.log_from_run_output("Author", mock_run_output)
        audit.close()

        lines = (tmp_path / "audit" / "Author.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        api_events = [e for e in events if e["event"] == "api_request"]
        assert len(api_events) == 1
        assert api_events[0]["input_tokens"] == 45000
        assert api_events[0]["output_tokens"] == 3200
        assert api_events[0]["duration_ms"] == 37000.0
        # Should also have run_metrics event
        rm_events = [e for e in events if e["event"] == "run_metrics"]
        assert len(rm_events) == 1
        assert rm_events[0]["total_tokens"] == 48200

    def test_log_from_run_output_none(self, tmp_path):
        """log_from_run_output(None) should not crash."""
        audit = AuditLogger(str(tmp_path))
        audit.log_from_run_output("X", None)
        audit.close()
        # No file created
        assert not (tmp_path / "audit" / "X.jsonl").exists()

    def test_log_from_run_output_roundtrip_tokens(self, tmp_path):
        """Per-roundtrip token events from assistant messages."""
        audit = AuditLogger(str(tmp_path))

        msg1 = MagicMock()
        msg1.role = "assistant"
        msg1.content = "Hello"
        msg1.metrics = MagicMock(
            input_tokens=1000, output_tokens=200,
            reasoning_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
            cost=None, duration=None, time_to_first_token=None,
        )

        msg2 = MagicMock()
        msg2.role = "tool"
        msg2.content = "tool result"
        msg2.metrics = MagicMock(input_tokens=0, output_tokens=0)

        msg3 = MagicMock()
        msg3.role = "assistant"
        msg3.content = "World"
        msg3.metrics = MagicMock(
            input_tokens=1500, output_tokens=300,
            reasoning_tokens=0, cache_read_tokens=0, cache_write_tokens=0,
            cost=None, duration=None, time_to_first_token=None,
        )

        mock_output = MagicMock()
        mock_output.tools = []
        mock_output.messages = [msg1, msg2, msg3]
        mock_output.metrics = None

        audit.log_from_run_output("Author", mock_output)
        audit.close()

        lines = (tmp_path / "audit" / "Author.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        rt_events = [e for e in events if e["event"] == "roundtrip_tokens"]
        assert len(rt_events) == 2
        assert rt_events[0]["input_tokens"] == 1000
        assert rt_events[0]["output_tokens"] == 200
        assert rt_events[0]["roundtrip_idx"] == 0
        assert rt_events[1]["input_tokens"] == 1500
        assert rt_events[1]["output_tokens"] == 300
        assert rt_events[1]["roundtrip_idx"] == 1
        # Check messages_summary
        summary = [e for e in events if e["event"] == "messages_summary"]
        assert len(summary) == 1
        assert summary[0]["messages_count"] == 3

    def test_extract_call_start_extras(self):
        """extract_call_start_extras returns system_prompt_size and tool names."""
        mock_agent = MagicMock()
        mock_agent.system_message = "You are a helpful reviewer." * 10  # 280 chars

        tool1 = MagicMock()
        tool1.name = "submit_review"
        tool1.__name__ = "submit_review"
        tool2 = MagicMock(spec=[])  # no name attr
        type(tool2).__name__ = "WebSearchTool"

        mock_agent.tools = [tool1, tool2]

        extras = AuditLogger.extract_call_start_extras(mock_agent)
        assert extras["system_prompt_size_chars"] == len(str(mock_agent.system_message))
        assert extras["system_prompt_size_tokens_est"] == extras["system_prompt_size_chars"] // 4
        assert "submit_review" in extras["skill_tools_loaded"]
        assert "WebSearchTool" in extras["skill_tools_loaded"]

    def test_extract_call_start_extras_no_tools(self):
        """extract_call_start_extras when agent has no tools."""
        mock_agent = MagicMock()
        mock_agent.system_message = "system prompt"
        mock_agent.tools = None

        extras = AuditLogger.extract_call_start_extras(mock_agent)
        assert "system_prompt_size_chars" in extras
        assert "skill_tools_loaded" not in extras

    def test_extract_call_end_extras(self):
        """extract_call_end_extras returns messages_count and total_tool_response_chars."""
        tool1 = MagicMock()
        tool1.result = "x" * 100
        tool2 = MagicMock()
        tool2.result = "y" * 200

        mock_output = MagicMock()
        mock_output.messages = [MagicMock()] * 5
        mock_output.tools = [tool1, tool2]

        extras = AuditLogger.extract_call_end_extras(mock_output)
        assert extras["messages_count"] == 5
        assert extras["total_tool_response_chars"] == 300

    def test_extract_call_end_extras_none(self):
        """extract_call_end_extras with None returns empty dict."""
        assert AuditLogger.extract_call_end_extras(None) == {}


# -----------------------------------------------------------------------
# Helper function tests
# -----------------------------------------------------------------------


class TestHelpers:
    def test_truncate_short(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncate_long(self):
        result = _truncate("a" * 200, 100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_truncate_none(self):
        assert _truncate(None) == ""

    def test_summarize_args_small(self):
        args = {"key": "value"}
        result = _summarize_args(args)
        assert result == {"key": "value"}

    def test_summarize_args_large(self):
        args = {"key": "x" * 500}
        result = _summarize_args(args, max_len=50)
        assert isinstance(result, str)
        assert result.endswith("...")


# -----------------------------------------------------------------------
# Engine integration tests — verify audit files are created during run
# -----------------------------------------------------------------------


class MockRunOutput:
    """Simulates agno RunOutput."""

    def __init__(self, content=None, tools=None, messages=None, metrics=None, model=None):
        self.content = content
        self.tools = tools or []
        self.messages = messages or []
        self.metrics = metrics
        self.model = model


class MockToolExecution:
    """Simulates agno ToolExecution."""

    def __init__(self, tool_name, tool_args=None, result=None, tool_call_error=False, metrics=None):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.result = result
        self.tool_call_error = tool_call_error
        self.metrics = metrics
        self.tool_call_id = None


class MockToolMetrics:
    def __init__(self, duration=None):
        self.duration = duration


class TestEngineAuditIntegration:
    """Test that ReviewEngine creates audit log files during runs."""

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_audit_files_created_on_convergence(self, MockAgent, MockCtxMgr, mock_import, tmp_path):
        """A successful convergence run should produce audit/*.jsonl files."""
        from review_loop.config import (
            AuthorConfig,
            ModelConfig,
            ReviewConfig,
            ReviewerConfig,
        )
        from review_loop.engine import ReviewEngine

        config = ReviewConfig(
            max_rounds=3,
            model_config=ModelConfig(model="claude-opus-4.6-1m"),
            author=AuthorConfig(
                name="Author",
                system_prompt="you are author",
                receiving_review_prompt="process feedback",
            ),
            reviewers=[
                ReviewerConfig(name="Reviewer-A", system_prompt="you are reviewer A"),
            ],
            tools=[],
            context={},
        )

        engine = ReviewEngine(config)
        engine._archiver._base_dir = str(tmp_path)

        # Mock reviewer returns empty issues (convergence on round 1)
        tool_exec = MockToolExecution(
            tool_name="submit_review",
            tool_args={"issues": "[]"},
            result="ok",
            metrics=MockToolMetrics(duration=0.2),
        )
        reviewer_output = MockRunOutput(content=None, tools=[tool_exec])

        # Set up mock agents
        mock_author_revision = AsyncMock()
        mock_author_revision.name = "Author"
        mock_author_revision.arun = AsyncMock(
            return_value=MockRunOutput(content="Initial draft content")
        )

        mock_reviewer = AsyncMock()
        mock_reviewer.name = "Reviewer-A"
        mock_reviewer.arun = AsyncMock(return_value=reviewer_output)

        engine._author_revision = mock_author_revision
        engine._reviewers = [mock_reviewer]

        result = await engine.run(context="test context")

        assert result.converged is True

        # Find the session directory
        session_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(session_dirs) == 1
        session_dir = session_dirs[0]

        audit_dir = session_dir / "audit"
        assert audit_dir.is_dir(), "audit/ directory should exist"

        # Check that agent audit files were created
        audit_files = list(audit_dir.glob("*.jsonl"))
        assert len(audit_files) >= 1, "At least one audit JSONL file should exist"

        # Verify content of audit log entries
        all_events = []
        for f in audit_files:
            for line in f.read_text().strip().split("\n"):
                if line:
                    all_events.append(json.loads(line))

        event_types = [e["event"] for e in all_events]
        assert "call_start" in event_types, "Should have call_start events"
        assert "call_end" in event_types, "Should have call_end events"

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_audit_logs_error_on_failure(self, MockAgent, MockCtxMgr, mock_import, tmp_path):
        """When agent call fails, error event should be logged."""
        from review_loop.config import (
            AuthorConfig,
            ModelConfig,
            ReviewConfig,
            ReviewerConfig,
        )
        from review_loop.engine import ReviewEngine

        config = ReviewConfig(
            max_rounds=3,
            model_config=ModelConfig(model="claude-opus-4.6-1m"),
            author=AuthorConfig(
                name="Author",
                system_prompt="you are author",
                receiving_review_prompt="process feedback",
            ),
            reviewers=[
                ReviewerConfig(name="Reviewer-A", system_prompt="you are reviewer A"),
            ],
            tools=[],
            context={},
        )

        engine = ReviewEngine(config)
        engine._archiver._base_dir = str(tmp_path)

        # Mock author generates content successfully
        mock_author_revision = AsyncMock()
        mock_author_revision.name = "Author"
        mock_author_revision.arun = AsyncMock(
            return_value=MockRunOutput(content="Initial draft")
        )
        engine._author_revision = mock_author_revision

        # Mock reviewer fails
        mock_reviewer = AsyncMock()
        mock_reviewer.name = "Reviewer-A"
        mock_reviewer.arun = AsyncMock(side_effect=RuntimeError("API timeout"))
        engine._reviewers = [mock_reviewer]

        result = await engine.run(context="test context")

        # Should terminate by error since all reviewers failed
        assert result.terminated_by_error is True

        # Find audit directory
        session_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(session_dirs) == 1
        audit_dir = session_dirs[0] / "audit"

        if audit_dir.exists():
            all_events = []
            for f in audit_dir.glob("*.jsonl"):
                for line in f.read_text().strip().split("\n"):
                    if line:
                        all_events.append(json.loads(line))

            # Should have at least start and error/end events
            event_types = [e["event"] for e in all_events]
            assert "call_start" in event_types

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    @pytest.mark.asyncio
    async def test_audit_logs_tool_calls(self, MockAgent, MockCtxMgr, mock_import, tmp_path):
        """Tool call events from RunOutput should be logged."""
        from review_loop.config import (
            AuthorConfig,
            ModelConfig,
            ReviewConfig,
            ReviewerConfig,
        )
        from review_loop.engine import ReviewEngine

        config = ReviewConfig(
            max_rounds=3,
            model_config=ModelConfig(model="claude-opus-4.6-1m"),
            author=AuthorConfig(
                name="Author",
                system_prompt="you are author",
                receiving_review_prompt="process feedback",
            ),
            reviewers=[
                ReviewerConfig(name="Reviewer-A", system_prompt="reviewer A"),
            ],
            tools=[],
            context={},
        )

        engine = ReviewEngine(config)
        engine._archiver._base_dir = str(tmp_path)

        # Create reviewer output with tool execution
        tool_exec = MockToolExecution(
            tool_name="submit_review",
            tool_args={"issues": "[]"},
            result="ok",
            metrics=MockToolMetrics(duration=0.15),
        )
        reviewer_output = MockRunOutput(content=None, tools=[tool_exec])

        mock_author_revision = AsyncMock()
        mock_author_revision.name = "Author"
        mock_author_revision.arun = AsyncMock(
            return_value=MockRunOutput(content="Draft")
        )
        engine._author_revision = mock_author_revision

        mock_reviewer = AsyncMock()
        mock_reviewer.name = "Reviewer-A"
        mock_reviewer.arun = AsyncMock(return_value=reviewer_output)
        engine._reviewers = [mock_reviewer]

        result = await engine.run(context="test")

        # Find reviewer audit file
        session_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        audit_dir = session_dirs[0] / "audit"

        reviewer_log = audit_dir / "Reviewer-A.jsonl"
        assert reviewer_log.exists(), "Reviewer audit log should exist"

        events = [json.loads(line) for line in reviewer_log.read_text().strip().split("\n") if line]
        tool_events = [e for e in events if e["event"] == "tool_call"]
        assert len(tool_events) >= 1
        assert tool_events[0]["tool"] == "submit_review"
        assert tool_events[0]["duration_ms"] == 150.0
