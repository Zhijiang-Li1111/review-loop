"""Tests for resume and guidance features."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from review_loop.models import (
    AuthorVerdictItem,
    ReviewerFeedback,
    ReviewIssue,
    ReviewResult,
)


# ---------------------------------------------------------------------------
# Persistence: resume_session, load_history, load_context
# ---------------------------------------------------------------------------


class TestResumeSession:
    def test_sets_session_dir(self, tmp_path: Path):
        from review_loop.persistence import Archiver

        # Create a valid archive structure
        rounds = tmp_path / "session" / "rounds"
        rounds.mkdir(parents=True)

        archiver = Archiver()
        result = archiver.resume_session(str(tmp_path / "session"))
        assert result == str((tmp_path / "session").resolve())
        assert archiver._session_dir == result

    def test_missing_dir_raises(self):
        from review_loop.persistence import Archiver

        archiver = Archiver()
        with pytest.raises(FileNotFoundError, match="Archive directory not found"):
            archiver.resume_session("/nonexistent/path")

    def test_missing_rounds_raises(self, tmp_path: Path):
        from review_loop.persistence import Archiver

        session = tmp_path / "session"
        session.mkdir()
        # No rounds/ subdirectory

        archiver = Archiver()
        with pytest.raises(FileNotFoundError, match="rounds/ subdirectory not found"):
            archiver.resume_session(str(session))


class TestLoadHistory:
    def _make_archive(self, tmp_path: Path, num_rounds: int = 2):
        """Create a mock archive with the given number of rounds."""
        from review_loop.persistence import Archiver

        session = tmp_path / "session"
        rounds = session / "rounds"
        rounds.mkdir(parents=True)
        (session / "context.md").write_text("Test context")

        for rn in range(1, num_rounds + 1):
            (rounds / f"round_{rn}_author.md").write_text(f"Content v{rn}")
            (rounds / f"round_{rn}_reviewer_R1.json").write_text(
                json.dumps({"issues": [{"severity": "minor", "content": f"Issue {rn}"}]})
            )
            (rounds / f"round_{rn}_author_verdict.json").write_text(
                json.dumps([{"reviewer": "R1", "issue_index": 0, "verdict": "accept", "reason": "Ok"}])
            )
            (rounds / f"round_{rn}_author_response.json").write_text(
                json.dumps({"updated_content": f"Content v{rn + 1}"})
            )

        archiver = Archiver()
        archiver.resume_session(str(session))
        return archiver

    def test_loads_correct_round_count(self, tmp_path: Path):
        archiver = self._make_archive(tmp_path, num_rounds=3)
        history = archiver.load_history()
        assert len(history) == 3

    def test_round_numbers_are_sequential(self, tmp_path: Path):
        archiver = self._make_archive(tmp_path, num_rounds=3)
        history = archiver.load_history()
        assert [h["round_num"] for h in history] == [1, 2, 3]

    def test_loads_author_content(self, tmp_path: Path):
        archiver = self._make_archive(tmp_path, num_rounds=2)
        history = archiver.load_history()
        assert history[0]["author_content"] == "Content v1"
        assert history[1]["author_content"] == "Content v2"

    def test_loads_reviewer_feedbacks(self, tmp_path: Path):
        archiver = self._make_archive(tmp_path, num_rounds=1)
        history = archiver.load_history()
        assert "R1" in history[0]["reviewer_feedbacks"]
        assert history[0]["reviewer_feedbacks"]["R1"]["issues"][0]["content"] == "Issue 1"

    def test_loads_verdict(self, tmp_path: Path):
        archiver = self._make_archive(tmp_path, num_rounds=1)
        history = archiver.load_history()
        assert history[0]["verdict"][0]["verdict"] == "accept"

    def test_loads_response(self, tmp_path: Path):
        archiver = self._make_archive(tmp_path, num_rounds=1)
        history = archiver.load_history()
        assert history[0]["response"]["updated_content"] == "Content v2"


    def test_skips_incomplete_rounds(self, tmp_path: Path):
        """Rounds without reviewer feedback (phantom N+1 file) are excluded."""
        from review_loop.persistence import Archiver

        session = tmp_path / "session"
        rounds = session / "rounds"
        rounds.mkdir(parents=True)
        (session / "context.md").write_text("ctx")

        # Complete round 1
        (rounds / "round_1_author.md").write_text("Content v1")
        (rounds / "round_1_reviewer_R1.json").write_text(
            json.dumps({"issues": [{"severity": "minor", "content": "Fix"}]})
        )
        (rounds / "round_1_author_verdict.json").write_text(json.dumps([]))
        (rounds / "round_1_author_response.json").write_text(
            json.dumps({"updated_content": "Content v2"})
        )
        # Phantom round 2 (just the author file, no reviewer feedback)
        (rounds / "round_2_author.md").write_text("Content v2")

        archiver = Archiver()
        archiver.resume_session(str(session))
        history = archiver.load_history()

        assert len(history) == 1
        assert history[0]["round_num"] == 1


class TestLoadContext:
    def test_loads_context(self, tmp_path: Path):
        from review_loop.persistence import Archiver

        session = tmp_path / "session"
        rounds = session / "rounds"
        rounds.mkdir(parents=True)
        (session / "context.md").write_text("My context text")

        archiver = Archiver()
        archiver.resume_session(str(session))
        assert archiver.load_context() == "My context text"

    def test_missing_context_raises(self, tmp_path: Path):
        from review_loop.persistence import Archiver

        session = tmp_path / "session"
        rounds = session / "rounds"
        rounds.mkdir(parents=True)
        # No context.md

        archiver = Archiver()
        archiver.resume_session(str(session))
        with pytest.raises(FileNotFoundError, match="context.md not found"):
            archiver.load_context()


# ---------------------------------------------------------------------------
# Engine: resume mode
# ---------------------------------------------------------------------------


def _make_config(max_rounds=10, num_reviewers=1):
    """Create a minimal ReviewConfig for testing."""
    from review_loop.config import (
        AuthorConfig,
        ModelConfig,
        ReviewConfig,
        ReviewerConfig,
    )

    reviewers = [
        ReviewerConfig(name=f"Reviewer-{chr(65 + i)}", system_prompt=f"Review {i}")
        for i in range(num_reviewers)
    ]
    return ReviewConfig(
        max_rounds=max_rounds,
        model_config=ModelConfig(model="test-model"),
        author=AuthorConfig(
            name="Author",
            system_prompt="Write well",
            receiving_review_prompt="Review carefully",
        ),
        reviewers=reviewers,
        tools=[],
        context={},
    )


class TestEngineResume:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def _setup_engine(self, config, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        engine = ReviewEngine(config)
        engine._archiver = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_resume_loads_history_and_continues(self, tmp_path: Path):
        config = _make_config(max_rounds=10, num_reviewers=1)
        engine = self._setup_engine(config)

        # Set up archiver resume mocks
        engine._archiver.resume_session.return_value = str(tmp_path)
        engine._archiver.load_history.return_value = [
            {
                "round_num": 1,
                "author_content": "Content v1",
                "reviewer_feedbacks": {"Reviewer-A": {"issues": []}},
                "verdict": [],
                "response": {"updated_content": "Content v2"},
            },
            {
                "round_num": 2,
                "author_content": "Content v2",
                "reviewer_feedbacks": {"Reviewer-A": {"issues": []}},
                "verdict": [],
                "response": {"updated_content": "Content v3"},
            },
        ]
        engine._archiver.load_context.return_value = "Loaded context"

        # Mock review to converge immediately
        engine._review = AsyncMock(return_value=[
            ReviewerFeedback(reviewer_name="Reviewer-A", issues=[])
        ])

        result = await engine.run(resume_path=str(tmp_path), extra_rounds=2)

        engine._archiver.resume_session.assert_called_once_with(str(tmp_path))
        engine._archiver.load_history.assert_called_once()
        # Should NOT call start_session or load_context (context is not needed on resume)
        engine._archiver.start_session.assert_not_called()
        assert result.converged is True

    @pytest.mark.asyncio
    async def test_resume_round_num_continues(self, tmp_path: Path):
        """After 2 existing rounds, next round should be 3."""
        config = _make_config(max_rounds=10, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._archiver.resume_session.return_value = str(tmp_path)
        engine._archiver.load_history.return_value = [
            {"round_num": 1, "author_content": "v1", "reviewer_feedbacks": {},
             "verdict": [], "response": {"updated_content": "v2"}},
            {"round_num": 2, "author_content": "v2", "reviewer_feedbacks": {},
             "verdict": [], "response": {"updated_content": "v3"}},
        ]
        engine._archiver.load_context.return_value = "ctx"

        # Round 3: has issues, resolved in round 4
        call_count = {"n": 0}

        async def mock_review(content, per_reviewer_ctx, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [ReviewerFeedback(
                    reviewer_name="Reviewer-A",
                    issues=[ReviewIssue(severity="minor", content="Fix")],
                )]
            return [ReviewerFeedback(reviewer_name="Reviewer-A", issues=[])]

        engine._review = AsyncMock(side_effect=mock_review)
        engine._author_evaluate_feedback = AsyncMock(return_value=[
            AuthorVerdictItem(reviewer="Reviewer-A", issue_index=0,
                              verdict="accept", reason="Done")
        ])
        engine._author_apply_changes = AsyncMock(return_value="v4")

        result = await engine.run(resume_path=str(tmp_path), extra_rounds=2)

        # Check round 3 was saved (the first new round)
        save_calls = engine._archiver.save_reviewer_feedback.call_args_list
        round_nums = [c.args[0] for c in save_calls]
        assert 3 in round_nums

    @pytest.mark.asyncio
    async def test_resume_appends_to_same_dir(self, tmp_path: Path):
        config = _make_config(num_reviewers=1)
        engine = self._setup_engine(config)

        archive_dir = str(tmp_path / "my_session")
        engine._archiver.resume_session.return_value = archive_dir
        engine._archiver.load_history.return_value = [
            {"round_num": 1, "author_content": "v1", "reviewer_feedbacks": {},
             "verdict": [], "response": {"updated_content": "v2"}},
        ]
        engine._archiver.load_context.return_value = "ctx"

        engine._review = AsyncMock(return_value=[
            ReviewerFeedback(reviewer_name="Reviewer-A", issues=[])
        ])

        result = await engine.run(resume_path=archive_dir, extra_rounds=1)
        assert result.archive_path == archive_dir

    @pytest.mark.asyncio
    async def test_resume_without_rounds_raises(self):
        config = _make_config()
        engine = self._setup_engine(config)

        with pytest.raises(ValueError, match="extra_rounds must be a positive integer"):
            await engine.run(resume_path="/some/path")

    @pytest.mark.asyncio
    async def test_resume_with_zero_rounds_raises(self):
        config = _make_config()
        engine = self._setup_engine(config)

        with pytest.raises(ValueError, match="extra_rounds must be a positive integer"):
            await engine.run(resume_path="/some/path", extra_rounds=0)

    @pytest.mark.asyncio
    async def test_resume_with_empty_history_raises(self, tmp_path: Path):
        config = _make_config()
        engine = self._setup_engine(config)

        engine._archiver.resume_session.return_value = str(tmp_path)
        engine._archiver.load_history.return_value = []

        with pytest.raises(ValueError, match="No rounds found in archive"):
            await engine.run(resume_path=str(tmp_path), extra_rounds=2)


class TestRebuildReviewerCtx:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def _setup_engine(self, config, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine
        return ReviewEngine(config)

    def test_rebuilds_ctx_with_verdicts(self):
        config = _make_config(num_reviewers=1)
        engine = self._setup_engine(config)

        last_round = {
            "reviewer_feedbacks": {
                "Reviewer-A": {
                    "issues": [
                        {"severity": "critical", "content": "Data wrong", "why": "No source", "pattern": "Check all claims"},
                    ]
                }
            },
            "verdict": [
                {"reviewer": "Reviewer-A", "issue_index": 0, "verdict": "accept", "reason": "Will fix"}
            ],
        }

        ctx = engine._rebuild_reviewer_ctx_from_history(last_round)
        assert "Reviewer-A" in ctx
        text = ctx["Reviewer-A"]
        assert "issue 0 (critical): Data wrong" in text
        assert "why: No source" in text
        assert "pattern: Check all claims" in text
        assert "[ACCEPT] Will fix" in text

    def test_rebuilds_ctx_no_verdict(self):
        config = _make_config(num_reviewers=1)
        engine = self._setup_engine(config)

        last_round = {
            "reviewer_feedbacks": {
                "R1": {"issues": [{"severity": "minor", "content": "Typo"}]}
            },
            "verdict": [],
        }

        ctx = engine._rebuild_reviewer_ctx_from_history(last_round)
        assert "[未回应]" in ctx["R1"]

    def test_empty_feedbacks(self):
        config = _make_config(num_reviewers=1)
        engine = self._setup_engine(config)

        last_round = {
            "reviewer_feedbacks": {"R1": {"issues": []}},
            "verdict": [],
        }

        ctx = engine._rebuild_reviewer_ctx_from_history(last_round)
        assert ctx == {}


# ---------------------------------------------------------------------------
# Engine: guidance injection
# ---------------------------------------------------------------------------


class TestGuidance:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def _setup_engine(self, config, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        engine = ReviewEngine(config)
        engine._archiver = MagicMock()
        engine._archiver.start_session.return_value = "/tmp/session"
        engine._context_mgr = MagicMock()
        engine._context_mgr.build_initial_context = AsyncMock(return_value="")
        return engine

    @pytest.mark.asyncio
    async def test_guidance_injected_into_author_prompt(self):
        config = _make_config(max_rounds=1, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="content")

        # Capture the feedbacks arg passed to _author_evaluate_feedback
        captured_prompts = {}
        original_evaluate = engine._author_evaluate_feedback

        async def capture_evaluate(content, feedbacks, **kwargs):
            captured_prompts["guidance"] = kwargs.get("guidance")
            return []

        engine._review = AsyncMock(return_value=[
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="minor", content="X")],
            )
        ])
        engine._author_evaluate_feedback = AsyncMock(side_effect=capture_evaluate)
        engine._author_apply_changes = AsyncMock(return_value="updated")

        await engine.run(guidance="Focus on accuracy")

        # Verify guidance was passed through
        assert captured_prompts["guidance"] == "Focus on accuracy"

    @pytest.mark.asyncio
    async def test_guidance_injected_into_reviewer_prompt(self):
        config = _make_config(max_rounds=1, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="content")

        # Capture the guidance arg passed to _review
        captured = {}

        async def capture_review(content, per_reviewer_ctx, **kwargs):
            captured["guidance"] = kwargs.get("guidance")
            return [ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="minor", content="X")],
            )]

        engine._review = AsyncMock(side_effect=capture_review)
        engine._author_evaluate_feedback = AsyncMock(return_value=[])
        engine._author_apply_changes = AsyncMock(return_value="updated")

        await engine.run(guidance="Check data accuracy")

        assert captured["guidance"] == "Check data accuracy"

    @pytest.mark.asyncio
    async def test_no_guidance_when_empty(self):
        config = _make_config(max_rounds=1, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="content")

        captured = {}

        async def capture_review(content, per_reviewer_ctx, **kwargs):
            captured["guidance"] = kwargs.get("guidance")
            return [ReviewerFeedback(reviewer_name="Reviewer-A", issues=[])]

        engine._review = AsyncMock(side_effect=capture_review)

        await engine.run()

        assert captured["guidance"] is None

    @pytest.mark.asyncio
    async def test_guidance_without_resume(self):
        """Guidance works standalone (no --resume)."""
        config = _make_config(max_rounds=1, num_reviewers=1)
        engine = self._setup_engine(config)

        engine._author_generate = AsyncMock(return_value="content")
        engine._review = AsyncMock(return_value=[
            ReviewerFeedback(reviewer_name="Reviewer-A", issues=[])
        ])

        result = await engine.run(guidance="Please focus on clarity")
        assert result.converged is True


# ---------------------------------------------------------------------------
# Guidance prompt injection (unit level)
# ---------------------------------------------------------------------------


class TestGuidancePromptInjection:
    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.ContextManager")
    @patch("review_loop.engine.Agent")
    def _setup_engine(self, config, MockAgent, MockCtxMgr, mock_import):
        from review_loop.engine import ReviewEngine

        engine = ReviewEngine(config)
        return engine

    @pytest.mark.asyncio
    async def test_reviewer_prompt_contains_guidance(self):
        """The actual reviewer prompt should contain the guidance prefix."""
        config = _make_config(max_rounds=1, num_reviewers=1)
        engine = self._setup_engine(config)

        captured_prompts = []

        async def mock_agent_call(agent, prompt):
            captured_prompts.append(prompt)
            # Return a simple mock RunOutput
            mock_output = MagicMock()
            mock_output.tools = []
            mock_output.messages = []
            mock_output.content = '{"issues": []}'
            return mock_output

        engine._safe_agent_call_full = AsyncMock(side_effect=mock_agent_call)

        await engine._review("test content", {}, guidance="Use DeepSeek V4")

        assert len(captured_prompts) == 1
        assert "📋 主编指导意见（供审核参考）：Use DeepSeek V4" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_reviewer_prompt_no_guidance_when_none(self):
        config = _make_config(max_rounds=1, num_reviewers=1)
        engine = self._setup_engine(config)

        captured_prompts = []

        async def mock_agent_call(agent, prompt):
            captured_prompts.append(prompt)
            mock_output = MagicMock()
            mock_output.tools = []
            mock_output.messages = []
            mock_output.content = '{"issues": []}'
            return mock_output

        engine._safe_agent_call_full = AsyncMock(side_effect=mock_agent_call)

        await engine._review("test content", {})

        assert len(captured_prompts) == 1
        assert "主编指导意见" not in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_author_verdict_prompt_contains_guidance(self):
        config = _make_config(num_reviewers=1)
        engine = self._setup_engine(config)

        captured_prompts = []

        async def mock_agent_call(agent, prompt):
            captured_prompts.append(prompt)
            mock_output = MagicMock()
            mock_output.tools = []
            mock_output.messages = []
            mock_output.content = "[]"
            return mock_output

        engine._safe_agent_call_full = AsyncMock(side_effect=mock_agent_call)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="minor", content="Fix this")],
            )
        ]

        await engine._author_evaluate_feedback("content", feedbacks, guidance="Prioritize accuracy")

        assert len(captured_prompts) == 1
        assert "⚠️ 主编指导意见：Prioritize accuracy" in captured_prompts[0]
        assert "请在本轮修改中优先响应以上指导意见" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_author_verdict_prompt_no_guidance_when_none(self):
        config = _make_config(num_reviewers=1)
        engine = self._setup_engine(config)

        captured_prompts = []

        async def mock_agent_call(agent, prompt):
            captured_prompts.append(prompt)
            mock_output = MagicMock()
            mock_output.tools = []
            mock_output.messages = []
            mock_output.content = "[]"
            return mock_output

        engine._safe_agent_call_full = AsyncMock(side_effect=mock_agent_call)

        feedbacks = [
            ReviewerFeedback(
                reviewer_name="Reviewer-A",
                issues=[ReviewIssue(severity="minor", content="Fix this")],
            )
        ]

        await engine._author_evaluate_feedback("content", feedbacks)

        assert len(captured_prompts) == 1
        assert "主编指导意见" not in captured_prompts[0]


# ---------------------------------------------------------------------------
# CLI: --resume, --rounds, --guidance
# ---------------------------------------------------------------------------


class TestCLIResume:
    def test_resume_requires_rounds(self, tmp_path):
        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "review": {"model": "m"},
            "author": {"name": "A", "system_prompt": "s", "receiving_review_prompt": "r"},
            "reviewers": [{"name": "R", "system_prompt": "s"}],
        }))

        with patch("sys.argv", ["review_loop", str(cfg_path), "--resume", "/tmp/archive"]):
            from review_loop.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_rounds_requires_resume(self, tmp_path):
        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "review": {"model": "m"},
            "author": {"name": "A", "system_prompt": "s", "receiving_review_prompt": "r"},
            "reviewers": [{"name": "R", "system_prompt": "s"}],
        }))

        with patch("sys.argv", ["review_loop", str(cfg_path), "--rounds", "2"]):
            from review_loop.main import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_resume_and_rounds_passed(self, tmp_path):
        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "review": {"model": "m"},
            "author": {"name": "A", "system_prompt": "s", "receiving_review_prompt": "r"},
            "reviewers": [{"name": "R", "system_prompt": "s"}],
        }))

        mock_result = ReviewResult(
            converged=True, rounds_completed=3, archive_path="/tmp/out",
            final_content="done", unresolved_issues=[],
        )

        with (
            patch("sys.argv", ["review_loop", str(cfg_path), "--resume", "/tmp/archive", "--rounds", "2"]),
            patch("review_loop.main.ReviewEngine") as mock_engine_cls,
            patch("review_loop.main.ConfigLoader") as mock_loader,
        ):
            mock_loader.load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_result)
            mock_engine_cls.return_value = mock_engine

            from review_loop.main import main
            main()

            call_kwargs = mock_engine.run.call_args.kwargs
            assert call_kwargs["resume_path"] == "/tmp/archive"
            assert call_kwargs["extra_rounds"] == 2

    def test_guidance_passed(self, tmp_path):
        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({
            "review": {"model": "m"},
            "author": {"name": "A", "system_prompt": "s", "receiving_review_prompt": "r"},
            "reviewers": [{"name": "R", "system_prompt": "s"}],
        }))

        mock_result = ReviewResult(
            converged=True, rounds_completed=1, archive_path="/tmp/out",
            final_content="done", unresolved_issues=[],
        )

        with (
            patch("sys.argv", ["review_loop", str(cfg_path), "--guidance", "Focus on accuracy"]),
            patch("review_loop.main.ReviewEngine") as mock_engine_cls,
            patch("review_loop.main.ConfigLoader") as mock_loader,
        ):
            mock_loader.load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_result)
            mock_engine_cls.return_value = mock_engine

            from review_loop.main import main
            main()

            call_kwargs = mock_engine.run.call_args.kwargs
            assert call_kwargs["guidance"] == "Focus on accuracy"
