"""Tests for the CLI entry point."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from review_loop.models import ReviewResult


class TestCLI:
    def test_accepts_yaml_path(self, tmp_path):
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"review": {"model": "m"}, "author": {"name": "A", "system_prompt": "s", "receiving_review_prompt": "r"}, "reviewers": [{"name": "R", "system_prompt": "s"}]}))

        mock_result = ReviewResult(
            converged=True,
            rounds_completed=2,
            archive_path="/tmp/output/2026-04-10_1200",
            final_content="Final",
            unresolved_issues=[],
        )

        with (
            patch("sys.argv", ["review_loop", str(cfg_path)]),
            patch("review_loop.main.ReviewEngine") as mock_engine_cls,
            patch("review_loop.main.ConfigLoader") as mock_loader,
        ):
            mock_loader.load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_result)
            mock_engine_cls.return_value = mock_engine

            from review_loop.main import main
            main()

            mock_loader.load.assert_called_once_with(str(cfg_path))
            mock_engine_cls.assert_called_once()

    def test_prints_archive_path(self, tmp_path, capsys):
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"review": {"model": "m"}, "author": {"name": "A", "system_prompt": "s", "receiving_review_prompt": "r"}, "reviewers": [{"name": "R", "system_prompt": "s"}]}))

        mock_result = ReviewResult(
            converged=True,
            rounds_completed=2,
            archive_path="/tmp/output/2026-04-10_1200",
            final_content="Final",
            unresolved_issues=[],
        )

        with (
            patch("sys.argv", ["review_loop", str(cfg_path)]),
            patch("review_loop.main.ReviewEngine") as mock_engine_cls,
            patch("review_loop.main.ConfigLoader") as mock_loader,
        ):
            mock_loader.load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_result)
            mock_engine_cls.return_value = mock_engine

            from review_loop.main import main
            main()

            captured = capsys.readouterr()
            assert "/tmp/output/2026-04-10_1200" in captured.out

    def test_nonexistent_config_exits(self):
        with patch("sys.argv", ["review_loop", "/nonexistent/config.yaml"]):
            from review_loop.main import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_input_flag_passed(self, tmp_path, capsys):
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"review": {"model": "m"}, "author": {"name": "A", "system_prompt": "s", "receiving_review_prompt": "r"}, "reviewers": [{"name": "R", "system_prompt": "s"}]}))

        input_file = tmp_path / "draft.md"
        input_file.write_text("Initial draft content")

        mock_result = ReviewResult(
            converged=True, rounds_completed=1, archive_path="/tmp/out",
            final_content="done", unresolved_issues=[],
        )

        with (
            patch("sys.argv", ["review_loop", str(cfg_path), "--input", str(input_file)]),
            patch("review_loop.main.ReviewEngine") as mock_engine_cls,
            patch("review_loop.main.ConfigLoader") as mock_loader,
        ):
            mock_loader.load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_result)
            mock_engine_cls.return_value = mock_engine

            from review_loop.main import main
            main()

            call_kwargs = mock_engine.run.call_args
            assert call_kwargs.kwargs.get("initial_content") == "Initial draft content"

    def test_context_flag_passed(self, tmp_path, capsys):
        import yaml
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"review": {"model": "m"}, "author": {"name": "A", "system_prompt": "s", "receiving_review_prompt": "r"}, "reviewers": [{"name": "R", "system_prompt": "s"}]}))

        ctx_file = tmp_path / "context.md"
        ctx_file.write_text("Research context here")

        mock_result = ReviewResult(
            converged=True, rounds_completed=1, archive_path="/tmp/out",
            final_content="done", unresolved_issues=[],
        )

        with (
            patch("sys.argv", ["review_loop", str(cfg_path), "--context", str(ctx_file)]),
            patch("review_loop.main.ReviewEngine") as mock_engine_cls,
            patch("review_loop.main.ConfigLoader") as mock_loader,
        ):
            mock_loader.load.return_value = MagicMock()
            mock_engine = MagicMock()
            mock_engine.run = AsyncMock(return_value=mock_result)
            mock_engine_cls.return_value = mock_engine

            from review_loop.main import main
            main()

            call_kwargs = mock_engine.run.call_args
            assert call_kwargs.kwargs.get("context") == "Research context here"
