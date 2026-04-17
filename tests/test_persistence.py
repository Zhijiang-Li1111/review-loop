"""Tests for the Archiver class in review_loop.persistence."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml
import pytest


@dataclass
class _MockConfig:
    max_rounds: int = 3
    model: str = "claude-opus-4.6-1m"


class TestStartSession:
    def test_creates_dirs(self, tmp_path: Path):
        from review_loop.persistence import Archiver

        archiver = Archiver(base_dir=str(tmp_path))
        session_dir = archiver.start_session(_MockConfig())

        session_path = Path(session_dir)
        assert session_path.exists()
        assert (session_path / "rounds").exists()
        assert (session_path / "rounds").is_dir()

    def test_copies_config(self, tmp_path: Path):
        from review_loop.persistence import Archiver

        cfg = _MockConfig()
        archiver = Archiver(base_dir=str(tmp_path))
        session_dir = archiver.start_session(cfg)

        config_path = Path(session_dir) / "config.yaml"
        assert config_path.exists()
        loaded = yaml.safe_load(config_path.read_text())
        assert loaded == asdict(cfg)

    def test_masks_api_key(self, tmp_path: Path):
        from review_loop.persistence import Archiver

        @dataclass
        class _ConfigWithKey:
            model_config: dict

        cfg = _ConfigWithKey(model_config={"model": "m", "api_key": "sk-secret"})
        archiver = Archiver(base_dir=str(tmp_path))
        session_dir = archiver.start_session(cfg)

        config_path = Path(session_dir) / "config.yaml"
        loaded = yaml.safe_load(config_path.read_text())
        assert loaded["model_config"]["api_key"] == "***"


class TestSaveAuthorContent:
    def _make_archiver(self, tmp_path: Path):
        from review_loop.persistence import Archiver
        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_saves_author_md(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        archiver.save_author_content(1, "Draft v1 content")

        path = Path(archiver._session_dir) / "rounds" / "round_1_author.md"
        assert path.exists()
        assert path.read_text() == "Draft v1 content"


class TestSaveReviewerFeedback:
    def _make_archiver(self, tmp_path: Path):
        from review_loop.persistence import Archiver
        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_saves_reviewer_json(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        data = {"issues": [{"severity": "critical", "content": "Gap"}]}
        archiver.save_reviewer_feedback(1, "逻辑审核员", data)

        path = Path(archiver._session_dir) / "rounds" / "round_1_reviewer_逻辑审核员.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data


class TestSaveAuthorResponse:
    def _make_archiver(self, tmp_path: Path):
        from review_loop.persistence import Archiver
        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_saves_author_response_json(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        data = {"responses": [], "updated_content": "v2"}
        archiver.save_author_response(1, data)

        path = Path(archiver._session_dir) / "rounds" / "round_1_author_response.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data


class TestSaveFinal:
    def _make_archiver(self, tmp_path: Path):
        from review_loop.persistence import Archiver
        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_saves_final_md(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        archiver.save_final("Final content here")

        path = Path(archiver._session_dir) / "final.md"
        assert path.exists()
        assert path.read_text() == "Final content here"


class TestSaveUnresolved:
    def _make_archiver(self, tmp_path: Path):
        from review_loop.persistence import Archiver
        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_saves_unresolved_json(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        data = [{"reviewer_name": "R1", "issues": [{"severity": "critical", "content": "X"}]}]
        archiver.save_unresolved(data)

        path = Path(archiver._session_dir) / "unresolved_issues.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data


class TestSaveContextAndError:
    def _make_archiver(self, tmp_path: Path):
        from review_loop.persistence import Archiver
        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_save_context(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        archiver.save_context("Initial context text")

        path = Path(archiver._session_dir) / "context.md"
        assert path.exists()
        assert path.read_text() == "Initial context text"

    def test_save_error_log(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        archiver.save_error_log("All reviewers failed")

        path = Path(archiver._session_dir) / "error.log"
        assert path.exists()
        assert "All reviewers failed" in path.read_text()


class TestWorkspace:
    def _make_archiver(self, tmp_path: Path):
        from review_loop.persistence import Archiver
        archiver = Archiver(base_dir=str(tmp_path))
        archiver.start_session(_MockConfig())
        return archiver

    def test_start_session_creates_workspace_dir(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        workspace = Path(archiver._session_dir) / "workspace"
        assert workspace.exists()
        assert workspace.is_dir()

    def test_workspace_dir_property(self, tmp_path: Path):
        archiver = self._make_archiver(tmp_path)
        assert archiver.workspace_dir is not None
        assert archiver.workspace_dir == Path(archiver._session_dir) / "workspace"

    def test_workspace_dir_none_before_session(self):
        from review_loop.persistence import Archiver
        archiver = Archiver()
        assert archiver.workspace_dir is None

    def test_resume_creates_workspace_dir(self, tmp_path: Path):
        from review_loop.persistence import Archiver

        # Create a session first
        archiver = Archiver(base_dir=str(tmp_path))
        session_dir = archiver.start_session(_MockConfig())

        # Remove workspace dir to simulate old archives
        workspace = Path(session_dir) / "workspace"
        workspace.rmdir()
        assert not workspace.exists()

        # Resume should recreate it
        archiver2 = Archiver()
        archiver2.resume_session(session_dir)
        assert workspace.exists()
