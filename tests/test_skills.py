"""Tests for skill support in review-loop config and engine."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml
import pytest

from review_loop.config import (
    AuthorConfig,
    ConfigLoader,
    ReviewConfig,
    ReviewerConfig,
    SkillConfig,
    ToolConfig,
)

FIXTURES_DIR = str(Path(__file__).parent / "fixtures")
TEST_SKILL_PATH = str(Path(__file__).parent / "fixtures" / "test_skill")


# ---------------------------------------------------------------------------
# Config parsing: SkillConfig
# ---------------------------------------------------------------------------


class TestSkillConfigDataclass:
    def test_skill_config_has_path(self):
        sc = SkillConfig(path="/some/path")
        assert sc.path == "/some/path"


# ---------------------------------------------------------------------------
# Config parsing: global skills
# ---------------------------------------------------------------------------


class TestConfigLoaderGlobalSkills:
    @pytest.fixture
    def base_config(self):
        return {
            "review": {"max_rounds": 3, "model": "claude-opus-4.6-1m"},
            "author": {
                "name": "Author",
                "system_prompt": "Write.",
                "receiving_review_prompt": "Review.",
            },
            "reviewers": [
                {"name": "R1", "system_prompt": "Check."},
            ],
        }

    def test_no_skills_defaults_none(self, tmp_path, base_config):
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.skills is None

    def test_global_skills_parsed(self, tmp_path, base_config):
        base_config["skills"] = [{"path": "/path/to/skill"}]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.skills is not None
        assert len(cfg.skills) == 1
        assert cfg.skills[0].path == "/path/to/skill"

    def test_global_skills_missing_path_raises(self, tmp_path, base_config):
        base_config["skills"] = [{"name": "bad"}]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        with pytest.raises(ValueError, match="path"):
            ConfigLoader.load(str(path))


# ---------------------------------------------------------------------------
# Config parsing: author skills
# ---------------------------------------------------------------------------


class TestConfigLoaderAuthorSkills:
    @pytest.fixture
    def base_config(self):
        return {
            "review": {"max_rounds": 3, "model": "claude-opus-4.6-1m"},
            "author": {
                "name": "Author",
                "system_prompt": "Write.",
                "receiving_review_prompt": "Review.",
            },
            "reviewers": [
                {"name": "R1", "system_prompt": "Check."},
            ],
        }

    def test_author_skills_default_none(self, tmp_path, base_config):
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.author.skills is None

    def test_author_skills_parsed(self, tmp_path, base_config):
        base_config["author"]["skills"] = [{"path": "/skill/author"}]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.author.skills is not None
        assert len(cfg.author.skills) == 1
        assert cfg.author.skills[0].path == "/skill/author"

    def test_author_skills_missing_path_raises(self, tmp_path, base_config):
        base_config["author"]["skills"] = [{"bad": "value"}]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        with pytest.raises(ValueError, match="path"):
            ConfigLoader.load(str(path))


# ---------------------------------------------------------------------------
# Config parsing: reviewer skills
# ---------------------------------------------------------------------------


class TestConfigLoaderReviewerSkills:
    @pytest.fixture
    def base_config(self):
        return {
            "review": {"max_rounds": 3, "model": "claude-opus-4.6-1m"},
            "author": {
                "name": "Author",
                "system_prompt": "Write.",
                "receiving_review_prompt": "Review.",
            },
            "reviewers": [
                {"name": "R1", "system_prompt": "Check."},
            ],
        }

    def test_reviewer_skills_default_none(self, tmp_path, base_config):
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.reviewers[0].skills is None

    def test_reviewer_skills_parsed(self, tmp_path, base_config):
        base_config["reviewers"] = [
            {"name": "R1", "system_prompt": "Check.", "skills": [{"path": "/skill/r1"}]},
        ]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.reviewers[0].skills is not None
        assert len(cfg.reviewers[0].skills) == 1
        assert cfg.reviewers[0].skills[0].path == "/skill/r1"

    def test_reviewer_skills_missing_path_raises(self, tmp_path, base_config):
        base_config["reviewers"] = [
            {"name": "R1", "system_prompt": "Check.", "skills": [{"wrong": "x"}]},
        ]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(base_config, allow_unicode=True))
        with pytest.raises(ValueError, match="path"):
            ConfigLoader.load(str(path))


# ---------------------------------------------------------------------------
# Engine: skill loading
# ---------------------------------------------------------------------------


class TestEngineSkillLoading:
    """Test that skills are loaded and passed to Agent during engine init."""

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.build_claude")
    @patch("review_loop.engine.Agent")
    @patch("review_loop.engine.ContextManager")
    def test_author_agent_receives_skills(
        self, MockCtxMgr, MockAgent, mock_build_claude, mock_import
    ):
        from review_loop.engine import ReviewEngine

        mock_build_claude.return_value = MagicMock()

        config = ReviewConfig(
            max_rounds=3,
            model_config=MagicMock(),
            author=AuthorConfig(
                name="Author",
                system_prompt="Write.",
                receiving_review_prompt="Review.",
                skills=[SkillConfig(path=TEST_SKILL_PATH)],
            ),
            reviewers=[ReviewerConfig(name="R1", system_prompt="Check.")],
            tools=[],
            context={},
            skills=None,
        )

        engine = ReviewEngine(config)

        # The Agent constructor should have been called with skills != None
        # for author agents (verdict + revision = first 2 calls)
        calls = MockAgent.call_args_list
        assert len(calls) >= 2  # at least verdict + revision agents
        # Check verdict agent
        assert calls[0].kwargs.get("skills") is not None
        # Check revision agent
        assert calls[1].kwargs.get("skills") is not None

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.build_claude")
    @patch("review_loop.engine.Agent")
    @patch("review_loop.engine.ContextManager")
    def test_reviewer_agent_receives_skills(
        self, MockCtxMgr, MockAgent, mock_build_claude, mock_import
    ):
        from review_loop.engine import ReviewEngine

        mock_build_claude.return_value = MagicMock()

        config = ReviewConfig(
            max_rounds=3,
            model_config=MagicMock(),
            author=AuthorConfig(
                name="Author",
                system_prompt="Write.",
                receiving_review_prompt="Review.",
            ),
            reviewers=[
                ReviewerConfig(
                    name="R1",
                    system_prompt="Check.",
                    skills=[SkillConfig(path=TEST_SKILL_PATH)],
                ),
            ],
            tools=[],
            context={},
            skills=None,
        )

        engine = ReviewEngine(config)

        # 3rd Agent call should be the reviewer
        calls = MockAgent.call_args_list
        assert len(calls) >= 3
        assert calls[2].kwargs.get("skills") is not None

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.build_claude")
    @patch("review_loop.engine.Agent")
    @patch("review_loop.engine.ContextManager")
    def test_global_skills_shared(
        self, MockCtxMgr, MockAgent, mock_build_claude, mock_import
    ):
        from review_loop.engine import ReviewEngine

        mock_build_claude.return_value = MagicMock()

        config = ReviewConfig(
            max_rounds=3,
            model_config=MagicMock(),
            author=AuthorConfig(
                name="Author",
                system_prompt="Write.",
                receiving_review_prompt="Review.",
            ),
            reviewers=[ReviewerConfig(name="R1", system_prompt="Check.")],
            tools=[],
            context={},
            skills=[SkillConfig(path=TEST_SKILL_PATH)],
        )

        engine = ReviewEngine(config)

        # All Agent calls should have skills (from global)
        for call in MockAgent.call_args_list:
            assert call.kwargs.get("skills") is not None

    @patch("review_loop.engine.import_from_path")
    @patch("review_loop.engine.build_claude")
    @patch("review_loop.engine.Agent")
    @patch("review_loop.engine.ContextManager")
    def test_no_skills_passes_none(
        self, MockCtxMgr, MockAgent, mock_build_claude, mock_import
    ):
        from review_loop.engine import ReviewEngine

        mock_build_claude.return_value = MagicMock()

        config = ReviewConfig(
            max_rounds=3,
            model_config=MagicMock(),
            author=AuthorConfig(
                name="Author",
                system_prompt="Write.",
                receiving_review_prompt="Review.",
            ),
            reviewers=[ReviewerConfig(name="R1", system_prompt="Check.")],
            tools=[],
            context={},
            skills=None,
        )

        engine = ReviewEngine(config)

        for call in MockAgent.call_args_list:
            assert call.kwargs.get("skills") is None
