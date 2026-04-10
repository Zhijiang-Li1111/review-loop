"""Tests for review_loop.config — ConfigLoader and config dataclasses."""

import yaml
import pytest

from review_loop.config import (
    AuthorConfig,
    ConfigLoader,
    ModelConfig,
    ReviewConfig,
    ReviewerConfig,
    ToolConfig,
    resolve_env,
)


@pytest.fixture
def sample_config_dict():
    """Minimal valid config dict matching the YAML structure."""
    return {
        "review": {
            "max_rounds": 3,
            "model": "claude-opus-4.6-1m",
            "api_key": "test-key",
            "base_url": "http://localhost:23333/api/anthropic",
        },
        "author": {
            "name": "Author",
            "system_prompt": "You are an author.",
            "receiving_review_prompt": "Process feedback.",
        },
        "reviewers": [
            {"name": "逻辑审核员", "system_prompt": "Check logic."},
            {"name": "数据审核员", "system_prompt": "Check data."},
        ],
    }


@pytest.fixture
def sample_config_yaml(tmp_path, sample_config_dict):
    """Write sample config dict to a YAML file and return the path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(sample_config_dict, allow_unicode=True))
    return str(path)


class TestConfigLoaderFullParse:
    def test_load_returns_review_config(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert isinstance(cfg, ReviewConfig)

    def test_review_fields(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.max_rounds == 3
        assert cfg.model_config.model == "claude-opus-4.6-1m"
        assert cfg.model_config.api_key == "test-key"
        assert cfg.model_config.base_url == "http://localhost:23333/api/anthropic"

    def test_author(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert isinstance(cfg.author, AuthorConfig)
        assert cfg.author.name == "Author"
        assert cfg.author.system_prompt == "You are an author."
        assert cfg.author.receiving_review_prompt == "Process feedback."

    def test_reviewers(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert len(cfg.reviewers) == 2
        assert all(isinstance(r, ReviewerConfig) for r in cfg.reviewers)
        assert cfg.reviewers[0].name == "逻辑审核员"
        assert cfg.reviewers[1].name == "数据审核员"

    def test_tools_default_empty(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.tools == []

    def test_context_default_empty(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.context == {}

    def test_context_builder_default_none(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.context_builder is None


class TestConfigLoaderDefaults:
    def test_min_max_defaults(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["review"] = {"model": "claude-opus-4.6-1m"}
        path = tmp_path / "no_rounds.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))

        cfg = ConfigLoader.load(str(path))
        assert cfg.max_rounds == 10


class TestConfigLoaderValidation:
    def test_missing_review_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        del d["review"]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        with pytest.raises(ValueError, match="review"):
            ConfigLoader.load(str(path))

    def test_missing_author_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        del d["author"]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        with pytest.raises(ValueError, match="author"):
            ConfigLoader.load(str(path))

    def test_missing_reviewers_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        del d["reviewers"]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        with pytest.raises(ValueError, match="reviewers"):
            ConfigLoader.load(str(path))

    def test_empty_reviewers_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["reviewers"] = []
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        with pytest.raises(ValueError, match="reviewers"):
            ConfigLoader.load(str(path))

    def test_missing_model_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["review"] = {"max_rounds": 3}  # model missing
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        with pytest.raises(ValueError, match="model"):
            ConfigLoader.load(str(path))


class TestConfigLoaderOptionalFields:
    def test_tools_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["tools"] = [{"path": "pkg.MyTool"}]
        path = tmp_path / "tools.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert len(cfg.tools) == 1
        assert cfg.tools[0].path == "pkg.MyTool"

    def test_tool_missing_path_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["tools"] = [{"name": "bad"}]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        with pytest.raises(ValueError, match="path"):
            ConfigLoader.load(str(path))

    def test_context_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["context"] = {"research_dir": "~/data/", "days": 7}
        path = tmp_path / "ctx.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.context["research_dir"] == "~/data/"

    def test_context_builder_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["context_builder"] = "my_pkg.context.build"
        path = tmp_path / "cb.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.context_builder == "my_pkg.context.build"

    def test_reviewer_tools_parsed(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["reviewers"] = [
            {"name": "R1", "system_prompt": "Check.", "tools": [{"path": "pkg.SearchTools"}]},
            {"name": "R2", "system_prompt": "Review."},
        ]
        path = tmp_path / "rt.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.reviewers[0].tools is not None
        assert len(cfg.reviewers[0].tools) == 1
        assert cfg.reviewers[0].tools[0].path == "pkg.SearchTools"
        assert cfg.reviewers[1].tools is None

    def test_reviewer_tool_missing_path_raises(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["reviewers"] = [
            {"name": "R1", "system_prompt": "Check.", "tools": [{"name": "bad"}]},
        ]
        path = tmp_path / "bad_rt.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        with pytest.raises(ValueError, match="path"):
            ConfigLoader.load(str(path))


class TestResolveEnv:
    def test_env_prefix_resolves(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "sk-secret")
        assert resolve_env("env:TEST_KEY") == "sk-secret"

    def test_env_prefix_missing_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ValueError, match="MISSING_VAR"):
            resolve_env("env:MISSING_VAR")

    def test_literal_returned(self):
        assert resolve_env("sk-literal") == "sk-literal"

    def test_none_returns_none(self):
        assert resolve_env(None) is None


class TestModelConfig:
    def test_defaults(self):
        mc = ModelConfig(model="m")
        assert mc.api_key is None
        assert mc.base_url is None
        assert mc.temperature is None
        assert mc.max_tokens is None

    def test_to_safe_dict_masks_key(self):
        mc = ModelConfig(model="m", api_key="sk-secret")
        safe = mc.to_safe_dict()
        assert safe["api_key"] == "***"

    def test_to_safe_dict_none_preserved(self):
        mc = ModelConfig(model="m")
        safe = mc.to_safe_dict()
        assert safe["api_key"] is None


class TestConfigLoaderModelFields:
    def test_full_model_config(self, tmp_path, sample_config_dict, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-resolved")
        d = dict(sample_config_dict)
        d["review"] = {
            "model": "claude-opus-4.6-1m",
            "api_key": "env:MY_KEY",
            "base_url": "http://proxy:8080",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        path = tmp_path / "full.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.model_config.api_key == "sk-resolved"
        assert cfg.model_config.temperature == 0.7
        assert cfg.model_config.max_tokens == 4096


class TestAuthorInitialPrompt:
    def test_default_initial_prompt(self, sample_config_yaml):
        cfg = ConfigLoader.load(sample_config_yaml)
        assert cfg.author.initial_prompt == "请基于上述背景资料，生成初始内容。"

    def test_custom_initial_prompt(self, tmp_path, sample_config_dict):
        d = dict(sample_config_dict)
        d["author"]["initial_prompt"] = "请基于以上选题讨论结论，生成一份完整的文章大纲。"
        path = tmp_path / "custom_prompt.yaml"
        path.write_text(yaml.dump(d, allow_unicode=True))
        cfg = ConfigLoader.load(str(path))
        assert cfg.author.initial_prompt == "请基于以上选题讨论结论，生成一份完整的文章大纲。"
