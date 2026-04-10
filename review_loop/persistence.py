"""Archiver — review session persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml


def _mask_api_keys(d: dict) -> None:
    """Recursively mask any ``api_key`` values in a nested dict."""
    for key, value in d.items():
        if key == "api_key" and value is not None:
            d[key] = "***"
        elif isinstance(value, dict):
            _mask_api_keys(value)


class Archiver:
    """Persist review session artefacts to disk."""

    def __init__(self, base_dir: str = "output") -> None:
        self._base_dir = base_dir
        self._session_dir: str | None = None

    def start_session(self, config) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        session_path = Path(self._base_dir) / timestamp
        rounds_path = session_path / "rounds"
        os.makedirs(rounds_path, exist_ok=True)

        config_dict = asdict(config)
        _mask_api_keys(config_dict)
        config_file = session_path / "config.yaml"
        config_file.write_text(yaml.dump(config_dict, allow_unicode=True))

        self._session_dir = str(session_path.resolve())
        return self._session_dir

    def save_author_content(self, round_num: int, content: str) -> None:
        path = Path(self._session_dir) / "rounds" / f"round_{round_num}_author.md"
        path.write_text(content)

    def save_reviewer_feedback(self, round_num: int, name: str, data: dict) -> None:
        path = Path(self._session_dir) / "rounds" / f"round_{round_num}_reviewer_{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def save_author_verdict(self, round_num: int, data: list) -> None:
        path = Path(self._session_dir) / "rounds" / f"round_{round_num}_author_verdict.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def save_author_response(self, round_num: int, data: dict) -> None:
        path = Path(self._session_dir) / "rounds" / f"round_{round_num}_author_response.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def save_final(self, content: str) -> None:
        path = Path(self._session_dir) / "final.md"
        path.write_text(content)

    def save_unresolved(self, data: list) -> None:
        path = Path(self._session_dir) / "unresolved_issues.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def save_context(self, context: str) -> None:
        path = Path(self._session_dir) / "context.md"
        path.write_text(context)

    def save_error_log(self, error: str) -> None:
        path = Path(self._session_dir) / "error.log"
        path.write_text(f"Review terminated by error:\n{error}\n")
