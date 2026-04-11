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

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def resume_session(self, archive_path: str) -> str:
        """Point at an existing archive directory for appending new rounds."""
        session_path = Path(archive_path).resolve()
        if not session_path.is_dir():
            raise FileNotFoundError(f"Archive directory not found: {archive_path}")
        rounds_path = session_path / "rounds"
        if not rounds_path.is_dir():
            raise FileNotFoundError(
                f"rounds/ subdirectory not found in: {archive_path}"
            )
        self._session_dir = str(session_path)
        return self._session_dir

    def load_history(self) -> list[dict]:
        """Load round history from the current session's rounds/ directory.

        Returns a list of dicts (one per round) with keys:
        ``author_content``, ``reviewer_feedbacks``, ``verdict``, ``response``.
        Only includes complete rounds (those with reviewer feedback files).
        """
        rounds_dir = Path(self._session_dir) / "rounds"

        # Discover round numbers from author files
        round_nums: set[int] = set()
        for f in rounds_dir.iterdir():
            if f.name.startswith("round_") and f.name.endswith("_author.md"):
                try:
                    num = int(f.name.split("_")[1])
                except ValueError:
                    continue
                round_nums.add(num)

        history: list[dict] = []
        for rn in sorted(round_nums):
            author_file = rounds_dir / f"round_{rn}_author.md"

            # Check if this round has any reviewer feedback
            reviewer_files = list(rounds_dir.glob(f"round_{rn}_reviewer_*.json"))
            if not reviewer_files:
                # Incomplete round (e.g., the N+1 author file written after
                # the last completed round) — skip it.
                continue

            record: dict = {
                "round_num": rn,
                "author_content": author_file.read_text(),
                "reviewer_feedbacks": {},
                "verdict": None,
                "response": None,
            }

            # Load reviewer feedbacks
            for f in rounds_dir.glob(f"round_{rn}_reviewer_*.json"):
                # Extract reviewer name from filename
                prefix = f"round_{rn}_reviewer_"
                name = f.name[len(prefix):-len(".json")]
                record["reviewer_feedbacks"][name] = json.loads(f.read_text())

            # Load verdict
            verdict_file = rounds_dir / f"round_{rn}_author_verdict.json"
            if verdict_file.exists():
                record["verdict"] = json.loads(verdict_file.read_text())

            # Load response
            response_file = rounds_dir / f"round_{rn}_author_response.json"
            if response_file.exists():
                record["response"] = json.loads(response_file.read_text())

            history.append(record)

        return history

    def load_context(self) -> str:
        """Read context.md from the current session directory."""
        ctx_path = Path(self._session_dir) / "context.md"
        if not ctx_path.exists():
            raise FileNotFoundError(
                f"context.md not found in: {self._session_dir}"
            )
        return ctx_path.read_text()
