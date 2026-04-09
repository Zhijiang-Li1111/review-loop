"""CLI entry point for the review-loop framework."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from review_loop.config import ConfigLoader
from review_loop.engine import ReviewEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a write-review loop."
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--input", help="Path to initial content file")
    parser.add_argument("--context", help="Path to context file")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    config = ConfigLoader.load(args.config)
    engine = ReviewEngine(config)

    run_kwargs: dict = {}

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            run_kwargs["initial_content"] = f.read()

    if args.context:
        with open(args.context, "r", encoding="utf-8") as f:
            run_kwargs["context"] = f.read()

    result = asyncio.run(engine.run(**run_kwargs))

    print(f"Review archived at: {result.archive_path}")
    if result.converged:
        print(f"Status: converged in {result.rounds_completed} round(s)")
    elif result.terminated_by_error:
        print("Status: terminated by error")
    else:
        print(f"Status: max rounds ({result.rounds_completed}) reached without convergence")
