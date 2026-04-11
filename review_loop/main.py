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
    parser.add_argument(
        "--resume",
        help="Path to existing review archive to resume from",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Number of additional rounds to run (used with --resume)",
    )
    parser.add_argument(
        "--guidance",
        help="Editor guidance injected into Author and Reviewer prompts",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.resume and not args.rounds:
        print("Error: --rounds is required when using --resume", file=sys.stderr)
        sys.exit(1)

    if args.rounds and not args.resume:
        print("Error: --rounds can only be used with --resume", file=sys.stderr)
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

    if args.resume:
        run_kwargs["resume_path"] = args.resume
        run_kwargs["extra_rounds"] = args.rounds

    if args.guidance:
        run_kwargs["guidance"] = args.guidance

    result = asyncio.run(engine.run(**run_kwargs))

    print(f"Review archived at: {result.archive_path}")
    if result.converged:
        print(f"Status: converged in {result.rounds_completed} round(s)")
    elif result.terminated_by_error:
        print("Status: terminated by error")
    else:
        print(f"Status: max rounds ({result.rounds_completed}) reached without convergence")
