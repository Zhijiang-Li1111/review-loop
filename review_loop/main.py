"""CLI entry point for the review-loop framework."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from review_loop.config import ConfigLoader
from review_loop.engine import ReviewEngine

logger = logging.getLogger("review_loop")


def _setup_logging(session_dir: str | None = None) -> None:
    """Configure logging to stderr + optional file."""
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    root = logging.getLogger()

    if not root.handlers:
        # First call: set up stderr handler
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(formatter)
        root.addHandler(stderr_handler)

    if session_dir:
        log_path = os.path.join(session_dir, "run.log")
        os.makedirs(session_dir, exist_ok=True)
        formatter = logging.Formatter(fmt)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def _install_signal_handlers() -> None:
    """Log a message on SIGTERM before exiting."""
    def _on_sigterm(signum, frame):
        logger.critical("Received SIGTERM (signal %d) — exiting", signum)
        sys.exit(143)  # 128 + 15

    signal.signal(signal.SIGTERM, _on_sigterm)


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

    # Set up logging early (stderr only until we know the session dir)
    _setup_logging()
    _install_signal_handlers()

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

    try:
        result = asyncio.run(engine.run(**run_kwargs))
    except MemoryError:
        logger.critical("MemoryError — process running out of memory")
        sys.exit(137)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (SIGINT)")
        sys.exit(130)
    except Exception:
        logger.critical("Unhandled exception in review loop", exc_info=True)
        sys.exit(1)

    # Reconfigure logging to also write to session log file
    if result.archive_path:
        _setup_logging(result.archive_path)

    logger.info("Review archived at: %s", result.archive_path)
    print(f"Review archived at: {result.archive_path}")
    if result.converged:
        print(f"Status: converged in {result.rounds_completed} round(s)")
    elif result.terminated_by_error:
        print("Status: terminated by error")
    else:
        print(f"Status: max rounds ({result.rounds_completed}) reached without convergence")
