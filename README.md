# review-loop

[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet?logo=anthropic&logoColor=white)](https://claude.ai/code)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A generic write-review loop framework for iterative content refinement through multi-agent adversarial review.

## What It Does

Author writes content → multiple Reviewers audit in parallel → Author processes feedback (accept / reject with evidence) → rejected issues go back to the specific Reviewer → loop until all issues resolved.

**Key principle:** Adversarial by design. Reviewers hold firm positions. Only rigorous logic and evidence can overturn a judgment. No rubber-stamping.

## Architecture

Built on the same foundation as [discuss-agent](https://github.com/Zhijiang-Li1111/discuss-agent) — YAML-driven configuration, Agno + Claude for agents, structured output for review feedback.

## Install

```bash
pip install -e .
```

## Usage

```bash
python -m review_loop configs/outline_review.yaml
```

## Design

See [docs/DESIGN.md](docs/DESIGN.md) for the complete design document.

## License

MIT
