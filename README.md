# review-loop

[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet?logo=anthropic&logoColor=white)](https://claude.ai/code)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/Zhijiang-Li1111/review-loop)](https://github.com/Zhijiang-Li1111/review-loop/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Zhijiang-Li1111/review-loop)](https://github.com/Zhijiang-Li1111/review-loop/network)

A generic **write-review loop** framework for iterative content refinement through multi-reviewer adversarial review. Define an Author and multiple Reviewers in YAML, and the engine orchestrates an adversarial feedback loop until consensus or max rounds.

## Key Features

- **YAML-driven configuration** — define Author, Reviewers, model, tools, and context in a single config file
- **Multi-reviewer parallel review** — all Reviewers audit content concurrently each round
- **Structured feedback** — every issue includes `severity`, `content`, `why`, and `pattern` fields
- **Author verdict per issue** — Author independently accepts or rejects each issue with evidence; no rubber-stamping
- **Custom tools via Python dotted path** — extend with your own tool classes loaded at runtime
- **Custom context builder** — plug in any async function to build initial context from external sources
- **Automatic archiving** — every session saves config, per-round reviews, verdicts, revisions, and the final output
- **`--input` flag** — review existing content instead of generating from scratch
- **Adversarial by design** — Reviewers hold firm positions; only rigorous logic and evidence can overturn a judgment

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        review-loop Engine                       │
│                                                                 │
│  ┌─────────┐    ┌──────────────────────────────────────────┐    │
│  │         │    │          Parallel Review                  │    │
│  │  Author │    │  ┌────────────┐  ┌────────────┐          │    │
│  │         │──▶ │  │ Reviewer 1 │  │ Reviewer 2 │  · · ·   │    │
│  │ Generate│    │  └─────┬──────┘  └─────┬──────┘          │    │
│  │ Initial │    │        │               │                 │    │
│  │ Content │    │        ▼               ▼                 │    │
│  │         │    │  ┌─────────────────────────────┐         │    │
│  │         │    │  │ Structured Issues (per each) │         │    │
│  │         │    │  └──────────────┬──────────────┘         │    │
│  │         │    └────────────────┬┘                        │    │
│  │         │◀────────────────────┘                         │    │
│  │         │                                               │    │
│  │ Verdict │  accept / reject / unclear (per issue)        │    │
│  │         │                                               │    │
│  │ Revise  │──▶  Updated content  ──▶  Next round ─ ─ ─ ┐ │    │
│  └─────────┘                                             │ │    │
│       ▲                                                  │ │    │
│       └──────────────────────────────────────────────────┘ │    │
│                                                             │    │
│  Converge (all issues resolved) or reach max_rounds         │    │
└─────────────────────────────────────────────────────────────────┘
```

**Loop flow:**

1. **Author** generates initial content (or loads from `--input`)
2. **Reviewers 1..N** audit content **in parallel**, each submitting structured issues
3. If **zero issues** → converged, done ✅
4. **Author evaluates** each issue → `accept` / `reject` / `unclear` with evidence
5. **Author revises** content based on accepted issues
6. Rejected issues + context go back to the specific Reviewer for re-evaluation
7. **Repeat** until convergence or `max_rounds` reached

## Quick Start

### Install

```bash
git clone https://github.com/Zhijiang-Li1111/review-loop.git
cd review-loop
pip install -e .
```

### Minimal Configuration

Create `config.yaml`:

```yaml
review:
  max_rounds: 5
  model: "claude-sonnet-4-20250514"
  api_key: "env:ANTHROPIC_API_KEY"

author:
  name: "Author"
  system_prompt: |
    You are a technical writer. Your task is to produce clear,
    accurate, and well-structured content based on the given context.
  receiving_review_prompt: |
    You received reviewer feedback. For each issue, decide:
    - accept: the issue is valid, you will fix it
    - reject: the reviewer is wrong, provide evidence
    - unclear: need clarification

reviewers:
  - name: "Accuracy Reviewer"
    system_prompt: |
      You review content for factual accuracy. Check every claim
      for supporting evidence. Flag unsupported assertions.

  - name: "Clarity Reviewer"
    system_prompt: |
      You review content for clarity and readability. Flag jargon
      without explanation, ambiguous statements, and logical gaps.
```

### Run

```bash
# Generate and review content (Author writes first draft)
python -m review_loop config.yaml

# Review existing content
python -m review_loop config.yaml --input draft.md

# Provide additional context
python -m review_loop config.yaml --input draft.md --context background.md
```

## Configuration Reference

### Top-Level Structure

```yaml
review:          # Engine settings
author:          # Author agent configuration
reviewers:       # List of reviewer agents
tools:           # (optional) Global tools available to the Author
context:         # (optional) Key-value data passed to context builder
context_builder: # (optional) Python dotted path to async context builder function
```

### `review` — Engine Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | *required* | Model identifier (e.g., `claude-sonnet-4-20250514`) |
| `api_key` | string | `null` | API key. Supports `env:VAR_NAME` syntax to read from environment |
| `base_url` | string | `null` | Custom API base URL (for proxies). Supports `env:VAR_NAME` |
| `temperature` | float | `null` | Sampling temperature |
| `max_tokens` | int | `null` | Max tokens per response |
| `max_rounds` | int | `10` | Maximum review rounds before stopping |

### `author` — Author Agent

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | *required* | Display name for the Author agent |
| `system_prompt` | string | *required* | System prompt defining the Author's role and writing guidelines |
| `initial_prompt` | string | `"请基于上述背景资料，生成初始内容。"` | Prompt used when generating initial content |
| `receiving_review_prompt` | string | `""` | Prompt prepended when Author evaluates reviewer feedback |

### `reviewers` — Reviewer Agents

Each reviewer is an object in the list:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | *required* | Unique display name for this reviewer |
| `system_prompt` | string | *required* | System prompt defining the reviewer's focus area and standards |
| `tools` | list | `null` | Per-reviewer tools (list of `{path: "dotted.path.ClassName"}`) |

**Template variables:** Reviewer system prompts support `{{author.system_prompt}}` to reference the Author's system prompt.

### `tools` — Global Author Tools

```yaml
tools:
  - path: "my_package.tools.WebSearchTool"
  - path: "my_package.tools.DatabaseQueryTool"
```

Each tool is loaded via Python's `importlib` from the dotted path. Tool classes receive a `context` dict at initialization.

### `context` and `context_builder`

```yaml
context:
  source_dir: "./data"
  topic: "API design best practices"

context_builder: "my_package.context.build_context"
```

The `context_builder` function is called with the `context` dict and must return a string:

```python
async def build_context(context: dict) -> str:
    # Load files, query APIs, etc.
    return "Assembled context string..."
```

## Issue Structure

Every issue raised by a Reviewer has four fields:

| Field | Type | Description |
|-------|------|-------------|
| `severity` | string | Issue severity: `critical`, `major`, `minor`, or `suggestion` |
| `content` | string | What the issue is — a clear description of the problem found |
| `why` | string | *Why* it's a problem — what principle it violates or what consequence it causes. Helps the Author understand root cause |
| `pattern` | string | Similar pattern hint — suggests the Author check the entire content for similar occurrences of this issue type |

Example:

```json
{
  "severity": "major",
  "content": "The claim 'X improves performance by 10x' has no citation or benchmark data",
  "why": "Unsupported quantitative claims undermine credibility and may mislead readers",
  "pattern": "Check all performance claims in sections 3 and 5 for supporting data"
}
```

## CLI Usage

```
usage: python -m review_loop [-h] [--input INPUT] [--context CONTEXT]
                              [--resume RESUME] [--rounds ROUNDS]
                              [--guidance GUIDANCE] config

positional arguments:
  config             Path to YAML configuration file

options:
  -h, --help         show this help message and exit
  --input INPUT      Path to initial content file (skip Author generation)
  --context CONTEXT  Path to context file (override context builder)
  --resume RESUME    Path to existing review archive to resume from
  --rounds ROUNDS    Number of additional rounds to run (used with --resume)
  --guidance GUIDANCE  Editor guidance injected into Author and Reviewer prompts
```

### Examples

```bash
# Author generates content from context, reviewers refine it
python -m review_loop config.yaml

# Review an existing document
python -m review_loop config.yaml --input my_document.md

# Provide context from a file (bypasses context_builder)
python -m review_loop config.yaml --context research_notes.md

# Combine both: review existing content with additional context
python -m review_loop config.yaml --input draft.md --context notes.md
```

### Resume — Continue a Previous Session

Resume an existing review session and run additional rounds. New rounds append to the same archive directory.

```bash
# Resume from a previous session, run 2 more rounds
python -m review_loop config.yaml --resume output/2026-04-11_1645 --rounds 2
```

- `--resume <path>` points to an existing archive directory (e.g., `output/2026-04-11_1645`)
- `--rounds <n>` specifies how many additional rounds to run (required with `--resume`)
- Round numbering continues from where the previous session left off
- Context is loaded from the archive (not regenerated)
- The latest author content is reconstructed from the saved rounds

### Guidance — Editor Direction

Inject editor guidance into both Author and Reviewer prompts. The Author sees it as a priority directive; Reviewers see it as an auditing reference.

```bash
# Guidance with a fresh run
python -m review_loop config.yaml --guidance '请特别注意数据准确性'

# Guidance with resume
python -m review_loop config.yaml --resume output/2026-04-11_1645 --rounds 1 \
  --guidance '用 DeepSeek V4 做开篇叙事线索'
```

The Author prompt receives:
```
⚠️ 主编指导意见：{guidance}
请在本轮修改中优先响应以上指导意见。
```

Each Reviewer prompt receives:
```
📋 主编指导意见（供审核参考）：{guidance}
```

When `--guidance` is not provided, no injection occurs.

### Output

Each run creates a timestamped archive directory under `output/`:

```
output/2026-04-11_1430/
├── config.yaml              # Sanitized config (API keys masked)
├── context.md               # Initial context
├── rounds/
│   ├── round_1_author.md              # Author's content for round 1
│   ├── round_1_reviewer_Accuracy.json # Reviewer feedback
│   ├── round_1_reviewer_Clarity.json
│   ├── round_1_author_verdict.json    # Author's per-issue verdicts
│   ├── round_1_author_response.json   # Author's revised content
│   ├── round_2_author.md              # Content entering round 2
│   └── ...
├── final.md                 # Final content
└── unresolved_issues.json   # Issues remaining (if max_rounds hit)
```

## Custom Tools

Tools are Python classes loaded at runtime via dotted path. They receive a `context` dict on initialization.

### Writing a Tool

```python
# my_tools/search.py

class WebSearchTool:
    def __init__(self, context: dict):
        self.api_key = context.get("search_api_key", "")

    def search(self, query: str) -> str:
        """Search the web for information.

        Args:
            query: Search query string.

        Returns:
            Search results as formatted text.
        """
        # Your implementation here
        return f"Results for: {query}"
```

### Registering Tools

In your config:

```yaml
# Global tools (available to Author)
tools:
  - path: "my_tools.search.WebSearchTool"

# Per-reviewer tools
reviewers:
  - name: "Fact Checker"
    system_prompt: "..."
    tools:
      - path: "my_tools.search.WebSearchTool"
```

The `path` is a standard Python dotted import path: `package.module.ClassName`.

## Project Structure

```
review-loop/
├── review_loop/
│   ├── __init__.py        # Public API exports
│   ├── __main__.py        # python -m review_loop entry point
│   ├── main.py            # CLI argument parsing and orchestration
│   ├── config.py          # YAML config loading and validation
│   ├── engine.py          # Core ReviewEngine — the main loop
│   ├── models.py          # Data classes (ReviewIssue, AuthorVerdictItem, etc.)
│   ├── tools.py           # Built-in submission tools (submit_review/verdict/revision)
│   ├── context.py         # ContextManager — builds initial shared context
│   ├── persistence.py     # Archiver — saves session artifacts to disk
│   └── registry.py        # Dynamic import loader for dotted paths
├── tests/
│   ├── test_config.py
│   ├── test_context.py
│   ├── test_engine.py
│   ├── test_main.py
│   ├── test_models.py
│   ├── test_persistence.py
│   ├── test_registry.py
│   └── test_resume.py
├── configs/
│   └── outline_review.yaml  # Example configuration
├── pyproject.toml
├── LICENSE
└── README.md
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with verbose output
pytest -v
```

## License

[MIT](LICENSE) © 2026 Zhijiang Li
