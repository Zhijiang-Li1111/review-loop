# review-loop Implementation Plan

## Task Order

Tasks are ordered by dependency: data structures first, then config, then persistence, then context, then engine, then CLI, then integration.

### Task 1: pyproject.toml

Create project file with dependencies: `agno`, `pyyaml`. Dev deps: `pytest`, `pytest-asyncio`. Same patterns as discuss-agent.

### Task 2: registry.py

Direct copy from discuss-agent. `import_from_path(dotted_path)` utility.

### Task 3: models.py — Data Structures

Dataclasses: `ReviewIssue`, `ReviewerFeedback`, `AuthorVerdictItem`, `AuthorResponse`, `RoundRecord`, `ReviewResult`.

Tests: instantiation, asdict, defaults.

### Task 4: config.py — Configuration Loading

Dataclasses: `ModelConfig` (reuse), `ToolConfig` (reuse), `AuthorConfig`, `ReviewerConfig`, `ReviewConfig`.

`ConfigLoader.load(path)` — YAML parsing, validation, env resolution.

`build_claude(model_config)` — Agno Claude factory (reuse).

`resolve_env(value)` — env: prefix resolution (reuse).

Tests: full parse, defaults, validation errors, per-field tests.

### Task 5: persistence.py — Archiver

Reuse discuss-agent's `Archiver` base. Add review-loop specific methods:
- `save_author_content(round_num, content)` → `rounds/round_{n}_author.md`
- `save_reviewer_feedback(round_num, name, data)` → `rounds/round_{n}_reviewer_{name}.json`
- `save_author_response(round_num, data)` → `rounds/round_{n}_author_response.json`
- `save_final(content)` → `final.md`
- `save_unresolved(data)` → `unresolved_issues.json`

Tests: directory creation, file content verification.

### Task 6: context.py — Context Manager

Simplified from discuss-agent. Just `build_initial_context()` with pluggable builder. No compression needed (review-loop rounds are shorter than multi-agent discussions).

Tests: with builder, without builder.

### Task 7: engine.py — Review Loop Engine (CORE)

The main new code. Steps:

1. `__init__` — create Author agent (with tools), N Reviewer agents (no tools)
2. `_safe_agent_call` — retry wrapper (reuse pattern)
3. `_author_generate` — Author creates initial content from context
4. `_review` — N Reviewers audit in parallel, parse JSON issues
5. `_author_process_feedback` — Author reviews issues with receiving-review prompt, outputs verdict+updated content
6. `_build_reviewer_context` — Per-reviewer: only their issues + Author's responses
7. `run()` — Main loop orchestrating all steps

Tests (mocked, no real LLM calls):
- Agent creation (Author + N Reviewers)
- Author generates content
- Reviewers audit in parallel
- Convergence when all issues empty
- Author processes feedback correctly
- Rebuttals route only to originating reviewer
- max_rounds terminates with unresolved issues
- Single reviewer failure doesn't crash
- All reviewers fail raises error
- Error termination saves error log

### Task 8: main.py + __main__.py — CLI

argparse: `config` (required), `--input` (optional), `--context` (optional).

Tests: accepts yaml path, handles --input/--context, nonexistent config exits.

### Task 9: __init__.py — Package Exports

Export all public symbols.

## Implementation Strategy

- TDD for every task: write failing tests, then implement to make them pass
- Follow discuss-agent patterns exactly where reusing
- Mock all LLM calls in tests (no real API calls)
- Engine tests mock at the agent call level, not internal methods
