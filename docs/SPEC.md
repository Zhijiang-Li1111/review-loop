# review-loop Spec

## Overview

Write-review loop framework built on discuss-agent's architecture. Author generates content, N Reviewers audit in parallel with structured issue output, Author processes feedback using receiving-review methodology (accept/reject/unclear), rebuttals route only to the originating Reviewer. Loop until all Reviewers report zero issues or max_rounds reached.

## Modules

### 1. config.py — Configuration Loading

Reuse discuss-agent patterns. Load YAML with these top-level sections:

- `review` — `max_rounds` (default 10), `model`, `api_key`, `base_url`, `temperature`, `max_tokens`
- `author` — `name`, `system_prompt`, `receiving_review_prompt`
- `reviewers` — list of `{name, system_prompt}`
- `tools` — optional list of `{path}` for Author tools
- `context_builder` — optional dotted path to async context builder
- `context` — optional opaque dict for context builder

Dataclasses: `ModelConfig`, `ToolConfig`, `AuthorConfig`, `ReviewerConfig`, `ReviewConfig`. ConfigLoader validates required keys, resolves `env:VAR` references.

### 2. models.py — Data Structures

- `ReviewIssue` — `severity` (critical/major/minor), `content` (str)
- `ReviewerFeedback` — `reviewer_name` (str), `issues` (list[ReviewIssue])
- `AuthorVerdictItem` — `reviewer` (str), `issue_index` (int), `verdict` (accept/reject/unclear), `reason` (str)
- `AuthorResponse` — `responses` (list[AuthorVerdictItem]), `updated_content` (str)
- `RoundRecord` — `round_num` (int), `author_content` (str), `reviewer_feedbacks` (list[ReviewerFeedback]), `author_response` (AuthorResponse | None)
- `ReviewResult` — `converged` (bool), `rounds_completed` (int), `archive_path` (str), `final_content` (str | None), `unresolved_issues` (list[ReviewerFeedback]), `terminated_by_error` (bool)

### 3. context.py — Context Building

Reuse discuss-agent's ContextManager pattern. Accepts optional async context_builder callable. Returns initial context string.

### 4. persistence.py — Archiving

Reuse discuss-agent's Archiver pattern. Directory layout per design doc:

```
output_dir/YYYY-MM-DD_HHMM/
├── config.yaml
├── context.md
├── rounds/
│   ├── round_1_author.md
│   ├── round_1_reviewer_{name}.json
│   ├── round_1_author_response.json
│   ├── round_2_author.md
│   └── ...
├── final.md
└── unresolved_issues.json
```

New methods: `save_author_content`, `save_reviewer_feedback`, `save_author_response`, `save_final`, `save_unresolved`.

### 5. registry.py — Dynamic Import

Direct reuse of discuss-agent's `import_from_path`.

### 6. engine.py — Core Review Loop (NEW)

The only module that requires fresh implementation.

**`ReviewEngine.__init__(config)`:**
- Import tools via registry
- Import context_builder via registry
- Create Author agent (Agno Agent with tools)
- Create N Reviewer agents (no tools)

**`ReviewEngine.run(initial_content=None, context=None)`:**

1. Start session, build/load context
2. If no initial_content → Author generates v1 from context
3. For round 1..max_rounds:
   a. N Reviewers audit current content in parallel (`asyncio.gather`)
   b. Each Reviewer outputs `{"issues": [...]}` — structured JSON
   c. If all issues empty → save final, return converged
   d. Author processes all issues (receiving-review prompt injected)
   e. Author outputs `{"responses": [...], "updated_content": "..."}` — structured JSON
   f. Update content, build per-reviewer context for next round
   g. Per-reviewer context: only that reviewer's issues + Author's responses to them
4. If max_rounds reached → save final + unresolved issues, return not converged

**Reviewer prompt (subsequent rounds) injects:**
- That reviewer's previous issues
- Author's response to each (accept/reject/unclear with reason)
- Instruction to evaluate: close resolved, maintain unresolved, may add new

**Author receives `receiving_review_prompt` plus all issues from all reviewers.**

**Error handling:** Same pattern as discuss-agent — `_safe_agent_call` with 1 retry, single reviewer failure is non-fatal, all reviewers failing raises `AllReviewersFailedError`.

### 7. main.py — CLI

```
python -m review_loop config.yaml [--input file.md] [--context file.md]
```

### 8. __init__.py — Package Exports

Export all public classes.

## Acceptance Criteria

1. YAML config loads with all fields, validates required keys, resolves env vars
2. Author generates content from context when no initial_content provided
3. N Reviewers audit in parallel, each outputs structured `{issues: [...]}` JSON
4. When all issues empty → converged, final content saved
5. Author processes feedback with receiving-review methodology (accept/reject/unclear)
6. Rebuttals only sent to originating Reviewer (not broadcast)
7. Reviewer sees only its own previous issues + Author's responses in next round
8. Loop terminates at max_rounds with unresolved issues saved
9. All rounds persisted to disk in specified directory structure
10. CLI accepts config path + optional --input and --context flags
11. Tools/context_builder work when configured
12. Single reviewer failure doesn't crash the loop
13. All reviewers failing raises AllReviewersFailedError and terminates gracefully
