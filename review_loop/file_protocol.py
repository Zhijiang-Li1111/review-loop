"""File-based protocol for review-loop communication.

Defines the format and parsing logic for feedback files and verdict files,
replacing the old tool-based channel (submit_review / submit_verdict / submit_revision).

Feedback file format (feedback_R{round}_{reviewer_name}.md):
    ## Issue 1
    - severity: critical
    - content: Description of the issue
    - why: Why this is a problem
    - pattern: Similar pattern hint

    ## No Issues
    (if no issues found)

Verdict file format (verdict_R{round}.md):
    ## Issue 0 (ReviewerName)
    - verdict: accept
    - reason: Explanation
"""

from __future__ import annotations

import re
from review_loop.models import AuthorVerdictItem, ReviewIssue


# ---------------------------------------------------------------------------
# Feedback file format spec (for inclusion in reviewer prompts)
# ---------------------------------------------------------------------------

FEEDBACK_FORMAT_SPEC = """\
请将审核结果写入 feedback 文件，格式如下：

每个 issue 用 ## Issue N 开头（N 从 1 开始），包含以下字段：
- severity: critical / major / minor / suggestion
- content: 问题描述
- why: 为什么这是问题（违反什么原则/会导致什么后果）
- pattern: 同类问题提示（建议检查全文哪些地方有类似模式）

示例：
## Issue 1
- severity: critical
- content: 论点缺乏数据支撑
- why: 没有具体数据佐证会让读者质疑结论可靠性
- pattern: 检查全文其他论点是否也缺少数据

## Issue 2
- severity: minor
- content: 术语未解释
- why: 目标读者可能不了解该术语
- pattern: 检查全文是否有其他未解释的专业术语

如果没有发现问题，请写：
## No Issues
审核通过，未发现问题。
"""

VERDICT_FORMAT_SPEC = """\
请将你对每条审核意见的裁定写入 verdict 文件，格式如下：

每条裁定用 ## Issue N (审核员名) 开头，包含：
- verdict: accept / reject / unclear
- reason: 裁定理由

对于 accept 的 issue，reason 中请详细描述计划的修改（改什么、在哪里、怎么改）。
对于 reject 的 issue，请引用正文中的具体证据。

示例：
## Issue 0 (逻辑审核员)
- verdict: accept
- reason: 确实缺少论据，将在第二段补充具体数据

## Issue 1 (数据审核员)
- verdict: reject
- reason: 原文第三段已有相关数据支撑，审核员可能遗漏
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_field(block: str, field: str) -> str:
    """Extract a field value from a block of text.

    Looks for patterns like:
        - field: value
        - field：value  (Chinese colon)
    """
    pattern = rf"^[-*]\s*{re.escape(field)}\s*[:：]\s*(.+)$"
    m = re.search(pattern, block, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def parse_feedback_file(content: str) -> list[ReviewIssue]:
    """Parse a feedback markdown file into a list of ReviewIssue.

    Returns an empty list if the file contains '## No Issues' or no issue blocks.
    """
    if not content or not content.strip():
        return []

    # Check for "No Issues" marker
    if re.search(r"##\s*No\s*Issues", content, re.IGNORECASE):
        return []

    # Split into issue blocks by ## Issue N headers
    blocks = re.split(r"(?=^##\s*Issue\s+\d+)", content, flags=re.MULTILINE)

    issues: list[ReviewIssue] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Must start with ## Issue
        if not re.match(r"^##\s*Issue\s+\d+", block):
            continue

        severity = _extract_field(block, "severity") or "minor"
        issue_content = _extract_field(block, "content")
        why = _extract_field(block, "why")
        pattern = _extract_field(block, "pattern")

        if not issue_content:
            # Try to get content from the lines after the header that aren't field lines
            continue

        issues.append(ReviewIssue(
            severity=severity.lower(),
            content=issue_content,
            why=why,
            pattern=pattern,
        ))

    return issues


def parse_verdict_file(content: str) -> list[AuthorVerdictItem]:
    """Parse a verdict markdown file into a list of AuthorVerdictItem.

    Expected format:
        ## Issue 0 (ReviewerName)
        - verdict: accept
        - reason: explanation
    """
    if not content or not content.strip():
        return []

    # Split into verdict blocks
    blocks = re.split(r"(?=^##\s*Issue\s+\d+)", content, flags=re.MULTILINE)

    verdicts: list[AuthorVerdictItem] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Parse header: ## Issue N (ReviewerName)
        header_match = re.match(
            r"^##\s*Issue\s+(\d+)\s*\(([^)]+)\)", block
        )
        if not header_match:
            continue

        issue_index = int(header_match.group(1))
        reviewer = header_match.group(2).strip()
        verdict = _extract_field(block, "verdict") or "unclear"
        reason = _extract_field(block, "reason")

        verdicts.append(AuthorVerdictItem(
            reviewer=reviewer,
            issue_index=issue_index,
            verdict=verdict.lower(),
            reason=reason,
        ))

    return verdicts


def feedback_filename(round_num: int, reviewer_name: str) -> str:
    """Return the standard feedback filename for a given round and reviewer."""
    return f"feedback_R{round_num}_{reviewer_name}.md"


def verdict_filename(round_num: int) -> str:
    """Return the standard verdict filename for a given round."""
    return f"verdict_R{round_num}.md"
