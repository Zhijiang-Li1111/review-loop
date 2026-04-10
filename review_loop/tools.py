"""Structured submission tools for the review-loop framework.

Instead of relying on output_schema (which fails under API proxies like Agent Maestro),
agents call tool functions to submit structured results. The engine extracts data from
the tool call arguments rather than parsing free-form text.

Three tools:
- submit_review: Reviewers call this to submit their list of issues found.
- submit_verdict: The Author calls this to submit per-issue verdicts (accept/reject/unclear).
- submit_revision: The Author calls this to submit the full revised content.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def submit_review(issues: str) -> str:
    """Submit the final review result with a list of issues found.

    This tool MUST be called as the final action after completing your review.
    All review findings must be submitted through this tool — do not write
    issues in your text response; they will be ignored.

    Args:
        issues: A JSON array of issue objects. Each object must have:
            - "severity": one of "critical", "major", "minor", "suggestion"
            - "content": description of the issue found

            Pass "[]" (empty JSON array) if no issues were found.

    Returns:
        Confirmation message with the number of issues submitted.
    """
    try:
        parsed = json.loads(issues) if isinstance(issues, str) else issues
    except (json.JSONDecodeError, TypeError):
        return f"Error: 'issues' must be a valid JSON array. Got: {str(issues)[:200]}"

    if not isinstance(parsed, list):
        return f"Error: 'issues' must be a JSON array, got {type(parsed).__name__}"

    # Validate each issue has required fields
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            return f"Error: issue {i} must be an object, got {type(item).__name__}"
        if "severity" not in item:
            return f"Error: issue {i} missing 'severity' field"
        if "content" not in item:
            return f"Error: issue {i} missing 'content' field"

    return json.dumps({"status": "submitted", "issue_count": len(parsed)})


def submit_verdict(verdicts: str) -> str:
    """Submit verdicts on each reviewer issue without revising content.

    This tool MUST be called as the final action after evaluating reviewer
    feedback.  Only submit your verdict for each issue here — do NOT include
    updated content.  A separate revision step will follow.

    Args:
        verdicts: A JSON array of verdict objects. Each object must have:
            - "reviewer": name of the reviewer who raised the issue
            - "issue_index": integer index of the issue (0-based, matching the
              order issues were presented)
            - "verdict": one of "accept" (will fix), "reject" (disagree), or
              "unclear" (need clarification)
            - "reason": brief explanation of the verdict

    Returns:
        Confirmation message with verdict counts.
    """
    try:
        parsed_verdicts = json.loads(verdicts) if isinstance(verdicts, str) else verdicts
    except (json.JSONDecodeError, TypeError):
        return f"Error: 'verdicts' must be a valid JSON array. Got: {str(verdicts)[:200]}"

    if not isinstance(parsed_verdicts, list):
        return f"Error: 'verdicts' must be a JSON array, got {type(parsed_verdicts).__name__}"

    for i, item in enumerate(parsed_verdicts):
        if not isinstance(item, dict):
            return f"Error: verdict {i} must be an object, got {type(item).__name__}"
        for field in ("reviewer", "issue_index", "verdict", "reason"):
            if field not in item:
                return f"Error: verdict {i} missing '{field}' field"
        if item["verdict"] not in ("accept", "reject", "unclear"):
            return (
                f"Error: verdict {i} has invalid verdict '{item['verdict']}'. "
                f"Must be 'accept', 'reject', or 'unclear'."
            )

    counts = {"accept": 0, "reject": 0, "unclear": 0}
    for item in parsed_verdicts:
        v = item.get("verdict", "unclear")
        counts[v] = counts.get(v, 0) + 1

    return json.dumps({
        "status": "submitted",
        "verdict_counts": counts,
    })


def submit_revision(updated_content: str) -> str:
    """Submit the full revised content after incorporating accepted feedback.

    This tool MUST be called as the final action after applying changes.
    The updated_content parameter must contain the COMPLETE revised content — not a
    summary, pointer, or reference like "见下方". The engine reads updated_content
    directly and passes it to the next review round; anything omitted is lost.

    Args:
        updated_content: The FULL revised content after incorporating accepted
            feedback. Must be complete and self-contained — this exact string
            replaces the previous version. Do not write "见下方" or any other
            placeholder; paste the entire revised document here.

    Returns:
        Confirmation message with content length.
    """
    # Validate updated_content
    if not isinstance(updated_content, str) or len(updated_content.strip()) == 0:
        return "Error: 'updated_content' must be a non-empty string containing the full revised content."

    # Warn if content looks like a placeholder (heuristic: very short)
    content_len = len(updated_content.strip())
    if content_len < 100:
        return (
            f"Error: 'updated_content' is only {content_len} characters. "
            f"It must contain the COMPLETE revised content, not a summary or pointer. "
            f"Got: {updated_content[:100]!r}"
        )

    return json.dumps({
        "status": "submitted",
        "content_length": content_len,
    })
