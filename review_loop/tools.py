"""Review submission tool for structured reviewer output via tool calling.

Instead of relying on output_schema (which fails under API proxies like Agent Maestro),
reviewers call submit_review() as a tool to submit structured results.
The engine extracts the issues from the tool call arguments.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def submit_review(issues: str) -> str:
    """Submit the final review result with a list of issues found.

    This tool MUST be called as the final action after completing your review.
    All review findings must be submitted through this tool.

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
