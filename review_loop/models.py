"""Data structures for the review-loop framework."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReviewIssue:
    severity: str  # "critical" | "major" | "minor"
    content: str


@dataclass
class ReviewerFeedback:
    reviewer_name: str
    issues: list[ReviewIssue]


@dataclass
class AuthorVerdictItem:
    reviewer: str
    issue_index: int
    verdict: str  # "accept" | "reject" | "unclear"
    reason: str


@dataclass
class AuthorResponse:
    responses: list[AuthorVerdictItem]
    updated_content: str


@dataclass
class RoundRecord:
    round_num: int
    author_content: str
    reviewer_feedbacks: list[ReviewerFeedback]
    author_response: AuthorResponse | None = None


@dataclass
class ReviewResult:
    converged: bool
    rounds_completed: int
    archive_path: str
    final_content: str | None
    unresolved_issues: list[ReviewerFeedback]
    terminated_by_error: bool = False
