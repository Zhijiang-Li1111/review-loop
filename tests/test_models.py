"""Tests for review_loop.models data structures."""

import dataclasses

from review_loop.models import (
    AuthorResponse,
    AuthorVerdictItem,
    ReviewerFeedback,
    ReviewIssue,
    ReviewResult,
    RoundRecord,
)


class TestReviewIssue:
    def test_instantiation(self):
        issue = ReviewIssue(severity="critical", content="Missing logic step")
        assert issue.severity == "critical"
        assert issue.content == "Missing logic step"

    def test_why_pattern_defaults(self):
        """why and pattern default to empty string (backward compatible)."""
        issue = ReviewIssue(severity="minor", content="Typo")
        assert issue.why == ""
        assert issue.pattern == ""

    def test_why_pattern_explicit(self):
        """why and pattern can be set explicitly."""
        issue = ReviewIssue(
            severity="major",
            content="Missing data source",
            why="Violates evidence-based principle; readers cannot verify the claim",
            pattern="Check all statistical claims for source attribution",
        )
        assert issue.why == "Violates evidence-based principle; readers cannot verify the claim"
        assert issue.pattern == "Check all statistical claims for source attribution"

    def test_asdict(self):
        issue = ReviewIssue(severity="minor", content="Typo")
        d = dataclasses.asdict(issue)
        assert d == {"severity": "minor", "content": "Typo", "why": "", "pattern": ""}

    def test_asdict_with_why_pattern(self):
        """asdict includes why and pattern when set."""
        issue = ReviewIssue(
            severity="critical",
            content="Gap",
            why="Breaks logical chain",
            pattern="Check all causal arguments",
        )
        d = dataclasses.asdict(issue)
        assert d == {
            "severity": "critical",
            "content": "Gap",
            "why": "Breaks logical chain",
            "pattern": "Check all causal arguments",
        }


class TestReviewerFeedback:
    def test_instantiation(self):
        fb = ReviewerFeedback(
            reviewer_name="逻辑审核员",
            issues=[ReviewIssue(severity="critical", content="Gap")],
        )
        assert fb.reviewer_name == "逻辑审核员"
        assert len(fb.issues) == 1

    def test_empty_issues(self):
        fb = ReviewerFeedback(reviewer_name="R1", issues=[])
        assert fb.issues == []

    def test_asdict(self):
        fb = ReviewerFeedback(
            reviewer_name="R1",
            issues=[ReviewIssue(severity="major", content="Data missing")],
        )
        d = dataclasses.asdict(fb)
        assert d["reviewer_name"] == "R1"
        assert len(d["issues"]) == 1
        assert d["issues"][0]["severity"] == "major"


class TestAuthorVerdictItem:
    def test_instantiation(self):
        item = AuthorVerdictItem(
            reviewer="R1",
            issue_index=0,
            verdict="accept",
            reason="Fixed in paragraph 3",
        )
        assert item.reviewer == "R1"
        assert item.issue_index == 0
        assert item.verdict == "accept"
        assert item.reason == "Fixed in paragraph 3"


class TestAuthorResponse:
    def test_instantiation(self):
        resp = AuthorResponse(
            responses=[
                AuthorVerdictItem(
                    reviewer="R1", issue_index=0, verdict="reject", reason="Evidence provided"
                )
            ],
            updated_content="Updated text",
        )
        assert len(resp.responses) == 1
        assert resp.updated_content == "Updated text"


class TestRoundRecord:
    def test_instantiation_defaults(self):
        r = RoundRecord(
            round_num=1,
            author_content="Draft v1",
            reviewer_feedbacks=[],
        )
        assert r.round_num == 1
        assert r.author_content == "Draft v1"
        assert r.reviewer_feedbacks == []
        assert r.author_response is None

    def test_with_all_fields(self):
        r = RoundRecord(
            round_num=2,
            author_content="Draft v2",
            reviewer_feedbacks=[
                ReviewerFeedback(reviewer_name="R1", issues=[])
            ],
            author_response=AuthorResponse(
                responses=[], updated_content="Draft v2 revised"
            ),
        )
        assert r.author_response is not None
        assert r.author_response.updated_content == "Draft v2 revised"

    def test_asdict(self):
        r = RoundRecord(
            round_num=1,
            author_content="Content",
            reviewer_feedbacks=[],
        )
        d = dataclasses.asdict(r)
        assert d["round_num"] == 1
        assert d["author_content"] == "Content"
        assert d["author_response"] is None


class TestReviewResult:
    def test_converged(self):
        result = ReviewResult(
            converged=True,
            rounds_completed=2,
            archive_path="/tmp/archive",
            final_content="Final text",
            unresolved_issues=[],
        )
        assert result.converged is True
        assert result.terminated_by_error is False

    def test_not_converged(self):
        result = ReviewResult(
            converged=False,
            rounds_completed=3,
            archive_path="/tmp/archive",
            final_content="Last draft",
            unresolved_issues=[
                ReviewerFeedback(
                    reviewer_name="R1",
                    issues=[ReviewIssue(severity="critical", content="Unresolved")],
                )
            ],
        )
        assert result.converged is False
        assert len(result.unresolved_issues) == 1

    def test_error_termination(self):
        result = ReviewResult(
            converged=False,
            rounds_completed=1,
            archive_path="/tmp/err",
            final_content=None,
            unresolved_issues=[],
            terminated_by_error=True,
        )
        assert result.terminated_by_error is True
        assert result.final_content is None

    def test_asdict(self):
        result = ReviewResult(
            converged=True,
            rounds_completed=2,
            archive_path="/out",
            final_content="done",
            unresolved_issues=[],
        )
        d = dataclasses.asdict(result)
        assert d["converged"] is True
        assert d["terminated_by_error"] is False
