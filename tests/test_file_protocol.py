"""Tests for review_loop.file_protocol."""

import pytest
from review_loop.file_protocol import (
    parse_feedback_file,
    parse_verdict_file,
    feedback_filename,
    verdict_filename,
)
from review_loop.models import ReviewIssue, AuthorVerdictItem


class TestParseFeedbackFile:
    def test_empty_string(self):
        assert parse_feedback_file("") == []

    def test_none_like(self):
        assert parse_feedback_file("   ") == []

    def test_no_issues_marker(self):
        content = "## No Issues\n审核通过，未发现问题。"
        assert parse_feedback_file(content) == []

    def test_no_issues_case_insensitive(self):
        content = "## no issues\nAll good."
        assert parse_feedback_file(content) == []

    def test_single_issue(self):
        content = """\
## Issue 1
- severity: critical
- content: 论点缺乏数据支撑
- why: 没有具体数据佐证
- pattern: 检查全文其他论点
"""
        issues = parse_feedback_file(content)
        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert issues[0].content == "论点缺乏数据支撑"
        assert issues[0].why == "没有具体数据佐证"
        assert issues[0].pattern == "检查全文其他论点"

    def test_multiple_issues(self):
        content = """\
## Issue 1
- severity: critical
- content: First issue
- why: Reason 1
- pattern: Pattern 1

## Issue 2
- severity: minor
- content: Second issue
- why: Reason 2
- pattern: Pattern 2
"""
        issues = parse_feedback_file(content)
        assert len(issues) == 2
        assert issues[0].severity == "critical"
        assert issues[0].content == "First issue"
        assert issues[1].severity == "minor"
        assert issues[1].content == "Second issue"

    def test_missing_optional_fields(self):
        content = """\
## Issue 1
- severity: major
- content: Some problem
"""
        issues = parse_feedback_file(content)
        assert len(issues) == 1
        assert issues[0].why == ""
        assert issues[0].pattern == ""

    def test_chinese_colon(self):
        content = """\
## Issue 1
- severity：critical
- content：论点缺乏数据
- why：原因
- pattern：模式
"""
        issues = parse_feedback_file(content)
        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert issues[0].content == "论点缺乏数据"

    def test_severity_default_to_minor(self):
        content = """\
## Issue 1
- content: Problem without severity
"""
        issues = parse_feedback_file(content)
        assert len(issues) == 1
        assert issues[0].severity == "minor"

    def test_preamble_text_ignored(self):
        content = """\
Here is my review of the document.

## Issue 1
- severity: major
- content: Logic gap in paragraph 3
- why: Missing connection
- pattern: Check transitions

Overall the document is decent.
"""
        issues = parse_feedback_file(content)
        assert len(issues) == 1
        assert issues[0].content == "Logic gap in paragraph 3"

    def test_uppercase_severity_normalized(self):
        content = """\
## Issue 1
- severity: CRITICAL
- content: Big problem
"""
        issues = parse_feedback_file(content)
        assert issues[0].severity == "critical"


class TestParseVerdictFile:
    def test_empty_string(self):
        assert parse_verdict_file("") == []

    def test_single_verdict(self):
        content = """\
## Issue 0 (逻辑审核员)
- verdict: accept
- reason: 确实缺少论据
"""
        verdicts = parse_verdict_file(content)
        assert len(verdicts) == 1
        assert verdicts[0].reviewer == "逻辑审核员"
        assert verdicts[0].issue_index == 0
        assert verdicts[0].verdict == "accept"
        assert verdicts[0].reason == "确实缺少论据"

    def test_multiple_verdicts(self):
        content = """\
## Issue 0 (逻辑审核员)
- verdict: accept
- reason: Will fix

## Issue 1 (数据审核员)
- verdict: reject
- reason: Already addressed in paragraph 3
"""
        verdicts = parse_verdict_file(content)
        assert len(verdicts) == 2
        assert verdicts[0].verdict == "accept"
        assert verdicts[1].verdict == "reject"
        assert verdicts[1].reviewer == "数据审核员"

    def test_verdict_default_to_unclear(self):
        content = """\
## Issue 0 (SomeReviewer)
- reason: Not sure about this
"""
        verdicts = parse_verdict_file(content)
        assert len(verdicts) == 1
        assert verdicts[0].verdict == "unclear"

    def test_chinese_colon(self):
        content = """\
## Issue 0 (逻辑审核员)
- verdict：accept
- reason：确实缺少论据
"""
        verdicts = parse_verdict_file(content)
        assert len(verdicts) == 1
        assert verdicts[0].verdict == "accept"


class TestFilenames:
    def test_feedback_filename(self):
        assert feedback_filename(1, "逻辑审核员") == "feedback_R1_逻辑审核员.md"
        assert feedback_filename(3, "Editor") == "feedback_R3_Editor.md"

    def test_verdict_filename(self):
        assert verdict_filename(1) == "verdict_R1.md"
        assert verdict_filename(5) == "verdict_R5.md"
