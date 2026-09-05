"""Evaluation error types."""

from __future__ import annotations


class EvalError(Exception):
    """Base class for evaluation errors."""


class EvalLoadError(EvalError):
    """Failed to load session JSONL or eval suite."""


class EvalCaseError(EvalError):
    """Invalid or missing eval case configuration."""


class BaselineMismatchError(EvalError):
    """Eval report does not match committed baseline."""
