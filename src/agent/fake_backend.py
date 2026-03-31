"""Fake backend utilities for Task 3 policy tests."""

from __future__ import annotations


class FakeBackend:
    """Simple response-queue backend used to test policy flow."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize the backend with a finite sequence of responses."""

        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        """Return the next queued response for a given prompt."""

        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("FakeBackend has no responses remaining.")
        return self._responses.pop(0)
