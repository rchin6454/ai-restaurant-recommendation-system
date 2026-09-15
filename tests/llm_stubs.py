"""A stand-in for `groq.Groq` shaped like the parts `src.llm.ranker` touches. No network."""

from __future__ import annotations

import json
from types import SimpleNamespace


class _Completions:
    def __init__(self, responses: tuple) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[min(len(self.calls), len(self.responses)) - 1]  # last one repeats
        if isinstance(response, BaseException):
            raise response
        return response


class StubClient:
    """`StubClient(r1, r2, …)` answers call n with r_n (an exception is raised instead of returned)."""

    def __init__(self, *responses) -> None:
        self.chat = SimpleNamespace(completions=_Completions(responses))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


def completion(content: str | None, *, finish_reason: str = "stop", prompt_tokens: int = 2500,
               cached_tokens: int = 0, completion_tokens: int = 700) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def picks_json(*ids: str, summary: str = "Picked for you.", caveats: tuple[str, ...] = ()) -> str:
    return json.dumps(
        {
            "picks": [
                {"id": rid, "rank": n, "explanation": f"Model prose for {rid}.", "match_highlights": ["model tag"]}
                for n, rid in enumerate(ids, start=1)
            ],
            "summary": summary,
            "caveats": list(caveats),
        }
    )
