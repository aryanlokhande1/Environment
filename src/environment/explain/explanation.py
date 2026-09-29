"""Human-readable rendering for audit records."""
from __future__ import annotations
import json
from typing import Any, Iterable

def render_explanations(records: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in records:
        detail = row.get("explanation", {})
        lines.extend([
            f"[{row.get('explanation_time')}] {row.get('category')}",
            f"APPLICATION: {row.get('application_id')} / CONTEXT: {row.get('context_id')}",
            "EVIDENCE: " + json.dumps(detail, sort_keys=True, default=str),
        ])
    return "\n".join(lines)
