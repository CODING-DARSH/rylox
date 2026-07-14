from __future__ import annotations

import re
from pathlib import Path

from rylox.assembler import AssembledContext

_MAX_SLUG_LENGTH = 50


def slugify_task(task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")
    slug = slug[:_MAX_SLUG_LENGTH].rstrip("_")
    return slug or "context"


def output_filename(task: str) -> str:
    return f"{slugify_task(task)}_context.md"


def render_markdown(ctx: AssembledContext) -> str:
    """Render an AssembledContext into the fixed markdown structure (spec
    §7). Every section header is always present, even when its content is
    empty — explicitly stated as empty via "(none)", never omitted, so the
    output shape is predictable for anything parsing it programmatically.
    """
    lines: list[str] = []

    lines.append("# Query")
    lines.append(ctx.query)
    lines.append("")

    lines.append("# Summary")
    lines.append(
        f"{len(ctx.included)} chunk(s) included, {len(ctx.related_files)} file(s) touched, "
        f"{ctx.total_tokens} / {ctx.max_tokens} tokens."
    )
    if ctx.skipped:
        lines.append(f"{len(ctx.skipped)} chunk(s) skipped (did not fit budget).")
    lines.append("")

    lines.append("# Entry Points")
    if ctx.entry_points:
        for entry in ctx.entry_points:
            lines.append(f"- `{entry.chunk.name}` — {entry.chunk.path}:{entry.chunk.start_line}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("# Execution Flow")
    if ctx.execution_flow:
        for entry in ctx.execution_flow:
            lines.append(
                f"- `{entry.chunk.name}` — {entry.chunk.path}:{entry.chunk.start_line} "
                f"({entry.reason})"
            )
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("# Included Files")
    if ctx.related_files:
        for path in ctx.related_files:
            lines.append(f"- {path}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("# Reasons")
    if ctx.included:
        for entry in ctx.included:
            lines.append(f"- `{entry.chunk.name}`: {entry.reason}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("# Code")
    if ctx.included:
        for entry in ctx.included:
            lines.append(f"### {entry.chunk.path}:{entry.chunk.start_line}-{entry.chunk.end_line}")
            lines.append(f"_{entry.reason}_")
            lines.append("")
            lines.append("```python")
            lines.append(entry.chunk.content)
            lines.append("```")
            lines.append("")
    else:
        lines.append("(none)")

    return "\n".join(lines).rstrip() + "\n"


def write_context_file(ctx: AssembledContext, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / output_filename(ctx.query)
    path.write_text(render_markdown(ctx), encoding="utf-8")
    return path