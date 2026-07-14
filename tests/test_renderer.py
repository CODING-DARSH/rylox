from __future__ import annotations

from pathlib import Path

from rylox.assembler import AssembledContext, AssembledEntry
from rylox.cache import CachedChunk
from rylox.renderer import output_filename, render_markdown, slugify_task, write_context_file


def _chunk(
    name: str, content: str, path: str = "a.py", start: int = 1, end: int = 2
) -> CachedChunk:
    return CachedChunk(
        path=path,
        kind="function",
        name=name,
        start_line=start,
        end_line=end,
        parent_class=None,
        docstring=None,
        content=content,
    )


def _entry(name: str, content: str, reason: str, **kwargs: object) -> AssembledEntry:
    return AssembledEntry(
        chunk=_chunk(name, content, **kwargs), reason=reason, tokens=len(content)
    )


def test_all_seven_headers_always_present_even_when_empty() -> None:
    ctx = AssembledContext(query="nothing found", max_tokens=1000)
    md = render_markdown(ctx)
    for header in (
        "# Query",
        "# Summary",
        "# Entry Points",
        "# Execution Flow",
        "# Included Files",
        "# Reasons",
        "# Code",
    ):
        assert header in md


def test_empty_sections_show_explicit_none_not_omitted() -> None:
    ctx = AssembledContext(query="nothing found", max_tokens=1000)
    md = render_markdown(ctx)
    assert md.count("(none)") == 5


def test_query_section_contains_exact_task_string() -> None:
    ctx = AssembledContext(query="How does authentication work?", max_tokens=1000)
    md = render_markdown(ctx)
    assert "How does authentication work?" in md


def test_summary_reports_counts_and_token_budget() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    ctx.entry_points.append(_entry("login", "def login(): pass", "direct match"))
    ctx.total_tokens = 18
    md = render_markdown(ctx)
    assert "1 chunk(s) included" in md
    assert "1 file(s) touched" in md
    assert "18 / 1000 tokens" in md


def test_summary_mentions_skipped_count_when_present() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    ctx.skipped.append(_entry("big_fn", "x" * 50, "direct match"))
    md = render_markdown(ctx)
    assert "1 chunk(s) skipped" in md


def test_summary_omits_skipped_line_when_nothing_skipped() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    md = render_markdown(ctx)
    assert "skipped" not in md


def test_entry_points_section_lists_name_and_location() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    ctx.entry_points.append(_entry("login", "def login(): pass", "direct match", path="auth.py"))
    md = render_markdown(ctx)
    assert "login" in md
    assert "auth.py:1" in md


def test_execution_flow_section_shows_reason_label() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    ctx.execution_flow.append(
        _entry("check_credentials", "def check_credentials(): pass", "callee of login")
    )
    md = render_markdown(ctx)
    assert "callee of login" in md


def test_included_files_deduplicates_across_sections() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    ctx.entry_points.append(_entry("a", "def a(): pass", "direct match", path="same.py"))
    ctx.execution_flow.append(_entry("b", "def b(): pass", "callee of a", path="same.py"))
    md = render_markdown(ctx)
    included_files_section = md.split("# Included Files")[1].split("# Reasons")[0]
    assert included_files_section.count("same.py") == 1


def test_reasons_section_lists_every_included_chunk() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    ctx.entry_points.append(_entry("a", "def a(): pass", "direct match"))
    ctx.supporting_symbols.append(_entry("b", "def b(): pass", "caller of a"))
    md = render_markdown(ctx)
    assert "`a`: direct match" in md
    assert "`b`: caller of a" in md


def test_code_section_contains_exact_chunk_content() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    ctx.entry_points.append(_entry("login", "def login():\n    return True", "direct match"))
    md = render_markdown(ctx)
    assert "def login():\n    return True" in md


def test_code_section_never_includes_skipped_chunks() -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    ctx.entry_points.append(_entry("kept", "def kept(): pass", "direct match"))
    ctx.skipped.append(
        _entry("dropped_chunk_unique_marker", "def dropped(): pass", "direct match")
    )
    md = render_markdown(ctx)
    assert "dropped_chunk_unique_marker" not in md


def test_slugify_handles_empty_string() -> None:
    assert slugify_task("") == "context"


def test_slugify_handles_punctuation_only_string() -> None:
    assert slugify_task("???!!!") == "context"


def test_slugify_truncates_long_task_strings() -> None:
    slug = slugify_task("a" * 200)
    assert len(slug) <= 50


def test_slugify_replaces_special_characters() -> None:
    assert slugify_task("C++ / weird-chars!!") == "c_weird_chars"


def test_slugify_is_lowercase() -> None:
    assert slugify_task("CamelCase Task") == "camelcase_task"


def test_output_filename_appends_context_suffix() -> None:
    assert output_filename("checkout flow") == "checkout_flow_context.md"


def test_write_context_file_creates_real_file(tmp_path: Path) -> None:
    ctx = AssembledContext(query="How does auth work?", max_tokens=1000)
    path = write_context_file(ctx, tmp_path)
    assert path.exists()
    assert path.name == "how_does_auth_work_context.md"
    assert path.read_text(encoding="utf-8").startswith("# Query")


def test_write_context_file_creates_output_dir_if_missing(tmp_path: Path) -> None:
    ctx = AssembledContext(query="q", max_tokens=1000)
    target = tmp_path / "nested" / "output"
    path = write_context_file(ctx, target)
    assert path.exists()


def test_write_context_file_overwrites_existing_file(tmp_path: Path) -> None:
    ctx1 = AssembledContext(query="q", max_tokens=1000)
    ctx1.entry_points.append(_entry("first", "def first(): pass", "direct match"))
    write_context_file(ctx1, tmp_path)

    ctx2 = AssembledContext(query="q", max_tokens=1000)
    ctx2.entry_points.append(_entry("second", "def second(): pass", "direct match"))
    path = write_context_file(ctx2, tmp_path)

    content = path.read_text(encoding="utf-8")
    assert "second" in content
    assert "first" not in content