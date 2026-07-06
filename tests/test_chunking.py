from __future__ import annotations

from pathlib import Path

from rylox.chunking import Chunk, parse_file, parse_source


def _kinds(chunks: list[Chunk]) -> list[tuple[str, str]]:
    return [(c.kind, c.name) for c in chunks]


def _names(chunks: list[Chunk]) -> list[str]:
    return [c.name for c in chunks]


def test_top_level_function_is_chunked_as_function() -> None:
    src = "def greet():\n    pass\n"
    chunks = parse_source(src, Path("mod.py"))
    assert _kinds(chunks) == [("function", "greet")]
    assert chunks[0].parent_class is None
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2


def test_class_and_its_methods_are_both_chunked() -> None:
    src = (
        "class Foo:\n"
        "    def bar(self):\n"
        "        pass\n"
        "    def baz(self):\n"
        "        pass\n"
    )
    chunks = parse_source(src, Path("mod.py"))
    assert _kinds(chunks) == [("class", "Foo"), ("method", "bar"), ("method", "baz")]
    assert chunks[0].parent_class is None
    assert chunks[1].parent_class == "Foo"
    assert chunks[2].parent_class == "Foo"


def test_class_docstring_extracted() -> None:
    src = 'class Foo:\n    """Foo does things."""\n    def bar(self):\n        pass\n'
    chunks = parse_source(src, Path("mod.py"))
    class_chunk = next(c for c in chunks if c.kind == "class")
    assert class_chunk.docstring == "Foo does things."


def test_method_docstring_extracted_single_and_triple_quotes() -> None:
    src = (
        "class Foo:\n"
        "    def bar(self):\n"
        "        '''Bar does things.'''\n"
        "        return 1\n"
        "    def baz(self):\n"
        '        "single quoted"\n'
        "        return 2\n"
    )
    chunks = parse_source(src, Path("mod.py"))
    bar = next(c for c in chunks if c.name == "bar")
    baz = next(c for c in chunks if c.name == "baz")
    assert bar.docstring == "Bar does things."
    assert baz.docstring == "single quoted"


def test_multiline_docstring_extracted_with_internal_newlines() -> None:
    src = (
        "def f():\n"
        '    """\n'
        "    Line1\n"
        "    Line2\n"
        "    Line3\n"
        '    """\n'
        "    pass\n"
    )
    chunks = parse_source(src, Path("mod.py"))
    assert chunks[0].docstring is not None
    assert "Line1" in chunks[0].docstring
    assert "Line2" in chunks[0].docstring
    assert "Line3" in chunks[0].docstring


def test_function_without_docstring_is_none() -> None:
    src = "def f():\n    return 1\n"
    chunks = parse_source(src, Path("mod.py"))
    assert chunks[0].docstring is None


def test_module_level_constants_produce_no_chunks() -> None:
    src = "CONST = 42\nOTHER = 'x'\n"
    chunks = parse_source(src, Path("mod.py"))
    assert chunks == []


def test_imports_do_not_become_chunks() -> None:
    src = "import os\nfrom pathlib import Path\n\ndef f():\n    pass\n"
    chunks = parse_source(src, Path("mod.py"))
    assert _kinds(chunks) == [("function", "f")]


def test_comments_and_blank_lines_do_not_affect_chunking() -> None:
    src = "# a comment\n\n\ndef f():\n    # inner comment\n    pass\n"
    chunks = parse_source(src, Path("mod.py"))
    assert _kinds(chunks) == [("function", "f")]
    assert chunks[0].start_line == 4
    assert chunks[0].end_line == 6


def test_type_hints_do_not_affect_parsing() -> None:
    src = "def f(x: int, y: str = 'a') -> bool:\n    return True\n"
    chunks = parse_source(src, Path("mod.py"))
    assert _kinds(chunks) == [("function", "f")]


def test_nested_function_inside_function_has_no_parent_class() -> None:
    src = "def outer():\n    def inner():\n        pass\n    return inner\n"
    chunks = parse_source(src, Path("mod.py"))
    assert _kinds(chunks) == [("function", "outer"), ("function", "inner")]
    assert all(c.parent_class is None for c in chunks)


def test_multiple_classes_are_independent() -> None:
    src = (
        "class A:\n"
        "    def m(self):\n"
        "        pass\n"
        "class B:\n"
        "    def m(self):\n"
        "        pass\n"
    )
    chunks = parse_source(src, Path("mod.py"))
    methods = [c for c in chunks if c.kind == "method"]
    assert methods[0].parent_class == "A"
    assert methods[1].parent_class == "B"


def test_duplicate_method_names_across_classes_distinguished_by_parent() -> None:
    src = (
        "class A:\n"
        "    def save(self):\n"
        "        pass\n"
        "class B:\n"
        "    def save(self):\n"
        "        pass\n"
    )
    chunks = parse_source(src, Path("mod.py"))
    saves = [c for c in chunks if c.name == "save"]
    assert len(saves) == 2
    assert {c.parent_class for c in saves} == {"A", "B"}
    assert all(c.path == Path("mod.py") for c in saves)


def test_nested_class_chunked_with_immediate_parent() -> None:
    src = "class Outer:\n    class Inner:\n        def m(self):\n            pass\n"
    chunks = parse_source(src, Path("mod.py"))
    outer = next(c for c in chunks if c.name == "Outer")
    inner = next(c for c in chunks if c.name == "Inner")
    method = next(c for c in chunks if c.name == "m")
    assert outer.parent_class is None
    assert inner.parent_class == "Outer"
    assert method.parent_class == "Inner"


def test_async_function_is_chunked_like_a_regular_function() -> None:
    src = 'async def fetch():\n    """Fetch docstring."""\n    pass\n'
    chunks = parse_source(src, Path("mod.py"))
    assert _kinds(chunks) == [("function", "fetch")]
    assert chunks[0].docstring == "Fetch docstring."
    assert chunks[0].content.startswith("async def fetch")


def test_async_method_is_chunked_as_method() -> None:
    src = "class Client:\n    async def fetch(self):\n        pass\n"
    chunks = parse_source(src, Path("mod.py"))
    method = next(c for c in chunks if c.name == "fetch")
    assert method.kind == "method"
    assert method.parent_class == "Client"


def test_decorated_function_span_includes_decorator_line() -> None:
    src = "@cache\ndef foo():\n    pass\n"
    chunks = parse_source(src, Path("mod.py"))
    assert _kinds(chunks) == [("function", "foo")]
    assert chunks[0].start_line == 1  # decorator line, not the def line
    assert chunks[0].end_line == 3
    assert chunks[0].content == "@cache\ndef foo():\n    pass"


def test_decorated_method_span_includes_decorator_line() -> None:
    src = "class Foo:\n    @staticmethod\n    def bar():\n        pass\n"
    chunks = parse_source(src, Path("mod.py"))
    bar = next(c for c in chunks if c.name == "bar")
    assert bar.kind == "method"
    assert bar.parent_class == "Foo"
    assert bar.start_line == 2  # the @staticmethod line
    assert "@staticmethod" in bar.content


def test_multiple_decorators_all_included_in_span() -> None:
    src = "@cache\n@retry\ndef foo():\n    pass\n"
    chunks = parse_source(src, Path("mod.py"))
    assert chunks[0].start_line == 1
    assert "@cache" in chunks[0].content
    assert "@retry" in chunks[0].content


def test_chunk_content_matches_source_exactly() -> None:
    src = "def foo():\n    pass\n"
    chunks = parse_source(src, Path("mod.py"))
    assert chunks[0].content == "def foo():\n    pass"


def test_chunk_path_is_preserved() -> None:
    path = Path("some/nested/module.py")
    chunks = parse_source("def f():\n    pass\n", path)
    assert chunks[0].path == path


def test_large_file_smoke_test_survives_many_functions() -> None:
    """Not a correctness test per se — just confirms the parser doesn't
    choke, mis-walk, or silently drop chunks at scale."""
    src = "\n".join(f"def f_{i}():\n    return {i}\n" for i in range(1000))
    chunks = parse_source(src, Path("big.py"))
    assert len(chunks) == 1000
    assert _names(chunks)[0] == "f_0"
    assert _names(chunks)[-1] == "f_999"


def test_parse_file_reads_real_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.py"
    file_path.write_text("def greet():\n    '''hi'''\n    pass\n", encoding="utf-8")
    result = parse_file(file_path)
    assert result.error is None
    assert len(result.chunks) == 1
    assert result.chunks[0].docstring == "hi"


def test_parse_file_missing_file_returns_error_not_exception(tmp_path: Path) -> None:
    result = parse_file(tmp_path / "does_not_exist.py")
    assert result.chunks == []
    assert result.error is not None


def test_parse_file_invalid_utf8_returns_error_not_exception(tmp_path: Path) -> None:
    file_path = tmp_path / "bad_encoding.py"
    file_path.write_bytes(b"\xff\xfe\x00\x01 not valid utf8 \x80\x81")
    result = parse_file(file_path)
    assert result.chunks == []
    assert result.error is not None


def test_parse_file_syntactically_broken_does_not_crash(tmp_path: Path) -> None:
    """tree-sitter is error-tolerant; the bar here is 'doesn't raise', not
    'produces zero chunks' — partial/best-effort results are acceptable."""
    file_path = tmp_path / "broken.py"
    file_path.write_text("def f(:\n    !!! not python ###\n", encoding="utf-8")
    result = parse_file(file_path)
    assert isinstance(result.chunks, list)  # did not raise


def test_empty_file_produces_no_chunks(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.py"
    file_path.write_text("", encoding="utf-8")
    result = parse_file(file_path)
    assert result.chunks == []
    assert result.error is None


def test_decorated_class_span_includes_decorator_line() -> None:
    src = "@dataclass\nclass Point:\n    x: int\n    y: int\n"
    chunks = parse_source(src, Path("mod.py"))
    assert chunks[0].kind == "class"
    assert chunks[0].start_line == 1
    assert chunks[0].content.startswith("@dataclass")


def test_crlf_line_endings_produce_clean_content_and_correct_lines() -> None:
    src = "def foo():\r\n    pass\r\n\r\ndef bar():\r\n    pass\r\n"
    chunks = parse_source(src, Path("mod.py"))
    assert chunks[0].name == "foo"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2
    assert chunks[0].content == "def foo():\n    pass"  # normalized, no stray \r
    assert chunks[1].name == "bar"
    assert chunks[1].start_line == 4


def test_utf8_bom_stripped_and_does_not_leak_into_content(tmp_path: Path) -> None:
    """Windows editors commonly write UTF-8 files with a BOM. Left unhandled,
    the BOM character silently prepends onto the first chunk's content,
    which would corrupt embedding/search without ever raising an error."""
    file_path = tmp_path / "bom.py"
    file_path.write_bytes(b"\xef\xbb\xbf" + b"def foo():\n    pass\n")
    result = parse_file(file_path)
    assert result.error is None
    assert result.chunks[0].content == "def foo():\n    pass"
    assert "\ufeff" not in result.chunks[0].content


def test_windows_style_path_preserved() -> None:
    """Chunk.path should just carry whatever Path it was given — including
    a Windows-style path object — without transformation."""
    windows_path = Path(r"C:\Users\dev\project\module.py")
    chunks = parse_source("def f():\n    pass\n", windows_path)
    assert chunks[0].path == windows_path