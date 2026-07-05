from __future__ import annotations

from pathlib import Path

from rylox.chunking import Chunk, parse_file, parse_source


def _kinds(chunks: list[Chunk]) -> list[tuple[str, str]]:
    return [(c.kind, c.name) for c in chunks]


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


def test_function_without_docstring_is_none() -> None:
    src = "def f():\n    return 1\n"
    chunks = parse_source(src, Path("mod.py"))
    assert chunks[0].docstring is None


def test_module_level_constants_produce_no_chunks() -> None:
    src = "CONST = 42\nOTHER = 'x'\n"
    chunks = parse_source(src, Path("mod.py"))
    assert chunks == []


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


def test_chunk_path_is_preserved() -> None:
    path = Path("some/nested/module.py")
    chunks = parse_source("def f():\n    pass\n", path)
    assert chunks[0].path == path


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