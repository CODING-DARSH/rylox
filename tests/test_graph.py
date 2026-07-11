from __future__ import annotations

from typing import Optional

from rylox.graph import build_reference_graph, expand_one_hop

from rylox.cache import CachedChunk


def _chunk(name: str, content: str, parent_class: Optional[str] = None) -> CachedChunk:
    return CachedChunk(
        path="a.py",
        kind="method" if parent_class else "function",
        name=name,
        start_line=1,
        end_line=2,
        parent_class=parent_class,
        docstring=None,
        content=content,
    )


def test_direct_call_creates_forward_and_reverse_edge() -> None:
    chunks = [
        _chunk("login", "def login():\n    check_credentials()"),
        _chunk("check_credentials", "def check_credentials():\n    return True"),
    ]
    graph = build_reference_graph(chunks)
    assert graph.forward[0] == {1}
    assert graph.reverse[1] == {0}


def test_unrelated_chunk_has_no_edges() -> None:
    chunks = [
        _chunk("login", "def login():\n    check_credentials()"),
        _chunk("check_credentials", "def check_credentials():\n    return True"),
        _chunk("unrelated", "def unrelated():\n    pass"),
    ]
    graph = build_reference_graph(chunks)
    assert 2 not in graph.forward
    assert 2 not in graph.reverse


def test_self_reference_is_excluded() -> None:
    chunks = [_chunk("recurse", "def recurse():\n    return recurse()")]
    graph = build_reference_graph(chunks)
    assert graph.forward.get(0, set()) == set()


def test_class_construction_counted_as_a_reference() -> None:
    chunks = [
        _chunk("build", "def build():\n    return Widget()"),
        _chunk("Widget", "class Widget:\n    pass"),
    ]
    graph = build_reference_graph(chunks)
    assert graph.forward[0] == {1}


def test_expand_one_hop_finds_direct_callee_only_not_transitive() -> None:
    chunks = [
        _chunk("login", "def login():\n    check_credentials()"),
        _chunk("check_credentials", "def check_credentials():\n    return db_lookup()"),
        _chunk("db_lookup", "def db_lookup():\n    return True"),
    ]
    graph = build_reference_graph(chunks)
    expanded = expand_one_hop([0], graph, chunks)
    names = {chunks[e.index].name for e in expanded}
    assert names == {"check_credentials"}  # not db_lookup — that's two hops away


def test_expand_one_hop_finds_caller_only_not_transitive() -> None:
    chunks = [
        _chunk("login", "def login():\n    check_credentials()"),
        _chunk("check_credentials", "def check_credentials():\n    return db_lookup()"),
        _chunk("db_lookup", "def db_lookup():\n    return True"),
    ]
    graph = build_reference_graph(chunks)
    expanded = expand_one_hop([2], graph, chunks)
    names = {chunks[e.index].name for e in expanded}
    assert names == {"check_credentials"}  # not login — that's two hops away


def test_expand_one_hop_labels_reason_correctly() -> None:
    chunks = [
        _chunk("login", "def login():\n    check_credentials()"),
        _chunk("check_credentials", "def check_credentials():\n    return True"),
    ]
    graph = build_reference_graph(chunks)
    expanded = expand_one_hop([0], graph, chunks)
    assert expanded[0].reason == "callee of login"

    expanded_reverse = expand_one_hop([1], graph, chunks)
    assert expanded_reverse[0].reason == "caller of check_credentials"


def test_expand_one_hop_excludes_items_already_in_top_set() -> None:
    chunks = [
        _chunk("login", "def login():\n    check_credentials()"),
        _chunk("check_credentials", "def check_credentials():\n    return db_lookup()"),
        _chunk("db_lookup", "def db_lookup():\n    return True"),
    ]
    graph = build_reference_graph(chunks)
    expanded = expand_one_hop([0, 1], graph, chunks)
    names = [chunks[e.index].name for e in expanded]
    assert names == ["db_lookup"]  # login and check_credentials don't re-appear


def test_expand_one_hop_deduplicates_shared_expansion_targets() -> None:
    chunks = [
        _chunk("a", "def a():\n    shared_helper()"),
        _chunk("b", "def b():\n    shared_helper()"),
        _chunk("shared_helper", "def shared_helper():\n    pass"),
    ]
    graph = build_reference_graph(chunks)
    expanded = expand_one_hop([0, 1], graph, chunks)
    assert len([e for e in expanded if chunks[e.index].name == "shared_helper"]) == 1


def test_empty_chunk_list_produces_empty_graph() -> None:
    graph = build_reference_graph([])
    assert graph.forward == {}
    assert graph.reverse == {}


def test_expand_one_hop_with_no_edges_returns_empty_list() -> None:
    chunks = [_chunk("lonely", "def lonely():\n    pass")]
    graph = build_reference_graph(chunks)
    assert expand_one_hop([0], graph, chunks) == []