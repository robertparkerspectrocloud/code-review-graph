"""Shared utilities for tool sub-modules."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from ..graph import GraphStore
from ..incremental import find_project_root, get_db_path

logger = logging.getLogger(__name__)


def _error_response(
    message: str, status: str = "error", **extra: Any,
) -> dict[str, Any]:
    """Build a standardised error response dict."""
    return {"status": status, "error": message, "summary": message, **extra}


# Metadata keys written by a build/update run. Their presence proves the graph
# was built at least once, even if the resulting graph legitimately has 0 nodes
# (e.g. an empty repo). Used to distinguish "never built" from "built, empty".
_BUILD_MARKER_KEYS: tuple[str, ...] = ("last_updated", "last_build_type")


def _graph_is_built(store: GraphStore) -> bool:
    """Return True if the graph has been built at least once.

    A graph counts as built when it contains any node, or when a build marker
    metadata row is present. The marker check ensures a legitimately empty repo
    that was actually built is not mistaken for a never-built repo.
    """
    try:
        if store.get_stats().total_nodes > 0:
            return True
        return any(store.get_metadata(key) for key in _BUILD_MARKER_KEYS)
    except sqlite3.Error:
        logger.warning("Failed to read graph build state", exc_info=True)
        # Be conservative: if we cannot tell, do not block the read tool.
        return True


def _not_built_response() -> dict[str, Any]:
    """Standard response returned by read tools when the graph is not built."""
    return {
        "status": "not_built",
        "message": "Graph not built — run build_or_update_graph_tool first.",
        "summary": "Graph not built — run build_or_update_graph_tool first.",
        "next_tool_suggestions": ["build_or_update_graph_tool"],
    }


def _get_store_for_read(
    repo_root: str | None = None,
) -> tuple[GraphStore | None, Path | None, dict[str, Any] | None]:
    """Open the store for a read tool, guarding against an unbuilt graph.

    Returns ``(store, root, None)`` when the graph has been built; the caller
    owns ``store`` and must close it. Returns ``(None, None, not_built)`` when
    the graph is missing or never built, in which case the caller should return
    the ``not_built`` dict immediately (no store to close).
    """
    root = _resolve_root(repo_root)
    db_path = get_db_path(root)
    # A missing db file means the graph was never built. Opening GraphStore
    # would silently create an empty db, so short-circuit before that happens.
    if not Path(db_path).exists():
        return None, None, _not_built_response()
    store = GraphStore(db_path)
    if not _graph_is_built(store):
        store.close()
        return None, None, _not_built_response()
    return store, root, None

# Common JS/TS builtin method names filtered from callers_of results.
# "Who calls .map()?" returns hundreds of hits and is never useful.
# These are kept in the graph (callees_of still shows them) but excluded
# when doing reverse call tracing to reduce noise.
_BUILTIN_CALL_NAMES: set[str] = {
    "map", "filter", "reduce", "reduceRight", "forEach", "find", "findIndex",
    "some", "every", "includes", "indexOf", "lastIndexOf",
    "push", "pop", "shift", "unshift", "splice", "slice",
    "concat", "join", "flat", "flatMap", "sort", "reverse", "fill",
    "keys", "values", "entries", "from", "isArray", "of", "at",
    "trim", "trimStart", "trimEnd", "split", "replace", "replaceAll",
    "match", "matchAll", "search", "substring", "substr",
    "toLowerCase", "toUpperCase", "startsWith", "endsWith",
    "padStart", "padEnd", "repeat", "charAt", "charCodeAt",
    "assign", "freeze", "defineProperty", "getOwnPropertyNames",
    "hasOwnProperty", "create", "is", "fromEntries",
    "log", "warn", "error", "info", "debug", "trace", "dir", "table",
    "time", "timeEnd", "assert", "clear", "count",
    "then", "catch", "finally", "resolve", "reject", "all", "allSettled", "race", "any",
    "parse", "stringify",
    "floor", "ceil", "round", "random", "max", "min", "abs", "pow", "sqrt",
    "addEventListener", "removeEventListener", "querySelector", "querySelectorAll",
    "getElementById", "createElement", "appendChild", "removeChild",
    "setAttribute", "getAttribute", "preventDefault", "stopPropagation",
    "setTimeout", "clearTimeout", "setInterval", "clearInterval",
    "toString", "valueOf", "toJSON", "toISOString",
    "getTime", "getFullYear", "now",
    "isNaN", "parseInt", "parseFloat", "toFixed",
    "encodeURIComponent", "decodeURIComponent",
    "call", "apply", "bind", "next",
    "emit", "on", "off", "once",
    "pipe", "write", "read", "end", "close", "destroy",
    "send", "status", "json", "redirect",
    "set", "get", "delete", "has",
    "findUnique", "findFirst", "findMany", "createMany",
    "update", "updateMany", "deleteMany", "upsert",
    "aggregate", "groupBy", "transaction",
    "describe", "it", "test", "expect", "beforeEach", "afterEach",
    "beforeAll", "afterAll", "mock", "spyOn",
    "require", "fetch",
}


def _validate_repo_root(path: "Path | str") -> Path:
    """Validate that a path is a plausible project root.

    Ensures the path is an existing directory that contains a ``.git``,
    ``.svn``, or ``.code-review-graph`` directory, preventing arbitrary
    file-system traversal via the ``repo_root`` parameter.
    """
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"repo_root is not an existing directory: {resolved}"
        )
    has_vcs = (
        (resolved / ".git").exists()
        or (resolved / ".svn").exists()
        or (resolved / ".code-review-graph").exists()
    )
    if not has_vcs:
        raise ValueError(
            f"repo_root does not look like a project root "
            f"(no .git, .svn, or .code-review-graph directory found): "
            f"{resolved}"
        )
    return resolved


def _resolve_root(repo_root: str | None = None) -> Path:
    """Resolve and validate the repository root without opening a store."""
    return _validate_repo_root(Path(repo_root)) if repo_root else find_project_root()


def _get_store(repo_root: str | None = None) -> tuple[GraphStore, Path]:
    """Resolve repo root and open the graph store.

    Callers own the returned store and must close it (try/finally or
    context manager) to avoid leaking SQLite file descriptors.
    """
    root = _resolve_root(repo_root)
    db_path = get_db_path(root)
    return GraphStore(db_path), root


def _resolve_graph_file_paths(
    store: GraphStore, root: Path, file_paths: list[str],
) -> list[str]:
    """Resolve user-facing file paths to the paths stored in the graph.

    Graphs may contain absolute paths, repo-relative paths, or cwd-relative
    paths depending on how they were built. Tool inputs are usually relative to
    repo root, so exact matching alone can miss existing graph nodes.
    """
    resolved: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        if path not in seen:
            resolved.append(path)
            seen.add(path)

    for file_path in file_paths:
        raw = file_path.replace("\\", "/")
        candidates = [raw]
        path = Path(file_path)
        if path.is_absolute():
            try:
                candidates.append(str(path.resolve().relative_to(root)).replace("\\", "/"))
            except ValueError:
                pass
        else:
            candidates.append(str(root / path))

        for candidate in candidates:
            if store.get_nodes_by_file(candidate):
                add(candidate)

        suffixes = []
        for candidate in candidates:
            normalized = candidate.replace("\\", "/")
            if normalized not in suffixes:
                suffixes.append(normalized)

        for suffix in suffixes:
            for matched_path in store.get_files_matching(suffix):
                add(matched_path)

    return resolved


def compact_response(
    summary: str,
    key_entities: list[str] | None = None,
    risk: str = "unknown",
    communities: list[str] | None = None,
    flows_affected: list[str] | None = None,
    next_tool_suggestions: list[str] | None = None,
    data: dict[str, Any] | None = None,
    detail_level: str = "minimal",
) -> dict[str, Any]:
    """Standard compact response format for token efficiency."""
    resp: dict[str, Any] = {
        "status": "ok",
        "summary": summary,
    }
    if key_entities:
        resp["key_entities"] = key_entities[:10]
    if risk != "unknown":
        resp["risk"] = risk
    if communities:
        resp["communities"] = communities[:5]
    if flows_affected:
        resp["flows_affected"] = flows_affected[:5]
    if next_tool_suggestions:
        resp["next_tool_suggestions"] = next_tool_suggestions[:3]
    if detail_level != "minimal" and data:
        resp["data"] = data
    return resp
