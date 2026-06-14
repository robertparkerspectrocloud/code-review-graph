"""Tests for the MCP server entry point.

Focused on the ``_resolve_repo_root`` helper that threads the
``serve --repo <X>`` CLI flag into every tool wrapper, and on the
set of tools that must be registered as async coroutines so the MCP
stdio event loop stays responsive during long-running operations.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from code_review_graph import main as crg_main


@pytest.fixture(autouse=True)
def _isolate_crg_tools_env(monkeypatch):
    """Always strip CRG_TOOLS / CRG_DETAIL_LEVEL so that any test invoking
    ``crg_main.main`` does not accidentally permanently shrink the global tool
    registry when the suite runs under a developer environment that exports
    ``CRG_TOOLS``.  Without this the snapshot/restore in
    ``TestApplyToolFilter._restore_tools`` only sees the already-filtered
    set and cannot restore the dropped tools."""
    monkeypatch.delenv("CRG_TOOLS", raising=False)
    monkeypatch.delenv("CRG_DETAIL_LEVEL", raising=False)


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Snapshot the full tool registry before each test and restore after.

    ``main()`` now trims to the lean set by default (and ``_apply_tool_filter``
    removes tools permanently via ``remove_tool``), so any test that starts the
    server would otherwise leave the global registry shrunk for every later
    test.  Re-register anything that was dropped, and reset the detail-level
    override."""
    import asyncio

    original = asyncio.run(crg_main.mcp.list_tools())
    saved_override = crg_main._detail_level_override
    yield
    crg_main._detail_level_override = saved_override
    current_names = {t.name for t in asyncio.run(crg_main.mcp.list_tools())}
    for tool in original:
        if tool.name not in current_names:
            crg_main.mcp.add_tool(tool)


class TestResolveRepoRoot:
    """Precedence rules for _resolve_repo_root (see #222 follow-up)."""

    @pytest.fixture(autouse=True)
    def _reset_default(self):
        """Save and restore the module-level default before/after each test."""
        original = crg_main._default_repo_root
        yield
        crg_main._default_repo_root = original

    def test_none_when_neither_is_set(self):
        crg_main._default_repo_root = None
        assert crg_main._resolve_repo_root(None) is None

    def test_empty_string_treated_as_unset(self):
        """Empty string from an MCP client should not shadow the --repo flag."""
        crg_main._default_repo_root = "/tmp/flag-repo"
        assert crg_main._resolve_repo_root("") == "/tmp/flag-repo"

    def test_flag_used_when_client_omits_repo_root(self):
        crg_main._default_repo_root = "/tmp/flag-repo"
        assert crg_main._resolve_repo_root(None) == "/tmp/flag-repo"

    def test_client_arg_wins_over_flag(self):
        crg_main._default_repo_root = "/tmp/flag-repo"
        assert crg_main._resolve_repo_root("/explicit") == "/explicit"

    def test_client_arg_used_when_no_flag(self):
        crg_main._default_repo_root = None
        assert crg_main._resolve_repo_root("/explicit") == "/explicit"


class TestServeMainTransport:
    """``main()`` wires FastMCP to stdio or Streamable HTTP."""

    def test_stdio_calls_mcp_run_stdio(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(crg_main.mcp, "run", fake_run)
        crg_main.main(repo_root=None)
        assert calls == [{"transport": "stdio", "show_banner": False}]

    def test_http_calls_mcp_run_with_host_port(self, monkeypatch):
        calls: list[dict] = []

        def fake_run(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(crg_main.mcp, "run", fake_run)
        crg_main.main(
            repo_root="/tmp/r",
            transport="streamable-http",
            host="127.0.0.1",
            port=5555,
        )
        assert calls == [
            {
                "transport": "streamable-http",
                "host": "127.0.0.1",
                "port": 5555,
            }
        ]

    def test_streamable_http_without_host_port_raises(self):
        with pytest.raises(ValueError, match="requires host and port"):
            crg_main.main(transport="streamable-http", host=None, port=5555)
        with pytest.raises(ValueError, match="requires host and port"):
            crg_main.main(transport="streamable-http", host="127.0.0.1", port=None)


class TestLongRunningToolsAreAsync:
    """Long-running MCP tools must be registered as coroutines so the
    asyncio event loop stays responsive while the work runs in a
    background thread via ``asyncio.to_thread``. Without this, Windows
    MCP clients hang on ``build_or_update_graph_tool`` and
    ``embed_graph_tool`` — see #46, #136.
    """

    HEAVY_TOOLS = {
        "build_or_update_graph_tool",
        "run_postprocess_tool",
        "embed_graph_tool",
        "detect_changes_tool",
        "generate_wiki_tool",
    }

    def test_heavy_tools_are_coroutines(self):
        """Regression guard for #46/#136: the 5 long-running MCP tools must
        stay ``async def`` so FastMCP can offload their blocking work via
        ``asyncio.to_thread`` and keep the stdio event loop responsive.

        The original implementation of this test went through
        ``crg_main.mcp.get_tools()``, which does not exist in the FastMCP
        2.14+ API pinned in pyproject.toml (``list_tools()`` replaces it and
        returns MCP protocol ``Tool`` objects, which do not expose the
        underlying Python function at all).  The sibling test
        ``test_heavy_tool_source_uses_to_thread`` already resolves each
        tool by ``getattr(crg_main, name)``; we do the same here so this
        guard is independent of any FastMCP internal surface.  See #239.
        """
        missing: list[str] = []
        not_async: list[str] = []

        for tool_name in self.HEAVY_TOOLS:
            fn = getattr(crg_main, tool_name, None)
            if fn is None:
                missing.append(tool_name)
                continue
            # The @mcp.tool() decorator wraps the function; FunctionTool
            # stores the underlying callable on ``.fn`` on current FastMCP
            # 2.x but we fall back to the wrapper itself for resilience.
            underlying = getattr(fn, "fn", None) or fn
            if not asyncio.iscoroutinefunction(underlying):
                not_async.append(tool_name)

        assert not missing, f"heavy tool(s) not registered at all: {missing}"
        assert not not_async, (
            f"these tools must be async but were registered as sync, "
            f"which will hang the stdio event loop on Windows: {not_async}"
        )

    def test_heavy_tool_source_uses_to_thread(self):
        """Defense in depth: the source of every heavy tool wrapper must
        literally call asyncio.to_thread so we don't accidentally turn
        a tool async without offloading the blocking work."""
        for tool_name in self.HEAVY_TOOLS:
            fn = getattr(crg_main, tool_name, None)
            assert fn is not None, f"{tool_name} not found on module"
            # The @mcp.tool() decorator wraps the original function; walk
            # through the wrapper to find the underlying source.
            underlying = getattr(fn, "fn", None) or fn
            source = inspect.getsource(underlying)
            assert "asyncio.to_thread" in source, (
                f"{tool_name} must call asyncio.to_thread to offload its "
                f"blocking work; otherwise Windows MCP clients will hang. "
                f"See #46, #136."
            )

    @pytest.mark.asyncio
    async def test_detect_changes_timeout_uses_error_response_shape(
        self, monkeypatch
    ):
        async def fake_wait_for(coro, timeout):
            coro.close()
            raise asyncio.TimeoutError

        monkeypatch.setenv("CRG_TOOL_TIMEOUT", "1")
        monkeypatch.setattr(crg_main.asyncio, "wait_for", fake_wait_for)

        tool = getattr(crg_main.detect_changes_tool, "fn", None)
        underlying = tool or crg_main.detect_changes_tool

        result = await underlying()

        assert result["status"] == "error"
        assert "timed out after 1s" in result["error"]
        assert result["summary"] == result["error"]

    def test_regression_guard_does_not_depend_on_fastmcp_internals(self):
        """Regression guard for #239 bug 3: ensure the async guards above
        resolve heavy tools by module attribute lookup, NOT through a
        FastMCP internal API that may drift between releases.

        The original ``test_heavy_tools_are_coroutines`` called an API on
        the mcp instance that does not exist in ``fastmcp>=2.14.0``.  It
        died with ``AttributeError`` at runtime on every platform,
        silently disabling the async-regression guard that was supposed
        to protect #46/#136 from regressing.  This test locks in the
        module-lookup approach so the guards keep working regardless of
        internal FastMCP surface changes.
        """
        import ast as _ast

        # Every heavy tool must be reachable by plain getattr on the
        # module — that's the only API surface the guards are allowed to
        # use.  No mcp internals.
        for tool_name in self.HEAVY_TOOLS:
            fn = getattr(crg_main, tool_name, None)
            assert fn is not None, (
                f"{tool_name} must be reachable via "
                f"getattr(crg_main, tool_name) so the async guards "
                f"do not depend on any FastMCP internal API"
            )

        # And the guards themselves must not reference renamed/removed
        # APIs on the mcp instance.  We check the parsed AST of the
        # function bodies (not the docstrings) so an explanatory comment
        # mentioning an old API name doesn't trip this guard.
        forbidden_mcp_attrs = {
            "get_tools", "_tools", "tool_manager", "_tool_manager",
        }
        for guard_fn in (
            self.test_heavy_tools_are_coroutines,
            self.test_heavy_tool_source_uses_to_thread,
        ):
            source = inspect.getsource(guard_fn).lstrip()
            tree = _ast.parse(source)
            for node in _ast.walk(tree):
                # We want chained attributes like ``crg_main.mcp.get_tools``.
                # That's an Attribute whose value is also an Attribute whose
                # attr == "mcp".
                if (
                    isinstance(node, _ast.Attribute)
                    and node.attr in forbidden_mcp_attrs
                    and isinstance(node.value, _ast.Attribute)
                    and node.value.attr == "mcp"
                ):
                    raise AssertionError(
                        f"{guard_fn.__name__} references mcp.{node.attr} — "
                        f"this attribute drifts across FastMCP releases "
                        f"and will silently break the guard.  Use "
                        f"getattr(crg_main, tool_name) instead."
                    )

class TestApplyToolFilter:
    """Tests for _apply_tool_filter (``serve --tools`` / ``CRG_TOOLS``).

    The filter removes MCP tools not present in the allow-list.
    This dramatically reduces per-turn token overhead in LLM-backed
    MCP clients by pruning unused tool descriptions.
    """

    @pytest.fixture(autouse=True)
    def _restore_tools(self):
        """Snapshot registered tools before test, restore after.

        ``_apply_tool_filter`` calls ``mcp.remove_tool()`` which is
        permanent.  We snapshot the list of Tool objects via the public
        ``list_tools()`` async API (FastMCP >=3) and re-register them
        after the test body runs.
        """
        import asyncio

        original = asyncio.run(crg_main.mcp.list_tools())
        yield
        current_names = {
            t.name for t in asyncio.run(crg_main.mcp.list_tools())
        }
        for tool in original:
            if tool.name not in current_names:
                crg_main.mcp.add_tool(tool)

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Ensure CRG_TOOLS is not set from the outer environment."""
        monkeypatch.delenv("CRG_TOOLS", raising=False)

    @staticmethod
    async def _tool_names() -> set[str]:
        return {t.name for t in await crg_main.mcp.list_tools()}

    @pytest.mark.asyncio
    async def test_default_is_lean_set(self):
        """With nothing specified, the server trims to the curated lean set."""
        before = await self._tool_names()
        assert before == set(await self._tool_names())  # sanity
        crg_main._apply_tool_filter(None)
        after = await self._tool_names()
        assert after == set(crg_main.LEAN_TOOLS)
        # Lean is strictly a subset and never larger than the full registry.
        assert after < before
        assert len(crg_main.LEAN_TOOLS) == 7

    @pytest.mark.asyncio
    async def test_all_keyword_restores_full_set(self):
        """``--tools all`` (or CRG_TOOLS=all) keeps every registered tool."""
        before = await self._tool_names()
        assert len(before) == 30
        crg_main._apply_tool_filter("all")
        after = await self._tool_names()
        assert after == before
        assert len(after) == 30

    @pytest.mark.asyncio
    async def test_all_keyword_case_insensitive(self):
        before = await self._tool_names()
        crg_main._apply_tool_filter("ALL")
        assert await self._tool_names() == before

    @pytest.mark.asyncio
    async def test_lean_keyword_uses_curated_set(self):
        crg_main._apply_tool_filter("lean")
        assert await self._tool_names() == set(crg_main.LEAN_TOOLS)

    @pytest.mark.asyncio
    async def test_filter_via_argument(self):
        """The ``tools`` argument keeps only the listed tools."""
        keep = "query_graph_tool,semantic_search_nodes_tool"
        crg_main._apply_tool_filter(keep)
        remaining = await self._tool_names()
        assert remaining == {"query_graph_tool", "semantic_search_nodes_tool"}

    @pytest.mark.asyncio
    async def test_filter_via_env_var(self, monkeypatch):
        """The ``CRG_TOOLS`` env var works as fallback."""
        monkeypatch.setenv("CRG_TOOLS", "query_graph_tool")
        crg_main._apply_tool_filter(None)
        remaining = await self._tool_names()
        assert remaining == {"query_graph_tool"}

    @pytest.mark.asyncio
    async def test_env_var_all_restores_full_set(self, monkeypatch):
        before = await self._tool_names()
        monkeypatch.setenv("CRG_TOOLS", "all")
        crg_main._apply_tool_filter(None)
        assert await self._tool_names() == before

    @pytest.mark.asyncio
    async def test_argument_takes_precedence_over_env(self, monkeypatch):
        """CLI --tools wins over CRG_TOOLS env var."""
        monkeypatch.setenv("CRG_TOOLS", "list_repos_tool")
        crg_main._apply_tool_filter("query_graph_tool")
        remaining = await self._tool_names()
        assert remaining == {"query_graph_tool"}

    @pytest.mark.asyncio
    async def test_unknown_names_ignored_gracefully(self):
        """Unknown tool names don't error; only the valid ones survive."""
        crg_main._apply_tool_filter(
            "query_graph_tool,this_tool_does_not_exist,another_bogus_tool"
        )
        remaining = await self._tool_names()
        assert remaining == {"query_graph_tool"}

    @pytest.mark.asyncio
    async def test_all_unknown_names_removes_everything(self):
        """A spec of only-unknown names is honoured (removes all real tools).

        This is intentional: the names were explicit, just wrong. It is the
        caller's responsibility to pass real names; we never *expand* output.
        """
        crg_main._apply_tool_filter("nonexistent_a,nonexistent_b")
        assert await self._tool_names() == set()

    @pytest.mark.asyncio
    async def test_empty_string_keeps_all(self):
        """An explicit empty string should not remove all tools."""
        before = await self._tool_names()
        crg_main._apply_tool_filter("")
        after = await self._tool_names()
        assert before == after

    @pytest.mark.asyncio
    async def test_whitespace_handling(self):
        """Spaces around tool names are stripped."""
        crg_main._apply_tool_filter(" query_graph_tool , semantic_search_nodes_tool ")
        remaining = await self._tool_names()
        assert remaining == {"query_graph_tool", "semantic_search_nodes_tool"}

    def test_stderr_notice_printed_when_trimming(self, capsys):
        """Trimming to lean prints a one-line stderr notice; stdout stays clean."""
        crg_main._apply_tool_filter("query_graph_tool")
        captured = capsys.readouterr()
        assert captured.out == ""  # stdout must stay clean for JSON-RPC
        assert "lean tool mode" in captured.err
        assert "--tools all" in captured.err

    def test_no_stderr_notice_when_keeping_all(self, capsys):
        """``all`` keeps everything and prints nothing."""
        crg_main._apply_tool_filter("all")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_lean_tools_are_all_registered(self):
        """Every name in LEAN_TOOLS must be a real registered tool."""
        import asyncio

        registered = {t.name for t in asyncio.run(crg_main.mcp.list_tools())}
        for name in crg_main.LEAN_TOOLS:
            assert name in registered, f"LEAN_TOOLS lists unknown tool {name!r}"


class TestResolveDetailLevel:
    """Server-wide detail-level override (``CRG_DETAIL_LEVEL`` / --detail)."""

    @pytest.fixture(autouse=True)
    def _reset_override(self, monkeypatch):
        original = crg_main._detail_level_override
        monkeypatch.delenv("CRG_DETAIL_LEVEL", raising=False)
        crg_main._detail_level_override = None
        yield
        crg_main._detail_level_override = original

    def test_no_override_returns_argument(self):
        assert crg_main._resolve_detail_level("standard") == "standard"
        assert crg_main._resolve_detail_level("minimal") == "minimal"

    def test_module_override_wins(self):
        crg_main._detail_level_override = "minimal"
        assert crg_main._resolve_detail_level("standard") == "minimal"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("CRG_DETAIL_LEVEL", "standard")
        assert crg_main._resolve_detail_level("minimal") == "standard"

    def test_module_override_beats_env(self, monkeypatch):
        monkeypatch.setenv("CRG_DETAIL_LEVEL", "standard")
        crg_main._detail_level_override = "minimal"
        assert crg_main._resolve_detail_level("standard") == "minimal"

    def test_unknown_override_ignored(self, monkeypatch):
        """An invalid override never silently expands output."""
        monkeypatch.setenv("CRG_DETAIL_LEVEL", "bogus")
        assert crg_main._resolve_detail_level("minimal") == "minimal"

    def test_override_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("CRG_DETAIL_LEVEL", "  MINIMAL  ")
        assert crg_main._resolve_detail_level("standard") == "minimal"


class TestMcpWrapperDefaults:
    """The MCP tool wrappers default to ``minimal`` so the token moat holds."""

    @staticmethod
    def _wrapper(name):
        fn = getattr(crg_main, name)
        return getattr(fn, "fn", None) or fn

    def test_query_graph_tool_defaults_minimal(self):
        sig = inspect.signature(self._wrapper("query_graph_tool"))
        assert sig.parameters["detail_level"].default == "minimal"

    def test_semantic_search_defaults_minimal(self):
        sig = inspect.signature(self._wrapper("semantic_search_nodes_tool"))
        assert sig.parameters["detail_level"].default == "minimal"

    def test_impact_radius_defaults_minimal(self):
        sig = inspect.signature(self._wrapper("get_impact_radius_tool"))
        assert sig.parameters["detail_level"].default == "minimal"

    def test_detect_changes_defaults_minimal(self):
        sig = inspect.signature(self._wrapper("detect_changes_tool"))
        assert sig.parameters["detail_level"].default == "minimal"

    def test_query_graph_tool_has_max_results_cap(self):
        sig = inspect.signature(self._wrapper("query_graph_tool"))
        assert "max_results" in sig.parameters
        assert sig.parameters["max_results"].default == 100

    def test_review_tools_have_max_tokens(self):
        for name in ("get_review_context_tool", "detect_changes_tool"):
            sig = inspect.signature(self._wrapper(name))
            assert "max_tokens" in sig.parameters
            assert sig.parameters["max_tokens"].default == 6000


class TestServeDetailFlag:
    """``serve --detail`` sets the module-level override before the loop."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        original = crg_main._detail_level_override
        yield
        crg_main._detail_level_override = original

    def test_detail_flag_sets_override(self, monkeypatch):
        monkeypatch.setattr(crg_main.mcp, "run", lambda **kw: None)
        crg_main.main(repo_root=None, detail_level="minimal")
        assert crg_main._detail_level_override == "minimal"

    def test_invalid_detail_flag_ignored(self, monkeypatch):
        monkeypatch.setattr(crg_main.mcp, "run", lambda **kw: None)
        crg_main._detail_level_override = None
        crg_main.main(repo_root=None, detail_level="loud")
        assert crg_main._detail_level_override is None
