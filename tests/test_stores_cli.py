"""Tests for run stores and the command-line interface."""

from __future__ import annotations

import io
import json

import pytest

from pyagents import Agent, MemoryStore, RunResult, RunStatus, SQLiteRunStore, store_from_uri
from pyagents.cli import build_parser, main, resolve_tools
from pyagents.errors import RunNotFound
from pyagents.stores import _normalise_status
from pyagents.testing import ScriptedProvider, text_response


def sample_result() -> RunResult:
    return RunResult(output="done", status=RunStatus.COMPLETED, agent="a", finished_at=1.0)


class TestMemoryStore:
    def test_save_and_load_round_trip(self):
        store = MemoryStore()
        result = sample_result()
        store.save(result)
        loaded = store.load(result.run_id)
        assert loaded.output == "done"
        assert loaded.status is RunStatus.COMPLETED
        assert loaded is not result

    def test_saves_are_isolated_copies(self):
        store = MemoryStore()
        result = sample_result()
        store.save(result)
        result.output = "mutated"
        assert store.load(result.run_id).output == "done"

    def test_list_and_filter(self):
        store = MemoryStore()
        ok = sample_result()
        failed = sample_result()
        failed.status = RunStatus.FAILED
        store.save(ok)
        store.save(failed)
        assert len(store.list_runs()) == 2
        assert [row["run_id"] for row in store.list_runs(status="failed")] == [failed.run_id]

    def test_has_delete_and_missing(self):
        store = MemoryStore()
        result = sample_result()
        store.save(result)
        assert store.has(result.run_id)
        store.delete(result.run_id)
        assert not store.has(result.run_id)
        with pytest.raises(RunNotFound):
            store.load(result.run_id)

    def test_store_is_truthy_when_empty(self):
        store = MemoryStore()
        assert bool(store) is True
        assert len(store) == 0


class TestSQLiteStore:
    def test_round_trip(self, tmp_path):
        store = SQLiteRunStore(tmp_path / "runs.db")
        result = sample_result()
        store.save(result)
        assert store.load(result.run_id).output == "done"
        assert [row["run_id"] for row in store.list_runs()] == [result.run_id]
        store.delete(result.run_id)
        assert store.list_runs() == []
        store.close()

    def test_overwrites_on_repeated_saves(self, tmp_path):
        store = SQLiteRunStore(tmp_path / "runs.db")
        result = sample_result()
        store.save(result)
        result.output = "updated"
        store.save(result)
        assert store.load(result.run_id).output == "updated"
        assert len(store.list_runs()) == 1
        store.close()

    def test_status_filter_and_limit(self, tmp_path):
        store = SQLiteRunStore(tmp_path / "runs.db")
        for status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.COMPLETED):
            result = sample_result()
            result.status = status
            store.save(result)
        assert len(store.list_runs(status="completed")) == 2
        assert len(store.list_runs(limit=1)) == 1
        store.close()

    def test_in_memory_database(self):
        store = SQLiteRunStore(":memory:")
        store.save(sample_result())
        assert len(store.list_runs()) == 1
        store.close()

    def test_missing_run(self, tmp_path):
        store = SQLiteRunStore(tmp_path / "runs.db")
        with pytest.raises(RunNotFound):
            store.load("nope")
        store.close()

    def test_usable_from_another_thread(self, tmp_path):
        import threading

        store = SQLiteRunStore(tmp_path / "runs.db")
        errors = []

        def worker():
            try:
                store.save(sample_result())
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert errors == []
        assert len(store.list_runs()) == 1
        store.close()

    def test_context_manager(self, tmp_path):
        with SQLiteRunStore(tmp_path / "runs.db") as store:
            store.save(sample_result())
            assert len(store.list_runs()) == 1


class TestStoreFactories:
    def test_store_from_uri(self):
        assert store_from_uri(None) is None
        assert isinstance(store_from_uri("memory:"), MemoryStore)
        assert isinstance(store_from_uri(":memory:"), MemoryStore)

    def test_store_from_uri_sqlite(self, tmp_path):
        store = store_from_uri(str(tmp_path / "x.db"))
        assert isinstance(store, SQLiteRunStore)
        store.close()

    def test_status_normalisation(self):
        assert _normalise_status("RunStatus.FAILED") == "failed"
        assert _normalise_status(None) == ""


class TestCliParsing:
    def test_parser_accepts_every_command(self):
        parser = build_parser()
        for argv in (
            ["run", "hi"],
            ["chat"],
            ["tools", "list"],
            ["providers"],
            ["runs", "list"],
            ["info"],
        ):
            assert parser.parse_args(argv)

    def test_resolve_tools(self):
        assert [tool.name for tool in resolve_tools("calculator,current_time")] == [
            "calculator",
            "current_time",
        ]
        assert resolve_tools(None) == []
        assert len(resolve_tools("all")) > 5

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert "pyagents" in capsys.readouterr().out


class TestCliCommands:
    def test_run_streams_answer(self, capsys):
        assert main(["run", "hello", "--provider", "echo"]) == 0
        out = capsys.readouterr().out
        assert 'You said: "hello"' in out

    def test_run_json(self, capsys):
        assert main(["run", "hi", "--provider", "echo", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] == "completed"
        assert payload["run_id"].startswith("run_")

    def test_run_no_stream_and_verbose(self, capsys):
        assert main(["run", "hi", "--provider", "echo", "--no-stream", "--verbose"]) == 0
        captured = capsys.readouterr()
        assert "status: completed" in captured.err
        assert "You said" in captured.out

    def test_run_from_stdin(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("piped prompt"))
        assert main(["run", "-", "--provider", "echo"]) == 0
        assert "piped prompt" in capsys.readouterr().out

    def test_run_without_prompt(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        assert main(["run", "-", "--provider", "echo"]) == 2
        assert "provide a prompt" in capsys.readouterr().err

    def test_run_with_tools(self, capsys):
        assert main(["run", "hi", "--provider", "echo", "--tools", "calculator"]) == 0

    def test_run_with_json_tools_names_invalid(self, capsys):
        assert main(["run", "hi", "--provider", "echo", "--tools", "nope"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_run_missing_credentials(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert main(["run", "hi", "--provider", "openai"]) == 1
        assert "API key" in capsys.readouterr().err

    def test_tools_listing(self, capsys):
        assert main(["tools", "list"]) == 0
        out = capsys.readouterr().out
        assert "calculator" in out
        assert "python_repl" not in out

    def test_tools_all_and_json(self, capsys):
        assert main(["tools", "list", "--all", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        names = [item["name"] for item in payload]
        assert "python_repl" in names

    def test_tools_verbose(self, capsys):
        assert main(["tools", "list", "--verbose"]) == 0
        assert "required:" in capsys.readouterr().out

    def test_providers(self, capsys):
        assert main(["providers"]) == 0
        out = capsys.readouterr().out
        assert "openai" in out and "ollama" in out

    def test_providers_json(self, capsys):
        assert main(["providers", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert any(
            item["name"] == "echo" and item["credential"] == "not required" for item in payload
        )

    def test_info(self, capsys):
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        assert "pyagents:" in out
        assert "default_provider" in out

    def test_info_json(self, capsys):
        assert main(["info", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert "env" in payload

    def test_bad_flag_exits(self):
        with pytest.raises(SystemExit):
            main(["run", "hi", "--nope"])

    def test_chat_non_interactive(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("hello\n"))
        assert main(["chat", "--provider", "echo"]) == 0
        out = capsys.readouterr().out
        assert "agent>" in out

    def test_chat_commands(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "sys.stdin", io.StringIO("/help\n/tools\n/reset\n/instruction be terse\nhello\n")
        )
        monkeypatch.setattr("sys.stdin", io.StringIO("/help\n/exit\n"))
        assert main(["chat", "--provider", "echo"]) == 0
        assert "Commands:" in capsys.readouterr().out


class TestCliRuns:
    def _make_store(self, tmp_path):
        store = SQLiteRunStore(tmp_path / "runs.db")
        agent = Agent(
            "cli", provider=ScriptedProvider([text_response("stored answer")]), store=store
        )
        result = agent.run("remember this")
        store.close()
        return tmp_path / "runs.db", result

    def test_list_show_delete(self, tmp_path, capsys):
        path, result = self._make_store(tmp_path)
        assert main(["runs", "list", "--store", str(path)]) == 0
        assert result.run_id in capsys.readouterr().out

        assert main(["runs", "show", result.run_id, "--store", str(path)]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["output"] == "stored answer"

        assert main(["runs", "delete", result.run_id, "--store", str(path)]) == 0
        assert "deleted" in capsys.readouterr().out

    def test_list_json_and_missing_store(self, tmp_path, capsys):
        path, _ = self._make_store(tmp_path)
        assert main(["runs", "list", "--store", str(path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)

        assert main(["runs", "show", "nope", "--store", str(tmp_path / "missing.db")]) == 1
        assert "no run store" in capsys.readouterr().err
