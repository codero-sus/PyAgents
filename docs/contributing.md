# Contributing

Thanks for taking the time to improve PyAgents. This page covers the workflow
and the conventions the project follows.

## Development setup

```bash
git clone https://github.com/codero-sus/PyAgents.git
cd PyAgents

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e ".[dev]"              # core + httpx, pydantic, pytest, ruff, mypy
```

Optional, for building these docs:

```bash
pip install -e ".[docs]"
mkdocs serve                         # http://127.0.0.1:8000
```

## The checks

Everything CI runs, you can run locally. All four must pass before you open a
pull request:

```bash
python -m pytest tests -q            # ~1 second, no network required
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy
```

Handy variations:

```bash
python -m pytest tests -q --cov=pyagents --cov-report=term-missing
python -m pytest tests/test_agent.py -k hooks -v
python -m ruff check --fix src tests examples
python -m ruff format src tests examples
```

The test suite never touches the network: providers are exercised through
`FakeTransport`, `ScriptedProvider` and the `echo` provider. If your change
needs a new HTTP interaction, add it to a fake rather than an integration test.

## Repository layout

```
src/pyagents/      the package (src layout, no runtime dependencies)
tests/             pytest suite, mirrors the module layout
examples/          runnable, offline scripts (01..08)
docs/              mkdocs-material documentation
.github/workflows/ CI
```

Guidelines:

* **Keep the core dependency-free.** `urllib`, `json`, `asyncio` and friends
  only. Third-party imports belong behind the `http` / `pydantic` extras and
  must be optional at runtime.
* **Type everything.** The package ships `py.typed` and is checked with mypy.
  Prefer `Optional[X]`-style annotations over `X | None` in public signatures:
  the project supports Python 3.9, where `get_type_hints()` would choke on PEP
  604 unions inside dataclasses.
* **Document behaviour with runnable examples.** Docstrings in `types.py`,
  `utils.py`, `errors.py`, `memory.py`, `stores.py`, `parsers.py`,
  `embeddings.py`, `workflows.py`, `teams.py`, `tools/*` and `llm/*` are
  executed by `tests/test_doctests.py`, so keep the examples true.
* **Errors:** raise a subclass of `PyAgentsError`; add the new class to
  `errors.py` and to `__all__` there and in `pyagents/__init__.py`.
* **Public API:** anything intended for users is listed in the module's
  `__all__` and re-exported from `pyagents/__init__.py`.

## Adding a tool

1. Write a typed function with a Google-style docstring.
2. Decorate it with `@tool(tags=[...])` in `src/pyagents/tools/builtin.py`
   (or keep it in your own module).
3. Export the name from `pyagents/tools/__init__.py` and, if it should be part
   of the default set, from `default_tools()`.
4. Add tests in `tests/test_tools.py` - success, validation failure and, if it
   touches the filesystem or the network, the unhappy path too.

Dangerous tools (`write_file`, `http_request`, `python_repl`) stay out of the
default set; add them to the `include=` list instead.

## Adding a provider

1. Subclass `BaseProvider` and implement the wire-format hooks: `_endpoint`,
   `_auth_headers`, `_payload`, `_parse_response` and, for streaming,
   `_parse_stream_data` (or `_iter_events` when the API is not SSE).
2. Implement `_new_stream_state()` so the base class can aggregate chunks.
3. Register the class with `register_provider("name", MyProvider)` and export it
   from `pyagents/llm/__init__.py`.
4. Add tests in `tests/test_providers.py` using `FakeTransport` - assert on the
   payload you send, the parsing of the response, retry behaviour on 429/5xx,
   and that control options never leak into the payload.

## Documentation

* `docs/*.md` is the source of truth; `README.md` is the front door.
* When you change a CLI flag, a constructor argument or an error message,
  update the docs in the same commit.
* Examples must run offline: guard real providers with
  `examples/_bootstrap.demo_provider()`, or use `ScriptedProvider`.
* Build locally with `mkdocs build --strict` - CI fails on warnings.

## Commits and pull requests

* Small, focused commits with an imperative subject line
  (`add ollama keep-alive option`, not `changes`).
* Reference issues in the body: `Closes #42`.
* A pull request should describe *what changed and why*, mention any new
  public API, and note the commands you ran (`pytest`, `ruff`, `mypy`).
* Keep unrelated refactors out of feature branches - they make review and
  bisecting harder.

## Commit checklist

```bash
python -m pytest tests -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy
for example in examples/0*.py; do python "$example" > /dev/null; done
```

* [ ] tests pass, new behaviour is covered
* [ ] `ruff check` and `ruff format --check` are clean
* [ ] `mypy` is clean
* [ ] examples still run
* [ ] docs and docstrings updated
* [ ] `__all__` and the top-level re-exports updated if the public API changed

## License

By contributing you agree that your work is licensed under the MIT License, as
described in [LICENSE](https://github.com/codero-sus/PyAgents/blob/main/LICENSE).
