# Memory & state

Two different problems, two different tools:

* **Memory** decides *what the model sees* on the next turn.
* **Run stores** decide *what you can inspect later*.

## Memory

Every memory implements the same small interface:

```python
class Memory(ABC):
    def add(self, message: Message) -> None: ...
    def messages(self) -> List[Message]: ...
    def extend(self, messages: Iterable[Message]) -> None: ...
    def clear(self) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data) -> "Memory": ...
```

Attach one to an agent and it is used automatically:

```python
from pyagents import Agent, BufferMemory

memory = BufferMemory(max_messages=20)
agent = Agent("assistant", provider="openai", memory=memory)

agent.run("My name is Ada.")
agent.run("What is my name?")     # remembers
```

The agent writes both sides of the conversation (user messages and assistant
answers, tool observations included) and replays `memory.messages()` at the
start of the next run. A memory backend that raises does not kill the run -
errors are recorded in `result.metadata["memory_errors"]`.

### Built-in strategies

| class | keeps | constructor |
|---|---|---|
| `BufferMemory` | the last N messages | `max_messages=20` (or `None` for everything) |
| `WindowMemory` | last N, always keeping the system prompt and first user message | `max_messages=20, keep_system=True, keep_first_user=True` |
| `TokenWindowMemory` | messages that fit a token budget | `max_tokens=4000, token_counter=None` |
| `SummaryMemory` | a rolling summary plus the most recent messages | `provider, summary_threshold=12, keep_recent=6` |
| `VectorMemory` | messages most similar to the current task | `embedder, top_k=6, always_keep=2` |
| `CompositeMemory` | the union of several memories (deduplicated) | `memories=[...]` |

`SummaryMemory` compresses old turns with an LLM: when more than
`summary_threshold` messages are pending it asks the provider for a summary and
replaces them with a single synthetic message, keeping the last `keep_recent`
messages verbatim. The agent uses the async path (`aadd` /
`summarize_async`) automatically, so summarisation never blocks the loop.

`VectorMemory` retrieves by semantic similarity. The agent calls
`memory.set_query(task)` at the start of each run, so you only have to build it
once:

```python
from pyagents import VectorMemory
from pyagents.embeddings import HashEmbedder

memory = VectorMemory(HashEmbedder(512), top_k=4)
agent = Agent("librarian", provider="openai", memory=memory)
```

Trade-offs between strategies:

| strategy | cost | good for |
|---|---|---|
| buffer / window | none | short chats, explicit control |
| token window | none | staying under a context limit without an extra LLM call |
| summary | one extra call per threshold | long conversations |
| vector | embedding calls | "find the relevant past discussion" workflows |

### Persisting memory

Every strategy round-trips through plain dictionaries:

```python
import json

data = memory.to_dict()                       # {"kind": "buffer", "messages": [...], ...}
Path("memory.json").write_text(json.dumps(data))

from pyagents.memory import memory_from_dict
restored = memory_from_dict(json.loads(Path("memory.json").read_text()))
```

## Run stores

A run store persists whole `RunResult`s - outputs, steps, usage and metadata -
so you can audit, resume inspection or build a dashboard.

```python
from pyagents import Agent, MemoryStore, SQLiteRunStore, store_from_uri

store = SQLiteRunStore("runs.sqlite3")       # or MemoryStore() for tests
agent = Agent("worker", provider="openai", store=store)

result = agent.run("Do the thing")
store.has(result.run_id)                     # True
store.load(result.run_id)                    # RunResult (rebuilt from JSON)
store.list_runs(limit=10, status="completed")  # summary dicts, newest first
store.delete(result.run_id)
store.close()
```

* `SQLiteRunStore(path, check_same_thread=False)` is safe to share between the
  worker thread used by `run()` and your main thread.
* `MemoryStore` is for tests and short-lived processes; it still copies
  results, so later mutation of a `RunResult` cannot corrupt history.
* `store_from_uri(None)` returns `None`; `"memory:"` / `":memory:"` returns a
  `MemoryStore`; anything else is treated as a SQLite path. That is exactly
  what the CLI's `--store` flag does.

The agent checkpoints during long runs (`checkpoint=True` by default), so a
crashed or cancelled run still has a record of the steps that completed.
Failure paths save with `RunStatus.FAILED`; cancellation saves
`RunStatus.CANCELLED`.

### Custom stores

Subclass `RunStore` to put runs anywhere - S3, Postgres, a JSON API:

```python
from pyagents.stores import RunStore
from pyagents.types import RunResult

class JsonlRunStore(RunStore):
    def __init__(self, path):
        self.path = path

    def save(self, result: RunResult, *, event: str = "checkpoint") -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict()) + "\n")

    def load(self, run_id: str) -> RunResult:
        ...
```

Note that `bool(store)` is `True` even when the store is empty - the agent
tests `is not None`, never truthiness.

## Putting them together

```python
from pyagents import Agent, BufferMemory, SQLiteRunStore

agent = Agent(
    "assistant",
    provider="openai",
    memory=BufferMemory(max_messages=40),   # what the model sees
    store=SQLiteRunStore("runs.sqlite3"),   # what you can audit
)
```

For multi-agent systems, memory is usually shared per-agent (each specialist
keeps its own short history) while the run store is shared by the team, which
saves the composite `TeamResult` under its own id.
