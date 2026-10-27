# Command line

Installing PyAgents puts a `pyagents` command on your PATH.

```bash
$ pyagents --version
pyagents 0.1.0
$ pyagents --help
```

## `pyagents run`

Run a single agent once:

```bash
pyagents run "Summarise the key points of the attached brief"
pyagents run - < prompt.txt                 # read the task from stdin
echo "what is 2+2?" | pyagents run -
```

Useful flags:

| flag | effect |
|---|---|
| `--provider NAME` | `openai`, `anthropic`, `claude`, `ollama`, `openai-compatible`, `echo` |
| `--model NAME` | model id; defaults to the provider's |
| `--base-url URL` | override the API endpoint |
| `--instructions TEXT` | system prompt |
| `--tools LIST` | comma-separated built-in tools, or `all` |
| `--max-steps N` | reasoning step budget (default 10) |
| `--store URI` | SQLite path or `memory:` to persist the run |
| `--history N` | keep N messages of memory between invocations |
| `--no-stream` | wait for the whole answer instead of streaming tokens |
| `--json` | print the full `RunResult` as JSON |
| `--quiet` | print nothing but errors |
| `--verbose` | print a one-line run summary to stderr |
| `--human-review` | prompt a human to approve or edit the agent's answer before finishing |

```bash
$ pyagents run "hello world" --provider echo
You said: "hello world". (This is the offline echo provider.)

$ pyagents run "hello" --provider echo --no-stream --verbose
You said: "hello". (This is the offline echo provider.)
status: completed
run_id: run_5f1c8f0b0e04
steps: 1
tool calls: 0
tokens: 16 (prompt 2, completion 14)
duration: 1 ms

$ pyagents run "17 * 23 + 100" --tools calculator --provider openai --json
{
  "run_id": "run_...",
  "output": "The answer is 491.",
  "status": "completed",
  ...
}

Human review (pipe your answer in):

```bash
$ echo "Approved." | pyagents run "Summarise this" --provider echo --human-review
You said: "Summarise this". (This is the offline echo provider.)

Your response (approve by pressing Enter with no changes,
or type a revised answer): you> Approved.
Approved.
```

Exit codes: `0` success, `1` for configuration or provider errors (including a
missing API key), `2` for argument errors such as forgetting the prompt.

## `pyagents chat`

Interactive session with memory:

```
$ pyagents chat --provider openai
pyagents chat — openai/gpt-4o-mini.
Commands: /help, /tools, /reset, /instruction <text>, /exit

you> my name is Ada
Nice to meet you, Ada.
you> /tools
- calculator: Evaluate a mathematical expression and return the result.
- current_time: Return the current date and time.
you> /reset
(memory cleared)
you> /exit
```

Commands: `/help`, `/tools`, `/reset`, `/instruction <text>`, `/exit` (or
Ctrl-D). `--history N` controls how many messages stay in memory; when stdin is
not a terminal the prompt is omitted so the command can be scripted.

## `pyagents tools`

Inspect the built-in tool library:

```bash
$ pyagents tools list
calculator: Evaluate a mathematical expression and return the result. Supports ...
current_time: Return the current date and time.
read_file: Read a UTF-8 text file from disk.
list_directory: List the entries of a directory.
http_get: Fetch a URL with a GET request.

$ pyagents tools list --all          # include write_file, http_request, python_repl
$ pyagents tools list --verbose      # full schemas
$ pyagents tools list --json
```

## `pyagents providers`

Show every registered provider and whether its credentials are present:

```bash
$ pyagents providers
Providers (name → adapter, credential)
  anthropic          AnthropicProvider          missing
  claude             AnthropicProvider          missing
  echo               EchoProvider               not required
  ollama             OllamaProvider             not required
  openai             OpenAIProvider             missing
  openai-compatible  OpenAICompatibleProvider   not required

Default provider: ollama (llama3.2)
```

## `pyagents runs`

Query a SQLite run store:

```bash
pyagents runs list --store runs.sqlite3 --limit 5
pyagents runs list --store runs.sqlite3 --status failed
pyagents runs show run_ab12cd34 --store runs.sqlite3
pyagents runs delete run_ab12cd34 --store runs.sqlite3
pyagents runs show run_ab12cd34 --store runs.sqlite3 --json
```

```
$ pyagents runs list --store runs.sqlite3
run_ab12cd34  completed  steps=2   'The answer is 491.'
```

`--json` on the same command emits the full run summaries; with a run id,
`runs show` prints the complete `RunResult` JSON.

## `pyagents info`

Environment diagnostics - Python version, installed extras, provider defaults,
store paths - handy when filing a bug report:

```bash
$ pyagents info
$ pyagents info --json
```

## Configuration

The CLI reads the same environment variables as the library:

```bash
export OPENAI_API_KEY=sk-...
export PYAGENTS_PROVIDER=ollama
export PYAGENTS_MODEL=llama3.2
export OLLAMA_HOST=localhost:11434
```

`PYAGENTS_PROVIDER` wins over everything, which is the easiest way to pin a
default for scripts and CI.
