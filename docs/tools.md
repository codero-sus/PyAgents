# Tools

Tools are ordinary Python functions with type hints and a docstring. PyAgents
derives the JSON schema from them, validates the model's arguments, runs the
call, and turns failures into observations instead of crashes.

## Declaring a tool

```python
from pyagents.tools import tool

@tool
def convert(amount: float, currency: str = "USD") -> str:
    """Convert an amount into the given currency.

    Args:
        amount: How much money to convert.
        currency: Target currency code.
    """
    return f"{amount * 0.92:.2f} {currency}"
```

The generated schema:

```json
{
  "name": "convert",
  "description": "Convert an amount into the given currency.",
  "parameters": {
    "type": "object",
    "properties": {
      "amount": {"type": "number", "description": "How much money to convert."},
      "currency": {"type": "string", "description": "Target currency code.", "default": "USD"}
    },
    "required": ["amount"]
  }
}
```

Configuration is optional:

```python
@tool(name="fx", description="Convert currencies.", tags=["finance"], strict=True)
def convert(...): ...
```

| option | effect |
|---|---|
| `name` | tool name sent to the model (defaults to the function name) |
| `description` | overrides the first docstring paragraph |
| `parameters` | supply the JSON schema by hand (skips inference) |
| `strict` | reject unknown arguments with `ToolValidationError` |
| `output_errors` | return failures as error observations (default `True`); set `False` to raise |
| `tags` / `metadata` | free-form labels for your own filtering |

## Supported annotations

| annotation | JSON schema |
|---|---|
| `str`, `int`, `float`, `bool` | string / integer / number / boolean |
| `list[str]`, `List[int]` | `array` with typed `items` |
| `dict`, `Dict[str, int]` | `object` |
| `Optional[T]` | `T`, not required |
| `Literal["a", "b"]` | `enum` |
| `enum.Enum` subclasses | `enum` of the member values |
| `dataclasses.dataclass` | nested object schema, recursively |
| pydantic `BaseModel` | its own `model_json_schema()` |
| `Annotated[int, "hint"]` | integer with the hint as `description` |

Parameter descriptions come from Google-style, NumPy-style and reST docstrings
(`utils.parse_docstring`). ``*args``/``**kwargs`` are ignored by default;
``**kwargs`` sets `additionalProperties: true`.

## Calling tools directly

```python
from pyagents.tools import calculator

result = calculator.invoke({"expression": "2 + 2"})

result.content        # "4"
result.output         # 4 (the raw return value)
result.is_error       # False
result.duration_ms
```

`invoke` / `ainvoke` never raise for bad arguments - they return a
`ToolResult` with `is_error=True` and a human-readable message that also goes
back to the model. Use `tool.func(**kwargs)` when you want the raw exception.

## Bundles

```python
from pyagents.tools import Toolkit, default_toolkit

kit = Toolkit([...])          # or default_toolkit()
kit.get("calculator")         # -> Tool
kit.names()                   # ["calculator", ...]
kit.filter(tags=["web"])      # -> a new Toolkit
kit.to_specs()                # the schema list sent to providers
agent.add_tool(kit)           # a Toolkit can be added as a whole
```

## Built-in tools

| tool | what it does | notes |
|---|---|---|
| `calculator` | safe arithmetic (`+ - * / // % **`, parentheses, `sqrt`, `log`, `sin`, `factorial`, `pi`, ...) | parsed with `ast`; no `eval` |
| `current_time` | ISO timestamp, local or for an IANA timezone | uses `zoneinfo`, falls back to local time |
| `read_file` | read a UTF-8 text file (truncated after `max_bytes`) | 20 kB default |
| `write_file` | write text to a path | creates parent directories |
| `list_directory` | list the entries of a directory | non-recursive |
| `http_get` | GET a URL and return status + body | needs network |
| `http_request` | any method, with headers/body/timeout | needs network |
| `python_repl` | execute Python and capture stdout | runs in-process with full privileges |

```python
from pyagents.tools import default_tools, default_toolkit

default_tools()                      # calculator, current_time, files, http_get
default_tools(include=["python_repl"])
default_toolkit(include=["python_repl", "http_request"])
```

`default_tools()` intentionally excludes `write_file`, `http_request` and
`python_repl` - add them explicitly when you want them.

## Security notes

* `calculator` never calls `eval`; it walks an AST allow-list.
* `python_repl` executes arbitrary code **in-process** with the permissions
  of your program. It is not a sandbox - do not expose it to untrusted models.
* HTTP tools will happily fetch any URL the process can reach. Put your own
  allow-list in a `before_tool` hook if that matters to you.
* File tools operate with the permissions of the host process.

## Wrapping stateful objects

Bound methods work exactly like plain functions, which is the easiest way to
expose a client, cache or connection pool as a tool:

```python
from pyagents.tools import Tool

class WeatherService:
    def __init__(self, client):
        self.client = client

    async def weather(self, city: str) -> str:
        """Look up the current weather for a city.

        Args:
            city: City name, e.g. ``"Delhi"``.
        """
        return await self.client.current_weather(city)

service = WeatherService(client)
get_weather = Tool(service.weather, name="get_weather", tags=["weather"])
```

The schema, description and async handling all come from the bound method, and
the instance is captured for you. Override `parameters` only when you need a
hand-written schema.
