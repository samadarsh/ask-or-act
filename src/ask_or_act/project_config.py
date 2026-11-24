"""The project config file (docs/cli-design.md §3.10): ``ask-or-act.toml``, or
``[tool.ask-or-act]`` in ``pyproject.toml``.

* The search starts in the current folder and goes up to the repository root (the
  nearest folder with ``.git``), never above it, so a config in an unrelated parent
  folder is never picked up. Outside a repository only the current folder is searched.
* In each folder ``ask-or-act.toml`` wins over ``pyproject.toml``, as with uv.
* Paths inside the file (``baseline``, ``changelog``) are relative to the file's folder.
* Unknown keys are an error, so a typo can't silently fall back to a default.

Only the ``[gate]`` table is read so far. ``[defaults]`` for the other commands comes
later.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FILE_NAME = "ask-or-act.toml"


class ConfigError(ValueError):
    """A config file that can't be used (CLI exit 2)."""


@dataclass(frozen=True)
class ProjectConfig:
    path: Path  # the file it came from
    tables: dict[str, Any]

    @property
    def folder(self) -> Path:
        return self.path.parent

    def table(self, name: str) -> dict[str, Any]:
        value = self.tables.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"{self.path}: [{name}] must be a table")
        return value


def _read(path: Path) -> dict[str, Any] | None:
    """The ask-or-act tables in ``path``, or None if a pyproject has none."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: can't read it as TOML: {exc}") from exc
    if path.name == "pyproject.toml":
        tool = data.get("tool", {})
        found = tool.get("ask-or-act") if isinstance(tool, dict) else None
        if found is None:
            return None
        if not isinstance(found, dict):
            raise ConfigError(f"{path}: [tool.ask-or-act] must be a table")
        return found
    return data


def load(path: Path) -> ProjectConfig:
    """A config file named explicitly (``--policy``)."""
    if not path.is_file():
        raise ConfigError(f"{path}: no such file")
    tables = _read(path)
    if tables is None:
        raise ConfigError(f"{path}: has no [tool.ask-or-act] table")
    return ProjectConfig(path, tables)


def find(start: Path | None = None) -> ProjectConfig | None:
    """The nearest config between ``start`` (default: cwd) and the repository root."""
    here = (start or Path.cwd()).resolve()
    chain = [here, *here.parents]
    root = next((d for d in chain if (d / ".git").exists()), None)
    folders = chain[: chain.index(root) + 1] if root is not None else [here]
    for folder in folders:
        for name in (FILE_NAME, "pyproject.toml"):
            path = folder / name
            if path.is_file():
                tables = _read(path)
                if tables is not None:
                    return ProjectConfig(path, tables)
    return None


def check_keys(
    where: str, table: dict[str, Any], allowed: dict[str, type | tuple[type, ...]]
) -> None:
    """Unknown keys and wrong types are errors, with every problem named."""
    problems = [f"unknown key {k!r}" for k in table if k not in allowed]
    for k, want in allowed.items():
        if k in table:
            v = table[k]
            types = want if isinstance(want, tuple) else (want,)
            # bool is an int in Python, so `true` must not pass for a number
            ok = isinstance(v, types) and not (isinstance(v, bool) and bool not in types)
            if not ok:
                names = " or ".join(t.__name__ for t in types)
                problems.append(f"{k!r} must be {names}, got {type(v).__name__}")
    if problems:
        known = ", ".join(sorted(allowed))
        raise ConfigError(f"{where}: {'; '.join(problems)} (known keys: {known})")
