"""The analysis commands never open a network connection (docs/cli-design.md §5.4).

Every way Python code reaches the network goes through the socket module, so the
socket layer is replaced for the whole test. Each replacement records the attempt and
raises. A command that tried to connect would either fail or be recorded, and the
test checks both. Commands join ``COMMANDS`` as they are built.
"""

import socket
from pathlib import Path
from typing import Any

import pytest

from ask_or_act.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "predictions"

COMMANDS: list[list[str]] = [
    ["check", str(FIXTURES / "support-bot.csv")],
    ["thresholds", str(FIXTURES / "support-bot-large.csv")],
    ["thresholds", str(FIXTURES / "support-bot.csv"), str(FIXTURES / "support-bot-large.csv")],
    ["metrics", str(FIXTURES / "support-bot.csv"), "--n-boot", "50"],
    [
        "compare",
        str(FIXTURES / "support-bot.csv"),
        str(FIXTURES / "support-bot-large.csv"),
        "--n-boot",
        "50",
    ],
    ["check", str(FIXTURES / "scores.csv"), "--json"],
    [
        "check",
        str(FIXTURES / "moderation.jsonl"),
        "--confidence-of",
        "positive",
        "--positive",
        "unsafe",
    ],
]


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []

    def refuse(name: str) -> Any:
        def blocked(*args: object, **kwargs: object) -> None:
            attempts.append(f"{name}{args[1:] if name.endswith('connect') else args}")
            raise OSError(f"network blocked by test_local_only: {name}")

        return blocked

    monkeypatch.setattr(socket.socket, "connect", refuse("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", refuse("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    return attempts


def test_the_blocker_itself_works(no_network: list[str]) -> None:
    with pytest.raises(OSError, match="network blocked"):
        socket.create_connection(("example.com", 443))
    assert no_network == ["create_connection(('example.com', 443),)"]


@pytest.mark.parametrize("argv", COMMANDS, ids=[" ".join(a[:1] + a[2:]) or a[0] for a in COMMANDS])
def test_analysis_commands_make_no_network_calls(
    argv: list[str], no_network: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(argv)
    capsys.readouterr()
    assert no_network == []
    assert code in (0, 6, 7)  # ran to completion: never a crash (70) from a blocked socket
