"""Static checks on the GitHub Action and the workflows that use it. No network.

They hold the security rules the Action promises (action.yml's header): actions
pinned to full commit SHAs, no `${{ }}` expanded inside a shell script (script
injection), no pull_request_target, never contents: write.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
FILES = [
    ROOT / "action.yml",
    ROOT / ".github" / "workflows" / "gate.yml",
    ROOT / "examples" / "github" / "gate.yml",
]


def scripts(text: str) -> list[str]:
    """The body of every `run:` step, block (`run: |`) or one-line."""
    out: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(?:- )?run: ?(.*)$", line)
        if not m:
            continue
        if m.group(2).strip() not in ("|", ">", "|-", ">-"):
            out.append(m.group(2))
            continue
        indent = len(m.group(1))
        body = []
        for nxt in lines[i + 1 :]:
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                break
            body.append(nxt)
        out.append("\n".join(body))
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_expression_inside_a_script(path: Path) -> None:
    for body in scripts(path.read_text()):
        assert "${{" not in body, f"pass it through env: instead\n{body}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_actions_are_pinned_to_a_commit(path: Path) -> None:
    for ref in re.findall(r"^\s*(?:- )?uses:\s*(\S+)", path.read_text(), re.MULTILINE):
        if ref == "./":
            continue
        name, _, version = ref.partition("@")
        if name == "samadarsh/ask-or-act":  # the example's placeholder for users
            assert version == "COMMIT_SHA"
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", version), ref


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_write_to_contents_and_no_pull_request_target(path: Path) -> None:
    text = "\n".join(line.split("#")[0] for line in path.read_text().splitlines())
    assert "pull_request_target" not in text
    assert not re.search(r"contents:\s*write", text)


def test_the_comment_skips_forks_and_edits_its_own_comment() -> None:
    from ask_or_act.gate_markdown import MARKER

    text = (ROOT / "action.yml").read_text()
    assert "github.event.pull_request.head.repo.full_name == github.repository" in text
    assert f'startswith("{MARKER}")' in text
    assert "--no-record" in text  # CI never appends to the changelog
