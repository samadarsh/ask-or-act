"""Reading and checking predictions files (docs/cli-design.md §4). No network.

Table-driven: one small input per error and warning, so every message in §4.2, §4.5
and §4.6 is exercised.
"""

import io
import math
from pathlib import Path
from typing import Any

import pytest

from ask_or_act.guarantees.sample_size import min_rows
from ask_or_act.predictions import (
    InputError,
    Scale,
    check,
    hashed_split,
    read_predictions,
    suggest_scale,
)

HEADER = "id,prediction,confidence,human_label"


def _csv(*rows: str, header: str = HEADER) -> str:
    return "\n".join([header, *rows]) + "\n"


def _many(n: int, conf: str = "0.9", label: str = "a", pred: str = "a") -> list[str]:
    return [f"r{i},{pred},{conf},{label}" for i in range(n)]


def _distinct(n: int) -> list[str]:
    """n rows with n distinct confidences in [0.6, 0.99] (no sparse-confidence warning)."""
    return [f"r{i},a,{0.6 + 0.39 * i / max(1, n - 1):.4f},a" for i in range(n)]


def _run(tmp_path: Path, text: str, name: str = "p.csv", **kw: Any) -> Any:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return check(path, **kw)


def _codes(result: Any, severity: str) -> set[str]:
    return {p.code for p in result.problems if p.severity == severity}


ERRORS: list[tuple[str, str, str, dict[str, Any], str]] = [
    # (case, file name, contents, options, expected error code)
    ("missing file", "nope.csv", "", {"_missing": True}, "unreadable"),
    ("empty file", "p.csv", "", {}, "empty"),
    ("header only", "p.csv", HEADER + "\n", {}, "empty"),
    ("json array", "p.json", "[]", {}, "format"),
    ("unknown extension", "p.txt", "x", {}, "format"),
    ("duplicate header", "p.csv", _csv("1,a,0.9,a,a", header=HEADER + ",id"), {}, "header"),
    ("too many fields", "p.csv", _csv("1,a,0.9,a,extra"), {}, "fields"),
    (
        "missing column",
        "p.csv",
        _csv("1,a,0.9", header="id,prediction,confidence"),
        {},
        "missing_column",
    ),
    (
        "bad --columns target",
        "p.csv",
        _csv("1,a,0.9,a"),
        {"columns": {"prediction": "nope"}},
        "columns",
    ),
    ("unknown --columns name", "p.csv", _csv("1,a,0.9,a"), {"columns": {"nope": "id"}}, "columns"),
    ("empty id", "p.csv", _csv(",a,0.9,a"), {}, "empty_id"),
    ("duplicate id", "p.csv", _csv("1,a,0.9,a", "1,a,0.8,a"), {}, "duplicate_id"),
    ("confidence not a number", "p.csv", _csv("1,a,high,a"), {}, "conf_not_number"),
    ("confidence NaN", "p.csv", _csv("1,a,nan,a"), {}, "conf_not_number"),
    ("confidence empty with a prediction", "p.csv", _csv("1,a,,a"), {}, "conf_empty"),
    ("confidence out of range", "p.csv", _csv("1,a,1.5,a"), {}, "conf_range"),
    (
        "percent value too big",
        "p.csv",
        _csv("1,a,150,a"),
        {"confidence_scale": "percent"},
        "conf_range",
    ),
    ("votes not whole", "p.csv", _csv("1,a,2.5,a"), {"confidence_scale": "votes:5"}, "conf_range"),
    ("logprob positive", "p.csv", _csv("1,a,0.3,a"), {"confidence_scale": "logprob"}, "conf_range"),
    (
        "prediction not an option",
        "p.csv",
        _csv("1,c,0.9,a,a|b", header=HEADER + ",options"),
        {},
        "pred_not_option",
    ),
    (
        "label not an option",
        "p.csv",
        _csv("1,a,0.9,c,a|b", header=HEADER + ",options"),
        {},
        "label_not_option",
    ),
    ("empty split", "p.csv", _csv("1,a,0.9,a,", header=HEADER + ",split"), {}, "split_empty"),
    (
        "unknown split",
        "p.csv",
        _csv("1,a,0.9,a,dev", header=HEADER + ",split"),
        {},
        "split_unknown",
    ),
    (
        "group in both splits",
        "p.csv",
        _csv("1,a,0.9,a,calibration,g", "2,a,0.9,a,test,g", header=HEADER + ",split,group"),
        {},
        "group_leak",
    ),
    (
        "positive with three labels",
        "p.csv",
        _csv("1,a,0.9,a", "2,b,0.9,b", "3,c,0.9,c"),
        {"confidence_of": "positive"},
        "positive_not_binary",
    ),
    (
        "positive label unknown",
        "p.csv",
        _csv("1,a,0.9,a", "2,b,0.9,b"),
        {"confidence_of": "positive", "positive": "z"},
        "positive_unknown",
    ),
    (
        "positive label needed",
        "p.csv",
        _csv("1,safe,0.9,safe", "2,unsafe,0.9,unsafe"),
        {"confidence_of": "positive"},
        "positive_missing",
    ),
    ("jsonl not an object", "p.jsonl", "[1, 2]\n", {}, "jsonl"),
    ("jsonl not json", "p.jsonl", "{oops\n", {}, "jsonl"),
    (
        "jsonl list in a text column",
        "p.jsonl",
        '{"id": 1, "prediction": ["a"], "confidence": 0.9, "human_label": "a"}\n',
        {},
        "not_text",
    ),
]


@pytest.mark.parametrize(("case", "name", "text", "kw", "code"), ERRORS, ids=[e[0] for e in ERRORS])
def test_each_error_is_reported_and_blocks_the_table(
    tmp_path: Path, case: str, name: str, text: str, kw: dict[str, Any], code: str
) -> None:
    if kw.pop("_missing", False):
        result = check(tmp_path / name)
    else:
        result = _run(tmp_path, text, name, **kw)
    assert code in _codes(result, "error"), (case, result.problems)
    assert result.table is None
    assert not result.ok
    for p in result.errors:
        assert p.message
    with pytest.raises(InputError) as info:
        read_predictions(tmp_path / name, **kw)
    assert any(p.code == code for p in info.value.problems)


WARNINGS: list[tuple[str, str, dict[str, Any], str]] = [
    ("failed prediction", _csv("f,,,a", *_distinct(60)), {}, "failed"),
    ("unlabelled rows", _csv("u,a,0.9,", *_distinct(60)), {}, "unlabelled"),
    ("no labels at all", _csv("1,a,0.9,", "2,a,0.8,"), {}, "no_labels"),
    ("constant confidence", _csv(*_many(60)), {}, "constant_conf"),
    (
        "few distinct confidences",
        _csv(*[f"r{i},a,0.{5 + i % 4},a" for i in range(60)]),
        {},
        "few_conf",
    ),
    ("case mismatch", _csv("c,Refund,0.9,refund", *_distinct(60)), {}, "case_mismatch"),
    (
        "looks like P(positive)",
        _csv(
            *[f"n{i},safe,0.{10 + i},safe" for i in range(20)],
            *[f"p{i},unsafe,0.{70 + i},unsafe" for i in range(20)],
        ),
        {},
        "looks_like_positive",
    ),
    (
        "below 1/k",
        _csv(*[f"r{i},{'abc'[i % 3]},0.{10 + i},{'abc'[i % 3]}" for i in range(40)]),
        {},
        "below_chance",
    ),
    ("small test half", _csv(*_distinct(60)), {}, "small_test"),
    ("too few calibration rows", _csv(*_distinct(20)), {}, "few_calibration"),
]


@pytest.mark.parametrize(("case", "text", "kw", "code"), WARNINGS, ids=[w[0] for w in WARNINGS])
def test_each_warning_is_reported_and_the_table_still_builds(
    tmp_path: Path, case: str, text: str, kw: dict[str, Any], code: str
) -> None:
    result = _run(tmp_path, text, **kw)
    assert code in _codes(result, "warning"), (case, result.problems)
    assert result.ok
    assert result.table is not None


def test_clean_file_has_no_problems_and_converts_every_row(tmp_path: Path) -> None:
    rows = [
        f"r{i},{'ab'[i % 2]},{0.5 + i / 2000:.4f},{'ab'[(i // 3) % 2]},a|b,"
        f"{'calibration' if i % 2 else 'test'},g{i}"
        for i in range(400)
    ]
    result = _run(tmp_path, _csv(*rows, header=HEADER + ",options,split,group"))
    assert result.problems == ()
    t = result.table
    assert t is not None
    assert len(t) == 400
    assert t.rows[1].split == "calibration"
    assert t.rows[1].options == ("a", "b")
    assert t.rows[1].group == "g1"
    assert result.labelled == {"calibration": 200, "test": 200}


def test_every_problem_is_found_in_one_pass(tmp_path: Path) -> None:
    """Missing columns with an obvious match are still checked under that match, so a
    user sees the percent problem too, not one error per run."""
    text = _csv("1,a,87,a", "2,b,91,b", header="id,pred,conf,label")
    result = _run(tmp_path, text)
    assert _codes(result, "error") == {"missing_column", "conf_range"}
    by_code = {p.code: p for p in result.errors}
    assert (
        "--columns prediction=pred,confidence=conf,human_label=label"
        in by_code["missing_column"].hint
    )
    assert "--confidence-scale percent" in by_code["conf_range"].hint


def test_messages_quote_short_snippets_only(tmp_path: Path) -> None:
    long = "x" * 200
    result = _run(tmp_path, _csv(f"1,a,{long},a"))
    message = result.errors[0].message
    assert long not in message
    assert "x" * 19 + "…" in message


def test_line_numbers_are_file_lines_and_capped(tmp_path: Path) -> None:
    result = _run(tmp_path, _csv(*[f"r{i},a,9,a" for i in range(8)]))
    p = next(p for p in result.errors if p.code == "conf_range")
    assert p.lines == (2, 3, 4, 5, 6)
    assert p.count == 8
    assert "lines 2, 3, 4, 5, 6, …" in p.message


@pytest.mark.parametrize(
    ("values", "scale"),
    [
        ([0.1, 0.9], "probability"),
        ([12.0, 87.0, 99.5], "percent"),
        ([0.0, 3.0, 5.0], "votes:5"),
        ([-0.01, -2.3], "logprob"),
        ([-3.2, 4.1], ""),  # logits
        ([0.5, 250.0], ""),
    ],
)
def test_scale_detection_only_suggests(values: list[float], scale: str) -> None:
    suggested, hint = suggest_scale(values)
    assert suggested == scale
    if scale not in ("", "probability"):
        assert f"--confidence-scale {scale.split(':')[0]}" in hint
    if scale == "":
        assert "calibrate" in hint


@pytest.mark.parametrize(
    ("scale", "raw", "expected"),
    [
        ("probability", 0.25, 0.25),
        ("percent", 87.0, 0.87),
        ("votes:4", 3.0, 0.75),
        ("logprob", math.log(0.8), 0.8),
        ("logprob", 0.0, 1.0),
    ],
)
def test_scales_convert_to_probabilities(scale: str, raw: float, expected: float) -> None:
    got = Scale.parse(scale).convert(raw)
    assert got == pytest.approx(expected)


def test_scales_pick_the_frozen_pre_registered_grids() -> None:
    from ask_or_act.guarantees.fixed_sequence import DEFAULT_STARTS, grid_for

    assert Scale.parse("percent").grid().grid == grid_for("verbalized")
    assert Scale.parse("votes:5").grid().grid == grid_for("self_consistency", samples=5)
    assert Scale.parse("probability").grid().starts == DEFAULT_STARTS["continuous"]
    with pytest.raises(ValueError, match="unknown confidence scale"):
        Scale.parse("votes:0")


def test_percent_scale_reads_the_scores_file(tmp_path: Path) -> None:
    text = _csv("1,a,87,a", "2,b,91,b", header="id,pred,conf,label")
    table = read_predictions(
        _write(tmp_path, text),
        columns={"prediction": "pred", "confidence": "conf", "human_label": "label"},
        confidence_scale="percent",
    )
    assert [r.confidence for r in table.rows] == pytest.approx([0.87, 0.91])
    assert table.grid.text == "percent steps"


def _write(tmp_path: Path, text: str, name: str = "p.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_confidence_of_positive_converts_to_confidence_in_the_prediction(
    tmp_path: Path,
) -> None:
    text = _csv("1,unsafe,0.9,unsafe", "2,safe,0.2,safe", "3,,,safe")
    table = read_predictions(_write(tmp_path, text), confidence_of="positive", positive="unsafe")
    assert [r.confidence for r in table.rows] == pytest.approx([0.9, 0.8, None])


@pytest.mark.parametrize(("labels", "positive"), [(("0", "1"), "1"), (("false", "True"), "True")])
def test_positive_label_is_inferred_only_for_0_1_and_true_false(
    tmp_path: Path, labels: tuple[str, str], positive: str
) -> None:
    neg, pos = labels
    text = _csv(f"1,{pos},0.9,{pos}", f"2,{neg},0.3,{neg}")
    result = _run(tmp_path, text, confidence_of="positive")
    assert result.positive == positive
    assert result.table is not None
    assert result.table.rows[1].confidence == pytest.approx(0.7)


def test_hashed_split_is_reproducible_and_stable_when_rows_change(tmp_path: Path) -> None:
    rows = _distinct(400)
    a = _run(tmp_path, _csv(*rows), "a.csv").table
    b = _run(tmp_path, _csv(*reversed(rows[:300])), "b.csv").table  # fewer rows, reversed
    assert a is not None
    assert b is not None
    split_a = {r.id: r.split for r in a.rows}
    assert all(split_a[r.id] == r.split for r in b.rows)
    assert 150 < sum(s == "calibration" for s in split_a.values()) < 250
    other = _run(tmp_path, _csv(*rows), "c.csv", split_seed=1).table
    assert other is not None
    assert {r.id: r.split for r in other.rows} != split_a
    assert hashed_split("x", 0, 0.5) == hashed_split("x", 0, 0.5)


def test_hashed_split_keeps_groups_together(tmp_path: Path) -> None:
    rows = [f"r{i},a,{0.5 + i / 1000:.3f},a,g{i // 4}" for i in range(200)]
    table = _run(tmp_path, _csv(*rows, header=HEADER + ",group")).table
    assert table is not None
    by_group: dict[str | None, set[str | None]] = {}
    for r in table.rows:
        by_group.setdefault(r.group, set()).add(r.split)
    assert all(len(s) == 1 for s in by_group.values())


def test_split_map_renames_values(tmp_path: Path) -> None:
    text = _csv("1,a,0.9,a,dev", "2,a,0.8,a,holdout", header=HEADER + ",split")
    table = _run(tmp_path, text, split_map={"dev": "calibration", "holdout": "test"}).table
    assert table is not None
    assert [r.split for r in table.rows] == ["calibration", "test"]


def test_jsonl_tsv_stdin_and_column_mappings_all_read_the_same(tmp_path: Path) -> None:
    jsonl = (
        '{"id": 1, "prediction": "a", "confidence": 0.9, "human_label": "a",'
        ' "options": ["a", "b"]}\n'
        "\n"
        '{"id": 2, "prediction": true, "confidence": "0.8", "human_label": null}\n'
    )
    t1 = read_predictions(_write(tmp_path, jsonl, "p.jsonl"))
    assert [(r.id, r.prediction, r.confidence, r.label) for r in t1.rows] == [
        ("1", "a", 0.9, "a"),
        ("2", "true", 0.8, None),
    ]
    assert t1.rows[0].options == ("a", "b")
    tsv = "id\tprediction\tconfidence\thuman_label\n1\ta\t0.9\ta\n"
    t2 = read_predictions(_write(tmp_path, tsv, "p.tsv"))
    t3 = read_predictions(io.StringIO(tsv), format="tsv")
    t4 = read_predictions(
        {"id": ["1"], "prediction": ["a"], "confidence": [0.9], "human_label": ["a"]}
    )
    assert t2.rows == t3.rows == t4.rows


def test_utf8_bom_is_tolerated_and_other_encodings_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    path.write_bytes(("﻿" + _csv("1,a,0.9,a")).encode("utf-8"))
    assert read_predictions(path).rows[0].id == "1"
    path.write_bytes(_csv("1,café,0.9,café").encode("latin-1"))
    result = check(path)
    assert {p.code for p in result.errors} == {"encoding"}


def test_unlabelled_rows_stay_in_the_table_but_are_never_scored(tmp_path: Path) -> None:
    table = read_predictions(_write(tmp_path, _csv("1,a,0.9,a", "2,a,0.8,")))
    assert len(table) == 2
    assert table.n_labelled == 1
    assert table.scored.ids == ("1",)


@pytest.mark.parametrize(
    ("alpha", "starts", "rate", "expected"),
    [
        (0.1, 4, 0.0, 36),
        (0.1, 4, 0.05, 114),
        (0.05, 4, 0.0, 72),
        (0.2, 4, 0.0, 17),
        (0.1, 1, 0.0, 22),  # self-consistency: one scan start
        (0.01, 4, 0.0, 368),
    ],
)
def test_min_rows_matches_the_table_in_the_design_doc(
    alpha: float, starts: int, rate: float, expected: int
) -> None:
    assert min_rows(alpha, 0.1, starts=starts, error_rate=rate) == expected


def test_min_rows_edge_cases() -> None:
    assert min_rows(0.1, 0.1, error_rate=0.1) is None  # can't certify at the target rate
    with pytest.raises(ValueError, match="need 0 < alpha"):
        min_rows(1.5, 0.1)
