"""File-reading half of `hub`: tiny local files shaped like the real ones, no network."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ask_or_act.data import hub


def write_banking_parquet(path: Path) -> None:
    table = pa.table({"text": ["where is my card", "top up"], "label": [1, 0]})
    hf_meta = {"info": {"features": {"label": {"names": ["top_up", "card_arrival"]}}}}
    table = table.replace_schema_metadata({"huggingface": json.dumps(hf_meta)})
    pq.write_table(table, path)


def test_read_parquet_returns_rows_and_hf_features(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    write_banking_parquet(path)
    rows, features = hub.read_parquet(path)
    assert rows == [{"text": "where is my card", "label": 1}, {"text": "top up", "label": 0}]
    assert features["label"]["names"] == ["top_up", "card_arrival"]


def test_read_parquet_without_hf_metadata(tmp_path: Path) -> None:
    path = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"a": [1]}), path)
    assert hub.read_parquet(path) == ([{"a": 1}], {})


@pytest.mark.parametrize("content", ['[{"id": "1"}, {"id": "2"}]', '{"id": "1"}\n\n{"id": "2"}\n'])
def test_read_json_rows_accepts_array_and_jsonl(tmp_path: Path, content: str) -> None:
    path = tmp_path / "rows.json"
    path.write_text(content)
    assert hub.read_json_rows(path) == [{"id": "1"}, {"id": "2"}]


def test_load_candidates_banking77(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    write_banking_parquet(path)
    cands = hub.load_candidates("banking77", [path])
    assert [c.label for c in cands] == ["card_arrival", "top_up"]


def test_load_candidates_aegis(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    row = {"id": "a1", "prompt": "p", "prompt_label": "safe", "prompt_label_source": "human"}
    path.write_text(json.dumps([row]))
    (c,) = hub.load_candidates("aegis", [path])
    assert (c.id, c.label, c.meta["source_file"]) == ("a1", "safe", "test.json")


def test_load_candidates_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        hub.load_candidates("imagenet", [])
