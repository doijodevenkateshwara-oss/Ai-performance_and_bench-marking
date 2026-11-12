import json
import random

from llm_perf.config import DatasetConfig
from llm_perf.datasets import Dataset, import_traces, write_dataset


def _cfg(tmp_path, rows, **kw):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return DatasetConfig(source=str(p), format="jsonl", **kw)


def test_load_jsonl(tmp_path):
    cfg = _cfg(tmp_path, [{"prompt": "a"}, {"prompt": "b"}])
    ds = Dataset.load("d", cfg)
    assert len(ds.rows) == 2


def test_column_mapping(tmp_path):
    cfg = _cfg(tmp_path, [{"q": "hello", "a": "world"}], columns={"prompt": "q", "expected": "a"})
    ds = Dataset.load("d", cfg)
    assert ds.rows[0]["prompt"] == "hello"
    assert ds.rows[0]["expected"] == "world"


def test_sample_first_n(tmp_path):
    rows = [{"prompt": str(i)} for i in range(10)]
    cfg = _cfg(tmp_path, rows, sample={"strategy": "first", "count": 3})
    ds = Dataset.load("d", cfg)
    assert len(ds.rows) == 3


def test_render_vars(tmp_path):
    cfg = _cfg(tmp_path, [{"prompt": "Tell me about {topic}."}], vars={"topic": ["k8s", "finops"]})
    ds = Dataset.load("d", cfg)
    rng = random.Random(0)
    row = ds.sample(rng)
    assert "{topic}" not in row["prompt"]
    assert row["prompt"] in ("Tell me about k8s.", "Tell me about finops.")


def test_shard_deterministic(tmp_path):
    rows = [{"prompt": str(i)} for i in range(10)]
    cfg = _cfg(tmp_path, rows)
    ds = Dataset.load("d", cfg)
    s0 = ds.shard(0, 2)
    s1 = ds.shard(1, 2)
    assert len(s0.rows) + len(s1.rows) == len(ds.rows)
    assert s0.rows[0]["prompt"] == "0"
    assert s1.rows[0]["prompt"] == "1"


def test_import_traces_generic(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text(
        "\n".join(json.dumps({"inputs": {"question": f"q{i}"}, "outputs": {"answer": "a"}}) for i in range(3))
    )
    rows = import_traces(
        str(p), fmt="jsonl", mapping={"prompt": "inputs.question", "expected": "outputs.answer"}
    )
    assert len(rows) == 3
    assert rows[0]["prompt"] == "q0"


def test_import_traces_langsmith(tmp_path):
    p = tmp_path / "ls.jsonl"
    p.write_text(json.dumps({"inputs": {"question": "hi"}, "outputs": {"answer": "yo"}}))
    rows = import_traces(str(p), fmt="langsmith")
    assert rows[0]["prompt"] == "hi"
    assert rows[0]["expected"] == "yo"


def test_write_dataset_jsonl(tmp_path):
    dest = tmp_path / "out.jsonl"
    write_dataset([{"prompt": "x"}, {"prompt": "y"}], str(dest), "jsonl")
    assert len(dest.read_text().strip().splitlines()) == 2
