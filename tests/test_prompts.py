import random

from llm_perf.prompts import PromptPicker, load_prompts


def test_load_prompts_list():
    assert load_prompts(["a", "b"]) == ["a", "b"]


def test_load_prompts_yaml(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("prompts:\n  - hello\n  - world\n")
    assert load_prompts(str(p)) == ["hello", "world"]


def test_load_prompts_txt(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("first prompt\n\nsecond prompt\n")
    assert load_prompts(str(p)) == ["first prompt", "second prompt"]


def test_picker_vars_render():
    picker = PromptPicker(prompts=["Tell me about {topic}."],
                          vars_={"topic": ["k8s", "ai"]})
    rng = random.Random(1)
    out = picker.pick(rng)
    assert "{topic}" not in out
    assert out.startswith("Tell me about ")


def test_picker_row_priority():
    picker = PromptPicker(prompts=["fallback"])
    row = {"prompt": "from row"}
    assert picker.pick(random.Random(0), row) == "from row"
