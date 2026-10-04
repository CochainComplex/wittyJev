"""lora.py's text loss (the Tractatus form rows) equals transformers' own causal-LM loss on the target tokens.
Needs torch and transformers; skipped without them."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
lora = pytest.importorskip("lora")


def tiny_qwen():
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
    )
    return transformers.Qwen3_5ForCausalLM(config).eval()


def test_text_loss_is_the_causal_lm_loss_on_the_target():
    model = tiny_qwen()
    rows = [{"ids": [5, 9, 11, 3, 7, 2, 8], "start": 4}, {"ids": [1, 4, 6, 20, 30], "start": 2}]
    dm = SimpleNamespace(model=model, device="cpu")
    with torch.no_grad():
        ours = lora.lm_loss(dm, rows)
        theirs = []
        for r in rows:
            ids = torch.tensor([r["ids"]])
            labels = ids.clone()
            labels[0, : r["start"]] = -100  # the prompt is not a target
            theirs.append(model(input_ids=ids, labels=labels).loss)
    assert ours.item() == pytest.approx(torch.stack(theirs).mean().item(), rel=1e-4)
