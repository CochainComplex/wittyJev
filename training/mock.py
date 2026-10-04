"""A tiny, randomly initialised stand-in for JevK5: same architecture, tokenizer and layer pattern as the 4B.

    python mock.py ../data/mock/base

Every training and inference path (lora.py, jevk5.runtime.JevK5, the GGUF conversion) runs on it in seconds
on a CPU, so the Tractatus pipeline can be tested end to end before the real run. Swapping in the real
model means passing its Hub id or folder instead of this one; nothing else changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REAL = "alibiserikbay/JevK5"
TINY = {  # everything else, the rope settings included, stays as in the 4B
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "mtp_num_hidden_layers": 0,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    torch.manual_seed(args.seed)
    config = json.loads(Path(hf_hub_download(REAL, "config.json")).read_text())
    config.update(TINY)
    config = transformers.Qwen3_5TextConfig(**{k: v for k, v in config.items() if k != "architectures"})
    model = transformers.Qwen3_5ForCausalLM(config)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    transformers.AutoTokenizer.from_pretrained(REAL).save_pretrained(args.out)
    (args.out / "jevk5_config.json").write_text(json.dumps({"temperature": 1.0}) + "\n")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
