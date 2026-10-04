"""LoRA-tune a causal LM as a one-pass decision model: cross-entropy on the option-letter logits.

    cd training && python lora.py ../data/gate3 ../models/jevk5 --epochs 2 --lr 3e-5 --warmup 20

Targets are the gold option (one-hot) or, for items with gold_probs, the exact distribution, so
probability questions learn to spread mass as the evidence says. Only the last position's hidden
state is read, and only the option letters' rows of the output layer, as at inference. The
adapter is merged into the base weights at the end, so inference costs exactly the base model.
Evaluates on dev.jsonl before and after training (accuracy, NLL, ECE per source).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from batching import plan_batches

from jevk5.runtime import JevK5, decision_options

TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkvz",
    "in_proj_ba",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


def fla_backward_ok() -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        major, minor, patch = (int(x) for x in version("triton").split(".")[:3])
    except (PackageNotFoundError, ValueError):
        return False
    return (major, minor, patch) >= (3, 7, 1)


def encode_items(dm: JevK5, items: list[dict], max_len: int) -> list[dict]:
    out = []
    for item in items:
        options = decision_options(item["question"])
        keys = [k for k, _ in options]
        if item["expected"] not in keys or len(keys) > 16:
            continue
        ids = dm.encode(item["state"], item["question"]["instructions"], [t for _, t in options])
        if len(ids) > max_len:
            continue
        target = np.zeros(len(keys), dtype=np.float32)
        gold = item.get("gold_probs")
        if gold and set(gold) == set(keys):
            target[:] = [gold[k] for k in keys]
        else:
            target[keys.index(item["expected"])] = 1.0
        out.append(
            {
                "ids": ids,
                "target": target / target.sum(),
                "source": item["source"],
                "kind": item["question"]["type"],
                "id": item.get("id"),
            }
        )
    return out


def encode_lm(dm: JevK5, items: list[dict], max_len: int) -> list[dict]:
    """Rows that teach the model to write text (the Tractatus form): its chat prompt, then the target."""
    out = []
    for item in items:
        prompt = dm.tok.apply_chat_template(
            item["lm"]["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        head = dm.tok.encode(prompt, add_special_tokens=False)
        tail = dm.tok.encode(item["lm"]["target"] + "<|im_end|>", add_special_tokens=False)
        if len(head) + len(tail) <= max_len:
            out.append({"ids": head + tail, "start": len(head), "source": item["source"], "id": item.get("id")})
    return out


def lm_loss(dm: JevK5, rows: list[dict]) -> torch.Tensor:
    """Mean token cross-entropy of each row's target, read from the hidden states at the target positions
    only, so the prompt never costs full-vocabulary logits."""
    lengths = [len(r["ids"]) for r in rows]
    ids = torch.zeros((len(rows), max(lengths)), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = torch.tensor(r["ids"])
    ids = ids.to(dm.device)
    hidden = dm.model.model(input_ids=ids, use_cache=False).last_hidden_state
    losses = [
        F.cross_entropy(dm.model.lm_head(hidden[i, r["start"] - 1 : n - 1]).float(), ids[i, r["start"] : n])
        for i, (r, n) in enumerate(zip(rows, lengths, strict=True))
    ]
    return torch.stack(losses).mean()


def batch_logits(dm: JevK5, rows: list[dict]) -> torch.Tensor:
    lengths = torch.tensor([len(r["ids"]) for r in rows], device=dm.device)
    ids = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r["ids"])] = torch.tensor(r["ids"])
    return dm._slot_logits(ids.to(dm.device), lengths - 1).float()


def loss_of(logits: torch.Tensor, rows: list[dict]) -> torch.Tensor:
    k = max(len(r["target"]) for r in rows)
    target = torch.zeros((len(rows), k), device=logits.device)
    for i, r in enumerate(rows):
        target[i, : len(r["target"])] = torch.from_numpy(r["target"])
    mask = (
        torch.arange(k, device=logits.device)[None]
        < torch.tensor([len(r["target"]) for r in rows], device=logits.device)[:, None]
    )
    logp = F.log_softmax(logits[:, :k].masked_fill(~mask, -1e4), -1)
    return -(target * logp).sum(-1).mean()


@torch.no_grad()
def evaluate(dm: JevK5, data: list[dict], token_budget: int = 32768) -> dict:
    dm.model.eval()
    probs: list[np.ndarray] = [np.zeros(0)] * len(data)
    for rows in plan_batches(
        [len(r["ids"]) for r in data],
        token_budget=token_budget,
        max_batch=64,
        bucket=16,
        max_length=10**6,
    ):
        logits = batch_logits(dm, [data[i] for i in rows]).cpu().numpy()
        for j, i in enumerate(rows):
            z = logits[j, : len(data[i]["target"])]
            p = np.exp(z - z.max())
            probs[i] = p / p.sum()
    by: dict[str, list] = {}
    for r, p in zip(data, probs, strict=True):
        by.setdefault(r["source"], []).append((r["target"], p))
    report = {}
    for source, pairs in [*sorted(by.items()), ("ALL", [x for v in by.values() for x in v])]:
        acc = np.mean([t.argmax() == p.argmax() for t, p in pairs])
        nll = -np.mean([np.log(max(float(p[t.argmax()]), 1e-9)) for t, p in pairs])
        conf = np.array([p.max() for _, p in pairs])
        hit = np.array([t.argmax() == p.argmax() for t, p in pairs], dtype=float)
        bins = np.minimum((conf * 10).astype(int), 9)
        ece = sum(
            abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
            for b in range(10)
            if (bins == b).any()
        )
        report[source] = {
            "n": len(pairs),
            "acc": round(float(acc), 4),
            "nll": round(float(nll), 4),
            "ece": round(float(ece), 4),
        }
    dm.model.train()
    return {"report": report, "probs": probs}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--base", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-before", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lm-weight", type=float, default=0.5, help="weight of the form-writing rows' loss")
    parser.add_argument("--steps", type=int, help="train exactly this many steps (overrides --epochs)")
    args = parser.parse_args(argv)
    from peft import LoraConfig, get_peft_model

    if not fla_backward_ok():
        # fla refuses its backward pass on Hopper with Triton < 3.7.1 (wrong gradients, fla #640);
        # hiding it makes transformers use its PyTorch reference kernels: exact, but slower.
        sys.modules["fla"] = None  # type: ignore[assignment]

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dm = JevK5(args.base, device=args.device, graphs=False, temperature=1.0)  # SemIf prompt, as at inference
    load = lambda name: [json.loads(x) for x in (args.data / name).open()]  # noqa: E731
    items = load("train.jsonl")[: args.limit]
    train = encode_items(dm, [i for i in items if "lm" not in i], args.max_len)
    train_lm = encode_lm(dm, [i for i in items if "lm" in i], args.max_len)
    dev = encode_items(dm, load("dev.jsonl"), args.max_len)
    print(f"train {len(train)} decisions, {len(train_lm)} texts; dev {len(dev)} decisions", flush=True)
    log: dict = {"args": {k: str(v) for k, v in vars(args).items()}, "steps": []}
    if args.eval_before:
        log["before"] = evaluate(dm, dev)["report"]
        print("before", json.dumps(log["before"]["ALL"]), flush=True)

    names = {n.rsplit(".", 1)[-1] for n, _ in dm.model.named_modules()}
    config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=[t for t in TARGETS if t in names],
    )
    model = get_peft_model(dm.model, config)
    model.print_trainable_parameters()
    dm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dm.model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    def epoch_plan() -> list[tuple[str, list[int]]]:
        """One shuffled pass over all rows as (kind, row indices); every batch holds one kind of row."""
        plan = []
        for kind, data in (("decision", train), ("text", train_lm)):
            if not data:
                continue
            lengths = np.array([len(r["ids"]) for r in data])
            batches = plan_batches(
                (lengths + rng.integers(0, 16, len(lengths))).tolist(),
                token_budget=args.token_budget,
                max_batch=64,
                bucket=16,
                max_length=10**6,
            )
            plan += [(kind, rows) for rows in batches]
        rng.shuffle(plan)
        return plan

    plans = epoch_plan()
    per_epoch = len(plans)
    # --steps fixes the budget (the answer-only control matches the Tractatus LoRA's step count)
    target = args.steps or per_epoch * math.ceil(args.epochs)
    while len(plans) < target:
        plans += epoch_plan()
    total = args.steps or int(per_epoch * args.epochs)
    log["total_steps"] = total
    schedule = lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * s / total))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    started, window = time.perf_counter(), []
    for step, (kind, rows) in enumerate(plans[:total]):
        with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16):
            if kind == "decision":
                batch = [train[i] for i in rows]
                loss = loss_of(batch_logits(dm, batch), batch)
            else:
                loss = args.lm_weight * lm_loss(dm, [train_lm[i] for i in rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        window.append((kind, loss.item()))
        if (step + 1) % 25 == 0:
            eta = (total - step - 1) * (time.perf_counter() - started) / (step + 1) / 60
            losses = {k: float(np.mean([v for w, v in window if w == k])) for k in {w for w, _ in window}}
            log["steps"].append({"step": step + 1, **{f"loss_{k}": v for k, v in losses.items()}})
            shown = ", ".join(f"{k} {v:.4f}" for k, v in sorted(losses.items()))
            print(f"step {step + 1}/{total} loss {shown} ~{eta:.0f} min left", flush=True)
            window = []
    dm.model.gradient_checkpointing_disable()
    merged = model.merge_and_unload()
    dm.model = merged
    dm.slot_weight = merged.lm_head.weight[dm.slots].detach().contiguous()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out)
    dm.tok.save_pretrained(args.out)
    log["after"] = evaluate(dm, dev)["report"]
    (args.out / "train_log.json").write_text(json.dumps(log, indent=1))
    print("after", json.dumps(log["after"]["ALL"]), flush=True)
    before = log.get("before", {})
    for source, r in log["after"].items():
        b = before.get(source)
        was = f"acc {b['acc']:.3f} ece {b['ece']:.3f} -> " if b else ""
        print(f"  {source:<42} {was}acc {r['acc']:.3f} ece {r['ece']:.3f}  (n={r['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
