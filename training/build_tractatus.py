"""Training rows for the Tractatus-supervised LoRA, or for its answer-only control, from tractatus_data.py.

    python build_tractatus.py ../data/tractatus/train.jsonl [more.jsonl ...] ../data/lora-tractatus --variant tractatus
    python build_tractatus.py ../data/tractatus/train.jsonl ../data/lora-answers --variant answers

Writes <out>/train.jsonl and <out>/dev.jsonl in lora.py's format. Every kept teacher question gives its
answer row (source "teacher"). With --variant tractatus, a question with a checked form also gives one
yes/no row per atom, labelled with the teacher's verdict (source "atom"), and one row that writes the form
(source "form": jevk5.decompose's prompt as input, the form's JSON without verdicts as the target, the way
the model writes it in explicit mode). --dev-share of the documents, chosen by a hash of their id, goes to
dev.jsonl as answer rows only, for calibration and model selection. --replay appends rows from
build_train.py's train.jsonl as they are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jevk5.decompose import Decomposer

VARIANTS = ("tractatus", "answers")


def held_out(doc: str, share: float) -> bool:
    return int(hashlib.sha256(doc.encode()).hexdigest()[:8], 16) / 16**8 < share


def rows(record: dict, variant: str) -> list[dict]:
    """The rows one kept teacher question contributes."""
    out = [{"id": record["id"], "state": record["state"], "question": record["question"],
            "expected": str(record["expected"]), "source": record.get("source", "teacher")}]
    form = record.get("tractatus")
    if variant == "answers" or not form:
        return out
    for i, (atom, truth) in enumerate(zip(form["atoms"], form["truth"], strict=True)):
        out.append({"id": f"{record['id']}-a{i + 1}", "state": record["state"],
                    "question": {"type": "noul", "instructions": atom},
                    "expected": "true" if truth else "false", "source": "atom"})
    target = {k: v for k, v in form.items() if k != "truth"}
    out.append({"id": f"{record['id']}-form", "source": "form",
                "lm": {"messages": Decomposer.messages(record["state"], record["question"]),
                       "target": json.dumps(target, ensure_ascii=False)}})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", type=Path, nargs="+", help="tractatus_data.py outputs")
    parser.add_argument("out", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="tractatus")
    parser.add_argument("--dev-share", type=float, default=0.1)
    parser.add_argument("--replay", type=Path, help="build_train.py's train.jsonl")
    args = parser.parse_args(argv)

    train, dev = [], []
    for record in (json.loads(line) for path in args.sources for line in path.open()):
        if held_out(record.get("doc", record["id"]), args.dev_share):
            dev += rows(record, "answers")
        else:
            train += rows(record, args.variant)
    if args.replay:
        train += [json.loads(line) for line in args.replay.open()]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in (("train", train), ("dev", dev)):
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))
    kinds = {k: sum(r["source"] == k for r in train) for k in ("teacher", "atom", "form")}
    print(f"train {len(train)} rows {kinds}, dev {len(dev)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
