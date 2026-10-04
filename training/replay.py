"""A small public replay for the Tractatus LoRA: train splits of openly licensed multiple-choice and yes/no
datasets as JevK5 decision rows, so easy decisions don't regress.

    python replay.py ../data/tractatus/replay.jsonl --rows 1000 --jevbench ../data/jevbench

v0.3's own replay (30,052 rows from 26 train splits, see MODEL_CARD.md) was built outside this repo. This
uses six of the same train splits, the ones whose shape needs no dataset-specific logic beyond field names.
A row sharing an 8-word sequence with any of JevBench's public items is dropped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

LETTERS = "abcdefgh"
ANSWER = "Which option answers the question correctly?"
# name: (hub id, config, license, record -> (state, criterion, options, answer index) or None)
SOURCES = {
    "boolq": ("google/boolq", None, "CC BY-SA 3.0",
              lambda r: (r["passage"], r["question"].rstrip("?") + "?", None, int(r["answer"]))),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "CC BY-SA 4.0",
            lambda r: (r["question"], ANSWER, r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))),
    "commonsense_qa": ("tau/commonsense_qa", None, "MIT",
                       lambda r: (r["question"], ANSWER, r["choices"]["text"],
                                  r["choices"]["label"].index(r["answerKey"]))),
    "openbookqa": ("allenai/openbookqa", "main", "Apache-2.0",
                   lambda r: (r["question_stem"], ANSWER, r["choices"]["text"],
                              r["choices"]["label"].index(r["answerKey"]))),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "Apache-2.0",
                   lambda r: (r["sentence"], "Which option fills the blank (_) correctly?",
                              [r["option1"], r["option2"]], int(r["answer"]) - 1)),
    "hellaswag": ("Rowan/hellaswag", None, "MIT",
                  lambda r: (r["ctx"], "Which ending continues the text most plausibly?", r["endings"],
                             int(r["label"]))),
}


def shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def row(source: str, i: int, state: str, criterion: str, options: list[str] | None, answer: int) -> dict:
    if options is None:  # a yes/no question; `answer` is 1 for yes
        question = {"type": "noul", "instructions": criterion}
        expected = "true" if answer else "false"
    else:
        question = {"type": "choice", "instructions": criterion,
                    "criteria": {LETTERS[k]: text for k, text in enumerate(options)}}
        expected = LETTERS[answer]
    return {"id": f"replay-{source}-{i}", "state": state, "question": question, "expected": expected,
            "source": f"replay-{source}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--rows", type=int, default=1000, help="per source")
    parser.add_argument("--jevbench", type=Path, default=Path("../data/jevbench"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from datasets import load_dataset

    public = set()
    for path in sorted((args.jevbench / "datasets" / "public").glob("*.jsonl")):
        for item in map(json.loads, path.open()):
            public |= shingles(json.dumps(item["state"]) + " " + item["question"]["instructions"])
    rng, rows = random.Random(args.seed), []
    for name, (hub, config, license_, parse) in SOURCES.items():
        try:
            data = load_dataset(hub, config, split="train")
        except Exception as error:  # noqa: BLE001 - a source that no longer loads is left out, loudly
            print(f"{name}: skipped ({type(error).__name__}: {str(error)[:120]})")
            continue
        kept, dropped = [], 0
        for i in rng.sample(range(len(data)), min(len(data), args.rows * 2)):
            state, criterion, options, answer = parse(data[i])
            if public & shingles(state + " " + criterion):
                dropped += 1
                continue
            kept.append(row(name, i, state, criterion, options, answer))
            if len(kept) == args.rows:
                break
        rows += kept
        print(f"{name} ({license_}): {len(kept)} rows, {dropped} dropped for overlap with JevBench")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
