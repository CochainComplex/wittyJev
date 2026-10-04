"""JevBench's public items as gold-labelled records for training/tractatus_data.py (comparison only, never
training): the teacher's forms and verdicts are then checked against JevBench's own answers.

    python bench/jevbench_records.py data/jevbench data/tractatus/jevbench.jsonl
"""

from __future__ import annotations

import json
import sys

from tractatus_eval import TIERS, load


def record(task, tier: str) -> dict:
    expected = {"yes": "true", "no": "false"}.get(str(task.expected), str(task.expected))
    return {"id": task.id, "state": task.state, "question": task.question, "expected": expected,
            "source": f"jevbench-{tier}", "family": task.family}


def main(argv: list[str]) -> int:
    tasks, tier_of, _, _ = load(argv[0], TIERS, None)
    with open(argv[1], "w") as out:
        for task in tasks:
            out.write(json.dumps(record(task, tier_of[task.id]), ensure_ascii=False) + "\n")
    print(f"{len(tasks)} records -> {argv[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
