"""Tractatus supervision for teacher questions: the teacher's logical form of each kept question, with a
verdict per atom, kept only when the verdicts compose to the question's agreed answer.

    LLM_URL=... LLM_MODEL=... python tractatus_data.py ../data/teacher/test.jsonl ../data/tractatus/test.jsonl \
        --crosscheck <a second model on the same server>

Reads teacher.py's JSONL (records with state, question, expected, agree), or gold-labelled records without
"agree" (replay rows, benchmark items). For every agreed question the
teacher (jevk5.decompose on an OpenAI-compatible server: --url, --model or $LLM_URL, $LLM_MODEL; a key, if
needed, in $LLM_API_KEY) writes atoms, the logical form and a verdict per atom. The form must pass
jevk5.tractatus.validate, and its verdicts must compose to `expected` (TLP 4.1212: the form shows the
answer); otherwise the question keeps "tractatus": null and trains on its answer alone. `--crosscheck MODEL`
asks a second model to answer too and records whether it agrees: independent labels for a test split.
Output is resumable: records already written are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from jevk5.decompose import INSTRUCTIONS, Decomposer, parse
from jevk5.tractatus import compose, validate

TEACHER_SYSTEM = (
    "You analyse typed decision questions into elementary propositions, after Wittgenstein's Tractatus: "
    "the answer is a truth-function of simple facts about the case. Judge each fact from the evidence."
)
TEACHER = INSTRUCTIONS.replace(
    "Never include the answer, your reasoning, or which atoms are true.",
    "Never include the answer or your reasoning.",
) + '\nAlso add "truth": a list of true or false, one per atom in order, as the evidence settles each atom.'


def tractatus(teacher: Decomposer, record: dict) -> dict | None:
    """The teacher's form with its verdicts, if the verdicts compose to the agreed answer."""
    question = record["question"]
    form = parse(teacher.ask(Decomposer.messages(record["state"], question, TEACHER_SYSTEM, TEACHER)))
    if not form or form.get("atomic"):
        return None
    truth = form.pop("truth", None)
    try:
        validate(form, question)
    except ValueError:
        return None
    if not (isinstance(truth, list) and len(truth) == len(form["atoms"])
            and all(isinstance(t, bool) for t in truth)):
        return None
    shown = compose(form, question, [float(t) for t in truth])
    return {**form, "truth": truth} if max(shown, key=shown.get) == str(record["expected"]) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="teacher.py output")
    parser.add_argument("out", type=Path)
    parser.add_argument("--url", default=os.environ.get("LLM_URL"), required="LLM_URL" not in os.environ)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"), required="LLM_MODEL" not in os.environ)
    parser.add_argument("--effort", default="xhigh", help="the teacher's reasoning effort")
    parser.add_argument("--max-tokens", type=int, default=32000, help="per request, thinking included")
    parser.add_argument("--crosscheck", help="a second model that answers each question")
    parser.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args(argv)

    # teacher questions need both solves to agree; gold-labelled records (replay, benchmarks) carry no "agree"
    records = [r for r in map(json.loads, args.source.open()) if r.get("agree", True) and "error" not in r]
    done = {json.loads(line)["id"] for line in args.out.open()} if args.out.exists() else set()
    todo = [r for r in records if r["id"] not in done]
    cache = args.out.with_suffix(".llm-cache.jsonl")
    teacher = Decomposer(url=args.url, model=args.model, effort=args.effort, cache=str(cache),
                         options={"max_tokens": args.max_tokens})
    checker = (Decomposer(url=args.url, model=args.crosscheck, effort="high", cache=str(cache), template=False)
               if args.crosscheck else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock, counts = threading.Lock(), {"records": 0, "forms": 0, "crosschecked": 0}

    def run(record: dict) -> dict:
        out = {**record, "tractatus": tractatus(teacher, record)}
        if checker:
            answer = checker.answer(record["state"], record["question"])
            out["crosscheck"] = {"model": args.crosscheck, "answer": answer,
                                 "agree": answer == str(record["expected"])}
        return out

    with ThreadPoolExecutor(args.parallel) as pool:
        for future in as_completed([pool.submit(run, r) for r in todo]):
            try:
                out = future.result()
            except Exception as error:  # noqa: BLE001 - a failed request leaves the record for a rerun
                print(f"\nskipped: {error!r}", flush=True)
                continue
            with lock, args.out.open("a") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                counts["records"] += 1
                counts["forms"] += out["tractatus"] is not None
                counts["crosschecked"] += bool(out.get("crosscheck", {}).get("agree"))
                print(f"\r{counts} of {len(todo)}", end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
