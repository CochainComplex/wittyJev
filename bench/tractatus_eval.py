"""JevK5 through a logical form (jevk5.tractatus) on JevBench's public items, against its single pass.

    export LLM_URL=... LLM_MODEL=... LLM_API_KEY=...   # the LLM that writes the forms and answers "direct"
    docker run -d --gpus all -p 127.0.0.1:8080:8080 -v "$PWD/models:/models:ro" \
        ghcr.io/ggml-org/llama.cpp:server-cuda -m /models/jevk5-4b-v0.3-Q8_0.gguf -c 16384 -ngl 99 \
        --host 0.0.0.0 --port 8080
    python bench/tractatus_eval.py --jevbench data/jevbench --out results/tractatus

Systems, all on the same items:
  holistic   the unchanged 4B, one pass (the baseline)
  checklist  the 4B, one pass, the form's atoms appended to the question
  compose    the 4B judges every atom; the answer is computed from the form
  direct     the LLM that writes the forms, answering alone (one-hot; accuracy only)
Checklist and compose read the same cached form, so they differ only in how it is used. The LLM sees
the state and the question, never a task's provenance or answer. Scoring is JevBench's own
(jevbench.scoring.score_task). Writes one JSONL per system and summary.json, and prints the summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from jevk5.decompose import Decomposer
from jevk5.gguf import JevK5GGUF
from jevk5.prompt import decision_options
from jevk5.tractatus import Tractatus

TIERS = ("easy", "original", "hard")
SYSTEMS = ("holistic", "checklist", "compose", "direct")


def labelled(question: dict, probs: dict[str, float]) -> dict[str, float]:
    """Option-id probabilities under JevBench's labels (noul: yes/no), as bench/jevk5_direct.py maps them."""
    if question["type"] == "noul":
        return {"yes": probs["true"], "no": probs["false"]}
    return probs


def retried(call, attempts: int = 3):
    for attempt in range(attempts):
        try:
            return call()
        except Exception:  # noqa: BLE001 - network errors; the last one propagates
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))


def direct(decomposer: Decomposer, task) -> dict[str, float] | None:
    """The LLM's own answer as a one-hot distribution, or None if it names no option."""
    choice = retried(lambda: decomposer.answer(task.state, task.question))
    ids = [key for key, _ in decision_options(task.question)]
    return {key: float(key == choice) for key in ids} if choice else None


def mcnemar(fixed: int, broken: int) -> float:
    """Exact two-sided p of the discordant pairs."""
    n = fixed + broken
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(fixed, broken) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(rows: list[dict], base: list[dict], metrics) -> dict:
    by_id = {r["task_id"]: r for r in base}

    def acc(items):
        return round(sum(r["correct"] for r in items) / len(items), 3) if items else None

    hard = [r for r in rows if r["tier"] == "hard"]
    fixed = sum(r["correct"] and not by_id[r["task_id"]]["correct"] for r in rows)
    broken = sum(by_id[r["task_id"]]["correct"] and not r["correct"] for r in rows)
    return {
        "n": len(rows),
        "accuracy": acc(rows),
        **{tier: acc([r for r in rows if r["tier"] == tier]) for tier in TIERS},
        "hard_families": {f: acc([r for r in hard if r["family"] == f])
                          for f in sorted({r["family"] for r in hard})},
        "hard_ece": round(metrics.ece_top_label(
            [(max(r["probs"].values()), r["correct"]) for r in hard if r["probs"]])["ece"], 3),
        "families": {f: acc([r for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})},
        "ece": round(metrics.ece_top_label(
            [(max(r["probs"].values()), r["correct"]) for r in rows if r["probs"]])["ece"], 3),
        "brier": round(sum(r["brier"] for r in rows) / len(rows), 3),
        "with_form": sum(r.get("form") is not None for r in rows),
        "fixed_vs_holistic": fixed,
        "broken_vs_holistic": broken,
        "mcnemar_p": round(mcnemar(fixed, broken), 3),
    }


def load(jevbench: str, tiers, limit: int | None, records: str | None = None):
    """JevBench's metrics and scorer, and its public tasks with their tiers (the benchmark is a checkout,
    not a dependency). With `records`, gold- or teacher-labelled records (teacher.py, tractatus_data.py
    output) become the tasks instead, their split as the tier."""
    sys.path.insert(0, jevbench)
    from jevbench import metrics  # noqa: PLC0415
    from jevbench.scoring import score_task  # noqa: PLC0415
    from jevbench.tasks import Task  # noqa: PLC0415

    tasks, tier_of = [], {}
    for r in map(json.loads, open(records)) if records else []:
        if not r.get("agree", True) or "error" in r:
            continue
        q, expected = r["question"], str(r["expected"])
        noul = q["type"] == "noul"
        task = Task(id=r["id"], family=r.get("spec", {}).get("family") or r.get("family", ""), state=r["state"],
                    question=q, labels=["no", "yes"] if noul else [k for k, _ in decision_options(q)],
                    expected={"true": "yes", "false": "no"}[expected] if noul
                    else int(expected) if q["type"] == "score" else expected, split="public")
        tasks.append(task)
        tier_of[task.id] = r.get("split", "records")
    for tier in tiers if not records else []:
        path = Path(jevbench) / "datasets" / "public" / f"{tier}.jsonl"
        for line in path.read_text().splitlines()[:limit]:
            task = Task.from_dict(json.loads(line))
            tasks.append(task)
            tier_of[task.id] = tier
    return tasks, tier_of, metrics, score_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jevbench", default="data/jevbench")
    parser.add_argument("--out", default="results/tractatus")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--tiers", nargs="+", default=list(TIERS))
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--form-url", default=os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1"),
                        help="who writes the forms (OpenAI-compatible)")
    parser.add_argument("--form-model", default=os.environ.get("LLM_MODEL", "local"))
    parser.add_argument("--form-key-env", default="LLM_API_KEY")
    parser.add_argument("--form-effort", default="medium", help='reasoning effort, or "none"')
    parser.add_argument("--blind", action="store_true", help="forms from the question alone")
    parser.add_argument("--form-no-template", action="store_true", help="for models that reject chat_template_kwargs")
    parser.add_argument("--records", help="evaluate on these labelled records instead of JevBench's public items")
    args = parser.parse_args(argv)

    tasks, tier_of, metrics, score_task = load(args.jevbench, args.tiers, args.limit, args.records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = {"blind": args.blind, "effort": None if args.form_effort == "none" else args.form_effort,
              "template": not args.form_no_template}  # saved in summary.json: no endpoint, no model id
    decomposer = Decomposer(url=args.form_url, model=args.form_model, key_env=args.form_key_env,
                            cache=str(out / "llm-cache.jsonl"), **source)
    model = JevK5GGUF(args.url, temperature=1.22, knockout_temperature=0.93)  # v0.3 4B GGUF values
    current: dict = {}  # the form of the item being read; None (read whole) if decomposing failed

    def form_of(state, question):
        return current["form"]

    local = {
        "holistic": model,
        "checklist": Tractatus(model, form_of, "checklist"),
        "compose": Tractatus(model, form_of, "compose"),
    }
    local = {name: runtime for name, runtime in local.items() if name in args.systems}
    rows: dict[str, list[dict]] = {name: [] for name in args.systems}

    def row(task, probs, extra: dict) -> dict:
        scored = score_task(probs, task) if probs else {"correct": False, "predicted": None}
        return {
            "task_id": task.id, "tier": tier_of[task.id], "family": task.family,
            "type": task.question["type"], "expected": task.expected,
            "correct": bool(scored["correct"]), "predicted": scored["predicted"], "probs": probs,
            "brier": metrics.brier_score(probs, str(task.expected), task.labels) if probs else 2.0,
            **extra,
        }

    def decompose(task):
        return retried(lambda: decomposer(task.state, task.question))

    started = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool:
        needs_form = {"checklist", "compose"} & set(rows)
        forms = {pool.submit(decompose if needs_form else lambda t: None, t): t for t in tasks}  # gate GPU passes
        directs = {pool.submit(direct, decomposer, t): t for t in tasks} if "direct" in rows else {}
        for done, future in enumerate(as_completed(forms), 1):
            task = forms[future]
            error = future.exception()
            current["form"] = None if error else future.result()
            for name, runtime in local.items():
                t0 = time.perf_counter()
                probs, tokens = runtime.probabilities(task.state, task.question)
                extra = {"tokens": tokens, "seconds": round(time.perf_counter() - t0, 3)}
                if name != "holistic":
                    extra.update(form=runtime.last_form, atoms=runtime.last_atoms,
                                 form_error=repr(error) if error else None)
                rows[name].append(row(task, labelled(task.question, probs), extra))
            print(f"\r{done}/{len(tasks)} items, {time.perf_counter() - started:.0f} s", end="", flush=True)
        for future in as_completed(directs):
            task = directs[future]
            probs = None if future.exception() else future.result()
            rows["direct"].append(row(task, probs and labelled(task.question, probs), {}))
    print()

    for name, items in rows.items():
        items.sort(key=lambda r: r["task_id"])
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items))
    base = rows.get("holistic") or next(iter(rows.values()))
    summary = {name: summarize(items, base, metrics) for name, items in rows.items()}
    summary["form_source"] = source
    summary["rejected_forms"] = decomposer.rejected
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rejected_forms"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
