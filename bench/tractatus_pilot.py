"""Pilot: the 4B analyses its own uncertain answers, all local (llama.cpp), on JevBench's public items.

    python bench/tractatus_pilot.py --out results/tractatus-self     # freeze PREREGISTRATION.md there first

Every item gets the one pass. Items whose one-pass top probability is below --gate also get, from the same 4B:
  analysis   it writes atoms + a logical form (thinking on; jevk5.decompose's prompt)
  compose    it judges each atom in its own pass; the form's truth table gives the composed answer
  checklist  it re-reads the question with the atoms appended
  think      control: it thinks with JevK5's own prompt; its answer letters are read after the thinking
Rules, all scored from the same runs; anything missing falls back to the one pass:
  agree            switch to the checklist only if the composed answer agrees (primary)
  agree_decisive   agree, and the composed answer's probability >= --decisive
  decisive         the composed answer if its probability >= --decisive, else the checklist
  checklist        the checklist whenever there is a form (control)
  think            the thinking control
Rows are appended as items finish, so a rerun resumes; summary.json at the end. Scoring is JevBench's own.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from tractatus_eval import TIERS, labelled, load, mcnemar

from jevk5.decompose import Decomposer, reply
from jevk5.gguf import JevK5GGUF
from jevk5.prompt import LETTERS, decision_options, messages, prompt_text
from jevk5.tractatus import checklist, compose

RULES = ("one", "agree", "agree_decisive", "decisive", "checklist", "think")
EMPTY_THINK = "<think>\n\n</think>\n\n"


def top(probs: dict[str, float]) -> str:
    return max(sorted(probs), key=probs.get)


def think(model: JevK5GGUF, url: str, state, question: dict, options: dict) -> dict[str, float] | None:
    """The one pass's own prompt, read after the 4B has thought about it (None past 16 options)."""
    texts = [text for _, text in decision_options(question)]
    if len(texts) > len(LETTERS):
        return None
    message = reply(url, "local", "", messages(state, question["instructions"], texts), 900.0,
                    chat_template_kwargs={"enable_thinking": True}, **options)
    thought = (message.get("reasoning_content") or "").strip()
    if not thought:
        return None
    prompt = prompt_text(state, question["instructions"], texts)
    probs, _ = model.letters(prompt.replace(EMPTY_THINK, f"<think>\n{thought}\n</think>\n\n", 1), len(texts))
    return dict(zip((key for key, _ in decision_options(question)), probs))


def analyse(model: JevK5GGUF, decomposer: Decomposer, url: str, options: dict, task) -> dict:
    """Everything the rules need for one uncertain item; failures are recorded, not raised."""
    out: dict = {}
    t0 = time.perf_counter()
    try:
        form = decomposer(task.state, task.question)
    except Exception as error:  # noqa: BLE001 - a failed analysis falls back to the one pass
        form, out["error"] = None, repr(error)
    out["analysis_s"] = round(time.perf_counter() - t0, 1)
    if form:
        p = [model.probabilities(task.state, {"type": "noul", "instructions": a})[0]["true"]
             for a in form["atoms"]]
        out.update(form=form, atoms=p,
                   compose=labelled(task.question, compose(form, task.question, p)),
                   checklist=labelled(task.question, model.probabilities(task.state, checklist(task.question, form))[0]))
    t1 = time.perf_counter()
    try:
        probs = think(model, url, task.state, task.question, options)
        out["think"] = probs and labelled(task.question, probs)
    except Exception as error:  # noqa: BLE001
        out["think_error"] = repr(error)
    out["think_s"] = round(time.perf_counter() - t1, 1)
    return out


def decide(row: dict, rule: str, decisive: float) -> dict[str, float]:
    one = row["one"]
    if rule == "one" or not row["gated"]:
        return one
    if rule == "think":
        return row.get("think") or one
    comp, chk = row.get("compose"), row.get("checklist")
    if comp is None or chk is None:
        return one
    agree, sure = top(comp) == top(chk), max(comp.values()) >= decisive
    return {"agree": chk if agree else one, "agree_decisive": chk if agree and sure else one,
            "decisive": comp if sure else chk, "checklist": chk}[rule]


def summarize(rows: list[dict], tasks: dict, rule: str, decisive: float, metrics, score_task) -> dict:
    scored = []
    for row in rows:
        probs = decide(row, rule, decisive)
        task = tasks[row["task_id"]]
        p_true = probs.get(str(task.expected), 0.0)
        scored.append({**row, "probs": probs, "correct": bool(score_task(probs, task)["correct"]),
                       "switched": top(probs) != top(row["one"]),
                       "brier": metrics.brier_score(probs, str(task.expected), task.labels),
                       "logloss": -math.log(max(p_true, 1e-12))})
    base = {r["task_id"]: bool(score_task(r["one"], tasks[r["task_id"]])["correct"]) for r in rows}

    def acc(items):
        return round(sum(r["correct"] for r in items) / len(items), 3) if items else None

    fixed = sum(r["correct"] and not base[r["task_id"]] for r in scored)
    broken = sum(base[r["task_id"]] and not r["correct"] for r in scored)
    gated = [r for r in scored if r["gated"]]
    return {
        "accuracy": acc(scored), **{tier: acc([r for r in scored if r["tier"] == tier]) for tier in TIERS},
        "families": {f: acc([r for r in scored if r["family"] == f]) for f in sorted({r["family"] for r in scored})},
        "fixed": fixed, "broken": broken, "mcnemar_p": round(mcnemar(fixed, broken), 4),
        "ece": round(metrics.ece_top_label([(max(r["probs"].values()), r["correct"]) for r in scored])["ece"], 3),
        "ece_hard": round(metrics.ece_top_label(
            [(max(r["probs"].values()), r["correct"]) for r in scored if r["tier"] == "hard"])["ece"], 3),
        "brier": round(sum(r["brier"] for r in scored) / len(scored), 4),
        "logloss": round(sum(r["logloss"] for r in scored) / len(scored), 4),
        "gated": len(gated), "switched": sum(r["switched"] for r in scored),
        "kept_one_pass_when_gated": sum(not r["switched"] for r in gated),
        "confident_mistakes_untouched": sum(not r["correct"] for r in scored if not r["gated"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jevbench", default="data/jevbench")
    parser.add_argument("--out", default="results/tractatus-self")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--gate", type=float, default=0.8)
    parser.add_argument("--decisive", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    tasks, tier_of, metrics, score_task = load(args.jevbench, TIERS, args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    options = {"max_tokens": args.max_tokens, "temperature": 0.6, "seed": 1}
    model = JevK5GGUF(args.url, temperature=1.22, knockout_temperature=0.93)  # v0.3 4B GGUF values
    decomposer = Decomposer(url=args.url + "/v1", model="local", key_env="NO_KEY", effort="medium",
                            cache=str(out / "llm-cache.jsonl"), options=options)
    path = out / "rows.jsonl"
    done = {json.loads(line)["task_id"] for line in path.read_text().splitlines()} if path.exists() else set()
    started = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        if task.id in done:
            continue
        one, _ = model.probabilities(task.state, task.question)
        row = {"task_id": task.id, "tier": tier_of[task.id], "family": task.family,
               "type": task.question["type"], "one": labelled(task.question, one)}
        row["gated"] = max(one.values()) < args.gate
        if row["gated"]:
            row.update(analyse(model, decomposer, args.url + "/v1", options, task))
        with path.open("a") as rows:
            rows.write(json.dumps(row) + "\n")
        print(f"\r{i}/{len(tasks)} items, {time.perf_counter() - started:.0f} s", end="", flush=True)
    print()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_id = {t.id: t for t in tasks}
    summary = {rule: summarize(rows, by_id, rule, args.decisive, metrics, score_task) for rule in RULES}
    summary["settings"] = vars(args) | {"options": options, "rejected_forms": decomposer.rejected}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({r: {k: summary[r][k] for k in ("accuracy", "hard", "fixed", "broken", "mcnemar_p", "ece")}
                      for r in RULES}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
