"""The acceptance test of docs/tractatus-lora.md: v0.3, the answer-only control and the Tractatus LoRA, paired.

    python bench/acceptance.py run --name v03 --url http://127.0.0.1:8080 \
        --test data/tractatus/test.jsonl --dev data/lora-tractatus/dev.jsonl --out results/acceptance
    python bench/acceptance.py compare --out results/acceptance --baseline v03 --control answers --candidate tractatus

`run` reads every test question once through one model's llama-server (one pass, JevK5GGUF), after fitting
that model's temperature on the calibration hold-out (dev), never on the test split. Only test questions
whose label an independent model confirmed are used. `compare` applies the five criteria: overall gain over
v0.3 (exact McNemar p < 0.05), no family with a Holm-corrected significant loss, NLL and ECE within 0.01,
a gain over the answer-only control, and JevBench's public tiers (optional, from bench/tractatus_eval.py).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tractatus_eval import mcnemar

from jevk5.gguf import JevK5GGUF

TEMPERATURES = [round(0.5 + 0.05 * i, 2) for i in range(51)]  # 0.5 .. 3.0


def logits(model: JevK5GGUF, record: dict) -> dict[str, float]:
    """Log-probabilities per option at temperature 1, so temperatures can be refitted without re-reading."""
    probs, _ = model.probabilities(record["state"], record["question"])
    return {k: math.log(max(v, 1e-12)) for k, v in probs.items()}


def at(z: dict[str, float], temperature: float) -> dict[str, float]:
    top = max(z.values())
    w = {k: math.exp((v - top) / temperature) for k, v in z.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def nll(rows: list[dict], temperature: float) -> float:
    return -sum(math.log(max(at(r["z"], temperature)[r["expected"]], 1e-12)) for r in rows) / len(rows)


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        cell = [(c, ok) for c, ok in pairs if min(int(c * bins), bins - 1) == b]
        if cell:
            total += len(cell) / len(pairs) * abs(sum(ok for _, ok in cell) / len(cell) - sum(c for c, _ in cell) / len(cell))
    return total


def run(args) -> int:
    model = JevK5GGUF(args.url, temperature=1.0, top_k=args.top_k)
    read = lambda path, keep: [  # noqa: E731
        {"id": r["id"], "family": r.get("spec", {}).get("family", r.get("source", "")),
         "expected": str(r["expected"]), "z": logits(model, r)}
        for r in map(json.loads, open(path)) if keep(r)
    ]
    dev = read(args.dev, lambda r: "lm" not in r and r.get("source", "teacher") == "teacher")
    temperature = min(TEMPERATURES, key=lambda t: nll(dev, t))
    test = read(args.test, lambda r: r.get("crosscheck", {}).get("agree", True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"temperature": temperature, "dev_n": len(dev),
                                                       "test": test}) + "\n")
    print(f"{args.name}: temperature {temperature} (dev n={len(dev)}), {len(test)} test questions")
    return 0


def scored(path: Path) -> dict:
    data = json.loads(path.read_text())
    rows = {}
    for r in data["test"]:
        p = at(r["z"], data["temperature"])
        rows[r["id"]] = {"family": r["family"], "correct": max(p, key=p.get) == r["expected"],
                         "conf": max(p.values()), "nll": -math.log(max(p[r["expected"]], 1e-12))}
    return rows


def compare(args) -> int:
    out = Path(args.out)
    base, ctrl, cand = (scored(out / f"{n}.json") for n in (args.baseline, args.control, args.candidate))
    ids = sorted(set(base) & set(ctrl) & set(cand))

    def pair(a: dict, b: dict, keys) -> tuple[int, int, float]:
        fixed = sum(b[i]["correct"] and not a[i]["correct"] for i in keys)
        broken = sum(a[i]["correct"] and not b[i]["correct"] for i in keys)
        return fixed, broken, mcnemar(fixed, broken)

    def summary(rows: dict, keys) -> dict:
        return {"acc": sum(rows[i]["correct"] for i in keys) / len(keys),
                "nll": sum(rows[i]["nll"] for i in keys) / len(keys),
                "ece": ece([(rows[i]["conf"], rows[i]["correct"]) for i in keys])}

    report = {n: summary(r, ids) for n, r in ((args.baseline, base), (args.control, ctrl), (args.candidate, cand))}
    fixed, broken, p = pair(base, cand, ids)
    families = sorted({base[i]["family"] for i in ids})
    losses = []  # Holm over the families' one-sided losses of the candidate against the baseline
    for f in families:
        keys = [i for i in ids if base[i]["family"] == f]
        f_fixed, f_broken, f_p = pair(base, cand, keys)
        losses.append((f_p if f_broken > f_fixed else 1.0, f, f_fixed, f_broken, len(keys)))
    losses.sort()
    holm_loss = [f for k, (p_f, f, *_rest) in enumerate(losses) if p_f < 0.05 / (len(losses) - k)]
    c_fixed, c_broken, c_p = pair(ctrl, cand, ids)
    b, c = report[args.baseline], report[args.candidate]
    criteria = {
        "1 overall gain over v0.3": fixed > broken and p < 0.05,
        "2 no family with a significant loss": not holm_loss,
        "3 NLL and ECE within 0.01": c["nll"] <= b["nll"] + 0.01 and c["ece"] <= b["ece"] + 0.01,
        "4 gain over the answer-only control": c_fixed > c_broken and c_p < 0.05,
    }
    result = {"n": len(ids), "report": report, "vs_baseline": {"fixed": fixed, "broken": broken, "p": p},
              "vs_control": {"fixed": c_fixed, "broken": c_broken, "p": c_p},
              "families": {f: {"fixed": fx, "broken": bk, "n": n, "p_loss": p_f} for p_f, f, fx, bk, n in losses},
              "holm_losses": holm_loss, "criteria": criteria, "accepted": all(criteria.values())}
    (out / "compare.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("n", "report", "vs_baseline", "vs_control", "criteria", "accepted")},
                     indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--url", default="http://127.0.0.1:8080")
    r.add_argument("--test", required=True)
    r.add_argument("--dev", required=True)
    r.add_argument("--out", default="results/acceptance")
    r.add_argument("--top-k", type=int, default=40)
    c = sub.add_parser("compare")
    c.add_argument("--out", default="results/acceptance")
    c.add_argument("--baseline", default="v03")
    c.add_argument("--control", default="answers")
    c.add_argument("--candidate", default="tractatus")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
