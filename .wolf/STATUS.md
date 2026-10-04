---
description: session handoff, regenerate with /handoff when a quest finishes
budget_tokens: 1000
---
# STATUS — wittyJev (JevK5 fork × Tractatus)

> Single source of truth for resuming work. Read this FIRST when starting a session.
> Last updated: 2026-09-29

---

## ✅ Done

- Cloned github.com/CochainComplex/wittyJev (fork of allebee/jevk5; HEAD f26426d, runtime 0.3.3).
- Submodule `tractatus-thinking/` → github.com/CochainComplex/tractatus-thinking @ bad95a9 (v4.0.8). Staged, not committed.
- `openwolf init` (claude + codex + opencode; pm2 missing, so no daemon). Untracked, not committed.
- Goal mapped (tractatus session de2bc832), 5 scoping questions answered, answers sorted (tractatus
  session 7978ad29; sessions expire after 24 h). Parked ideas written to `IDEAS.md`.
- End goal confirmed: improve the quality of answers overall. Principles: KISS, YAGNI, SOLID, DRY, focus.
- GPT-6 Astra cross-check (codex): plausible targeted gains, not automatic; add a checklist control
  (same form, one pass) and a 27B-direct control; arithmetic needs execution, not reworded atoms.
- Built: `jevk5/tractatus.py` (composer: truth-functions, decision lists, checklist mode, `Tractatus`
  runtime), `jevk5/decompose.py` (the teacher server forms, cached), `tests/test_tractatus.py` (16 tests; 80 with
  test_prompt pass), `bench/tractatus_eval.py` (4 systems on JevBench public items, JevBench scoring).
- llama.cpp in Docker (container `wittyjev-llama`, 4B Q8_0 on the 6 GB GPU); venv `.venv` (pytest).

---

## 🚀 Current quest: Tractatus-supervised LoRA (go given 2026-09-29; plan + frozen criteria: docs/tractatus-lora.md)

**Built and tested on a mock-up** (training/mock.py: tiny random Qwen3.5 with JevK5's real tokenizer):
- training/tractatus_data.py (teacher form + per-atom verdicts, kept only if they compose to the answer;
  Mistral cross-check for test labels), training/build_tractatus.py (answer / atom / form rows; 10% dev
  hold-out), training/lora.py (+ form-writing loss, per-kind loss log, --device, --steps for equal budget),
  training/tractatus_pipeline.sh (mock by default; real run = env vars), bench/acceptance.py (the 5 criteria).
- Checks: text loss == transformers' causal-LM loss (tests/test_lora_text.py); GGUF conversion parity
  0.0008 logprob; equal budget 120/120 steps; explicit mode falls back cleanly; acceptance harness runs.
- xhigh cross-check (35 train questions): solves vs author+Mistral reference 0.95 -> 1.00, +39% tokens.
- Running: xhigh data generation (data/gen.log): test split 450 docs (held-out domains, Mistral-checked),
  then train split 1100 docs; ~14 h at 8 parallel the teacher server requests.
- Next: test/train overlap scan; replay mix (build_train.py, on the GPU machine); H100 training via
  `BASE=alibiserikbay/JevK5 DATA=../data/tractatus/train.jsonl OUT=../data/lora DEVICE=cuda EPOCHS=1
  LR=3e-5 RANK=16 WARMUP=20 OUTTYPE=q8_0 bash training/tractatus_pipeline.sh`; then bench/acceptance.py.
- Replay: training/replay.py -> data/tractatus/replay.jsonl (6,000 rows, 6 public train splits, no JevBench
  overlap), labelled by the teacher -> replay-tractatus.jsonl (forms checked against human gold).
- JevBench comparison (never trained on): bench/jevbench_records.py -> tractatus_data.py ->
  data/tractatus/jevbench-labels.jsonl; share of items whose teacher analysis reproduces JevBench's answer. Train data: data/tractatus/train-all.jsonl once the running generation ends (scratchpad runner).
- Compare to Mistral (results/compare-mistral/): JevBench public 231 (gold): JevK5 one pass 0.883, JevK5+checklist
  0.931, Mistral Small 4 alone 0.879 (8 answers were option numbers, upper bound 0.913), Qwen3.8 27B 0.987.
  Test split 574 (teacher labels): JevK5 0.817, Mistral 0.859. JevK5 ahead on adversarial, ambiguous, rubric and
  long policies; Mistral ahead on judging answers, dates/numbers, routing, trap, multi-hop.
- Paused (quota): data generation (test 574 kept, train-s2 242 kept), replay labels, JevBench labels (20/231),
  Qwen checklist on the test split (91/574 forms cached). All resumable.
- 2026-10-02 new teacher candidate Qwen3.8 flash-next (results/compare-qwen-next/): alone 0.991 on JevBench
  public (27B 0.987; thinking off 0.896), but its atoms help the 4B less: checklist 0.900 vs 0.931 with 27B atoms
  (10/6 vs 15/4 fixed/broken), compose 0.840 vs 0.870. Efforts: low/medium/xhigh; separate per-model quota.
- Training runs elsewhere (not from this repo). Real-run settings for training/tractatus_pipeline.sh:
  BASE=alibiserikbay/JevK5 DATA="../data/tractatus/train-all.jsonl ../data/tractatus/replay-tractatus.jsonl"
  DEVICE=cuda EPOCHS=1 LR=3e-5 RANK=16 WARMUP=20 OUTTYPE=q8_0 (DEV_SHARE defaults to the plan's 0.1).

## 🗄️ Earlier phase (pilot on JevBench public items)

**Pilot result** (JevBench's 231 public items, GGUF 4B Q8_0 in Docker, forms by the teacher server Qwen3.8-27B;
`results/tractatus/`, rerun from cache with `bench/tractatus_eval.py`):

| system | overall | standard | hard | long_policy | judge_hard | temporal_numeric | multi_hop | hard ECE | Brier | fixed/broken vs holistic |
|---|---|---|---|---|---|---|---|---|---|---|
| holistic 4B | 0.883 | 0.944 | 0.793 | 0.737 | 0.647 | 0.467 | 0.833 | 0.063 | 0.182 | — |
| checklist | **0.931** | 1.000 | 0.856 | 0.842 | 0.882 | 0.667 | 0.722 | 0.084 | 0.115 | 15 / 4 (p = 0.019) |
| compose | 0.870 | 0.889 | 0.802 | 0.632 | 0.941 | 0.667 | 0.667 | 0.055 | 0.194 | 17 / 20 (p = 0.74) |
| 27B direct | 0.987 | 1.000 | 0.973 | 1.000 | 0.941 | 0.933 | 0.944 | one-hot | 0.026 | 25 / 1 |

- Holistic reproduces the published v0.3 numbers (hard 0.793 vs 0.784), so the runtime is sound.
- Checklist (one pass, atoms appended) is the only significant gain for the 4B. Compose wins judge_hard and
  temporal_numeric but loses long_policy, multi_hop, standard: atoms that are themselves computations, atoms
  enumerating the options, long conjunctions that amplify one wrong atom.
- The 27B alone beats every 4B variant, so the gain cannot yet be credited to the structure rather than to
  the 27B's reading (caveat: public items may be in its training data; its answers carry no probabilities).

**Blind test** (`results/tractatus-blind/`, forms from the question alone): checklist 0.848 (2/10,
p = 0.039), compose 0.723 (12/49). Structure without reading the case hurts; the gain is case-grounded analysis.

**Universal design** (tractatus session 305a268c, cross-checked by GPT-6 Astra; awaiting a go):
- Principle that carries: "say clearly what can be said, stay silent otherwise" (TLP 4.116, 5.6, 7): the 4B's
  calibrated confidence is the gate. Post hoc: escalating conf < 0.9 to the 27B gives 0.983 on 39% of items, no
  family worse; a checklist stage before it ties (0.978, one item).
- Decomposition only where it is cheaper or more reliable than the strong reader: computed calculations, reusable
  rule templates, disagreement signals, audits, or inside the 4B via distillation (parked idea 2).
- Acceptance contract first (Astra): pre-registered families, per-family non-inferiority margins with simultaneous
  bounds, calibration tolerance, coverage, cost; fresh, independently labeled data split by template into
  calibration and test, plus a shifted domain; routing frozen before testing.
- Local-only proposal (no external model): confidence-gated self-analysis (4B writes its own atoms only when
  conf < 0.8). Proxy with 27B atoms at 0.8: replace 0.922 (11/2), blend 0.913 (9/2), switch-only-if-re-read>=0.8
  0.909 (6/0, zero breaks). Default policy proposed: gate + low-confidence flag on by default; analysis opt-in with
  a time budget and fallback to the one-pass answer, until it passes the acceptance contract.
- Decomposition integrated (tractatus session 73b74ede, all method mechanics mapped): on uncertain items,
  composition with decisive atom verdicts (most likely truth row >= 0.5) was right 19/19 vs one pass 13/19;
  checklist-only-if-composition-agrees: 0.926, 10 fixed, 0 broken. Design: say (one pass) -> analyse (4B writes
  atoms + form; reject bundled/name-only atoms, tautologies, contradictions) -> prove (decisive composition) ->
  refine (split an undecided pivotal atom once, untested) -> read (checklist, switch only if composition agrees or
  confident) -> stay silent (keep one pass, flag undecided, name the pivotal atom).
- GPT-6 Astra on a UNIVERSAL gain (4th cross-check): generation-based decomposition is multi-pass and gated, so
  it cannot fix confident mistakes. Universal route: keep JevK5 frozen, one pass; add small trainable
  "Tractatus heads" on its hidden states (form, evidence-span/relation, correction) that output
  softmax((z0 + g*dz)/T), gate g trained on correction benefit at ALL confidence levels (g = 0 is the exact
  fallback); train offline with teacher decompositions as auxiliary objectives plus answer loss (vs answer-only
  adapter); arithmetic via bounded extraction plus code. Acceptance: fresh items per family, powered paired
  test, per-family bounds (accuracy >= 0, NLL/Brier <= 0). This draws on parked ideas 1 and 2 (needs the go).
- Clarified 2026-09-29: "no external model" applies to the FINAL model at inference; teachers are allowed offline.
  Best route (tractatus session 54366dcd): a Tractatus-supervised LoRA continued from v0.3, trained on (1) answers,
  (2) elementary atom verdicts, (3) writing the form-typed decomposition; one-pass default, explicit local mode
  from the same weights; control = answer-only LoRA, same data and budget; replay v0.3's mix; acceptance on fresh
  per-family data. Needs: the teacher server for data (offline), an H100 for ~2x1-2 h of training.
- Pilot script ready (`bench/tractatus_pilot.py`, gated self-analysis + think control), not started.
- Proposed next step: write the contract, build the held-out set (Decision Index rows JevK5 never trained on,
  labels independent of the 27B), then run holistic vs confidence cascade vs checklist cascade.

### Closed decisions (answers of 2026-09-29)
- [1b] Method = tractatus-thinking's (atomic decomposition, multiplicative requirements, numbered hierarchy),
  derived from the book: the book's logic defines composition (TLP 5, 5.15); mathematical atoms are computed,
  not judged (TLP 6.2).
- [2a→c] Gate = accuracy on the same benchmarks. New outputs (trees, abstention, intervals, extra passes, API
  changes) only where they raise accuracy. One pass is a preference, not a gate.
- [3a/3b] Inference-time mechanism, weights unchanged; logical form from the caller or an LLM. Reading a
  tractatus-thinking JSON export as caller-supplied form is allowed (a format, not a live coupling).
- [4a/4b] Submodule = reference + code to port to Python where helpful.
- [5] Model = JevK5 v0.3 4B. LLM = the teacher LLM (an OpenAI-compatible server, Qwen3.8 27B); it writes decompositions only, trains nothing.
- [1c, 3c/3d, 4c] Parked in `IDEAS.md`: picture-theory lens, distillation, live MCP coupling. Not pursued.

### Defaults for unanswered points (user may override)
- Hardware: 4B GGUF (alibiserikbay/JevK5-GGUF) under llama.cpp on the 6 GB laptop GPU. llama.cpp not installed yet.
- End goal: measure on JevBench's public items; decide on a submission later.
- tractatus-thinking bugs: recorded below, not fixed; ports must not copy them.

### Sorted, not now
- Only if they raise accuracy: epistemic silence as a fallback trigger, Fréchet intervals, the tree as API output.
- Later, in scope: logical-space check on options, analysing long evidence into facts, prefix caching.
- Outside the method: multi_hop (needs intermediate values; a process problem, no gain expected).

---

## 📁 Active architecture

- `jevk5/prompt.py`: torch-free prompt, option mapping, `spread` (many-option readout). `runtime.py`: CUDA-graph
  JevK5. `gguf.py`: llama-server front end. `lite.py`: DeBERTa CPU model. `server.py`: /v1/systemone.
- `training/`: teacher.py (OpenAI-compatible teacher, 11 families), build_train.py, lora.py.
- `bench/`: JevBench adapters. `results/public231/*.jsonl`: per-item results (`task_id` starts with the tier;
  `family`, `correct`, `probs`).

---

## ⚠️ External blockers (don't block coding)

- No torch or pytest in system python3.12. Local GPU: RTX PRO 500, 6 GB (bf16 4B needs ~10 GB).
- tractatus-thinking bugs (not fixed):
  - `move` with `new_position` onto an occupied number overwrites that sibling (`src/tractatus/engine.ts:1977`
    `calculateNewNumber` never shifts): silent data loss, duplicate child refs, inflated totals.
  - Silence detector matches substrings (`src/utils/silence-detector.ts:215`): "single"/"since" hit "sin",
    "values" hits "value".
  - Circular-definition regex `(.+) is \1` fires on "default is to" ("t is t").

---

## 🔧 Useful commands

```bash
git submodule update --init          # after a fresh clone
```

---

## 📚 References (read IF needed)

- `IDEAS.md`: parked ideas (not pursued)
- `.wolf/cerebrum.md`: User Preferences, Do-Not-Repeat, Decision Log
- `.wolf/anatomy.md`: token-efficient file index
