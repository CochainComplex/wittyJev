# Tractatus-supervised LoRA: plan and acceptance criteria

Frozen on 2026-09-29, before any test item was generated. Changes after that date are listed at the end.

## Goal

Better decisions from the final JevK5 model across a defined set of question families. The final model is
self-contained: no external model at inference. External teachers are used offline, for training data only.

## Method

- Start from JevK5 v0.3 (4B), merged weights, and train one LoRA on top. This is not a full retraining.
- One adapter, three objectives:
  1. the answer letter of each teacher question (JevK5's own loss);
  2. elementary atom verdicts: every atom as a yes/no question, with the teacher's verdict as its label;
  3. writing the decomposition: causal-LM loss on the form's JSON, under `jevk5.decompose`'s prompt.
- Replay JevK5 v0.3's public train-split mix (`training/build_train.py`), so earlier skills stay.
- Control: an answer-only LoRA on the same questions (objective 1 plus replay), same steps, same learning rate.

## Data

- Teacher: Qwen3.8 27B with xhigh reasoning on an OpenAI-compatible server, through `training/teacher.py`.
  It writes documents with three hard typed questions each, solves every question twice, and keeps a
  question only if both solves match the author's answer.
  `training/tractatus_data.py` then adds the teacher's form with a verdict per atom, kept only if the verdicts
  compose to that answer.
- Families: teacher.py's 11 default families, equal weights: long_policy, temporal_numeric, multi_hop, judge,
  ambiguous, trap, adversarial, tradeoff, routing, extraction, rubric.
- Train split: teacher.py's train domains. 10% of the train documents are held out for calibration and model
  selection.
- Test split: teacher.py's three held-out domains, never used in training, so the test domain is shifted by
  construction. A test label also needs an independent model (Mistral Small 4) to give the same answer;
  items it disputes are excluded.
- JevBench's 231 public items: development only, never trained on.

## Acceptance (the one-pass decision path)

v0.3 (baseline), the answer-only control and the Tractatus LoRA are compared on the test split, paired, all
on the same GGUF runtime. Temperatures are fitted on the calibration hold-out, never on the test split.

1. Overall accuracy of the Tractatus LoRA is above v0.3's, exact McNemar p < 0.05.
2. No family shows a significant accuracy loss against v0.3 (per-family exact McNemar, Holm-corrected over the
   11 families, p < 0.05 counts as a loss).
3. Overall NLL and ECE are no worse than v0.3's by more than 0.01.
4. The Tractatus LoRA beats the answer-only control overall (exact McNemar p < 0.05); otherwise the gain is
   not credited to the structure.
5. On JevBench's public items (development), no tier is worse than v0.3's by more than one item.

About 100 test questions per family is a pilot-scale test. A pass is evidence, not proof of universality. A
loss that the test cannot detect is reported as inconclusive, not as "no loss".

## Explicit mode (reported separately)

The same weights write their own decomposition; composition, checklist and the "undecided" rule run on it.
Evaluated separately, and not part of the one-pass acceptance above.

## Changes after freezing

- 2026-09-29: the teacher's reasoning effort is xhigh. On 35 train questions its solves matched the
  answer that author and Mistral agree on 64/64 times, against 61/64 at the default effort (+39% tokens).
- 2026-09-29: a second train split with another seed (more documents, same pipeline and checks).
- 2026-09-29: v0.3's replay mix cannot be rebuilt here (its builder is not in the repo). The replay is
  `training/replay.py` instead: 6,000 rows from six of the same public train splits (BoolQ, ARC, CommonsenseQA,
  OpenBookQA, WinoGrande, HellaSwag), none sharing an 8-word sequence with JevBench's public items.
- 2026-09-29: the replay questions also get the teacher's forms and verdicts, kept only if they compose to the
  human gold label, so they train all three objectives. JevBench's public items are decomposed the same way
  for comparison only (how often the teacher's analysis reproduces JevBench's answers); never trained on.
