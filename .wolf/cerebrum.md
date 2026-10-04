---
description: learned preferences, project conventions, and Do-Not-Repeat rules
budget_tokens: 2000
---
# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-09-29

## User Preferences

<!-- How the user likes things done. Code style, tools, patterns, communication. -->

- For open-ended design goals: map the problem with the tractatus_thinking MCP, think it through deeply, then ask clarifying questions before implementing.
- When something stays unclear after the user answers, resolve it with the tractatus_thinking tool and record a default instead of asking again.
- Ideas the user designates go to `IDEAS.md` and are not pursued without an explicit decision.
- Principles: KISS, YAGNI, SOLID, DRY, stay focused. End goal (2026-09-29): improve the quality of answers overall.
- Cross-check ideas with GPT-6 Astra: `codex exec -m gpt-6-astra -s read-only --skip-git-repo-check --ephemeral -o out.md "<prompt>"` (the sandbox can't read the repo, so make the prompt self-contained).
- No institution- or service-specific details in repo files (hostnames, URLs, model ids with service prefixes, key paths, user paths); endpoints come from LLM_URL / LLM_MODEL / LLM_API_KEY / TEACHER_KEY; private runners live outside the repo.
- Don't touch the GPU cluster setup unless asked; it is handled separately.
- Compare before training (e.g. JevK5 against the second model first); spare a model's quota when asked; parallel requests are fine when allowed.

## Key Learnings

- **Project:** wittyJev = the user's fork of allebee/jevk5 (JevK5: Qwen3.5-4B/9B + merged LoRA, one-pass typed decisions). Name = Witt(genstein) + Jev.
- **Goal:** enhance JevK5 with Tractatus ideas; `tractatus-thinking/` (git submodule) is the user's own MCP server.
- OpenWolf condenses Bash output over ~4k tokens; the full text is in `.wolf/cache/bash/<id>.log`. Use Read with offset/limit for large files.
- tractatus_thinking MCP: atomic `add` is rejected on " and ", " plus ", " with ", or a comma/semicolon followed by a capital letter. Rephrase instead of fighting it.
- llama.cpp runs in Docker: `ghcr.io/ggml-org/llama.cpp:server-cuda`, container `wittyjev-llama`, port 8080, `-c 16384 -ngl 99`; the 4B Q8_0 uses 5.0 of 6.1 GB. JevK5GGUF temperatures for v0.3 4B: 1.22 / knockout 0.93.
- JevBench's `tasks.py`, `scoring.py`, `metrics.py` are stdlib-only: import them from the `data/jevbench` checkout instead of reimplementing. Never send a task's `provenance` (it has the rationale) to any model.
- Mock-up first: training/mock.py builds a tiny random Qwen3.5 (JevK5's tokenizer, same layer pattern) so the whole pipeline runs on CPU in minutes; swap BASE/DATA for the real run.
- llama.cpp's converter needs its `conversion/` package too (sparse checkout: gguf-py + conversion); pass --no-mtp for Qwen3.5.
- the teacher server's Mistral rejects chat_template_kwargs: Decomposer(template=False).
- Tractatus "silence" (TLP 7: a question without sense) is not abstention on thin evidence (TLP 5.156: incomplete knowledge). Keep the two apart.
- The teacher server's Qwen models accept reasoning_effort low, medium, xhigh (default); 'high' is rejected. Thinking off costs ~10 output tokens per answer.
- The teacher server enforces a per-model token quota over a trailing 24 h; every response's usage block reports it. Check it before large runs.
- bench/tractatus_eval.py caches every LLM reply per output folder: rerunning after a crash costs no LLM tokens.
- The local llama.cpp container stops on restarts: `docker start wittyjev-llama` before evaluations (otherwise: Connection refused).

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

- [2026-09-29] tractatus_thinking `move` with `new_position` overwrote two siblings (engine.ts:1977). Add propositions in their final order; don't reorder with `move`.

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->

- [2026-09-29] Scope: inference-time mechanism (decompose → unchanged 4B judges atoms → logic composes); accuracy on JevBench's public items is the gate. Why: user answers 1b, 2a→c, 3a/3b, 5 (details in STATUS.md).
- [2026-09-29] Pilot on 231 public items: checklist 0.931 vs holistic 0.883 (15/4, p=0.019); compose 0.870 (17/20); 27B direct 0.987. Compose is not the default; attribution (structure vs the 27B's reading) is the open question.
- [2026-09-29] Universal design: calibrated-confidence routing (answer / escalate / undecided) is the carrying principle; decomposition only where cheaper or more reliable than the strong reader. Astra: "universal" needs a pre-registered acceptance contract and fresh, independently labeled, template-split data; the cascade-vs-escalation gap was one item.
- [2026-09-29] Blind test (27B writes forms WITHOUT the evidence): checklist 0.848 (2 fixed / 10 broken, p=0.039), compose 0.723. Generic structure hurts the 4B; the 0.931 gain came from the 27B's case-grounded reading.
- [2026-09-29] The JevK5 4B as its own decomposer (llama.cpp /v1/chat/completions): thinking off gives atoms that are just names (a1..); thinking on found the right structure on hard item 1 but took 165 s per item on the laptop GPU and ignored the syntax (names, AND/NOT). Validator now rejects name-only atoms.
- [2026-09-29] The decomposer (the teacher LLM) writes the logical form only, never truth values; runs include a 27B-direct reference. Why: otherwise a gain can't be attributed to the method.
