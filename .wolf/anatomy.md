# anatomy.md

> Auto-maintained by OpenWolf. Last scanned: 2026-09-29T14:15:38.781Z
> Files: 137 tracked | Anatomy hits: 0 | Misses: 0

> Project structure index. Auto-maintained by OpenWolf hooks and daemon.
> Run `openwolf scan` to generate, or wait for the first Claude Code session.
> Status: Pending initial scan

## ./

- `.gitignore` — Git ignore rules (~34 tok)
- `.gitmodules` (~34 tok)
- `AGENTS.md` — OpenWolf (~75 tok)
- `CHANGELOG.md` — Change log (~2967 tok)
- `CLAUDE.md` — OpenWolf (~99 tok)
- `IDEAS.md` — Ideas (parked) (~757 tok)
- `LICENSE` — Project license (~3029 tok)
- `MODEL_CARD.md` — JevK5 v0.3 — open-weight Jev alternative (~3731 tok)
- `NOTICE` — Declares calling (~638 tok)
- `pyproject.toml` — Python project configuration (~190 tok)
- `README.md` — Project documentation (~6582 tok)

## bench/

- `acceptance.py` — The acceptance test of docs/tractatus-lora.md: v0.3, the answer-only control and the Tractatus LoRA, (~1998 tok)
- `gguf_check.py` — Check a GGUF build against the bf16 model on JevBench's public items. (~1322 tok)
- `jevbench-registration.patch` (~1006 tok)
- `jevk5_direct.py` — In-process JevBench adapter for JevK5 (Qwen3.5-4B + distilled LoRA, one forward pass). (~1022 tok)
- `jevk5_lite.py` — In-process JevBench adapter for JevK5-Lite (a CPU label-conditioned encoder; preview). (~1536 tok)
- `many_options.py` — Questions with more than 16 options: BANKING77, CLINC150+OOS and MASSIVE, train splits only. (~5853 tok)
- `parity.py` — Check that a runtime change leaves questions of up to 16 options bit-identical. (~1088 tok)
- `SUBMISSION.md` — Running JevK5 on JevBench (~1195 tok)
- `tractatus_eval.py` — JevK5 through a logical form (jevk5.tractatus) on JevBench's public items, against its single pass. (~3168 tok)
- `tractatus_pilot.py` — Pilot: the 4B analyses its own uncertain answers, all local (llama.cpp), on JevBench's public items. (~2586 tok)
- `vision_probe.py` — Probe whether JevK5 text weights can make image decisions with Qwen3.5's vision tower. (~1793 tok)

## docs/

- `tractatus-lora.md` — Tractatus-supervised LoRA: plan and acceptance criteria (~814 tok)

## docs/game-demos/

- `ARENA-LICENSE.txt` (~269 tok)

## examples/

- `README.md` — Project documentation (~767 tok)
- `tetris.py` — JevK5 plays Tetris: every placement is two typed decisions about a board drawn as text. (~4764 tok)

## jevk5/

- `__init__.py` — JevK5: a fast, calibrated, open decision model (Qwen3.5-4B + distilled LoRA, one pass). (~212 tok)
- `decompose.py` — Logical forms for `jevk5.tractatus`, written by an LLM on an OpenAI-compatible server. (~2418 tok)
- `gguf.py` — JevK5 on llama.cpp: the same option-letter readout from a GGUF, on Apple silicon or CPU. (~2255 tok)
- `lite.py` — JevK5-Lite (preview): a CPU encoder that scores every label of every head in one forward pass. (~2230 tok)
- `prompt.py` — The prompt, the option mapping and the many-option readout, with no heavy dependencies. (~2187 tok)
- `runtime.py` — JevK5: typed decisions from Qwen3.5-4B (+ a distilled LoRA, merged) in one forward pass. (~2204 tok)
- `server.py` — A TypeSafe-style /v1/systemone server for JevK5. (~1164 tok)
- `tractatus.py` — Decisions through a logical form: the answer as a truth-function of elementary propositions. (~2181 tok)

## results/

- `l40s_probe_20260924.json` (~11663 tok)
- `mmlu_pro_training_ids.json` (~3438 tok)

## results/public231/

- `jevk5-9b-v0.3.3.jsonl` (~75781 tok)
- `jevk5-9b-v0.3.jsonl` (~75340 tok)
- `jevk5-lite-preview1.jsonl` (~59538 tok)
- `jevk5-lite-preview1.summary.json` (~14303 tok)
- `jevk5-v0.1.jsonl` (~59178 tok)
- `jevk5-v0.1.summary.json` (~14177 tok)
- `jevk5-v0.2.jsonl` (~59093 tok)
- `jevk5-v0.2.summary.json` (~14166 tok)
- `jevk5-v0.3.jsonl` (~75162 tok)
- `untrained-qwen3.5-4b.jsonl` (~74290 tok)

## tests/

- `test_lite_bench.py` — bench/jevk5_lite.py's mapping from JevBench questions to JevK5-Lite heads (no JevBench install needed). (~747 tok)
- `test_lite.py` — JevK5-Lite's runtime on a tiny random model built offline (no download, CPU). (~1569 tok)
- `test_lora_text.py` — lora.py's text loss (the Tractatus form rows) equals transformers' own causal-LM loss on the target  (~458 tok)
- `test_prompt.py` — The many-option readout in jevk5.prompt, checked against readers with known answers. (~2081 tok)
- `test_runtimes.py` — The CUDA runtime and the llama.cpp client read the same passes and combine them the same way. (~1918 tok)
- `test_tractatus.py` — The Tractatus readout in jevk5.tractatus (and the decomposer's parsing), checked against models with (~2098 tok)

## tractatus-thinking/

- `.eslintrc.json` — ESLint configuration (~517 tok)
- `.gitignore` — Git ignore rules (~1022 tok)
- `.prettierignore` (~9 tok)
- `.prettierrc` — Prettier configuration (~46 tok)
- `claude_desktop_config.example.json` (~44 tok)
- `claude-desktop-developer-config.json` (~102 tok)
- `comprehensive-performance-test.mjs` — Declares runPerformanceTests (~2849 tok)
- `LICENSE` — Project license (~286 tok)
- `mcp-config.json` (~61 tok)
- `package.json` — Node.js package manifest (~867 tok)
- `performance-test.mjs` — Declares startTime (~625 tok)
- `PUBLISH_CHECKLIST.md` — IMPROVED Ready to Publish - Final Checklist (~863 tok)
- `QUICK_REFERENCE.md` — MANDATORY Wittgenstein Thinking - Quick Reference Card (~709 tok)
- `QUICKSTART.md` — IMPROVED Wittgenstein Thinking - Quick Start Guide (~2170 tok)
- `README.md` — Project documentation (~2860 tok)
- `test-modern.js` — __filename: testModernServer (~795 tok)
- `tsconfig.base.json` (~118 tok)
- `tsconfig.json` — TypeScript configuration (~174 tok)
- `tsconfig.src.json` (~175 tok)
- `tsconfig.tests.json` — /*", "src/**/*.test.ts", "src/**/*.spec.ts", "vitest.config.ts"] (~173 tok)
- `vitest.config.ts` — Vitest test configuration (~176 tok)

## tractatus-thinking/.husky/

- `pre-commit` — Run lint-staged (~10 tok)
- `pre-push` — Run core quality check - focus on production code precision (~24 tok)

## tractatus-thinking/.vscodium/

- `launch.json` (~371 tok)
- `settings.json` (~145 tok)

## tractatus-thinking/examples/

- `advanced-features.md` — Advanced Features Examples (~1061 tok)
- `basic-usage.md` — Basic Usage Examples (~687 tok)
- `mcp-integration.md` — MCP Integration Guide (~1413 tok)

## tractatus-thinking/src/

- `index-modern.ts` — Import unified handler (~2437 tok)
- `index.ts` — tractusHandler: createCoreProperties, createStartProperties, createAddProperties + 5 more (~2867 tok)

## tractatus-thinking/src/handlers/

- `add-proposition.ts` — MCP tool handler for the add_insight tool. (~3530 tok)
- `clear_thinking.ts` — MCP tool handler for the clear_thinking tool. (~2091 tok)
- `export.ts` — Generates enhanced structure visualization for export. (~1961 tok)
- `hypothesis-test.ts` — Handler for hypothesis testing operations in enhanced Tractatus thinking. (~3161 tok)
- `index.ts` (~33 tok)
- `move.ts` — Handler for moving propositions to reflect evolved understanding. (~2260 tok)
- `navigate.ts` — Exports NavigationHandler (~2337 tok)
- `revise.ts` — Handler for revising proposition content while preserving logical structure. (~1726 tok)
- `session-manager.ts` — Manages persistent thinking sessions with lifecycle controls. (~1085 tok)
- `tractatus-think.ts` — Unified handler for the tractatus_thinking tool. (~4740 tok)
- `undo.ts` — Handler for undoing the last structural operation. (~1470 tok)

## tractatus-thinking/src/tractatus/

- `engine.ts` — The core philosophical reasoning engine that implements Wittgenstein's Tractatus method. (~20279 tok)

## tractatus-thinking/src/types/

- `index.ts` — Parameters for the add_proposition tool. (~3498 tok)

## tractatus-thinking/src/utils/

- `bundling-detector.ts` — Bundling detection utilities for helping LLMs identify compound concepts (~1968 tok)
- `cross-reference-manager.ts` — Manages cross-references between propositions, enabling richer (~2946 tok)
- `decomposition-advisor.ts` — Decomposition advisor utilities for helping LLMs determine when and how (~3247 tok)
- `formatter.ts` — Formats TractusTree structures for export in various formats. (~1934 tok)
- `independence-validator.ts` — Independence validation utilities for ensuring logical independence (~4002 tok)
- `numbering-system.ts` — Generate child proposition number following configured format. (~2656 tok)
- `silence-detector.ts` — Exports SilenceDetector (~1834 tok)
- `structure-visualizer.ts` — Structure visualization utilities for helping LLMs understand the logical (~4243 tok)
- `validation.ts` — Validation utilities for the Clear Thinking tool. (~2166 tok)

## tractatus-thinking/tests/enhancements/

- `enhanced-operations.test.ts` — Declares startParams (~3418 tok)

## tractatus-thinking/tests/integration/

- `full-workflow.test.ts` — Declares const (~1968 tok)
- `tractatus-think-unified.test.ts` — Declares startParams (~2196 tok)

## tractatus-thinking/tests/philosophy/

- `wittgenstein-principles.test.ts` — API routes: GET (2 endpoints) (~3972 tok)

## tractatus-thinking/tests/unit/

- `depth-consistency.test.ts` — Declares engine (~377 tok)
- `logic-detection.test.ts` — Declares tree (~1243 tok)
- `numbering-fix.test.ts` — Declares ns (~457 tok)
- `threshold-enforcement.test.ts` — Declares tree (~1281 tok)
- `tractatus-engine.test.ts` — API routes: GET (6 endpoints) (~2961 tok)
- `validation-fixes.test.ts` — Declares tree (~1958 tok)

## tractatus-thinking/tests/unit/handlers/

- `clear-thinking.test.ts` — Declares params (~2133 tok)
- `export.test.ts` — Declares sessionManager (~3599 tok)
- `navigate.test.ts` — Declares sessionManager (~2596 tok)
- `session-manager.test.ts` — Declares tree (~2846 tok)

## tractatus-thinking/tests/unit/utils/

- `bundling-detector.test.ts` — Declares content (~1043 tok)
- `decomposition-advisor.test.ts` — Declares content (~1087 tok)
- `independence-validator.test.ts` — Declares createMockProposition (~1685 tok)
- `structure-visualizer.test.ts` — Declares createMockTree (~2406 tok)

## tractatus-thinking/tests/utils/

- `test-utilities.ts` — Shared test utilities for Wittgenstein MCP Server tests (~261 tok)

## training/

- `batching.py` — Group sequences into length-sorted, token-budgeted batches with a small set of shapes. (~448 tok)
- `build_tractatus.py` — Training rows for the Tractatus-supervised LoRA, or for its answer-only control, from tractatus_data (~1021 tok)
- `build_train.py` — Assemble a training folder for laya_turbo.cuda.lora from teacher output plus a general replay. (~1246 tok)
- `hard_dev_items.py` — A small hand-written calibration set in the style of JevBench's hard tier (not its items). (~7854 tok)
- `lora.py` — LoRA-tune a causal LM as a one-pass decision model: cross-entropy on the option-letter logits. (~3588 tok)
- `mmlu_leak.py` — Did training on MMLU-Pro test items inflate JevK5's MMLU-Pro accuracy? (~768 tok)
- `mock.py` — A tiny, randomly initialised stand-in for JevK5: same architecture, tokenizer and layer pattern as t (~592 tok)
- `order_avg.py` — Does reading each question in two option orders and averaging help? (never on JevBench items) (~1056 tok)
- `README.md` — Project documentation (~470 tok)
- `replay.py` — A small public replay for the Tractatus LoRA: train splits of openly licensed multiple-choice and ye (~1370 tok)
- `teacher.py` — Hard decisions written and checked by a teacher LLM (any OpenAI-compatible server, e.g. vLLM). (~14697 tok)
- `temp_choice.py` — Compare temperature settings on held-out data only (never on JevBench items). (~710 tok)
- `tractatus_data.py` — Tractatus supervision for teacher questions: the teacher's logical form of each kept question, with  (~1496 tok)
- `tractatus_pipeline.sh` — The Tractatus LoRA pipeline, end to end: rows -> both LoRAs (Tractatus and answer-only control) -> G (~530 tok)
