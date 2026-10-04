# Ideas (parked)

Written down, not pursued. Each idea comes from the scoping answers of 2026-09-29. The current work is
the inference-time mechanism described in `.wolf/STATUS.md`: decompose the criterion, let the unchanged
JevK5 4B judge each atom, compose by logic. An idea is picked up only by an explicit decision.

## 1. Picture theory as a lens on the model's internal representation

*From answer 1c, the "something unmapped" option.*

TLP 2.1–2.225: a picture shares its logical form with what it depicts (2.18), and a proposition is a
picture of reality (4.01). The question: does JevK5's last-position hidden state picture the case? That
is, does it encode the elementary propositions of the evidence as recoverable, structured features?

- **How it could be tested:** linear probes for atom truth values on the hidden states of decomposed
  items, layer by layer. Then check whether a compound's truth value is a simple function of the atom
  directions.
- **Why it matters:** if the atoms can be decoded from one pass, a probe readout could replace the
  extra passes of the decomposition (see idea 2).

## 2. Distil the decompositions into the weights

*From answer 3c. The "then distil" half of option 3d belongs here too.* **Un-parked on 2026-09-29:** now the
main route, as the Tractatus-supervised LoRA in `docs/tractatus-lora.md`.

Keep one pass and ~13 ms, while the model internalizes the structure that the inference-time mechanism
makes explicit.

- **Teacher route:** A teacher model writes decompositions and atom labels for training documents (a new
  family in `training/teacher.py`). The LoRA then trains on atom questions next to the compound ones
  (`training/lora.py`).
- **Self-distillation route:** run the inference-time mechanism over train splits and distil its
  composed distributions into a one-pass model.
- **Needs:** an H100-class GPU, a new temperature fit, and the repo's overlap scans against JevBench
  and the Decision Index.

## 3. Live coupling of tractatus-thinking and JevK5

*From answer 4c. The current scope uses the submodule only as reference and as code to port. Reading
an exported JSON file as a caller-supplied logical form is in scope (answer 3a). This idea is the
live coupling.*

- **JevK5 as the MCP's truth checker:** an operation that sends a proposition and its evidence to
  `jevk5-serve` (`/v1/systemone`, a noul question) and stores the calibrated probability as the
  proposition's confidence, moving it from hypothesis to established or rejected.
- **The MCP as JevK5's decomposer:** a tree built interactively with tractatus_thinking is judged atom
  by atom and composed live. Its decomposition types already name the connectives: analysis = AND,
  cases = OR, negation = NOT, implication = IF-THEN.
- **Needs:** a TypeScript-to-Python bridge (HTTP to `jevk5-serve`), and first a fix for the `move`
  operation, which overwrites siblings (`tractatus-thinking/src/tractatus/engine.ts:1977`). A tree that
  silently loses propositions would be judged wrongly.
