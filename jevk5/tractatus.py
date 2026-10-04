"""Decisions through a logical form: the answer as a truth-function of elementary propositions.

After Wittgenstein's Tractatus (5: a proposition is a truth-function of elementary propositions). A
form names atoms, yes/no questions about single facts of the case, and how the answer depends on them:

    {"atoms": ["Was the house vacant as the policy defines it?", "Did the leak last 14 days or more?"],
     "formula": "not a1 and a2"}                                           # a noul question
    {"atoms": [...], "rules": [{"option": "deny", "when": "a1"}, ...], "otherwise": "pay"}   # choice

"compose" judges every atom in its own pass and computes the answer instead of judging it: each row of
the atoms' truth table is weighted by the product of their probabilities (atoms are taken to be
independent, TLP 4.211) and counted for the option it makes true (TLP 5.15). A choice's rules are a
decision list, evaluated jointly on each row: the row goes to the first rule that holds, else to
`otherwise`; a score question's decision list runs over its level indices ("0", "1", ...). "checklist"
reads the question whole once, with the atoms appended as points to check.

Questions without a form are read whole, exactly as before. No torch: any runtime with
`probabilities(state, question)` works (JevK5, JevK5GGUF).
"""

from __future__ import annotations

import ast
import itertools
import math
import re
from collections.abc import Callable

from .prompt import answer, decision_options

MAX_ATOMS = 8
MODES = ("compose", "checklist")
_ATOM = re.compile(r"a([1-9][0-9]*)")

Row = tuple[bool, ...]


def formula(expr: str, count: int) -> Callable[[Row], bool]:
    """A formula over a1 .. a<count> (and, or, not, parentheses) as a function of one truth-table row.
    Anything else raises ValueError; the expression is walked, never evaluated as Python. Upper-case
    AND, OR, NOT are read as and, or, not."""
    expr = re.sub(r"\b(AND|OR|NOT)\b", lambda match: match[1].lower(), expr)
    try:
        tree = ast.parse(expr, mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"not a formula: {expr!r}") from error

    def check(node: ast.AST) -> None:
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                check(value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            check(node.operand)
        elif not (
            isinstance(node, ast.Name)
            and (match := _ATOM.fullmatch(node.id))
            and int(match[1]) <= count
        ):
            raise ValueError(f"not a formula over a1..a{count}: {ast.unparse(node)!r}")

    def value(node: ast.AST, row: Row) -> bool:
        if isinstance(node, ast.BoolOp):
            parts = (value(v, row) for v in node.values)
            return all(parts) if isinstance(node.op, ast.And) else any(parts)
        if isinstance(node, ast.UnaryOp):
            return not value(node.operand, row)
        return row[int(node.id[1:]) - 1]

    check(tree)
    return lambda row: value(tree, row)


def validate(form: dict, question: dict) -> dict:
    """The form, if the composer can read it for this question; ValueError otherwise."""
    atoms = form.get("atoms")
    if not (
        isinstance(atoms, list)
        and 1 <= len(atoms) <= MAX_ATOMS
        and all(isinstance(a, str) and a.strip() for a in atoms)
    ):
        raise ValueError(f"atoms must be 1 to {MAX_ATOMS} non-empty strings")
    if any(_ATOM.fullmatch(a.strip()) for a in atoms):
        raise ValueError("atoms must be questions, not the names a1, a2, ...")
    if question["type"] == "noul":
        formula(str(form.get("formula", "")), len(atoms))
    elif question["type"] in ("choice", "score"):  # a score's options are its level indices
        ids = [key for key, _ in decision_options(question)]
        rules = form.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError("a choice form needs a non-empty list of rules")
        for rule in rules:
            if not isinstance(rule, dict) or rule.get("option") not in ids:
                raise ValueError(f"rule {rule!r} names no option of {ids}")
            formula(str(rule.get("when", "")), len(atoms))
        if form.get("otherwise") not in ids:
            raise ValueError(f"otherwise must be one of {ids}")
    else:
        raise ValueError(f"unknown question type {question['type']!r}")
    pick = _pick(form, question)
    if len({pick(row) for row in itertools.product((True, False), repeat=len(atoms))}) < 2:
        raise ValueError("the form gives one answer on every row, so it says nothing (TLP 4.461)")
    return form


def _pick(form: dict, question: dict) -> Callable[[Row], str]:
    """The option each truth-table row gives: the formula's truth value, or the decision list's first
    rule that holds (else `otherwise`)."""
    count = len(form["atoms"])
    if question["type"] == "noul":
        holds = formula(str(form["formula"]), count)
        return lambda row: "true" if holds(row) else "false"
    rules = [(rule["option"], formula(str(rule["when"]), count)) for rule in form["rules"]]
    return lambda row: next((option for option, when in rules if when(row)), form["otherwise"])


def compose(form: dict, question: dict, p: list[float]) -> dict[str, float]:
    """The option distribution the form implies, given each atom's probability of being true."""
    pick = _pick(form, question)
    out = dict.fromkeys((key for key, _ in decision_options(question)), 0.0)
    for row in itertools.product((True, False), repeat=len(form["atoms"])):
        out[pick(row)] += math.prod(q if x else 1.0 - q for q, x in zip(p, row, strict=True))
    return out


def checklist(question: dict, form: dict) -> dict:
    """The question with the form's atoms appended to its instructions, to be read whole in one pass."""
    points = " ".join(f"({i}) {atom}" for i, atom in enumerate(form["atoms"], 1))
    return {**question, "instructions": f"{question['instructions']} Points to check: {points}"}


class Tractatus:
    """A runtime that decides through a logical form. `decompose(state, question)` returns a form (see
    `validate`) or None; with None, `model` reads the question whole."""

    def __init__(self, model, decompose: Callable[[object, dict], dict | None], mode: str = "compose"):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; use one of {MODES}")
        self.model, self.decompose, self.mode = model, decompose, mode
        self.last_form: dict | None = None
        self.last_atoms: list[float] | None = None

    def probabilities(self, state, question: dict) -> tuple[dict[str, float], int]:
        """Probability per option id and the input tokens of every pass, as the runtimes return them."""
        form = self.decompose(state, question)
        self.last_form, self.last_atoms = form, None
        if form is None:
            return self.model.probabilities(state, question)
        validate(form, question)
        if self.mode == "checklist":
            return self.model.probabilities(state, checklist(question, form))
        p, tokens = [], 0
        for atom in form["atoms"]:
            probs, used = self.model.probabilities(state, {"type": "noul", "instructions": atom})
            p.append(probs["true"])
            tokens += used
        self.last_atoms = p
        return compose(form, question, p), tokens

    def decide(self, state, question: dict) -> dict:
        probs, tokens = self.probabilities(state, question)
        return answer(question, probs, tokens)
