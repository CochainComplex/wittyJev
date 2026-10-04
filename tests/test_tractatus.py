"""The Tractatus readout in jevk5.tractatus (and the decomposer's parsing), checked against models with
known answers. No GPU and no network."""

from __future__ import annotations

import itertools
import sys
import json
from pathlib import Path

import pytest

from jevk5.decompose import Decomposer, parse
from jevk5.tractatus import MAX_ATOMS, Tractatus, compose, formula, validate

NOUL = {"type": "noul", "instructions": "Is a refund permitted?"}
CHOICE = {
    "type": "choice",
    "instructions": "What happens to the claim?",
    "criteria": {"deny": "Denied.", "cap": "Paid up to the cap.", "pay": "Paid in full."},
}


class Model:
    """Answers each known atom with a fixed probability; reads any other question as `whole`."""

    def __init__(self, truth: dict[str, float], whole: dict[str, float] | None = None):
        self.truth, self.whole, self.asked = truth, whole or {"true": 0.4, "false": 0.6}, []

    def probabilities(self, state, question):
        self.asked.append(question)
        p = self.truth.get(question["instructions"])
        if p is None:
            return dict(self.whole), 100
        return {"true": p, "false": 1 - p}, 10


@pytest.mark.parametrize("expr", ["a1 and (a2 or not a3)", "a1 AND (a2 OR NOT a3)"])
def test_formula_reads_and_or_not(expr):
    holds = formula(expr, 3)
    for row in itertools.product((True, False), repeat=3):
        assert holds(row) == (row[0] and (row[1] or not row[2]))


@pytest.mark.parametrize(
    "expr",
    ["a1 + a2", "__import__('os')", "a4", "a0", "x", "a1 and", "", "lambda: a1", "a1.real",
     "a1 if a2 else a3", "a1 == a2", "True"],
)
def test_formula_rejects_anything_else(expr):
    with pytest.raises(ValueError):
        formula(expr, 3)


def test_noul_multiplies_independent_atoms():
    form = {"atoms": ["A", "B"], "formula": "a1 and not a2"}
    assert compose(form, NOUL, [0.9, 0.2]) == pytest.approx({"true": 0.72, "false": 0.28})
    form = {"atoms": ["A", "B"], "formula": "a1 or a2"}
    assert compose(form, NOUL, [0.9, 0.2])["true"] == pytest.approx(1 - 0.1 * 0.8)


def test_shared_atoms_are_evaluated_jointly():
    assert compose({"atoms": ["A"], "formula": "a1 and a1"}, NOUL, [0.7])["true"] == pytest.approx(0.7)
    form = {"atoms": ["A"], "rules": [{"option": "deny", "when": "a1"},
                                     {"option": "cap", "when": "a1"}], "otherwise": "pay"}
    assert compose(form, CHOICE, [0.7]) == pytest.approx({"deny": 0.7, "cap": 0.0, "pay": 0.3})


def test_decision_list_takes_the_first_rule_that_holds():
    form = {"atoms": ["A", "B"], "rules": [{"option": "deny", "when": "a1"},
                                          {"option": "cap", "when": "a2"}], "otherwise": "pay"}
    probs = compose(form, CHOICE, [0.3, 0.6])
    assert probs == pytest.approx({"deny": 0.3, "cap": 0.7 * 0.6, "pay": 0.7 * 0.4})
    assert sum(probs.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "form,question",
    [
        ({"atoms": [], "formula": "a1"}, NOUL),
        ({"atoms": ["A"] * (MAX_ATOMS + 1), "formula": "a1"}, NOUL),
        ({"atoms": ["A", 3], "formula": "a1"}, NOUL),
        ({"atoms": ["a1", "a2"], "formula": "a1"}, NOUL),
        ({"atoms": ["A"], "formula": "a1 or not a1"}, NOUL),
        ({"atoms": ["A"], "rules": [{"option": "pay", "when": "a1"}], "otherwise": "pay"}, CHOICE),
        ({"atoms": ["A"]}, NOUL),
        ({"atoms": ["A"], "rules": [{"option": "refund", "when": "a1"}], "otherwise": "pay"}, CHOICE),
        ({"atoms": ["A"], "rules": [{"option": "deny", "when": "a1"}]}, CHOICE),
        ({"atoms": ["A"], "rules": [], "otherwise": "pay"}, CHOICE),
        ({"atoms": ["A"], "formula": "a1"}, {"type": "score", "instructions": "How bad?",
                                             "criteria": ["low", "high"]}),
    ],
)
def test_validate_rejects_forms_the_composer_cannot_read(form, question):
    with pytest.raises(ValueError):
        validate(form, question)


def test_without_a_form_the_question_is_read_whole():
    model = Model({})
    assert Tractatus(model, lambda state, question: None).probabilities("s", NOUL) == (model.whole, 100)
    assert model.asked == [NOUL]


def test_compose_judges_each_atom_once():
    model = Model({"A": 0.9, "B": 0.2})
    form = {"atoms": ["A", "B"], "formula": "a1 and not a2"}
    tractatus = Tractatus(model, lambda state, question: form)
    probs, tokens = tractatus.probabilities("s", NOUL)
    assert probs == pytest.approx({"true": 0.72, "false": 0.28}) and tokens == 20
    assert [q["instructions"] for q in model.asked] == ["A", "B"]
    assert all(q["type"] == "noul" for q in model.asked)
    assert tractatus.last_atoms == [0.9, 0.2]
    assert tractatus.decide("s", NOUL)["noul"] == pytest.approx(0.72)


def test_checklist_reads_once_with_the_atoms_appended():
    model = Model({"A": 0.9, "B": 0.2})
    form = {"atoms": ["A", "B"], "formula": "a1 and not a2"}
    probs, tokens = Tractatus(model, lambda s, q: form, "checklist").probabilities("s", NOUL)
    assert (probs, tokens) == (model.whole, 100)
    assert model.asked[0]["instructions"] == "Is a refund permitted? Points to check: (1) A (2) B"


def test_a_supplied_form_that_does_not_fit_raises():
    with pytest.raises(ValueError):
        Tractatus(Model({}), lambda s, q: {"atoms": ["A"], "formula": "a2"}).probabilities("s", NOUL)


def test_parse_finds_the_json_object_in_a_reply():
    assert parse('```json\n{"atomic": true}\n```') == {"atomic": True}
    assert parse("no json here") is None
    assert parse('{"atoms": [') is None
    assert parse("[1, 2]") is None


def replying(reply: str) -> Decomposer:
    decomposer = Decomposer(key_env="NO_SUCH_KEY")
    decomposer.ask = lambda messages: reply
    return decomposer


def test_decomposer_returns_only_forms_the_composer_can_read():
    form = {"atoms": ["A"], "formula": "not a1"}
    assert replying(json.dumps(form))("s", NOUL) == form
    assert replying('{"atomic": true}')("s", NOUL) is None
    bad = replying('{"atoms": ["A"], "formula": "a1 + 1"}')
    assert bad("s", NOUL) is None and NOUL["instructions"] in bad.rejected


def test_a_blind_decomposer_never_sends_the_evidence():
    sent = []
    decomposer = Decomposer(key_env="NO_SUCH_KEY", blind=True)
    decomposer.ask = lambda messages: sent.append(messages) or '{"atomic": true}'
    decomposer("The secret evidence.", NOUL)
    assert "secret" not in json.dumps(sent) and "not shown" in json.dumps(sent)


def test_a_score_form_is_a_decision_list_over_the_levels():
    score = {"type": "score", "instructions": "How severe?", "criteria": ["minor", "major", "critical"]}
    form = {"atoms": ["Is data lost?", "Is service down?"],
            "rules": [{"option": "2", "when": "a1 and a2"}, {"option": "1", "when": "a1 or a2"}],
            "otherwise": "0"}
    assert replying(json.dumps(form))("s", score) == form
    assert compose(form, score, [0.5, 0.5]) == pytest.approx({"0": 0.25, "1": 0.5, "2": 0.25})


@pytest.mark.parametrize(
    "reply,expected",
    [('{"answer": "false: A condition is missing."}', "false"), ('{"answer": "Yes"}', "true"),
     ('{"answer": "cap"}', "cap"), ('{"answer": "refund"}', None), ("no json", None)],
)
def test_answer_reads_the_option_id(reply, expected):
    question = NOUL if expected in ("true", "false") else CHOICE
    assert replying(reply).answer("s", question) == expected


def test_replay_rows_are_decisions():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
    from replay import row

    yes_no = row("boolq", 7, "Some passage.", "Is it true?", None, 1)
    assert yes_no["question"]["type"] == "noul" and yes_no["expected"] == "true"
    choice = row("arc", 3, "Q?", "Which option?", ["x", "y", "z"], 2)
    assert choice["question"]["criteria"] == {"a": "x", "b": "y", "c": "z"} and choice["expected"] == "c"
    assert choice["source"] == "replay-arc"
