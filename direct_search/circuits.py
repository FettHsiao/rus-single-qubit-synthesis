"""BS12 labels are matrix products; Qiskit ``ops`` are in execution order.

For example, the paper's TH = T @ H is executed as h(), then t().
Convert a complete mathematical word once, reversing all its primitive factors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Clifford, Operator

DATA, ANC = 0, 1
TOL = 1e-10
TEMPLATE_CONVENTION = "bs12_matrix_products_v2"
SLOTS = (
    ("anc_g0", "C_a0", "anc_g1"),
    ("anc_g2", "C_a1", "anc_g3"),
    ("anc_g4", "C_a2", "anc_g5"),
    ("dat_g0", "C_d0", "dat_g1"),
)
# Factors below are written in matrix-product order, not execution order.
SYLLABLES = {"TH": ("t", "h"), "SHTH": ("s", "h", "t", "h")}
NAMED_FACTORS = {
    "I": (), "X": ("x",), "Y": ("y",), "Z": ("z",),
    "H": ("h",), "S": ("s",), "Sdg": ("sdg",),
    "HS": ("h", "s"), "HX": ("h", "x"), "SH": ("s", "h"), "SHX": ("s", "h", "x"),
    "HSH": ("h", "s", "h"), "HSHX": ("h", "s", "h", "x"),
    "XHS": ("x", "h", "s"), "XHSH": ("x", "h", "s", "h"),
}

# Appendix A of 1206.3223v1, indexed exactly as G0, ..., G23.
# Qiskit's H/T/S differ from the paper's SU(2) representatives only by phase.
BS12_CLIFFORD_WORDS = (
    "", "H", "HSSH", "SS", "S", "SSS", "HSS", "SSH",
    "SH", "SSSH", "SSHSSH", "SHSSH", "SSSHSSH", "HS", "HSSS",
    "SSHSS", "SHSS", "SSSHSS", "HSH", "HSSSH", "HSHSSH",
    "HSSSHSSH", "SSSHS", "SHSSS",
)


def matrix_factors_to_ops(factors) -> tuple[str, ...]:
    """Convert A @ B @ C to execution order (C, B, A); no adjoints."""
    return tuple(reversed(tuple(factors)))


NAMED = {label: matrix_factors_to_ops(factors) for label, factors in NAMED_FACTORS.items()}


def core_word_to_ops(word) -> tuple[str, ...]:
    """Compile a complete BS12 product, reversing both syllables and gates."""
    factors = []
    for token in word:
        if token not in SYLLABLES:
            raise ValueError(f"Unknown BS12 syllable: {token}")
        factors.extend(SYLLABLES[token])
    return matrix_factors_to_ops(factors)


def append_ops(qc: QuantumCircuit, qubit: int, ops) -> None:
    for op in ops:
        if op == "i":
            continue
        if op not in {"h", "s", "sdg", "t", "tdg", "x", "y", "z"}:
            raise ValueError(f"Unsupported gate: {op}")
        getattr(qc, op)(qubit)


def circuit_from_ops(ops, name="sequence") -> QuantumCircuit:
    qc = QuantumCircuit(1, name=name)
    append_ops(qc, 0, ops)
    return qc


def remove_global_phase(matrix, tol=TOL):
    matrix = np.asarray(matrix, dtype=complex)
    nonzero = np.flatnonzero(np.abs(matrix.ravel()) > tol)
    if not len(nonzero):
        return matrix.copy()
    pivot = matrix.ravel()[nonzero[0]]
    return matrix / (pivot / abs(pivot))


def phase_normalize_batch(matrices, tol=TOL):
    """Remove only global phase from each matrix, using its first nonzero entry."""
    matrices = np.asarray(matrices, dtype=complex)
    flat = matrices.reshape(*matrices.shape[:-2], -1)
    indices = np.argmax(np.abs(flat) > tol, axis=-1)
    pivots = np.take_along_axis(flat, indices[..., None], axis=-1)[..., 0]
    phases = np.ones_like(pivots)
    np.divide(pivots, np.abs(pivots), out=phases, where=np.abs(pivots) > tol)
    return matrices / phases[..., None, None]


def matrix_key(matrix, decimals=10) -> str:
    canon = remove_global_phase(matrix)
    values = np.stack((canon.real, canon.imag), axis=-1).ravel()
    values = np.round(values, decimals)
    values[values == 0] = 0  # Eliminate negative zero from serialized keys.
    return ",".join(f"{v:.{decimals}f}" for v in values)


def equivalent(a, b, tol=TOL) -> bool:
    return bool(Operator(a).equiv(Operator(b), rtol=0, atol=tol))


def t_count(qc: QuantumCircuit) -> int:
    counts = qc.count_ops()
    return int(counts.get("t", 0) + counts.get("tdg", 0))


@dataclass
class Sequence:
    label: str
    ops: tuple[str, ...]
    circuit: QuantumCircuit = field(init=False, repr=False)
    matrix: np.ndarray = field(init=False, repr=False)
    gate: Gate = field(init=False, repr=False)

    def __post_init__(self):
        self.circuit = circuit_from_ops(self.ops, self.label)
        self.matrix = Operator(self.circuit).data
        # Keep the native gate definition, so exported circuits contain H/S/T.
        # Operator is cached on this custom Gate for repeated candidate evaluation.
        self.gate = SequenceGate(self.circuit, self.matrix)


class SequenceGate(Gate):
    """A native circuit definition with its Qiskit-computed operator cached."""

    def __init__(self, circuit, matrix):
        super().__init__(circuit.name, 1, [])
        self.definition = circuit
        self._matrix = matrix

    def to_matrix(self):
        return self._matrix.copy()


@lru_cache(maxsize=1)
def clifford_group() -> tuple[Sequence, ...]:
    """Build Appendix A's G0...G23 and validate all 24 Qiskit tableaux."""
    entries = tuple(Sequence(f"G{index}", matrix_factors_to_ops(word.lower()))
                    for index, word in enumerate(BS12_CLIFFORD_WORDS))
    tableaux = {Clifford(entry.circuit).tableau.tobytes() for entry in entries}
    if len(tableaux) != 24:
        raise RuntimeError(f"Expected 24 distinct Cliffords; got {len(tableaux)}")
    return entries


@lru_cache(maxsize=1)
def clifford_matrices():
    return np.array([c.matrix for c in clifford_group()])


def canonical_words(t: int):
    """BS12 Sec. II, in matrix order: no SH before syllable five, end in TH."""
    if t < 0:
        raise ValueError("T count must be nonnegative")
    if t <= 4:
        yield ("TH",) * t
    else:
        for tail in product(("TH", "SHTH"), repeat=t - 4):
            yield ("TH",) * 4 + tail


def core_pool(max_t: int) -> dict[int, list[Sequence]]:
    if max_t < 0:
        raise ValueError("max_t must be nonnegative")
    return {t: [Sequence("C[" + "|".join(word) + "]", core_word_to_ops(word))
                for word in canonical_words(t)] for t in range(max_t + 1)}


def allowed_sets(mode="project") -> dict[str, list[Sequence]]:
    """Return historical sets or representatives with checked local coverage.

    Historical modes preserve their physical operation lists. Previously the
    time-ordered names SH/HS/SHX/HSHX denoted the matrices HS/SH/XHS/XHSH.
    The verified mode derives its sets from stabilizer and monomial cosets.
    """
    if mode not in {"project", "fig4", "paper", "verified"}:
        raise ValueError("clifford_sets must be project, fig4, paper or verified")
    named = {label: Sequence(label, ops) for label, ops in NAMED.items()}
    if mode == "verified":
        # D=<S> stabilizes |0>; B=<S,X> normalizes Z. Use genuine
        # C/D, C/B and B\C representatives in matrix-product order.
        # anc_g1 / anc_g4 absorb B gates moved through the respective CZ.
        start = [named[k] for k in ("I", "X", "H", "HX", "SH", "SHX")]
        before = [named[k] for k in ("I", "H", "HS")]
        after = [named[k] for k in ("I", "H", "SH")]
        result = {
            "anc_g0": start, "anc_g1": list(clifford_group()),
            "anc_g2": after, "anc_g3": before,
            "anc_g4": list(clifford_group()), "anc_g5": before,
            "dat_g0": after, "dat_g1": before,
        }
        if not position_coverage(result)["all_slots_cover_clifford_group"]:
            raise RuntimeError("Verified position sets fail Clifford coset coverage")
        return result
    start = [named[k] for k in ("I", "X", "HS", "XHS", "HSH", "XHSH")]
    before = [named[k] for k in ("H", "SH", "HSH")]
    after = [named[k] for k in ("I", "HS", "HSH")]
    if mode == "paper":
        # Fig. 4 labels interpreted as matrix products, just like BS12.
        start = [named[k] for k in ("I", "X", "SH", "SHX", "HSH", "HSHX")]
        before = [named[k] for k in ("H", "HS", "HSH")]
        after = [named[k] for k in ("I", "SH", "HSH")]
    return {
        "anc_g0": start,
        "anc_g1": list(clifford_group()) if mode == "project" else before,
        "anc_g2": after, "anc_g3": before,
        "anc_g4": list(clifford_group()) if mode == "project" else after,
        "anc_g5": before, "dat_g0": after, "dat_g1": before,
    }


def position_coverage(allowed):
    """Check the local absorption argument, beyond the number of labels.

    Coverage is modulo phases, ancilla outcome relabeling and monomial
    Cliffords on external data input/output. It is not exact-U completeness
    and does not identify the paper's original database representatives.
    """
    group = clifford_group()
    expected = {matrix_key(g.matrix) for g in group}
    diagonal = [g.matrix for g in group if abs(g.matrix[0, 1]) < TOL]
    monomial = [g.matrix for g in group if
                abs(g.matrix[0, 1]) < TOL or abs(g.matrix[0, 0]) < TOL]
    slots = {}
    for slot, pool in allowed.items():
        matrices = [g.matrix for g in pool]
        if slot == "anc_g0":
            products = [g @ d for g in matrices for d in diagonal]
            relation = "C = representatives @ <S>"
        elif slot in {"anc_g2", "dat_g0"}:
            products = [g @ b for g in matrices for b in monomial]
            relation = "C = representatives @ <S,X>"
        elif slot in {"anc_g3", "anc_g5", "dat_g1"}:
            products = [b @ g for g in matrices for b in monomial]
            relation = "C = <S,X> @ representatives"
        else:
            products = matrices
            relation = "C = unrestricted CZ absorption slot"
        covered = {matrix_key(matrix) for matrix in products}
        slots[slot] = {"relation": relation, "covered_cliffords": len(covered & expected),
                       "complete": covered == expected}
    return {"scope": "local Clifford absorption modulo external data Cliffords and outcome relabeling",
            "all_slots_cover_clifford_group": all(slot["complete"] for slot in slots.values()),
            "slots": slots}


@dataclass
class Segment:
    sequence: Sequence
    labels: dict[str, str]
    ops: dict[str, list[str]]


def segment_pool(slots, cores, allowed) -> list[Segment]:
    """Execute input -> core -> output, i.e. U = g_out @ C @ g_in.

    In BS12 Proposition 2, g1 = g_out and g2 = g_in. Input/output
    restrictions are supplied by ``allowed`` rather than by the core grammar.
    """
    left, middle, right = slots
    unique = {}
    for a, core, b in product(allowed[left], cores, allowed[right]):
        ops = a.ops + core.ops + b.ops
        seq = Sequence(f"{left}:{a.label}/{core.label}/{b.label}", ops)
        # At fixed T count, global-phase-equivalent segments are interchangeable.
        unique.setdefault(matrix_key(seq.matrix), Segment(
            seq, {left: a.label, middle: core.label, right: b.label},
            {left: list(a.ops), middle: list(core.ops), right: list(b.ops)}))
    return list(unique.values())


def validate_template_form(labels, ops, t_partition) -> None:
    """Verify the symbolic BS12 template as well as its stored execution order."""
    if len(t_partition) != len(SLOTS):
        raise ValueError("A Fig. 3 template requires four T-count partitions")
    cliffords = {entry.label: entry.matrix for entry in clifford_group()}
    cliffords.update({name: Operator(circuit_from_ops(gates)).data
                      for name, gates in NAMED.items()})
    for slots, t in zip(SLOTS, t_partition):
        before, middle, after = slots
        label = labels[middle]
        if not (label.startswith("C[") and label.endswith("]")):
            raise ValueError(f"{middle}: invalid BS12 core label")
        word = tuple(label[2:-1].split("|")) if label != "C[]" else ()
        if (len(word) != t or any(token not in SYLLABLES for token in word)
                or any(token != "TH" for token in word[:4])):
            raise ValueError(f"{middle}: violates BS12 canonical grammar or T count")
        if tuple(ops[middle]) != core_word_to_ops(word):
            raise ValueError(f"{middle}: execution order does not implement its BS12 matrix word")
        for slot in (before, after):
            target = cliffords.get(labels[slot])
            if target is None or not equivalent(Operator(circuit_from_ops(ops[slot])).data, target):
                raise ValueError(f"{slot}: Clifford operations disagree with their matrix label")


def candidate_circuit(a0: Segment, a1: Segment, a2: Segment, d0: Segment):
    qc = QuantumCircuit(2, name="W")
    qc.append(a0.sequence.gate, [ANC])
    qc.cz(ANC, DATA)
    qc.append(a1.sequence.gate, [ANC])
    qc.append(d0.sequence.gate, [DATA])
    qc.cz(ANC, DATA)
    qc.append(a2.sequence.gate, [ANC])
    return qc


def circuit_from_template(template_ops, measure=False) -> QuantumCircuit:
    qc = QuantumCircuit(2, int(measure), name="W")
    for slot in SLOTS[0]:
        append_ops(qc, ANC, template_ops[slot])
    qc.cz(ANC, DATA)
    for slots, qubit in ((SLOTS[1], ANC), (SLOTS[3], DATA)):
        for slot in slots:
            append_ops(qc, qubit, template_ops[slot])
    qc.cz(ANC, DATA)
    for slot in SLOTS[2]:
        append_ops(qc, ANC, template_ops[slot])
    if measure:
        qc.measure(ANC, 0)
    return qc


def rus_loop_circuit(template_ops, success_outcomes, failure_ops, *, attempt=None) -> QuantumCircuit:
    """Build a real Qiskit while_loop/if_test protocol; no hand-written QASM."""
    qc = QuantumCircuit(2, 1, name="RUS")
    # Bootstrap using only the ancilla, leaving the input data state untouched.
    # For two success outcomes, use outcome 0 as the loop's exit flag.
    failure_value = 1 - int(success_outcomes[0])
    qc.reset(ANC)
    if failure_value:
        qc.x(ANC)
    qc.measure(ANC, 0)
    with qc.while_loop((qc.clbits[0], failure_value)):
        qc.reset(ANC)
        qc.compose(circuit_from_template(template_ops) if attempt is None else attempt, inplace=True)
        qc.measure(ANC, 0)
        for outcome, ops in failure_ops.items():
            with qc.if_test((qc.clbits[0], int(outcome))):
                qc.compose(circuit_from_ops(ops).inverse(), [DATA], inplace=True)
        if len(success_outcomes) == 2:
            # Both branches induce the same U, so leave after exactly one attempt.
            qc.reset(ANC)
            if 1 - failure_value:
                qc.x(ANC)
            qc.measure(ANC, 0)
    return qc
