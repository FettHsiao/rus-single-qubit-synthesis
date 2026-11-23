"""Executable paper Fig. 8 / Fig. 9 reference circuits and section-6.1 costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from .circuits import equivalent, t_count, rus_loop_circuit
from .dsa import analyze_w, encode_matrix
from .research import v3_cost_estimate
from .validate_qasm import dump_and_validate, write_validation_reports


def paper_circuits():
    smallest = QuantumCircuit(2, name="paper_fig8")
    smallest.h(1)
    smallest.t(1)
    smallest.cx(1, 0)
    smallest.h(1)
    smallest.cx(1, 0)
    smallest.t(1)
    smallest.h(1)
    v3 = QuantumCircuit(2, name="paper_fig9_v3")
    v3.h(1)
    v3.t(1)
    v3.h(1)
    v3.cz(1, 0)
    v3.tdg(1)
    v3.h(1)
    v3.t(1)
    v3.cz(1, 0)
    v3.h(1)
    v3.t(1)
    v3.h(1)
    v3.z(0)
    X = np.array([[0, 1], [1, 0]])
    # V3 = (I + 2i Z)/sqrt(5); Fig. 9 costs 4/(5/8)=6.4 expected T.
    return (("fig8", smallest, (np.eye(2) + 1j*np.sqrt(2)*X)/np.sqrt(3), .75),
            ("fig9_v3", v3, (np.eye(2) + 2j*np.diag([1,-1]))/np.sqrt(5), .625))


def export_examples(output_dir, epsilon=1e-6):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, entries = [], []
    for name, circuit, target, probability in paper_circuits():
        analysis = analyze_w(Operator(circuit).data)
        if not analysis.accepted or not equivalent(analysis.unitary, target) or abs(analysis.probability-probability)>1e-10:
            raise ValueError(f"Paper reference circuit {name} failed analytic validation")
        entry = {"entry_id": name, "raw_t_count": t_count(circuit), "success_probability": probability,
                 "expected_t_count": t_count(circuit)/probability, "unitary_matrix": encode_matrix(target)}
        loop = rus_loop_circuit({}, analysis.success_outcomes, analysis.failure_ops, attempt=circuit)
        for kind, reference in (("w", circuit), ("rus_loop", loop)):
            rows.append(dump_and_validate(reference, output_dir/f"{name}_{kind}.qasm", "qasm3", kind, entry))
        entries.append(entry)
    report = {"circuits": entries, "qasm_validation": write_validation_reports(rows, output_dir),
              "section_6_1_paper_estimate": v3_cost_estimate(epsilon),
              "section_6_1_fig9_estimate": v3_cost_estimate(epsilon, entries[1]["expected_t_count"]),
              "note": "5.26 is the paper's two-ancilla Fig. 1c cost; exported one-ancilla Fig. 9 costs 6.4."}
    (output_dir/"paper_examples.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--epsilon",type=float,default=1e-6)
    args=parser.parse_args(argv)
    result=export_examples(args.output_dir,args.epsilon)
    print(json.dumps(result,indent=2))
    return 0 if result["qasm_validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
