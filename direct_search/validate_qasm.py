"""Read QASM with Qiskit, compare circuit semantics, and report every failure."""
from __future__ import annotations

import argparse
import csv
from importlib import metadata
import json
from pathlib import Path
import re
import sys

import qiskit
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit import ClassicalRegister, Clbit
from qiskit.quantum_info import Operator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from direct_search.circuits import TOL, circuit_from_template, rus_loop_circuit

REPORT_FIELDS = (
    "entry_id", "file_path", "format", "kind", "status", "stage", "read_function",
    "verification", "num_qubits", "num_clbits", "error_type", "error_message",
    "export_group", "raw_t_count", "success_probability", "unitary_key",
    "fig3_t_partition", "template_labels", "template_ops", "source_database",
)


def validation_record(path, version="", kind="", entry=None, source_database=""):
    """Create a CSV-ready record with enough metadata to identify a circuit."""
    entry = entry or {}
    row = dict.fromkeys(REPORT_FIELDS, "")
    row.update(entry_id=entry.get("entry_id", Path(path).stem),
               file_path=str(Path(path).resolve()), format=version, kind=kind,
               status="failed", source_database=str(source_database))
    for field in ("export_group", "raw_t_count", "success_probability", "unitary_key"):
        row[field] = entry.get(field, "")
    for field in ("fig3_t_partition", "template_labels", "template_ops"):
        if field in entry:
            row[field] = json.dumps(entry[field], ensure_ascii=False)
    return row


def record_error(row, stage, error):
    """Preserve exception type and message without stopping other circuits."""
    row.update(status="failed", stage=stage, error_type=type(error).__name__,
               error_message=str(error))
    return row


def _condition_key(condition, circuit, cmap):
    """Normalize imported bool/bit and single-bit register comparisons."""
    if not isinstance(condition, tuple) or len(condition) != 2:
        raise ValueError(f"Unsupported classical condition for verification: {condition!r}")
    target, value = condition
    if isinstance(target, Clbit):
        bits = (cmap[circuit.find_bit(target).index],)
    elif isinstance(target, ClassicalRegister):
        bits = tuple(cmap[circuit.find_bit(bit).index] for bit in target)
    else:
        raise ValueError(f"Unsupported classical condition target: {target!r}")
    return bits, int(value)


def _circuit_events(circuit, qmap=None, cmap=None, width=None):
    """Represent dynamic circuits as unitary runs and mapped control events.

    QASM importers can use different Gate classes or block-local bit widths.
    Compare operators between nonunitary boundaries, with all wires mapped to
    the root circuit. Equal loops and branch bodies imply equal repeated action;
    this does not require simulating an arbitrary number of retries.
    """
    qmap = list(range(circuit.num_qubits)) if qmap is None else qmap
    cmap = list(range(circuit.num_clbits)) if cmap is None else cmap
    width = circuit.num_qubits if width is None else width
    events = []
    run = QuantumCircuit(width)

    def flush():
        nonlocal run
        if run.data:
            events.append(("unitary", Operator(run)))
            run = QuantumCircuit(width)

    for instruction in circuit.data:
        op = instruction.operation
        wires = tuple(qmap[circuit.find_bit(q).index] for q in instruction.qubits)
        bits = tuple(cmap[circuit.find_bit(c).index] for c in instruction.clbits)
        if op.name == "barrier":
            continue
        if op.name in {"if_else", "while_loop"}:
            flush()
            condition = _condition_key(op.condition, circuit, cmap)
            blocks = [_circuit_events(block, list(wires), list(bits), width) for block in op.blocks]
            if op.name == "if_else" and len(blocks) == 1:
                blocks.append([])
            events.append((op.name, condition, blocks))
        elif op.name in {"reset", "measure"}:
            flush()
            events.append((op.name, wires, bits))
        else:
            if getattr(op, "blocks", None) or bits or getattr(op, "condition", None) is not None:
                raise ValueError(f"Unsupported operation for verification: {op.name}")
            # Operator raises for unsupported/nonunitary instructions. Never skip them.
            run.unitary(Operator(op), list(wires))
    flush()
    return events


def _compare_events(expected, actual, tol, location="circuit"):
    if len(expected) != len(actual):
        raise ValueError(f"{location}: event count differs ({len(expected)} != {len(actual)})")
    for index, (left, right) in enumerate(zip(expected, actual)):
        where = f"{location}/event[{index}]"
        if left[0] != right[0]:
            raise ValueError(f"{where}: operation differs ({left[0]} != {right[0]})")
        if left[0] == "unitary":
            if not left[1].equiv(right[1], rtol=0, atol=tol):
                raise ValueError(f"{where}: unitary matrix differs after QASM readback")
        elif left[0] in {"while_loop", "if_else"}:
            if left[1] != right[1]:
                raise ValueError(f"{where}: classical condition differs ({left[1]} != {right[1]})")
            if len(left[2]) != len(right[2]):
                raise ValueError(f"{where}: control-flow branch count differs")
            for branch, (a, b) in enumerate(zip(left[2], right[2])):
                _compare_events(a, b, tol, f"{where}/{left[0]}[{branch}]")
        elif left != right:
            raise ValueError(f"{where}: measurement/reset wire mapping differs ({left} != {right})")


def compare_circuits(expected, actual, tol=TOL):
    """Check unitary action, measured/reset wires, and recursive while/if logic."""
    if (expected.num_qubits, expected.num_clbits) != (actual.num_qubits, actual.num_clbits):
        raise ValueError("Quantum/classical bit counts differ after QASM readback")
    _compare_events(_circuit_events(expected), _circuit_events(actual), tol)


def reference_circuit(entry, kind):
    """Reconstruct the intended circuit from the stored primitive template."""
    if kind.startswith("amplified_"):
        from direct_search.amplification import amplified_entry_circuit
        attempt = amplified_entry_circuit(entry)
        if kind in {"amplified_w_qasm3", "amplified_w"}:
            return attempt
        loop = rus_loop_circuit(entry["template_ops"], entry["success_outcomes"],
                                entry["failure_outcome_ops"], attempt=attempt)
        if kind in {"amplified_rus_loop_qasm3", "amplified_rus_loop"}:
            return loop
        raise ValueError(f"Unsupported circuit kind: {kind}")
    if kind in {"w", "w_qasm3"}:
        return circuit_from_template(entry["template_ops"])
    if kind in {"attempt", "attempt_qasm3", "attempt_qasm2"}:
        return circuit_from_template(entry["template_ops"], measure=True)
    if kind in {"rus_loop", "rus_loop_qasm3"}:
        return rus_loop_circuit(entry["template_ops"], entry["success_outcomes"], entry["failure_outcome_ops"])
    raise ValueError(f"Unknown exported circuit kind: {kind}")


def validate_qasm_file(path, expected=None, version=None, kind="", entry=None, tol=TOL,
                       source_database=""):
    """Use the stable qasm2.load/qasm3.load API, then compare with the reference."""
    row = validation_record(path, version or "", kind, entry, source_database)
    try:
        if version is None:
            source = Path(path).read_text(encoding="utf-8-sig")
            match = re.search(r"^\s*OPENQASM\s+([23])(?:\.\d+)?\s*;", source, re.MULTILINE)
            if match is None:
                raise ValueError("Missing or unsupported OPENQASM 2/3 version declaration")
            version = "qasm" + match.group(1)
        if version not in {"qasm2", "qasm3"}:
            raise ValueError(f"Unsupported format: {version}")
        row.update(format=version, read_function=f"qiskit.{version}.load")
    except Exception as exc:
        return record_error(row, "detect_format", exc)
    try:
        loaded = (qasm2 if version == "qasm2" else qasm3).load(str(path))
        row.update(num_qubits=loaded.num_qubits, num_clbits=loaded.num_clbits)
    except Exception as exc:
        return record_error(row, "load", exc)
    if expected is not None:
        try:
            compare_circuits(expected, loaded, tol)
        except Exception as exc:
            return record_error(row, "compare", exc)
    row.update(status="passed", stage="complete",
               verification="control_flow_and_operator_equivalence" if expected is not None else "readback_only")
    return row


def dump_and_validate(circuit, path, version, kind, entry, tol=TOL):
    """Serialize via Qiskit dump(), close the file, then read that file back."""
    row = validation_record(path, version, kind, entry)
    try:
        writer = {"qasm2": qasm2.dump, "qasm3": qasm3.dump}[version]
        with Path(path).open("w", encoding="utf-8") as handle:
            writer(circuit, handle)
    except Exception as exc:
        return record_error(row, "export", exc)
    return validate_qasm_file(path, circuit, version, kind, entry, tol)


def write_validation_reports(rows, output_dir):
    """Always write all-results CSV, errors-only CSV and a JSON summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = [row for row in rows if row["status"] != "passed"]
    for filename, records in (("qasm_validation.csv", rows), ("qasm_errors.csv", failed)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(records)
    try:
        importer_version = metadata.version("qiskit-qasm3-import")
    except metadata.PackageNotFoundError:
        importer_version = None
    summary = {"ok": not failed, "records_checked": len(rows), "passed": len(rows) - len(failed),
               "failed": len(failed),
               "readback_only": sum(row["verification"] == "readback_only" for row in rows),
               "equivalence_verified": sum(row["verification"] == "control_flow_and_operator_equivalence"
                                            for row in rows),
               "qiskit_version": qiskit.__version__, "qasm3_importer_version": importer_version,
               "results_csv": str((output_dir / "qasm_validation.csv").resolve()),
               "errors_csv": str((output_dir / "qasm_errors.csv").resolve())}
    (output_dir / "qasm_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def audit_directory(input_dir, report_dir=None, tol=TOL):
    """Validate existing files and missing manifest paths, continuing on errors.

    Files listed in database.json are compared with their reconstructed circuit.
    Unlisted files are still read, but explicitly marked readback_only.
    """
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    rows, references = [], {}
    for database_path in sorted(root.rglob("database.json")):
        try:
            payload = json.loads(database_path.read_text(encoding="utf-8"))
            for entry in payload["entries"]:
                for kind, relative in entry.get("export_files", {}).items():
                    if kind.endswith(("qasm2", "qasm3")):
                        references[(database_path.parent / relative).resolve()] = (entry, kind, database_path)
        except Exception as exc:
            rows.append(record_error(validation_record(database_path, "manifest"), "manifest", exc))
    paths = {p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".qasm", ".qasm2", ".qasm3"}}
    for path in sorted(paths | set(references)):
        reference = references.get(path)
        if reference is None:
            rows.append(validate_qasm_file(path, tol=tol))
            continue
        entry, kind, database_path = reference
        version = "qasm2" if kind.endswith("qasm2") else "qasm3"
        try:
            expected = reference_circuit(entry, kind)
        except Exception as exc:
            rows.append(record_error(validation_record(path, version, kind, entry, database_path), "build_reference", exc))
            continue
        rows.append(validate_qasm_file(path, expected, version, kind, entry, tol, database_path))
    return write_validation_reports(rows, report_dir or root)


def main(argv=None):
    cli = argparse.ArgumentParser(description="Read exported QASM 2/3 with Qiskit and record every failure.")
    cli.add_argument("--input-dir", type=Path, required=True)
    cli.add_argument("--report-dir", type=Path, help="Default: input directory")
    args = cli.parse_args(argv)
    if not args.input_dir.is_dir():
        cli.error(f"Directory does not exist: {args.input_dir}")
    summary = audit_directory(args.input_dir, args.report_dir)
    print(json.dumps(summary, indent=2))
    if not summary["ok"]:
        print(f"QASM validation failed for {summary['failed']} record(s): {summary['errors_csv']}", file=sys.stderr)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
