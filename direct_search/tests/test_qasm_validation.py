"""Qiskit file readback, dynamic control-flow integrity and error reporting."""
import contextlib
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qiskit import qasm3

from direct_search.circuits import circuit_from_template, rus_loop_circuit
from direct_search.dsa import SearchConfig
from direct_search.export import export_database
from direct_search.main import main as search_main
from direct_search.validate_qasm import (
    audit_directory, compare_circuits, dump_and_validate,
    main as audit_main, reference_circuit,
)
from test_direct_search import execute_forced_path, known_entry
import numpy as np
from qiskit.quantum_info import Statevector


class QasmReadbackTests(unittest.TestCase):
    def test_all_formats_load_and_compare_for_both_success_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            for swap in (False, True):
                entry = known_entry(swap)
                for kind, version in (("w_qasm3", "qasm3"), ("attempt_qasm3", "qasm3"),
                                      ("rus_loop_qasm3", "qasm3"), ("attempt_qasm2", "qasm2")):
                    with self.subTest(swap=swap, kind=kind):
                        expected = reference_circuit(entry, kind)
                        path = Path(directory) / f"{swap}_{kind}.qasm"
                        result = dump_and_validate(expected, path, version, kind, entry)
                        self.assertEqual(result["status"], "passed", result)
                        self.assertEqual(result["read_function"], f"qiskit.{version}.load")
                        if kind == "rus_loop_qasm3":
                            # Exercise the circuit loaded from the physical file, not the original.
                            loaded = qasm3.load(str(path))
                            success = int(entry["success_outcomes"][0])
                            psi = np.array([np.sqrt(0.3), 1j * np.sqrt(0.7)])
                            state, count = execute_forced_path(loaded, psi, [1 - success] * 3 + [success])
                            U = np.diag([1, (-1 + 2j * np.sqrt(2)) / 3])
                            actual = state.data[success * 2:success * 2 + 2]
                            self.assertTrue(Statevector(actual).equiv(Statevector(U @ psi)))
                            self.assertEqual(count, 4)

    def test_wrong_gate_measurement_and_loop_condition_are_rejected(self):
        entry = known_entry()
        static = circuit_from_template(entry["template_ops"])
        changed = static.copy()
        changed.x(0)
        with self.assertRaisesRegex(ValueError, "unitary matrix differs"):
            compare_circuits(static, changed)
        attempt = circuit_from_template(entry["template_ops"], measure=True)
        changed = static.copy()
        changed.add_register(attempt.cregs[0])
        changed.measure(0, 0)
        with self.assertRaisesRegex(ValueError, "wire mapping differs"):
            compare_circuits(attempt, changed)
        loop = rus_loop_circuit(entry["template_ops"], entry["success_outcomes"], entry["failure_outcome_ops"])
        source = qasm3.dumps(loop)
        self.assertIn("while (c[0])", source)
        wrong = qasm3.loads(source.replace("while (c[0])", "while (!c[0])", 1))
        with self.assertRaisesRegex(ValueError, "classical condition differs"):
            compare_circuits(loop, wrong)

    def test_csv_collects_syntax_semantic_and_missing_file_errors(self):
        entry = known_entry()
        stats = {"config": asdict(SearchConfig())}
        with tempfile.TemporaryDirectory() as directory:
            summary = export_database({"example": entry}, stats, directory, export_qasm2=True)
            self.assertTrue(summary["export_ok"])
            root = Path(directory)
            saved = json.loads((root / "database.json").read_text())["entries"][0]
            files = saved["export_files"]
            # Syntactically invalid QASM 2.
            (root / files["attempt_qasm2"]).write_text('OPENQASM 2.0; this is not a circuit;')
            # Syntactically valid QASM 3, but different from the intended W.
            wpath = root / files["w_qasm3"]
            wpath.write_text(wpath.read_text() + '\nx q[0];\n')
            # A manifest path which is absent must not disappear from the audit.
            (root / files["attempt_qasm3"]).unlink()
            summary = audit_directory(root)
            self.assertEqual(summary["failed"], 3)
            self.assertEqual(summary["passed"], 1)  # physical RUS loop
            with (root / "qasm_errors.csv").open(newline='') as handle:
                errors = list(csv.DictReader(handle))
            self.assertEqual(len(errors), 3)
            self.assertEqual({row["stage"] for row in errors}, {"load", "compare"})
            self.assertIn("FileNotFoundError", {row["error_type"] for row in errors})
            for row in errors:
                self.assertTrue(row["entry_id"])
                self.assertTrue(row["error_message"])
                self.assertTrue(row["template_ops"])
                self.assertTrue(row["source_database"])

    def test_export_errors_do_not_abort_other_formats_and_cli_fails(self):
        entry = known_entry()
        stats = {"config": asdict(SearchConfig())}
        with tempfile.TemporaryDirectory() as directory:
            with patch("direct_search.validate_qasm.qasm3.dump", side_effect=RuntimeError("broken exporter")), \
                 patch("direct_search.main.build_database", return_value=({"example": entry}, stats)), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = search_main(["--max-raw-t", "2", "--export-attempt-qasm2", "--output-dir", directory])
            self.assertEqual(code, 1)
            root = Path(directory)
            summary = json.loads((root / "summary.json").read_text())
            self.assertFalse(summary["export_ok"])
            self.assertEqual(summary["qasm_validation"]["failed"], 3)
            self.assertEqual(summary["qasm_validation"]["passed"], 1)
            with (root / "qasm_errors.csv").open(newline='') as handle:
                errors = list(csv.DictReader(handle))
            self.assertTrue(all(row["error_message"] == "broken exporter" for row in errors))

    def test_missing_importer_is_reported_instead_of_skipped(self):
        entry = known_entry()
        with tempfile.TemporaryDirectory() as directory:
            with patch("direct_search.validate_qasm.qasm3.load", side_effect=ImportError("qiskit_qasm3_import is missing")):
                result = dump_and_validate(reference_circuit(entry, "w_qasm3"),
                                           Path(directory) / "test.qasm", "qasm3", "w_qasm3", entry)
            self.assertEqual((result["status"], result["stage"]), ("failed", "load"))
            self.assertIn("qiskit_qasm3_import", result["error_message"])

    def test_invalid_entries_are_reported_and_other_entries_export(self):
        good = known_entry()
        bad = dict(good, raw_t_count=200)
        with tempfile.TemporaryDirectory() as directory:
            summary = export_database({"bad": bad, "good": good},
                                      {"config": asdict(SearchConfig())}, directory)
            self.assertEqual(summary["invalid_entries"], 1)
            self.assertEqual(summary["validated_entries"], 1)
            self.assertEqual(summary["qasm_validation"]["failed"], 1)
            self.assertEqual(summary["qasm_validation"]["passed"], 3)

    def test_untracked_files_are_labelled_readback_only_and_errors_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'standalone.qasm').write_text('OPENQASM 3.0; include "stdgates.inc"; qubit q; h q;')
            summary = audit_directory(root)
            self.assertEqual(summary["readback_only"], 1)
            self.assertEqual(summary["equivalence_verified"], 0)
            self.assertTrue(summary["ok"])
            with (root / "qasm_errors.csv").open(newline='') as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            (root / 'bad.qasm').write_text('no QASM header')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit_main(["--input-dir", directory]), 1)


if __name__ == '__main__':
    unittest.main()
