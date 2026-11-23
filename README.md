# RUS Single-Qubit Synthesis

This project is an independent reproduction of Adam Paetznick and Krysta M. Svore's *Repeat-Until-Success: Non-deterministic decomposition of single-qubit unitaries*. It implements a Qiskit-based workflow for finding Clifford+T repeat-until-success (RUS) circuits, reducing their expected T count, and composing them to approximate target single-qubit unitaries. [Paper and bibliographic record](https://arxiv.org/abs/1311.1074v2)

**Author** · Hsien-Fu Hsiao  
**Computer Science Group, Graduate Institute of Electrical Engineering, National Taiwan University**

The workflow covers the one-ancilla direct search in §4, amplitude amplification in §3.3, the statistics in §5 and Fig. 5, and database-based synthesis in §6.2. It also exports the circuits from Figs. 8 and 9 and provides the §6.1 resource estimate. [Paper PDF](https://arxiv.org/pdf/1311.1074v2)

## Installation

Run the commands from the repository root in a POSIX shell. The verified environment is Linux with Python 3.11.15 and Qiskit 2.4.1. The output registry uses POSIX file locking. On Windows, use a Linux environment such as WSL.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Install from [requirements-lock.txt](requirements-lock.txt) in place of [requirements.txt](requirements.txt) to match the exact package versions used for verification. The dependencies include Qiskit's OpenQASM 3 importer and Matplotlib for headless plot generation.

## Quick start

Run a single two-T partition through the pipeline. This generates both direct-search datasets, the reduced dataset, Fig. 5-style plots, and axial and general composition databases.

```bash
OUT="$PWD/outputs/demo"
python run_all.py \
  --max-raw-t 2 --job 0,0,0,2 --workers 2 \
  --axial-max-class 3 --general-max-class 3 --max-pool 200 \
  --gap-fill-iterations 2 \
  --search-chunk-size 256 --reduction-chunk-size 1 \
  --composition-chunk-size 8 --export-chunk-size 8 \
  --direct-dir "$OUT/direct" --amp-dir "$OUT/amplified" \
  --reduction-dir "$OUT/reduced" --fig-dir "$OUT/fig5" \
  --axial-dir "$OUT/axial" --general-dir "$OUT/general"
```

This run checks 31,104 candidates and retains 16 distinct success unitaries per direct dataset. Reduction yields one Clifford class. The axial and general composition datasets contain 4 and 11 entries, respectively, including their zero-T Clifford entry. Completed datasets require a fresh output path for another run.

Query the axial database for the lowest expected T circuit within distance `0.1` of a Z rotation by `0.3` radians.

```bash
python -m composition lookup \
  --input-dir "$OUT/axial" --output-dir "$OUT/lookup" \
  --target-z 0.3 --epsilon 0.1 --workers 2
```

## Method

### Direct search and branch validation

The search partitions the T budget across three ancilla cores and one data core in the two-CZ template of Fig. 3. It enumerates the Bocharov–Svore canonical Clifford+T core words, builds the allowed Clifford-dressed segments, and deduplicates equivalent segments. The code treats gate labels as matrix products. For example, `TH` means `T @ H` and executes as H followed by T. Qiskit qubit 0 is the data qubit and qubit 1 is the ancilla, giving the matrix basis order `|ancilla, data⟩`. [RUS paper, §4](https://arxiv.org/pdf/1311.1074v2) · [Bocharov–Svore canonical form](https://arxiv.org/abs/1206.3223)

The CLI uses the `verified` Clifford sets. The ansatz uses representatives of cosets of the diagonal subgroup $\langle S\rangle$ and the monomial subgroup $\langle S,X\rangle$. The initial ancilla slot covers all six stabilizer states, and the measurement-side representatives cover all three Pauli bases up to outcome relabeling. Two slots retain all 24 single-qubit Cliffords to absorb local gates moved across the CZs. The checks and chosen position sets are recorded in `summary.json`. Slot names below match the code in [circuits.py](direct_search/circuits.py).

| Slots | Clifford representatives in matrix-product order |
| --- | --- |
| `anc_g0` | `I`, `X`, `H`, `HX`, `SH`, `SHX` |
| `anc_g1`, `anc_g4` | All 24 single-qubit Cliffords |
| `anc_g2`, `dat_g0` | `I`, `H`, `SH` |
| `anc_g3`, `anc_g5`, `dat_g1` | `I`, `H`, `HS` |

Given a two-qubit attempt unitary $W$, the measurement block for outcome $m$ is $B_m = \langle m|W|0\rangle$ on the data qubit. The validator checks $B_m^\dagger B_m = p_m I$ for every nonzero branch, so the outcome probabilities are independent of the input state. The default acceptance rule requires a non-Clifford success unitary and a nonzero, Clifford-correctable failure branch. The dynamic circuit measures the ancilla, applies the inverse failure Clifford when needed, resets the ancilla, and retries. The expected cost is $T/p$ for a single attempt with T count $T$ and success probability $p$.

The NumPy backend evaluates batches of candidate matrices. A per-candidate Qiskit backend provides a reference. Direct search compares implementations of the same physical success unitary up to global phase and retains the one with the lowest direct expected T cost. Candidate partitions, bounds, examined counts, and search completeness are recorded in each dataset.

### Oblivious amplitude amplification

Search exports the retained entries twice. The unamplified dataset uses $j=0$. The amplified dataset uses the minimum-cost $j$ in the inclusive range `0..max_j`, favoring fewer rounds on a tie. It builds the amplified attempt from $W$, its adjoint, and ancilla phase reflections. Each reflection is Clifford for one ancilla. With $\theta=\arcsin\sqrt p$, the success probability and cost are

$$
p_j = \sin^2((2j+1)\theta), \qquad T_j = (2j+1)T, \qquad E_j = T_j/p_j.
$$

The optional attempt cap constrains $T_j$. The optimizer compares only the circuits retained by direct search and the specified round range. The resulting amplifier and recovery are verified against the original success unitary and the predicted probability. [RUS paper, §3.3](https://arxiv.org/pdf/1311.1074v2)

### Clifford reduction and axial classification

Reduction groups unitaries under independent left and right single-qubit Cliffords, $V=LUR$ up to global phase. It phase-normalizes at the first nonzero element in the first row, tests all $24\times24$ pairs, and chooses the lexicographic minimum using the real and imaginary parts of the entries in row-major order, with a numerical tolerance. It selects the source circuit in each class by minimum amplified expected T cost and stores its Clifford transformations. The physical attempt and recovery stay paired inside the retry loop. For a representative $V=LUR$, execution order is $R$, the RUS loop for $U$, then $L$. [RUS paper, §6.2](https://arxiv.org/pdf/1311.1074v2)

The repository classifies an entry as axial if independent $L$ and $R$ can make it diagonal. For example, $H R_z(\theta)$ qualifies using $L=H$ and $R=I$. The axial witness is stored separately from the lexicographic representative. The same criterion is used across direct search, reduction, composition, and lookup.

The Fig. 5 plotter takes a completed reduction dataset. It compares the original and amplified expected T costs of the same winning circuits and plots their raw T counts by success-probability bin. It writes full-range and expected-T-through-100 panels in PNG, PDF, and SVG, alongside CSV distributions and `plot_summary.json`.

### Composition, lookup, and gearbox correction

A class-$k$ circuit concatenates $k$ RUS components. Class 0 represents free single-qubit Clifford operations. Each component retries locally, making the total expected T count the sum of its components' expected costs. The general builder expands products $A C B$ over all 24 middle Cliffords, absorbing the outer Cliffords into the canonical representative and the executable recipe. The `g2` canonicalizer uses 64 pairs and handles diagonal phases analytically, with a fallback to the full 576-pair calculation on singular boundaries. The axial builder combines angle sums and differences, folds them to $[0,\pi/4]$ with Clifford wrappers, and can selectively fill the largest gaps. [RUS paper, §§6.2.1 and 6.2.3](https://arxiv.org/pdf/1311.1074v2)

Lookup chooses the lowest expected T circuit within the requested phase-insensitive distance $D(U,V)=\sqrt{1-|\operatorname{Tr}(U^\dagger V)|/2}$ from Eq. 27. It uses sorted angles for the axial database and an SU(2) quaternion KD-tree over the target's Clifford images for the general database. The chosen recipe is mapped back to the target's Clifford frame. The `--target-json` option accepts a 2×2 matrix with each complex entry encoded as `[real, imag]`, either directly or in a `unitary_matrix` object field. [RUS paper, Eq. 27](https://arxiv.org/pdf/1311.1074v2)

The hybrid synthesizer starts with a coarse axial match and searches nested single-input gearboxes for a correction. A gearbox maps a seed angle $a$ to $2\arctan(\tan^2(a/2))$, with single-attempt probability $\cos^4(a/2)+\sin^4(a/2)$. Each attempt invokes two child circuits, making its expected cost $2E_{\rm child}/p$. The search chooses the lowest-cost feasible correction for the selected coarse match. The separate `hybrid-estimate` command evaluates the paper's resource formulas and reports `circuit_synthesized: false`. [RUS paper, §6.2.2](https://arxiv.org/pdf/1311.1074v2) · [Wiebe–Kliuchnikov gearbox construction](https://arxiv.org/abs/1305.5528)

## Running the stages

Run a command with `--help` for its complete argument list. Use an absolute output path when passing the same path to multiple stages. Relative direct-search outputs are based under `direct_search/`. Relative paths supplied to the other commands are based on the current working directory.

| Command | Inputs, outputs, and controls |
| --- | --- |
| `python -m direct_search` | Produces unamplified and amplified schema-3 datasets. Use `--max-raw-t`, `--job`, `--estimate-only`, `--backend`, `--amp-max-j`, and `--amp-max-raw-t` to control the search. `--export-attempt-qasm2` adds single-attempt QASM 2. `--no-qasm` is for JSON/CSV-only runs without circuit-dependent downstream stages. |
| `python -m amplify` | Takes a direct dataset via `--input-dir` and writes a new dataset to `--output-dir`. Use `--max-j` and `--max-raw-t` to recompute with different amplification bounds. |
| `python -m reduction` | Takes the amplified dataset via `--input-dir` and writes one winner per Clifford class to `--output-dir`. Requires a positive amplification search bound, even when $j=0$ won for every entry. Use `--stage analyze` and `--stage export` with the same paths to save and resume the analysis checkpoint. |
| `python -m reduction.plot_fig5` | Takes a completed reduction dataset via `--input-dir` and writes plots and tables to `--output-dir`. `--bin-width` controls the expected-T histogram. |
| `python -m composition build` | Takes a reduced or direct dataset with circuits via `--input-dir` and writes a composition dataset to `--output-dir`. Use `--mode axial\|all`, `--max-class`, `--max-expected-t`, `--max-pool`, and `--base-limit`. A value of 0 removes the pool or seed cap. This command also supports `--stage analyze` and `--stage export`. |
| `python -m composition lookup` | Takes a composition dataset and a target via `--target-z` in radians or `--target-json`. Writes the selected circuit and `lookup.json` to `--output-dir`. |
| `python -m composition hybrid` | Takes an axial composition dataset and `--target-z`. Searches up to `--max-gearbox-depth` with `0 < epsilon < coarse-epsilon <= 1`. Writes a dynamic QASM 3 circuit and QPY bundle on success. |
| `python -m composition hybrid-estimate` | Takes the same axial input, target, and tolerances. Writes `hybrid_estimate.json`. |

A lookup or hybrid search without a qualifying circuit writes `lookup.json` with `found: false` and exits with status 2. For a working hybrid example using the quick-start database, the target below is its class-1 axial angle plus a one-level gearbox correction from the same seed.

```bash
python -m composition hybrid \
  --input-dir "$OUT/axial" --output-dir "$OUT/hybrid" \
  --target-z 0.398694415401203 --coarse-epsilon 0.03 --epsilon 1e-9 \
  --max-gearbox-depth 2 --workers 2
```

Run `python run_all.py --dry-run` to inspect the default pipeline. With no options, it searches up to 15 T gates and builds class-6 axial and class-3 general pools with a 100,000-entry cap and an expected T cap of 30. Start with `python -m direct_search --max-raw-t 15 --estimate-only` to estimate the search size. Pool construction itself can be expensive at high T. Most search and export stages support `--workers 0` for automatic worker sizing and `--workers 1` for serial operation.

The pipeline supports `--skip-direct` to reuse existing data, `--skip-amplification` with `--skip-direct` to require an existing amplified dataset, and `--skip-reduction` to reuse a completed reduction. Set fresh output paths for the stages that will run again. To run optional synthesis from the pipeline, supply `--target-z` and `--epsilon`, with `--hybrid-epsilon` for tighter gearbox synthesis and resource estimation.

## Source layout

| Path | Role |
| --- | --- |
| [run_all.py](run_all.py) | Pipeline orchestration and stage reuse |
| [direct_search/](direct_search/) | Candidate generation, verification, and direct circuit export |
| [amplify.py](amplify.py) | Amplification of an existing direct dataset |
| [reduction/](reduction/) | Clifford equivalence, Fig. 5 plots, and dataset readers |
| [composition/](composition/) | Composition recipes, target lookup, and gearbox synthesis |
| [clean_outputs.py](clean_outputs.py) | Preview and removal of generated outputs |

## Dataset and circuit formats

Every database records the algorithm, schema version, entries, and run statistics in `database.json`. Summary JSON and CSV report counts and costs. Entries reference circuit files by relative path under the dataset directory. Unitary matrices use `[real, imag]` pairs. Reduction adds `analysis.json` and `membership.jsonl` with class assignments and the Clifford operations for restoring each source unitary. Composition databases include component references and an execution recipe.

| Field or export | Meaning |
| --- | --- |
| `raw_t_count`, `success_probability` | T gates and success probability of the original attempt in direct and reduced datasets. Composed circuits retry locally and track total cost separately. |
| `expected_t_count`, `effective_t_count` | Original $T/p$ and optimized $T_j/p_j$ in direct and reduced datasets. In a composed dataset, `expected_t_count` is the additive cost of the chosen components. |
| `amplification_rounds`, `amplified_raw_t_count`, `amplified_success_probability` | Selected $j$, amplified T gates per attempt, and amplified success probability. |
| `w_qasm2` or `w_qasm3` | Unmeasured two-qubit attempt unitary $W$. |
| `attempt_qasm2` or `attempt_qasm3` | One attempt ending in an ancilla measurement. |
| `rus_loop_qasm3`, `physical_rus_loop_qasm3`, `target_rus_loop_qasm3` | Dynamic retry loops. In reduction, the target loop includes the representative's Clifford wrappers. |
| `amplified_target_rus_loop_qasm3` | Target loop implementing the effective cost in a reduced entry with $j>0$. |
| `qpy`, `qpy_indices` | Bundle and circuit-name index in reduced, composed, and hybrid datasets. `target_unitary_reference` is a measurement-free reference for matrix comparisons. |

For example, read the reduced circuit with the implementation used for its effective cost. Run Python from the repository root.

```python
from pathlib import Path
from qiskit import qasm3
from reduction.dataset import load_database

root = Path("outputs/demo/reduced")
entry = load_database(root)["entries"][0]
kind = (
    "amplified_target_rus_loop_qasm3"
    if entry["amplification_rounds"] > 0
    else "target_rus_loop_qasm3"
)
circuit = qasm3.load(str(root / entry["export_files"][kind]))
print(circuit)
```

The loop is a dynamic Qiskit `QuantumCircuit`. For an `Operator` comparison, use the unmeasured $W$ or the QPY `target_unitary_reference`. The latter represents the resulting single-qubit matrix and serves as a numerical reference. Export validation reloads QASM and QPY files with Qiskit and compares operators and control flow, including measurement, reset, and recovery. [Qiskit OpenQASM 3 API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qasm3)

## Validation and diagnostics

Run the test suites from the repository root.

```bash
for suite in direct_search/tests reduction/tests composition/tests tests; do
  python -m unittest discover -s "$suite" -v || exit
done
```

The tests cover Clifford position coverage, matrix and branch validation, forced failure/recovery/retry paths, amplification, serial and parallel consistency, orbit reduction, approximation lookup, nested gearbox circuits, and circuit serialization. A separate six-T run with `--job 2,2,2,0` examined 466,560 candidates, retained 80 distinct success unitaries, and produced nine independently audited Clifford classes.

Run these diagnostics on the quick-start datasets.

```bash
python -m direct_search.validate_qasm --input-dir "$OUT/amplified" --report-dir "$OUT/readback"
python -m direct_search.audit_paper "$OUT/direct" --output-dir "$OUT/direct-audit"
python -m reduction.audit "$OUT/reduced"
python -m direct_search.examples --output-dir "$OUT/paper-examples"
python -m direct_search.benchmark \
  --max-raw-t 2 --job 0,0,0,2 --max-examined 9000 \
  --workers 1,2 --include-qiskit --output "$OUT/benchmark.json"
```

The direct audit reconstructs attempt matrices independently and reports different Clifford-orbit counts. The reduction audit checks class uniqueness in SO(3), the chosen cost winners, and the stored Clifford transformations. The benchmark compares the search backends and worker counts on identical inputs and excludes export time.

## Reproduction scope

The Fig. 4 Clifford labels, interpreted as matrix products in this implementation, omit two initial stabilizer states and leave gaps in several position cosets. This is why the CLI uses the checked position sets above. Those checks establish local coverage modulo phase, outcome relabeling, and allowed external Clifford dressing. They leave open which physical unitaries appear as the selected representatives. The repository's axial classification uses independent Clifford factors, while §5 describes the paper's reported classification in terms of Clifford conjugation. [RUS paper, §§4–5](https://arxiv.org/pdf/1311.1074v2)

The paper reports 2,194 circuits from its search, including circuits from preliminary searches outside Fig. 3. The generated databases and Fig. 5-style plots describe the search space, reduction, and limits used here. A `search_complete: true` flag refers to the scheduled partitions. A single `--job` can therefore complete successfully within one partition. Candidate caps and composition pool limits affect coverage, and minimum-cost lookup is relative to the retained database. Validation used bounded searches. The full verified search through 15 T gates and the paper's empirical scaling fits remain unverified here. [RUS paper, §§5–6](https://arxiv.org/pdf/1311.1074v2)

## Cleaning generated files

Default and demo output directories, Python caches, and the local output registry are excluded from Git. Preview cleanup from the repository root before applying it.

```bash
python -B clean_outputs.py --all
python -B clean_outputs.py --all --yes
```

`--all` discovers generated datasets and caches inside the repository. `--dirs PATH [PATH ...]` selects registered output directories explicitly. The commands above cover the registered pipeline outputs. For the quick-start run and standalone diagnostic files, the entire `outputs/demo/` directory is disposable.

## License

This implementation is released under the MIT License. See [LICENSE](LICENSE).

## Citation

Cite the original paper for the RUS method and credit Hsien-Fu Hsiao for this implementation.

```bibtex
@article{paetznick2014repeat,
  title   = {Repeat-Until-Success: Non-deterministic decomposition of single-qubit unitaries},
  author  = {Paetznick, Adam and Svore, Krysta M.},
  journal = {Quantum Information and Computation},
  volume  = {14},
  number  = {15--16},
  pages   = {1277--1301},
  year    = {2014},
  doi     = {10.26421/QIC14.15-16-2},
  eprint  = {1311.1074},
  archivePrefix = {arXiv}
}
```
