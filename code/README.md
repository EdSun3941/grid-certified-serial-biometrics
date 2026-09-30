# Code package: grid-certified posynomial FAR–FRR envelopes for serial multibiometric threshold design

Reproduces every number, table and figure of the manuscript "Grid-Certified Posynomial FAR–FRR Envelopes for Serial
Multibiometric Threshold Design via Lagrangian Relaxation-Based Branch-and-Bound" (Springer version for the International
Journal of Information Security, `../paper_ijis`, main text `main.tex`, Online Resource 1 `ESM_1.tex`) and of its
supplementary material, and of an earlier IEEE-format draft of the same work (`../paper`). All reported values come from CSV files written by these scripts; every table CSV carries a
`source` column naming its inputs, and `verify_numbers.py` recomputes the numbers quoted in the text.

## Environment
- Python 3.11 (tested 3.11.15), numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, scikit-learn 1.8.0, matplotlib 3.10.9.
- No commercial solver: quadratic programs and GPs (log space) by SciPy SLSQP, linear and mixed-integer linear programs by
  HiGHS through `scipy.optimize.linprog` / `scipy.optimize.milp`, non-negative least squares by `scipy.optimize.nnls`.
- Hardware used: 2 vCPU, 7.8 GB RAM (Linux). Timing experiments (E2, E2v2, E2miqp, E7, E7v2) ran on one core with no other job.
- IJIS v02: splits 12–29 of D2/D3 and splits 2–9 of D4 ran on a 24-core PC (Windows, Python 3.12.5, same numpy/scipy/
  scikit-learn versions, pandas 2.3.3) with `run_local2.py`; see `results/E3fresh_provenance.csv`.

## Data
NIST Biometric Scores Set Release 1 (BSSR1), free download from NIST
(https://www.nist.gov/itl/iad/image-group/nist-biometric-scores-set-bssr1). The raw score files are not redistributed.
```
python prepare_bssr1.py <bssr1_root> ../data/processed chunk <subset> <matcher> <row_start> <row_stop>   # repeat per chunk
python prepare_bssr1.py <bssr1_root> ../data/processed combine <subset>
```
Subsets: `fing_x_face` (D1: 517 subjects, face C/G + left/right index), `fing_x_fing` (D2: 6000 subjects, two index
fingers), `face_x_face` (D3: 6000 probes x 3000 enrollees, two face matchers). Scores are similarities (accept iff s >= t);
BSSR1 failure scores (-1) are kept as the lowest score. The scripts look for `bssr1_<subset>.npz` in `$BSSR1_DIR`, else
`../data`, else `../data/processed`. SHA-256 of the processed files used for the paper:
```
25d2b08f196301fd9d9584012838a5b4da9d228e2f74631edf4ecb1c4ed50fe9  bssr1_fing_x_face.npz
27936c42f43f9b012922d2f3e8f846b741b902844aedaa831c78a02824dea928  bssr1_fing_x_fing.npz
42bd6ca98015fd9e831e26469317be82b16e00156be00e7c6e56695b43919817  bssr1_face_x_face.npz
```

## One-command reproduction
```
bash run_all.sh [NPROC]     # default NPROC=2; resumable (finished outputs are skipped)
```
Order: E1+E3 (`jobs_main.txt`), E3-cal (`jobs_cal.txt`), ablations/sensitivity/SPRT (`jobs_abl.txt`), E3b
(`jobs_cal2.txt`), timing E2+E7 (`jobs_time.txt`) and E2v2+E7v2 (`jobs_time2.txt`) on one core, then the analysis,
table, figure and verification scripts listed in `run_all.sh`. Outputs: `../results/<experiment>/*.csv`,
`../results/tables/*.csv`, `../results/figures/*.pdf`, `../paper/tables/*.tex`, `../paper/supp_tables.tex`.
Wall-clock on the hardware above: roughly one day in total (E3b and the 600-s-limit timing runs dominate).

## Experiments and where they appear (numbering of the IJIS manuscript and Online Resource 1)
| Experiment | Script | Output | Paper |
|---|---|---|---|
| E1 envelope fitting, E3 uncalibrated designs | `run_main.py` | `results/E1`, `results/E3` | Sect. 6.1, Table 5, Tables S3, S8 |
| E3-cal pair-level Clopper–Pearson calibration | `run_cal.py` | `results/E3cal` | Fig. 5, Table S4 |
| E3b subject-bootstrap calibration, all baselines | `run_cal2.py` | `results/E3b` | Table 3, Figs. 4–5, Sects. 6.3–6.5 |
| E2 solver comparison (subgradient LR, enumeration, DE) | `run_e2.py` | `results/E2` | Table 2 |
| E2v2 exact-dual LR-BB and MILP on the same fitting sets | `run_e2v2.py` | `results/E2v2` | Table 2, Sect. 6.2, Table S1 |
| E7 scalability (subgradient LR-BB, enumeration) | `run_e7.py` | `results/E7` | Fig. 3, Table S9 |
| E7v2 scalability (exact-dual LR-BB, MILP) | `run_e7v2.py` | `results/E7v2` | Fig. 3, Sect. 6.2 |
| Ablations, sensitivity, SPRT (preliminary implementation) | `run_main.py` options, `run_sprt.py` | `results/E3`, `results/S3` | Sect. 6.6, Table S11 |
| Independent exact reference (all exponent pairs, N = 2) | `reference_exact.py` | `results/E2ref` | Table 2, Sect. 6.2 |
| Refit of the system envelopes by the exact-dual LR-BB | `refit_check.py` | `results/E1xd` | Sect. 5.3, Table S14 |
| E3b LR-BB rows with the refitted envelopes | `run_cal2_xd.py` | `results/E3b` (earlier rows kept in `results/E3b_round1`) | Table 3, Figs. 4–5 |
| E8 stage-budget designs (GP vs. SPRT) | `run_cost.py` | `results/E8` | Sect. 6.5, Table 7 |
| Spoof analysis per matcher | `spoof_rev.py` | `T_rev_spoof.csv` | Sect. 6.5 (per-algorithm values) |
| Spoof analysis per trait | `spoof_trait.py` | `T_rev_spoof_trait.csv`, `T_rev_cost_spoof_trait.csv` | Tables 6 and 7 |
| Calibration diagnostic | `calib_diag.py` | `T_rev_calib_diag.csv` | Sect. 6.4, Table S2 |
| Continuous monomial / convexified support | `check_monoc_weights.py` | `T_rev_monoc.csv` | Sect. 6.2 |

Solver corrections made during internal review: the coefficient subproblem (`solve_alpha`) now works on columns
scaled to a maximum of one, the exact-dual LR-BB bounds every leaf by the exact dual with singleton exponent sets, ranks
incumbent candidates by their contribution, and prunes a P2 node whose linear program is infeasible under the cutoff bounds.
The results of the preliminary implementation in E2/E7 (subgradient LR-BB, enumeration) are kept unchanged and compared
with the independent reference. `rerun_milp_e7.py` belongs to an earlier fix and is not needed for a fresh run;
`envelope_solvers.py` before these corrections is kept in `dev/envelope_solvers_before_round2.py`. `dev/` holds
diagnostics used during development (not needed for the paper) and `dev/verify_numbers_v1.py`, the number check of an
earlier manuscript version.

## Modules
| File | Content |
|---|---|
| `data.py` | data location, subject-disjoint splits, empirical ROC, operating staircase, corner points, log-FAR thinning |
| `reference_exact.py` | independent exact solution of every exponent pair (2-D QP in closed form, scaled LP) |
| `envelope_solvers.py` | coefficient QP/LP, subgradient LR and LR-BB of the earlier work, **exact (convexified) Lagrangian dual**, **exact-dual LR-BB** (`fit_exact_dual`), P2 MILP (`fit_milp_p2`), baselines (hyperbola, monomial, enumeration, NLS, differential evolution, max-monomial) |
| `serial_design.py` | GP serial dual-threshold model, threshold recovery, joint simulation, calibration (Clopper–Pearson stage margins, subject-level bootstrap of the joint FAR), serial rules (Marcialis, symmetric rejection, direct search incl. coordinate descent, SPRT) and parallel fusion (sum, LLR, logistic regression) |
| `experiment_core.py` | constraint-generation fitting, design/evaluation helpers |
| `analyze_rev.py` | order selection by training FRR with tie averaging, corrected resampled t-test (Nadeau–Bengio), Holm within the primary family |
| `conservativeness_rev.py`, `decomposition_rev.py`, `reject_far_check.py` | FAR compliance / prediction coverage, per-design error decomposition, reject-threshold location |
| `analyze.py`, `analyze_all.py`, `dataset_stats.py`, `prop1_check.py`, `ablation_compare.py` | first-version tables still used by the paper |
| `make_main_tables.py`, `make_supp_tables.py` | LaTeX table bodies generated directly from the CSV files |
| `figures.py` | all figures (`python figures.py all`), vector PDF, fonts >= 8 pt |
| `verify_numbers.py` | recomputes every number quoted in the manuscript and checks that generated tables are current |
| `prepare_bssr1.py` | BSSR1 parser |
| `launch.sh`, `run_all.sh`, `jobs_*.txt` | job runner and job lists |

## Seeds
Subject splits use `numpy.random.default_rng(seed)`, seeds 0–9 (E2/E2v2: 0–4; E7/E7v2: 0; sensitivity, training fraction
and N-ablation: 0–4). Differential evolution and NLS multistart use the split seed; the subject bootstrap uses seed
1000 + split seed with B = 300 replicates; the spoof analysis draws genuine scores with seed 7 + split seed.

## License and archiving
MIT license. Public repository: https://github.com/EdSun3941/grid-certified-serial-biometrics (the manuscript
sources `../paper` and `../paper_ijis` are not part of the public repository).

## Springer (IJIS) version
The IJIS manuscript uses the same experiments. Additional or format-specific steps (all in `run_all.sh`):
- `spoof_trait.py`: presentation-attack exposure per trait (a face artifact replaces both face scores, because face
  matchers C and G compare the same image); writes `T_rev_spoof_trait.csv` (Table 6) and `T_rev_cost_spoof_trait.csv`
  (spoof column of Table 7).
- `PAPER=../paper_ijis python3 make_main_tables.py` / `make_supp_tables.py`: table bodies with superscript-letter
  footnotes, the spoof and stage-cap tables in the main text, and Online Resource tables numbered S1-S18 in citation order.
- `FIGSTYLE=springer python3 figures.py all`: figures at 84 mm / 174 mm width with sans-serif lettering, copied to
  `../paper_ijis/figures/Fig1-Fig5` (Fig. 1 is `../paper_ijis/figsrc/Fig1_pipeline.tex`, compiled with pdflatex);
  EPS versions are produced with `pdftops -eps -level3`.
- `make_springer_refs.py`: `../paper_ijis/references.tex` (numbered in order of first citation, full journal titles,
  DOI links) from `../paper_ijis/refs.bib`.
- `PAPER=../paper_ijis python3 verify_numbers.py`: the same number checks against the IJIS text, plus checks of the
  claims added in the IJIS revision, the abstract length and keyword count, and the cross-references into Online Resource 1.

## IJIS v02 (retitled revision): additional analyses
| Analysis | Script | Output | Paper |
|---|---|---|---|
| MIQP reference for P1 by outer approximation (HiGHS), N = 2 and 3; exact-dual LR-BB with N = 3 | `miqp_oa.py`, `run_e2v2.py` (`jobs_miqp.txt`) | `results/E2miqp`, `results/E2v2/*_N3.csv` | Table 2, Sect. 6.2 |
| Matched-FAR comparison (final threshold re-set on the test half; oracle) | `matched_far.py` | `T_rev_matchedfar*.csv` | Sect. 6.3, Table S15 |
| Matched FAR on the new splits and D4 (each split on the machine that computed it) | `matched_far.py --fresh --part cloud/local/combine` (`--part all` on one machine) | `T_fresh_matchedfar*.csv` | Sect. 6.3, Table S18 |
| D4: LFW faces (YuNet + SFace, OpenCV 4.13) paired with BSSR1 D2 fingers (chimeric, 1680 subjects) | `build_lfw_chimeric.py` | `data/lfw_x_fing.npz`, `data/lfw_x_fing_build.json` | Sect. 5.1, Table 1 |
| New splits 10–29 of D1–D3 and splits 0–9 of D4, bootstrap and held-out calibration (`calib` = `boot` / `xfit`) | `run_fresh.py` (`jobs_fresh.txt`, `jobs_d4.txt`) | `results/E3fresh` | Sects. 6.3–6.4, Tables 3–4, S16–S17, Fig. 4(d) |
| Analysis of the new splits and D4 | `analyze_fresh.py` | `T_fresh_selected/system/stats/compliance.csv` | Tables 3–4, S16–S17 |
| Hyperbola constants of the D4 matchers | `d4_hyp_constants.py` | `T_d4_hyp_constants.csv` | Sect. 6.3 |
| Runs on a second machine; comparison of overlapping splits | `run_local2.py`, `merge_local.py`, `compare_runs.py` | `results/E3fresh_provenance.csv` | Sect. 5.4 |

D4 inputs (not redistributed): `lfw.tgz` (SHA-256 055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0, as
published with scikit-learn), `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` from the
OpenCV model zoo (their SHA-256 values are logged in `data/lfw_x_fing_build.json`). Extract LFW to `../data/lfw_src/lfw`,
put the two model files in `../data/lfw_src`, and run `python build_lfw_chimeric.py ../data/lfw_src`.

Additional seeds: the held-out calibration splits the training subjects with seed 5000 + split; the fold-A bootstrap uses seed
3000 + split and the fold-B bootstrap 4000 + split; D4 pairs LFW identities with D2 subjects using seeds 20260930 (subject
choice) and 20260931 (pairing); the per-trait spoof analysis draws genuine comparisons with seed 2026 + split.

Cross-platform note: of the seven splits computed on both machines, six agreed exactly in all error rates; in one D1 split
3 of 1484 designs differed in training FRR and impostor stages (a first-stage reject threshold at a GP reject rate of
numerically one fell just above the lowest score on one machine and just below it on the other), with identical test
FRR and FAR. The cloud results are used for the overlapping splits (rule fixed before the comparison). For the same reason the
matched-FAR analysis of the new splits is run per split on the machine that computed the designs: the SPRT thresholds are
cumulative log-likelihood-ratio values, and last-bit differences flip exact ties of the integer fingerprint scores.

## IJIS v04: MLP fusion baseline (secondary comparison)
| Analysis | Script | Output | Paper |
|---|---|---|---|
| Parallel fusion by a multilayer perceptron (2 x 16 ReLU units, Adam, L2 1e-4, at most 300 epochs; logit as fused score; same bootstrap threshold as all methods) | `run_mlp.py` (`jobs_mlp.txt`; on a multi-core PC `run_local_mlp.py`) | `results/E3mlp` | Sects. 5.3, 6.3, Table S19 |
| Comparison with the proposed design, logistic-regression and LLR fusion (corrected resampled t-test, unadjusted; outside the primary Holm family) | `analyze_mlp.py` | `T_mlp_system.csv`, `T_mlp_stats.csv` | Sect. 6.3, Table S19 |

The MLP uses the split seed as `random_state` and for the impostor subsample (at most 200000 impostor comparisons with all
genuine comparisons, class-balanced sample weights, as for logistic regression). Calibrations: `boot` for all splits,
`xfit` (held-out, fold split seed 5000 + split, fold-B bootstrap seed 4000 + split) for the new splits and D4.
All 100 MLP jobs ran on the 24-core PC.
