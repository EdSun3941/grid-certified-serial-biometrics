# Code package: grid-certified corner-dominating FAR–FRR envelopes for serial multibiometric threshold design

Reproduces every number, table and figure of the manuscript "Grid-Certified Corner-Dominating FAR–FRR Envelopes for Serial
Multibiometric Threshold Design via Lagrangian Relaxation-Based Branch-and-Bound" (titled "Grid-Certified Posynomial …" up to v1.1.0) (Springer version for the International
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
(https://www.nist.gov/itl/iad/btg/nist-biometric-scores-set-bssr1). The raw score files are not redistributed.
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
| E1 envelope fitting, E3 uncalibrated designs | `run_main.py` | `results/E1`, `results/E3` | Sect. 6.1, Table 7, Tables S3, S8 |
| E3-cal pair-level Clopper–Pearson calibration | `run_cal.py` | `results/E3cal` | Fig. 5, Table S4 |
| E3b subject-bootstrap calibration, all baselines | `run_cal2.py` | `results/E3b` | Table 5, Figs. 4–5, Sects. 6.3–6.6 |
| E2 solver comparison (subgradient LR, enumeration, DE) | `run_e2.py` | `results/E2` | Table 4 |
| E2v2 exact-dual LR-BB and MILP on the same fitting sets | `run_e2v2.py` | `results/E2v2` | Table 4, Sect. 6.2, Table S1 |
| E7 scalability (subgradient LR-BB, enumeration) | `run_e7.py` | `results/E7` | Fig. 3, Table S9 |
| E7v2 scalability (exact-dual LR-BB, MILP) | `run_e7v2.py` | `results/E7v2` | Fig. 3, Sect. 6.2 |
| Ablations, sensitivity, SPRT (preliminary implementation) | `run_main.py` options, `run_sprt.py` | `results/E3`, `results/S3` | Sect. 6.7, Table S11 |
| Independent exact reference (all exponent pairs, N = 2) | `reference_exact.py` | `results/E2ref` | Table 4, Sect. 6.2 |
| Refit of the system envelopes by the exact-dual LR-BB | `refit_check.py` | `results/E1xd` | Sect. 5.3, Table S14 |
| E3b LR-BB rows with the refitted envelopes | `run_cal2_xd.py` | `results/E3b` (earlier rows kept in `results/E3b_round1`) | Table 5, Figs. 4–5 |
| E8 stage-budget designs (GP vs. SPRT) | `run_cost.py` | `results/E8` | Sect. 6.6, Table 9 |
| Spoof analysis per matcher | `spoof_rev.py` | `T_rev_spoof.csv` | Sect. 6.6 (per-algorithm values) |
| Spoof analysis per trait | `spoof_trait.py` | `T_rev_spoof_trait.csv`, `T_rev_cost_spoof_trait.csv` | Tables 8 and 9 |
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
  matchers C and G compare the same image); writes `T_rev_spoof_trait.csv` (Table 8) and `T_rev_cost_spoof_trait.csv`
  (spoof column of Table 9).
- `PAPER=../paper_ijis python3 make_main_tables.py` / `make_supp_tables.py`: table bodies with superscript-letter
  footnotes, the spoof and stage-cap tables in the main text, and Online Resource tables numbered S1-S22 in citation order.
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
| MIQP reference for P1 by outer approximation (HiGHS), N = 2 and 3; exact-dual LR-BB with N = 3 | `miqp_oa.py`, `run_e2v2.py` (`jobs_miqp.txt`) | `results/E2miqp`, `results/E2v2/*_N3.csv` | Table 4, Sect. 6.2 |
| Matched-FAR comparison (final threshold re-set on the test half; oracle) | `matched_far.py` | `T_rev_matchedfar*.csv` | Sect. 6.3, Table S15 |
| Matched FAR on the new splits and D4 (each split on the machine that computed it) | `matched_far.py --fresh --part cloud/local/combine` (`--part all` on one machine) | `T_fresh_matchedfar*.csv` | Sect. 6.3, Table S18 |
| D4: LFW faces (YuNet + SFace, OpenCV 4.13) paired with BSSR1 D2 fingers (chimeric, 1680 subjects) | `build_lfw_chimeric.py` | `data/lfw_x_fing.npz`, `data/lfw_x_fing_build.json` | Sect. 5.1, Table 3 |
| New splits 10–29 of D1–D3 and splits 0–9 of D4, bootstrap and held-out calibration (`calib` = `boot` / `xfit`) | `run_fresh.py` (`jobs_fresh.txt`, `jobs_d4.txt`) | `results/E3fresh` | Sects. 6.3–6.4, Tables 5–6, S16–S17, Fig. 4(d) |
| Analysis of the new splits and D4 | `analyze_fresh.py` | `T_fresh_selected/system/stats/compliance.csv` | Tables 5–6, S16–S17 |
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

## IJIS v05 corrections (mathematical and data review)
- `envelope_solvers.py`: when the search is interrupted, the reported lower bound is the minimum of (1 - eps) U and the
  bounds of the open nodes and leaves (Prop. 5); complete runs are unchanged. No reported result was affected (no
  interrupted run had a bound between (1 - eps) U and U).
- `run_fresh.py` / `analyze_fresh.py`: parallel fusion acquires every modality, so its stages per claim equal the number
  of matchers; files written before this fix record one stage, which `analyze_fresh.py` corrects on loading (only the
  stage columns of `T_fresh_selected.csv` and `T_fresh_system.csv` change; Table 3).
- `figures.py`: the training staircase in Fig. 2 and Fig. S1 is drawn with level y_k ending at its corner x_(k+1)
  (`where="pre"`); the envelopes and all numbers are unchanged.
- `make_supp_tables.py`, `make_springer_refs.py`: Table S7 confidence limits to four decimals, Table S3 column "Sum",
  Springer basic reference style (no final period, no "?." , dated online document).
- `results/E3fresh_pc_overlap/`: the PC copies of the seven splits computed on both machines (the cloud copies in
  `results/E3fresh` are used; comparison in `results/E3fresh_provenance.csv`).

## IJIS v08: changes after the review of 2026-10-09 (release v1.1.0)
Table and section numbers in this file follow the v08 manuscript (new Tables 1 and 2; Sect. 6.5 is the controlled
simulation; Online Resource 1 Sect. S4 describes the simulation and Sect. S5 the reproducibility details).

| Change or analysis | Script | Output | Paper |
|---|---|---|---|
| Held-out calibration: the order of every method is now fixed on fold A (smallest fold-A FRR after the fold-A calibration, ties averaged) before fold B is used; if fold B cannot calibrate that design, no design is deployed in the split and no other order is tried. Previously, orders that failed on fold B were dropped before the selection. The runs are unchanged; only the analysis changed | `analyze_fresh.py` (`select`), `make_main_tables.py`, `make_supp_tables.py`, `analyze_mlp.py` | `T_fresh_selected.csv` (columns `n_tied`, `n_deployed`, `p_deploy`), `T_fresh_system.csv`, `T_fresh_stats.csv` (paired over splits in which both designs were deployed), `T_fresh_compliance.csv` (`n_deployed`, `n_met`) | Sects. 5.2, 6.3–6.4, Table 6, Tables S17, S19 |
| D4 candidate sets: face only, face + one finger, three matchers, all orders | `analyze_d4_subsets.py` | `T_d4_subsets.csv` | Sect. 6.3, Table S20 |
| Calibration sensitivity with order re-selection: B = 1000, bootstrap seeds 7000 + split and 8000 + split, and no step-(i) Clopper–Pearson floor for the GP designs; Proposition 1 checked at every deployed stage threshold | `calib_sensitivity_all.py` (`jobs_sensall.txt`), `analyze_sens.py` | `results/E3sensall`, `T_sens_system.csv`, `T_sens_stats.csv`, `T_sens_summary.csv`, `T_sens_lemma.csv` | Sects. 3.5, 6.4, Table S21 |
| Controlled simulation with known population error rates (bootstrap and held-out calibration; correlation 0–0.8) | `sim_controlled.py` (`jobs_sim.txt`; on a multi-core PC `run_local_sim.py`), `analyze_sim.py` | `results/E9sim`, `T_sim.csv` | Sect. 6.5, Online Resource 1 Sect. S4, Table S22 |
| `verify_numbers.py --numbers-only` | `verify_numbers.py` | -- | recomputes every number of the article and compares it with the value written into the script (the checklist), without the manuscript source |

`calib_sensitivity_all.py` reuses the stage thresholds of the decision-level rules from `results/E3b` for the designs that
are feasible there (identical to recomputing them; checked on D1 split 0) and recomputes the others. The simulation ran on
the 24-core PC (Windows, Python 3.12.5), the sensitivity analysis on the 2-vCPU Linux machine, so that its paper variant
reproduces `results/E3b` exactly. `dev/calib_sensitivity.py` is an earlier version restricted to the selected orders.

## IJIS v09: additional analyses (release v1.2.0)
Table and section numbers follow the v09 manuscript (Online Resource 1 Tables S23 and S24 are new).

| Analysis | Script | Output | Paper |
|---|---|---|---|
| Fold-A-only control of the held-out calibration: the fold-A designs of `run_fresh.py` (same fold split and seeds) deployed with their fold-A calibration, fold B unused; checks that every fold-A design reproduces the xfit arm of `results/E3fresh`, and compares the bootstrap, fold-A-only and held-out calibrations | `run_foldA.py` (`jobs_foldA.txt`; on a multi-core PC `run_local_foldA.py`), `analyze_foldA.py` | `results/E3foldA`, `E3foldA_provenance.csv`, `T_foldA_repro.csv`, `T_foldA_selected.csv`, `T_foldA_system.csv`, `T_foldA_paired.csv`, `T_foldA_methods.csv` | Sects. 5.2, 6.4, 7, Table S23 |
| Presentation-attack sensitivity: attack strength (score interpolation), scores shifted above the genuine distribution, PAD at every trait acquisition (APCER 0.2 / 0.05, BPCER 0.01), FRR with PAD and trait acquisitions per claim; lambda = 1 reproduces Table 8 | `spoof_sens.py` | `spoof_sens_raw_*.csv`, `T_spoof_sens.csv` | Sect. 6.6, Table S24 |

The control arm ran D1 and D4 on the 2-vCPU Linux machine and D2 and D3 on the 24-core PC; split 12 of D2 was run on
both machines and gave identical rates (`results/E3foldA_cloud_overlap`). Two of the 15,050 feasible fold-A designs (D1,
splits 17 and 27) differ from the original xfit runs by one genuine comparison of fold A; in the one setting where this
changes the tied best orders, `analyze_foldA.py` deploys the orders fixed in the xfit arm.

D4 spoof analyses (Table 8 D4 row, Table S24 D4 rows): the thresholds of D4 splits 2-9 were computed on the 24-core PC.
The SPRT re-estimates its log-likelihood ratios when a design is evaluated, and their last bits differ between the PC
and the Linux machine, which flipped a few test scores lying exactly at an SPRT threshold in splits 4 and 7 (SPRT test
FRR 0.0016 instead of 0.0011 on D4 at alpha = 1e-3). `spoof_trait.py lfw_x_fing` and `spoof_sens.py lfw_x_fing` were
therefore also run on the PC; `merge_d4_spoof.py` keeps the PC rows of splits 2-9 (all other rows are identical on the
two machines), keeps the local files as `*_local.csv`, writes `results/D4_spoof_provenance.csv`, and re-aggregates
`T_rev_spoof_trait.csv` and `T_spoof_sens.csv`. This corrects the D4 SPRT entry of Table 8 (0.956 / 0.587 in v08,
now 0.981 / 0.596); the evaluated test FRR now equals that of `results/E3fresh` for every method and split.

## IJIS v12: estimand of the held-out calibration and the fold-A-only control (release v1.3.0)
Changes after the second-round review of the v11 manuscript (items R1, R2, and the minor items). Table numbers follow the
v12 manuscript; Online Resource 1 Tables S25 and S26 are new.

* Tie-break (R1). All counts and means are expectations over a uniform random tie-break among the orders with the best
  training (or fold-A) FRR, drawn independently for every method and split before fold B is used. With the held-out
  calibration, a split j with m_j tied orders of which d_j can be calibrated on fold B deploys a design with probability
  p_j = d_j / m_j (`p_deploy` in `T_fresh_selected.csv`). Means over the deployed designs (test FRR, FAR, FAR / alpha,
  stages) are now the deployment-conditional means sum_j p_j r_j / sum_j p_j, where r_j is the mean over the deployable tied
  orders (`analyze_fresh.py`, `analyze_foldA.py`, `analyze_mlp.py`). Up to v1.2.0, every split with p_j > 0 had equal
  weight; deployment and compliance counts were already p-weighted and do not change. Only D1 and D4 have 0 < p_j < 1;
  their held-out FRRs change by -8.5 % to +10.4 %.
* Paired tests of the held-out calibration (Tables S17 and S19): conditional on both designs being deployed, with
  independent tie-breaks for the two methods, so split j has the weight w_j = p_j p'_j and the per-split difference is that
  of the means over the deployable tied orders. `corrected_t_w` in `analyze_rev.py` applies the Nadeau-Bengio correction
  with the effective number of splits K = (sum w)^2 / sum w^2 and K - 1 degrees of freedom; with all weights one it equals
  `corrected_t`. No Holm decision changed (checked in `verify_numbers.py` by recomputing the earlier unweighted tests).
* `analyze_foldA.py` (R2) adds `T_foldA_category_splits.csv` and `T_foldA_categories.csv` (Table S25 a: the 190
  combinations of the proposed design by the outcome of the held-out calibration, every / some / no tied order deployed,
  with the fold-A-only results of the withheld designs) and `T_foldA_pairdiff.csv` (Table S25 b: held-out minus
  fold-A-only test FAR / alpha and FRR of the deployed designs, weighted corrected resampled t per subset and requirement).
* `tiebreak_sens.py` (new, about 3 min): a single random tie-break per method and split instead of the expectation,
  1000 draws (`numpy.random.default_rng(9000 + draw)`); `T_tiebreak_summary.csv` (deployment, compliance, FRR and
  FAR / alpha of the proposed design) and `T_tiebreak_decisions.csv` (Holm decisions of the held-out comparisons against
  the expectation-based ones; a comparison with fewer than five jointly deployed splits is not tested in that draw).
* `analyze_sim.py` adds `frac_reps` and `xfit_frac_reps`, the replicates whose outcome is fractional because tied orders
  are averaged (13 and 38 of 3000). The Wilson interval is computed from the summed outcomes; for outcomes in [0, 1] the
  variance is at most p (1 - p), so the interval is conservative.
* `verify_numbers.py`: new checks for all numbers and statements of this revision. The check of the HiGHS and SciPy
  versions now distinguishes the paper from the running environment: the versions stated in the paper are checked in the
  text, and a different running environment is printed as INFO and not counted as a failure.

Release v1.3.1 changes only `CITATION.cff` (author order of the preferred citation) and the manuscript-text checks of
`verify_numbers.py` (revised author list and affiliation; the commit-placeholder check no longer matches `\@@input`).
