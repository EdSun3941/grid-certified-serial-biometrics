# Grid-certified corner-dominating FAR–FRR envelopes for serial multibiometric threshold design

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23072674.svg)](https://doi.org/10.5281/zenodo.23072674)

Code and per-split results for the article

> C.-H. Su, F. Y.-S. Lin, T.-L. Sun, C.-C. Yeh, C.-H. Hsiao, "Grid-Certified Corner-Dominating FAR–FRR Envelopes for
> Serial Multibiometric Threshold Design via Lagrangian Relaxation-Based Branch-and-Bound", submitted to the
> *International Journal of Information Security*.

The package designs the thresholds of a serial (sequential) multibiometric verification system by geometric
programming under a system-level false acceptance rate (FAR) requirement. Each matcher's empirical FAR–FRR trade-off is
replaced by a posynomial envelope that dominates the corner points of its operating staircase, which bounds the training
FRR of every threshold with positive FAR. The envelopes are fitted by a branch-and-bound search whose bounds come from the
exact Lagrangian dual of the fitting problem, which gives a numerical (double-precision) certificate of global optimality
over the exponent grid. The evaluation
compares the designs with decision-level serial rules, a sequential probability ratio test and parallel score fusion
(including a small multilayer perceptron) on three subsets of NIST BSSR1 and a chimeric LFW + BSSR1 subset.

## Contents

| Path | Content |
|---|---|
| `code/` | all scripts (Python 3), job lists, the one-command driver `run_all.sh`, and `code/README.md` with a table mapping every experiment to its script, output and place in the article |
| `code/dev/` | diagnostics used during development; not needed to reproduce the article |
| `results/` | per-split CSV outputs of every experiment (`E1`, `E1xd`, `E2`, `E2ref`, `E2v2`, `E2miqp`, `E3`, `E3b`, `E3cal`, `E3fresh`, `E3mlp`, `E3sensall`, `E7`, `E7v2`, `E8`, `E9sim`, `S3`), the table CSVs in `results/tables` (each with a `source` column naming its inputs), the figures, and the run logs |
| `results/E3b_round1` | E3b rows before the refit of the system envelopes by the exact-dual LR-BB (kept for comparison, see `code/README.md`) |
| `results/E3fresh_provenance.csv` | which machine computed each file of `E3fresh` and the outcome of the cross-machine comparison |
| `results/E3fresh_pc_overlap/` | the second machine's copies of the seven splits computed on both machines |
| `data/` | instructions for obtaining the data (the score files are not redistributed) and the build log of subset D4 |

## Requirements

Python 3.11 or 3.12 with the packages in `requirements.txt` (the versions used for the article are pinned there).
No commercial solver is needed: linear and mixed-integer programs are solved by HiGHS through SciPy. OpenCV is needed
only to rebuild subset D4 from the LFW images.

```
pip install -r requirements.txt
```

## Data

See `data/README.md`. In short, download NIST BSSR1 and convert it with `code/prepare_bssr1.py`; for subset D4,
download LFW and two OpenCV model-zoo models and run `code/build_lfw_chimeric.py`. SHA-256 checksums of all processed
files are given so that the inputs can be checked before a run.

## Reproduction

```
cd code
bash run_all.sh [NPROC]        # resumable: finished outputs are skipped
```

`run_all.sh` runs all experiments, then the analysis, table and figure scripts. On a 2-vCPU machine the experiments of the
first version alone took about one day (the subject-bootstrap calibration and the 600-second timing runs dominate); the
20 further splits, subset D4 and the MLP baseline add considerably more, and were run for the article largely on a
24-core computer with `run_local2.py` and `run_local_mlp.py` (see `code/README.md`). Timing experiments must run on one
core with no other job.

The released `results/` already contain every per-split output used in the article, so the tables can be regenerated
without rerunning the experiments, for example

```
cd code
python analyze_rev.py && python analyze_fresh.py && python analyze_mlp.py
```

`code/verify_numbers.py` recomputes every number quoted in the article from these CSV files and compares it with the
manuscript source. The manuscript source is not part of this repository; without it, `python verify_numbers.py
--numbers-only` (run with `PAPER=../paper_ijis`) compares every recomputed number with the printed value written into the
script and skips only the checks that need the LaTeX source (phrases, reference list, labels, regenerated table files).

## Seeds

Subject splits use `numpy.random.default_rng(seed)` with seeds 0–9 (original splits) and 10–29 (new splits of D1–D3;
all procedures were fixed before these splits were run); D4 uses seeds 0–9. The subject bootstrap uses seed
1000 + split (B = 300). The held-out calibration splits the training subjects with seed 5000 + split and bootstraps
fold A with seed 3000 + split and fold B with seed 4000 + split. Further seeds are listed in `code/README.md`.

## Version history

- **v1.3.0** (October 2026, second revision after review): estimand of the held-out calibration. All counts and means are
  expectations over a uniform random tie-break among the best fold-A orders, made before fold B is used; means over the
  deployed designs (test FRR, FAR, stages) are now weighted by the deployment probability p_j of each split
  (`analyze_fresh.py`, `analyze_foldA.py`, `analyze_mlp.py`), and the paired tests of the held-out calibration weight
  each split by the probability that both designs are deployed (`corrected_t_w` in `analyze_rev.py`). Only D1 and D4
  change (held-out FRRs by -8.5 % to +10.4 %); no Holm decision changes. New: the fold-A-only control by the outcome of
  the held-out calibration with paired differences (`T_foldA_categories.csv`, `T_foldA_pairdiff.csv`), a single random
  tie-break sensitivity analysis (`tiebreak_sens.py`), the count of fractional simulation outcomes (`analyze_sim.py`),
  and checks for the revised article in `verify_numbers.py`, whose solver-version check now reports a different
  running environment as information instead of a failure. Runs are unchanged. The preferred citation lists the six
  authors of the article. See `code/README.md`, section "IJIS v12".
- **v1.2.0** (October 2026, revision after review): new title of the article ("Grid-Certified Corner-Dominating
  FAR–FRR Envelopes ..."). New: a fold-A-only control of the held-out calibration (`run_foldA.py`, `analyze_foldA.py`,
  `results/E3foldA`; the fold-A designs reproduce those of `results/E3fresh`), and a sensitivity analysis of the
  presentation-attack stress test to attack strength, score shift and presentation-attack detection (`spoof_sens.py`).
  Correction: the D4 spoof analyses of splits 2–9 are now evaluated on the PC that computed their thresholds
  (`merge_d4_spoof.py`), which changes the D4 SPRT entry of the per-trait spoof table (`T_rev_spoof_trait.csv`) from
  0.956 / 0.587 to 0.981 / 0.596. `verify_numbers.py --numbers-only` now also runs without the processed data (the
  checks that read them are skipped). All other results are unchanged. See `code/README.md`, section "IJIS v09".
- **v1.1.0** (October 2026, revision after review): the held-out calibration now fixes the order of every method on
  fold A before fold B is used, and a split whose fold-A design cannot be calibrated on fold B has no deployed design
  (`analyze_fresh.py`; previously, orders that failed on fold B were dropped before the selection). The runs are
  unchanged; `T_fresh_*.csv` and `T_mlp_stats.csv` are recomputed. New: calibration sensitivity with order re-selection
  (`calib_sensitivity_all.py`, `analyze_sens.py`, `results/E3sensall`), a controlled simulation with known population
  error rates (`sim_controlled.py`, `analyze_sim.py`, `results/E9sim`), the D4 candidate-set analysis
  (`analyze_d4_subsets.py`), and `verify_numbers.py --numbers-only`. See `code/README.md`, section "IJIS v08".
- **v1.0.0** (October 2026): version of the first submission.

## Citation

See `CITATION.cff`. Version 1.0.0 (first submission) is archived at Zenodo, https://doi.org/10.5281/zenodo.23072674;
version 1.3.0 is the version used for the revised article (version 1.2.0 was used for the previous revision).
Please cite the article once it is published.

## License

MIT (see `LICENSE`). The BSSR1 and LFW data and the OpenCV models are subject to their own terms and are not included.
