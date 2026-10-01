# Grid-certified posynomial FAR–FRR envelopes for serial multibiometric threshold design

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23072674.svg)](https://doi.org/10.5281/zenodo.23072674)

Code and per-split results for the article

> C.-H. Su, F. Y.-S. Lin, T.-L. Sun, C.-C. Yeh, C.-H. Hsiao, "Grid-Certified Posynomial FAR–FRR Envelopes for Serial
> Multibiometric Threshold Design via Lagrangian Relaxation-Based Branch-and-Bound", submitted to the
> *International Journal of Information Security*.

The package designs the thresholds of a serial (sequential) multibiometric verification system by geometric
programming under a system-level false acceptance rate (FAR) requirement. Each matcher's empirical FAR–FRR trade-off is
replaced by a posynomial envelope that dominates the corner points of its operating staircase, which makes the design
safe for every recovered threshold. The envelopes are fitted by a branch-and-bound search whose bounds come from the
exact Lagrangian dual of the fitting problem, which certifies global optimality over the exponent grid. The evaluation
compares the designs with decision-level serial rules, a sequential probability ratio test and parallel score fusion
(including a small multilayer perceptron) on three subsets of NIST BSSR1 and a chimeric LFW + BSSR1 subset.

## Contents

| Path | Content |
|---|---|
| `code/` | all scripts (Python 3), job lists, the one-command driver `run_all.sh`, and `code/README.md` with a table mapping every experiment to its script, output and place in the article |
| `code/dev/` | diagnostics used during development; not needed to reproduce the article |
| `results/` | per-split CSV outputs of every experiment (`E1`, `E1xd`, `E2`, `E2ref`, `E2v2`, `E2miqp`, `E3`, `E3b`, `E3cal`, `E3fresh`, `E3mlp`, `E7`, `E7v2`, `E8`, `S3`), the table CSVs in `results/tables` (each with a `source` column naming its inputs), the figures, and the run logs |
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
manuscript source; the manuscript source is not part of this repository.

## Seeds

Subject splits use `numpy.random.default_rng(seed)` with seeds 0–9 (original splits) and 10–29 (new splits of D1–D3;
all procedures were fixed before these splits were run); D4 uses seeds 0–9. The subject bootstrap uses seed
1000 + split (B = 300). The held-out calibration splits the training subjects with seed 5000 + split and bootstraps
fold A with seed 3000 + split and fold B with seed 4000 + split. Further seeds are listed in `code/README.md`.

## Citation

See `CITATION.cff`. Version 1.0.0 of this repository, the version used for the article, is archived at Zenodo:
https://doi.org/10.5281/zenodo.23072674. Please cite the article once it is published.

## License

MIT (see `LICENSE`). The BSSR1 and LFW data and the OpenCV models are subject to their own terms and are not included.
