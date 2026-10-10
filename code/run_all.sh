#!/bin/bash
# One-command reproduction of every number, table and figure of the manuscript and its supplementary material.
# Usage: bash run_all.sh [NPROC]      (default 2; resumable: finished outputs are skipped)
# Prerequisite: processed BSSR1 files bssr1_{fing_x_face,fing_x_fing,face_x_face}.npz in $BSSR1_DIR, ../data or
#               ../data/processed (see README and prepare_bssr1.py).
set -e
cd "$(dirname "$0")"
NP=${1:-2}
python3 -c "from data import data_file; [data_file(d) for d in ('fing_x_face','fing_x_fing','face_x_face')]"
mkdir -p ../results/logs
# ---- first-version experiments (still used: fitting metrics, uncalibrated and Clopper-Pearson designs, ablations)
sed -E 's#^python3 run_main.py ([a-z_]+) ([0-9]+) (.*)$#test -f ../results/E3/main_\1_s\2.csv || python3 run_main.py \1 \2 \3#' \
    jobs_main.txt > ../results/logs/jobs_main_guarded.txt
./launch.sh ../results/logs/jobs_main_guarded.txt "$NP" main   # E1 envelope fitting + E3 uncalibrated designs (+E4 noise)
./launch.sh jobs_cal.txt "$NP" cal                              # E3-cal: pair-level Clopper-Pearson calibration
./launch.sh jobs_abl.txt "$NP" abl                              # E5 ablations, E6 sensitivity, training fractions, SPRT (S3)
# ---- revision experiments
./launch.sh jobs_cal2.txt "$NP" cal2                            # E3b: subject-bootstrap calibration, all methods and baselines
./launch.sh jobs_ref.txt "$NP" ref                              # E2ref: independent exact reference (all exponent pairs, N = 2)
./launch.sh jobs_refit.txt "$NP" refit                          # E1xd: envelopes refitted by the exact-dual LR-BB
./launch.sh jobs_cal2xd.txt "$NP" cal2xd                        # E3b: LR-BB rows recomputed with the E1xd envelopes
./launch.sh jobs_cost.txt "$NP" cost                            # E8: stage-budget designs (GP vs SPRT)
./launch.sh jobs_time.txt 1 time                                # E2/E7 (subgradient LR-BB, enumeration, DE): one core, nothing else running
./launch.sh jobs_time2.txt 1 time2                              # E2v2/E7v2 (exact-dual LR-BB, MILP): one core, nothing else running
# ---- tables, figures, checks
python3 dataset_stats.py         # T_datasets, T_correlations
python3 prop1_check.py           # T_prop1_check
python3 analyze_all.py           # first-version tables (T_fit, T_conservativeness, T_scalability, ...)
python3 ablation_compare.py      # T_ablation_paired
python3 analyze_rev.py           # T_rev_selected, T_rev_system, T_rev_stats (corrected resampled t-test, Holm)
python3 conservativeness_rev.py  # T_rev_conservativeness
python3 decomposition_rev.py     # T_rev_decomposition
python3 reject_far_check.py      # T_reject_far
python3 spoof_rev.py             # T_rev_spoof (single-modality spoof analysis)
python3 check_monoc_weights.py   # T_rev_monoc (continuous monomial, support of the convexified solution)
python3 calib_diag.py            # T_rev_calib_diag, T_rev_calib_boot (bootstrap spread vs. shift between halves)
if [ -d ../paper ]; then          # manuscript sources are not part of the public repository
  python3 make_main_tables.py    # ../paper/tables/*.tex (main-paper table bodies)
  python3 make_supp_tables.py    # ../paper/supp_tables.tex
fi
python3 figures.py all           # ../results/figures/*.pdf
[ -d ../paper/figures ] && cp ../results/figures/F2_*.pdf ../results/figures/*_rev.pdf ../paper/figures/
[ -f ../paper/main.tex ] && python3 verify_numbers.py   # recomputes every number quoted in the manuscript; exit code 1 on any mismatch
# ---- IJIS (Springer) version, v02 additions
./launch.sh jobs_miqp.txt 1 miqp                                # E2miqp: MIQP reference by outer approximation (N = 2, 3); E2v2 N = 3
python3 matched_far.py           # T_rev_matchedfar*: final threshold re-set on the test half to test FAR <= alpha (oracle)
if [ -d ../data/lfw_src ] && [ ! -f ../data/lfw_x_fing.npz ]; then python3 build_lfw_chimeric.py ../data/lfw_src; fi   # D4 (needs LFW + 2 ONNX files)
python3 dataset_stats.py         # T_datasets, T_correlations (now incl. D4)
./launch.sh jobs_fresh.txt "$NP" fresh                          # E3fresh: new splits 10-29 of D1-D3, paper + cross-fitted calibration
./launch.sh jobs_d4.txt "$NP" d4                                # E3fresh: D4, splits 0-9
#   (the same jobs can run on a multi-core PC: python run_local2.py [workers] in code/, then on this machine
#    python merge_local.py <PC results/E3fresh folder>, which compares overlapping splits and writes E3fresh_provenance.csv)
python3 d4_hyp_constants.py      # T_d4_hyp_constants (hyperbola constants of the D4 matchers)
python3 analyze_fresh.py         # T_fresh_*
python3 matched_far.py --fresh --part all   # T_fresh_matchedfar* (Table S18); with two machines: --part cloud / local, then --part combine
# ---- IJIS v04: secondary baseline, parallel fusion by a small MLP (outside the primary Holm family)
./launch.sh jobs_mlp.txt "$NP" mlp                              # E3mlp: D1-D3 splits 0-29, D4 splits 0-9 (boot; xfit for new splits and D4)
#   (or on a multi-core PC: python run_local_mlp.py [workers] in code/, then copy results/E3mlp here)
python3 analyze_mlp.py           # T_mlp_system, T_mlp_stats (Table S19)
# ---- IJIS v08: review of 2026-10-09 (held-out protocol fixed in analyze_fresh.py; sensitivity, simulation, D4 subsets)
python3 analyze_d4_subsets.py    # T_d4_subsets (Table S20): D4 face only, face + one finger, three matchers
./launch.sh jobs_sensall.txt "$NP" sensall                      # E3sensall: B = 1000, other seeds, no step-(i) floor, all orders
python3 analyze_sens.py          # T_sens_* (Table S21; Proposition 1 at all deployed GP stage thresholds)
./launch.sh jobs_sim.txt "$NP" sim                              # E9sim: controlled simulation (or python run_local_sim.py [workers])
python3 analyze_sim.py           # T_sim (Table S22)
# ---- IJIS v09: fold-A-only control of the held-out calibration (review M1) and presentation-attack sensitivity (review M8)
./launch.sh jobs_foldA.txt "$NP" foldA                          # E3foldA: fold A only, same seeds as E3fresh (or python run_local_foldA.py [workers])
python3 analyze_foldA.py         # T_foldA_* (Tables S23, S25); checks that the fold-A designs reproduce those of E3fresh
# ---- IJIS v12: deployment-weighted means and paired tests of the held-out calibration are computed by analyze_fresh.py,
#      analyze_mlp.py and analyze_foldA.py above (corrected_t_w in analyze_rev.py); tie-break sensitivity:
python3 tiebreak_sens.py         # T_tiebreak_summary, T_tiebreak_decisions (Table S26); 1000 draws, about 3 min
# ---- IJIS (Springer) version: per-trait spoof analysis, Springer tables/figures/references, checks
python3 spoof_trait.py fing_x_face fing_x_fing face_x_face lfw_x_fing   # T_rev_spoof_trait, T_rev_cost_spoof_trait
python3 spoof_sens.py            # T_spoof_sens (Table S24): attack strength, score shift, PAD; lambda = 1 reproduces Table 8
#   (D4 splits 2-9 were designed on the 24-core PC; for the article, spoof_trait.py and spoof_sens.py were also run for D4 on that PC
#    and merged with python3 merge_d4_spoof.py <folder with the PC raw files>; see code/README.md, IJIS v09)
if [ -d ../paper_ijis ]; then
  PAPER=../paper_ijis python3 make_main_tables.py
  PAPER=../paper_ijis python3 make_supp_tables.py
  FIGSTYLE=springer python3 figures.py all   # ../results/figures_springer (84 / 174 mm, Arial-metric lettering)
  S=../results/figures_springer; F=../paper_ijis/figures
  cp $S/F2_envelopes_fing_x_face_face_C_s0.pdf $F/Fig2.pdf; cp $S/F5_scalability_rev.pdf $F/Fig3.pdf
  cp $S/F3_system_frr_rev.pdf $F/Fig4.pdf; cp $S/F4_conservativeness_rev.pdf $F/Fig5.pdf; cp $S/F2_envelopes_fing_x_fing_li_V_s0.pdf $F/FigS1.pdf
  python3 make_springer_refs.py
  (cd ../paper_ijis && for i in 1 2; do pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode ESM_1.tex >/dev/null; done)
  PAPER=../paper_ijis python3 verify_numbers.py
fi
[ -f ../paper_ijis/main.tex ] || PAPER=../paper_ijis python3 verify_numbers.py --numbers-only   # without the manuscript: every number against the checklist in the script
echo "done: see ../results/tables, ../results/figures and ../paper/tables"
