# Repository preparation notes

This package was prepared from the provided `src (2).zip` archive and the associated TL-ESKF manuscript. The source archive itself is not modified.

## Changes made for GitHub presentation

1. **Network-size runner:** The archive contains both `run_large_scale_parallel.py` (K=40) and `run_large_scale_parallel_K30_replacement.py` (K=30). The manuscript specifies K=5,10,20,30. The already-supplied K=30 replacement has been copied into this package as `src/run_large_scale_parallel.py`. Apart from the K=30 configuration, the replacement contains an additional `--run-only` option. This is an explicit selection of a supplied version, not a newly implemented scientific algorithm change.
2. **Dependencies:** Original NumPy/SciPy version pins were retained; missing Pandas/Matplotlib dependencies were added to the root `requirements.txt`.
3. **Plotting path:** The default `--tables` path in `src/plot_paper_figures_publication_v2.py` was changed from a personal Windows path to `analysis_tables`. The script's numerical logic is otherwise unchanged.
4. **Directory structure:** Python code is retained together in `src/`; selected original experiment runbooks are in `docs/`, and a scenario image in `assets/`. Root README and `.gitignore` are new. The two v2 runbooks were adjusted to point to `../requirements.txt` from `src/` after moving the requirements file to the repository root.
5. **Files omitted:** `__pycache__`, historical plotting alternatives, `make_fake_results1.py` / `make_fake_tables.py` synthetic-test data generators, `apply_t1_t2_patches.py` local patch tool, personal-path legacy analyzer, historical patch notes and reports. No original simulation output data were included in the supplied archive.
6. **License:** No license was chosen on the owner's behalf. Add one after confirming authorization.

## Validation performed

- The selected K=30 runner was syntax-compiled.
- A **1 s / 1-seed C008** smoke test completed successfully across seven estimator runs (with diagnostics disabled).
- `collect_results.py` successfully aggregated the smoke-test results into schema-2.0 tables.
- **Not checked:** full 120 s × 50/30-seed Monte Carlo run; numerical reproduction of all manuscript figures/tables; clean-environment installation across Python versions; all auxiliary scripts.

## Potential future cleanup

- Confirm the exact paper-reproducing code snapshot and numeric outputs before stating bit-for-bit or figure-by-figure reproducibility.
- Harmonize README, manuscript version, and any remaining development-oriented naming inside code comments.
- Supply an explicit license and update citation metadata at publication.
