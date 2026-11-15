# Merged Channel-Flow PINN — Post-Weight-Search Update

`train_hybrid.py`'s `DEFAULT_WEIGHTS` is now `data=20, pde=1, inlet=5, wall=5,
outlet=1` — the winning candidate from `weight_search.py`
(`outputs/weight_search_results.csv`), which beat the old default
(`data=10, ...`), a BC-heavy variant, a data-light variant, and the
Wang et al. (2021) adaptive scheme on whole-domain relative-L2 error
averaged over u, v, p and over TRAIN + UNSEEN Re.

## ⚠️ One thing to fix in code before your next training run

`train_hybrid.py`'s module docstring says adaptive weighting is the
default, but the **no-argument** `python train_hybrid.py` path explicitly
calls `run_adam(..., adaptive=False)` — so it actually trains with the
**fixed** `data=20` weights, not the adaptive scheme. That's what produced
every result in the updated report (`Merged_Channel_PINN_Report.docx`), so
it's an accurate description of what you ran, but the docstring itself is
now wrong and should say so, or the no-arg path should be changed to call
`adaptive=True`. Neither was changed here since your file, your call —
flag it in the report as a known limitation instead.

## What the updated report covers

`Merged_Channel_PINN_Report.docx` was rewritten end-to-end against your
latest results (`evaluation_summary.txt`, `metrics_*.csv`,
`weight_search_results.csv`, and all 11 figures you sent):

- Every formula (governing equations, output normalization, the relL2
  error metric, the exact fully-developed baselines) is now attributed to
  a specific paper/book, with a working hyperlink both inline and in the
  reference list.
- A worked derivation + Table 1 showing the entrance_solver.py pressure
  sign fix is verified against the exact fully-developed dP/dx at every
  Re (all six match to 4 significant figures post-fix).
- Table 2 reproduces your full weight-search comparison and explains why
  `data=20` won and why the adaptive run in that comparison looks worse
  than it probably deserves (short search budget, not a general verdict
  on adaptive vs. fixed weighting).
- Section 7 ("Discussion") benchmarks your u/v/p numbers against
  comparable published PINN results, states plainly where accuracy is
  solid (v: 0.5–4%) and where it isn't (p: up to 18–24% on the
  interpolation Re), explains the three compounding reasons for the
  pressure gap, and gives four concrete, ranked next steps.
- Section 8 lists every known limitation as of this revision, including
  the docstring/default mismatch above.

## Re-running

Nothing about the run commands changed:

```bash
python generate_data.py
python train_hybrid.py
python evaluate.py
python evaluate_global.py
python weight_search.py            # optional, re-validates the weight choice
```

If you make the adaptive-vs-fixed docstring fix mentioned above and re-run,
send me the new `evaluation_summary.txt` / `metrics_*.csv` and I'll update
Section 4.4, 6, and 7 of the report to match — those are the sections tied
to which weighting scheme actually produced the numbers.
