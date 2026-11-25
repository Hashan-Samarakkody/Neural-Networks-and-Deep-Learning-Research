# Pre-audit outputs archive

Outputs produced by the pipeline at baseline commit **fb7930c** (24 Nov 2026), before the audit fixes were applied.

Move `pipeline/outputs/` here before the frozen clean run (R-05):

```
cp -r pipeline/outputs/* outputs_archive/pre_audit_fb7930c/
```

Do not delete these files from git history (hard rule 3 from the audit document).

## What was wrong with these outputs

- No TwoStage row in `experiment_results.csv`
- Bootstrap CIs computed with 200 resamples (not 1000)
- DSI fitted on all 65 events (data leakage, R-01)
- TWDI could be negative (R-02)
- Two spurious censored events from baseline fallback (R-03)
- NaivePrevious used a look-ahead T_rec (R-09)
- Inner CV was shuffled (R-08)
