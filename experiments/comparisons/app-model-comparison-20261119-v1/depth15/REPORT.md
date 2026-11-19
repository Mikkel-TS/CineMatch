# Top-15 evaluation follow-up

19 November 2026. User-requested extension of [the direct comparison](../REPORT.md).

**Only the recommendation cutoff changes from five to fifteen.** The models still receive the same five positive onboarding opinions. The same 2,000 users, 3,788 candidate movies, frozen model weights, scores, tie order, hidden targets and watched exclusions are retained. No model was retrained or combined.

The conditional method remains highest on NDCG@15: **0.514879**, a relative gain of **64.28% over Anton** and **25.27% over count popularity**.

| Method | NDCG@15 | HitRate@15 | Precision@15 | Average recorded likes in 15 | Distinct movies across all top fifteens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional neighbours | 0.514879 | 98.45% | 48.24% | 7.2360 | 117 |
| Rating-count popularity | 0.411025 | 96.30% | 37.95% | 5.6925 | 19 |
| Anton’s updated scorer | 0.313419 | 90.40% | 27.64% | 4.1465 | 386 |
| Smoothed mean rating | 0.286619 | 87.05% | 25.08% | 3.7625 | 19 |

HitRate@15 means at least one hidden recorded like among fifteen recommendations. The conditional method achieves this for 1,969 of 2,000 users. Its average list contains 7.236 recorded likes, versus Anton's 4.1465. A hidden rating of at least four is a like; unrated movies remain unknown and receive no recovery credit.

NDCG@15 uses the same logarithmic position discounts as NDCG@5, extended to rank fifteen, and divides by the ideal score at fifteen. The normalized score is not directly comparable with the old NDCG@5 number: it assesses a longer list with a different denominator. Higher hit rates are expected because each list includes ten more opportunities for a hit. Compare models at the same cutoff.

## Descriptive uncertainty

| Conditional minus reference, NDCG@15 | Difference | Paired user-bootstrap 95% interval |
| --- | ---: | --- |
| conditional-vs-anton | +0.201461 | [0.193544, 0.209561] |
| conditional-vs-rating_count | +0.103854 | [0.098806, 0.108793] |
| conditional-vs-rating_mean | +0.228260 | [0.220482, 0.235774] |

These use 5,000 paired resamples. This cutoff was requested after the original confirmation results were inspected, so this is **a descriptive follow-up on known test users**, not a new independent confirmation trial or a replacement for the original prespecified primary metric. No model selection or tuning used these results.

Anton continues to expose more distinct films and fewer explicitly low-rated targets: 0.94% of recommendation slots versus 2.09% for conditional neighbours. These are secondary descriptions, not proof of greater real-world enjoyment.

## Verification and reproduction

The runner first checked every original frozen source/artifact hash, then checked that **all four models' first ten recommendations match their saved original rankings for every one of the 2,000 users**. Six focused tests cover the rank-fifteen boundary, ideal normalization, exclusions, ties, missing relevance and short lists. The separate [independent audit](audit.json) checks the reported arithmetic from saved rankings and hidden labels.

[Results](results.json) and [execution plan](plan.json) retain the cutoff, source hashes and unchanged original result hash. Per-user rankings/metrics are under `artifacts/experiments/app-model-comparison-20261119-v1/depth15`. The old comparison files and app remain unchanged.

Implementation: [depth evaluator](../../../evaluate_app_comparison_depth.py). From the CineMatch repository root, the command is:

```sh
.venv/bin/python experiments/evaluate_app_comparison_depth.py --k 15
```

The existing output is protected against overwrite. Reproduction should use a separately copied run directory via `--run-dir`; reusing these labels does not create a fresh test cohort.
