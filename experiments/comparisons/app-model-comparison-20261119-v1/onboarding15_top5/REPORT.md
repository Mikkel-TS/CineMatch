# Fifteen positive onboarding opinions, five recommendations

19 November 2026. User-requested follow-up to the [direct model comparison](../REPORT.md).

**The model receives fifteen positive opinions; evaluation remains NDCG@5 and HitRate@5.** This differs from the previous [top-fifteen follow-up](../depth15/REPORT.md), where only recommendation-list depth changed.

The same 2,000 evaluation users, 3,788-movie catalogue and frozen model weights are used. The conditional method remains highest: **0.527139 NDCG@5**, **48.81% above Anton** and **22.13% above rating-count popularity**.

| Method | NDCG@5 | HitRate@5 | Precision@5 | Distinct films across all top fives |
| --- | ---: | ---: | ---: | ---: |
| Conditional neighbours | 0.527139 | 88.65% | 51.03% | 35 |
| Rating-count popularity | 0.431625 | 83.10% | 41.77% | 12 |
| Anton’s updated scorer | 0.354245 | 73.45% | 33.10% | 320 |
| Smoothed mean rating | 0.290871 | 65.95% | 25.35% | 12 |

Conditional neighbours retrieve at least one remaining recorded like for **1,773 of 2,000 users**, versus **1,469** for Anton. Recorded likes mean hidden raw MovieLens ratings of at least four; unknown movie relevance earns no recovery credit.

## What changed

The original fixed positive-seed order is preserved: each new profile uses the first fifteen movies from its previously prepared twenty-positive profile. The original five inputs remain identical. We reconstruct the complete raw rating history from the existing twenty disclosed opinions plus hidden labels, then replay the original deterministic profile selection at size fifteen.

The last five movies from the old twenty-opinion profile return to the hidden target set with their original raw ratings. All fifteen newly disclosed movies are watched exclusions and cannot appear as recommendations. Hidden labels are never passed to either scorer. Great/Fantastic conversion remains unchanged: raw 4/4.5 maps to Great 4, raw 5 to Fantastic 5; Anton receives the same linear conversion to 1–4 used previously.

Both algorithms and the two baselines are unchanged. Anton's weights remain the frozen validation fit at ten raw historical opinions; they were not refitted for this profile size. No retraining, parameter tuning, combined model or app change was performed.

## Interpretation

Conditional neighbours still outperform the other methods within this fifteen-input setting. Their score is lower than the earlier five-input score, but this **does not isolate the effect of additional onboarding information**: ten more known liked films now become watched exclusions and leave the hidden target set. A causal comparison of the value of extra inputs would hold recommendation candidates and evaluation targets fixed between profile sizes.

Anton continues to expose more different movies and fewer explicitly low-rated targets: **0.94%** of its top-five slots versus **2.18%** for conditional neighbours. Those are descriptive secondary outcomes, not a different prespecified winner criterion.

This is a descriptive follow-up on the same already-examined users, not a new independent confirmation cohort. It remains a historical recovery benchmark, not an estimate of real-world discovery, enjoyment or onboarding completion.

## Descriptive paired uncertainty

| Conditional minus reference, NDCG@5 | Difference | Paired bootstrap 95% interval |
| --- | ---: | --- |
| conditional-vs-anton | +0.172893 | [0.160657, 0.185178] |
| conditional-vs-rating_count | +0.095514 | [0.088489, 0.102314] |
| conditional-vs-rating_mean | +0.236268 | [0.225627, 0.246682] |

Intervals use 5,000 user-level paired resamples and describe this fixed analysis. They do not correct for choosing new analyses after inspecting the original results.

## Checks and records

Eight focused tests cover observation reconstruction, exact nested support prefixes, restored hidden raw ratings, preservation of half-star ratings, malformed overlaps and exclusion of all fifteen disclosed movies. The [independent audit](audit.json) recomputes the saved metrics and checks all 2,000 profile transformations.

The runner verifies all original frozen source/artifact hashes before scoring and keeps previous outputs untouched. [Results](results.json) and [plan](plan.json) record hashes, assumptions and timestamps. The new profiles and per-user rankings/metrics are under `artifacts/experiments/app-model-comparison-20261119-v1/onboarding15_top5`.

Implementation: [onboarding follow-up evaluator](../../../evaluate_app_onboarding.py). From the CineMatch repository root:

```sh
.venv/bin/python experiments/evaluate_app_onboarding.py --positive-opinions 15
```

Existing output directories are protected against overwrite; reproducing in a separate original run copy requires `--run-dir`. Replaying known labels does not create new independent evidence.
