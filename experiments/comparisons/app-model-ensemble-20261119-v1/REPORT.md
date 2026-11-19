# Conditional and Anton output combination: fifteen opinions, five recommendations

19 November 2026. Run `app-model-ensemble-20261119-v1`.

**The tested combination does not improve the conditional model.** Eight nonzero rank blends were compared on validation users. All scored below conditional alone. The best nonzero challenger—97.5% conditional and 2.5% Anton—was frozen and evaluated once on 2,000 newly reserved confirmation users. Its NDCG@5 was **0.517325 versus 0.518893** for conditional alone. The paired difference is **−0.001568**, with a 95% interval **[−0.002639, −0.000504]**. This is a small measured loss, not an improvement.

Anton has complementary recorded likes, but this fusion rule does not convert them into a better final list. Keep conditional alone on the declared objective. No app model, bundle or onboarding setting was changed.

## Fresh confirmation results

Every method receives the same fifteen positive opinions and recommends five unseen films from the same 3,788-movie catalogue.

| Method | NDCG@5 ↑ | Users with a known like in top five ↑ | Known low-rating slots ↓ | Distinct films across all users |
| --- | ---: | ---: | ---: | ---: |
| Conditional alone | **0.518893** | **87.35%** | 2.01% | 31 |
| Frozen blend: 97.5% conditional / 2.5% Anton | 0.517325 | 87.25% | 2.01% | 33 |
| Rating-count popularity | 0.423478 | 81.50% | 2.39% | 12 |
| Anton alone | 0.358378 | 73.90% | 0.96% | 316 |
| Mean-rating baseline | 0.289065 | 65.25% | 0.87% | 10 |

The blend's NDCG is 0.30% lower relative to conditional. Hit rate changes by −0.10 percentage points (paired 95% interval −0.40 to +0.15); known low-rating rate is unchanged (difference interval −0.04 to +0.04 percentage points). Its two additional distinct films do not establish a useful discovery benefit. Catalogue exposure counts are across the population, not within-list diversity or user enjoyment.

The sole primary test is blend minus conditional NDCG@5, using 5,000 paired whole-user bootstrap resamples, seed 2026111901. The predeclared useful-gain screening margin was +0.01 absolute NDCG; it is a convention, not a validated business value. The result fails even the weaker criterion of a positive gain. Other intervals in [test_results.json](test_results.json) are exploratory and unadjusted; they are not extra independent primary claims.

## Do the outputs contain complementary information?

Yes, descriptively. Their top-five lists overlap by an average of only **1.258 films**. For **1,155 of 2,000 users (57.75%)**, Anton's top five contains at least one recorded like absent from conditional's top five. Anton finds at least one known like for **76 users** whose conditional list has none. Conditional does the reverse for **345 users**.

This demonstrates potentially useful different recommendations, not a way to select them reliably at serving time. The final blend:

- Added **46** known-like recommendation slots and removed **61**, across 10,000 user/movie slots.
- Improved NDCG for **52** users, worsened it for **91**, and left it unchanged for **1,857**.
- Rescued three users with no conditional hit but lost the only hits for five others.
- Recommended **201** known low-rating slots, exactly the same total as conditional.

All these diagnostics were computed after the lists were fixed. No hidden ratings select recommendations. No label-aware oracle or hypothetical best-of-both list is presented as a deployable model.

## Blend definition and validation search

The two original numeric scores have incompatible meanings. We combine candidate ranks:

`fusion(movie) = (1 − alpha)/(60 + rank_conditional(movie)) + alpha/(60 + rank_Anton(movie))`

Ranks cover the entire unseen common catalogue. The fifteen disclosed films are removed before ranking. Equal original scores receive average ranks; final fusion ties use ascending MovieLens IDs. This avoids treating Anton's clipped-score ties as distinct preferences. Pure endpoints reproduce each standalone ranking exactly. The 97.5%/2.5% values are weights on reciprocal ranks, not quotas for the five output slots.

The offset 60 and eight candidate weights were specified before validation predictions. Selection maximizes validation NDCG@5 among nonzero blends, breaking ties toward less Anton. Conditional and Anton endpoints are explicit references. A nonzero challenger remains eligible for the planned confirmation even if conditional wins validation; it must then be reported as rejected on validation, as happened here. There was no test-based retuning.

| Anton weight | Validation NDCG@5 | Hit rate@5 | Known low-rating slots | Distinct films |
| ---: | ---: | ---: | ---: | ---: |
| 0%: conditional endpoint | **0.504996** | 87.80% | 2.14% | 28 |
| 2.5%: frozen challenger | 0.503636 | 87.90% | 2.14% | 28 |
| 5% | 0.497630 | 86.90% | 2.12% | 26 |
| 10% | 0.488458 | 85.30% | 2.06% | 34 |
| 15% | 0.478083 | 84.40% | 1.94% | 38 |
| 20% | 0.472897 | 84.00% | 1.90% | 41 |
| 30% | 0.462848 | 83.70% | 1.90% | 53 |
| 40% | 0.455013 | 83.10% | 1.84% | 64 |
| 50% | 0.446345 | 82.80% | 1.70% | 81 |
| 100%: Anton endpoint | 0.335223 | 72.20% | 0.98% | 265 |

Larger Anton contributions show a validation tradeoff: lower recovery of recorded likes, often wider exposure and fewer known low ratings. These larger blends were not separately tested on confirmation users. No acceptable quality sacrifice for variety was authorized or inferred.

## Training and data boundaries

The previous experiment's confirmation users already informed the combination hypothesis. Reusing them would not give a fresh confirmation. This experiment therefore starts from the previous comparison's remaining training population, preserving all 4,500 previous heldout exclusions, and reserves another 4,000 users before retraining.

| Group | Users | Purpose |
| --- | ---: | --- |
| Common training | 192,442 | Both models trained on the same 20,783,195 ratings |
| Anton calibration | 1,000 | Fit its native internal weights once |
| Blend validation | 1,000 | Select the external fusion weight |
| Fresh confirmation | 2,000 | Evaluate only the frozen challenger and fixed references |

All groups are disjoint. Tags exclude all **8,500** old/new heldout authors before vocabulary fitting. The independent metadata content index is reused; audience vectors, tags, film averages and the conditional graph are refitted. Neither previously trained recommendation model is reused for fresh-user evaluation.

Eligibility requires at least 35 observed ratings and 21 positives in the shared catalogue. Selection uses a seeded permutation of eligible ascending user IDs; the 2,000 development users are split again by a seeded permutation into calibration and blend-validation groups. Each ranking profile reveals fifteen random recorded positives (raw MovieLens rating ≥4), leaving at least six hidden positives. All remaining observed raw ratings are retained as hidden targets.

Positive opinions use the established app proxy: floor the MovieLens rating, so 4/4.5 becomes Great and 5 becomes Fantastic. Conditional weights are 1/2. Anton receives `1 + .75*(verdict−1)`. Native Anton calibration retains its original ten raw ratings and up to thirty target ratings per user; this is separate from the fifteen-positive recommendation benchmark. Its internal weights are recorded in [anton_fit.json](anton_fit.json).

Conditional retains the selected positive conditional-neighbour recipe: 100 neighbours, shrinkage 10, positive threshold 4. Anton uses the same pinned commit `6e85958f503525b3020ec80fe613934e3c723d7b` as the earlier comparison, including its actual clipped recommendation scorer. This experiment tests combining the previously compared versions, not a newly fetched upstream version.

## Interpretation limits

- These are newly reserved users from the same historical dataset, not newly collected external users. Earlier model development used the broader dataset. Fresh confirmation reduces direct reuse of previously evaluated outcomes; it does not remove historical development exposure or dataset bias.
- Recorded positives are recovery targets; unobserved films receive no recovery credit but are not proven bad. Recorded low ratings are raw ratings ≤2. Missing feedback and exposure bias affect both recovery and dislike statistics.
- Random historical positive opinions approximate onboarding. They do not reproduce self-selected app favourites, current availability filters or live watch decisions.
- This common catalogue has 3,788 films, rather than the full app catalogue. Results compare refitted algorithms under matched data, not untouched deployed files.
- The uncertainty interval conditions on this training run and profile sampling. It does not cover repeated training populations, alternative fusion families or live satisfaction.
- The finding rejects this tested rank-blending approach for improving NDCG@5. It does not prove that every possible combination must fail. No further alternatives were chosen using confirmation outcomes.

## Reproducibility and audit

The experiment runner is [run_app_ensemble.py](../../run_app_ensemble.py). Its stages are `prepare`, `train`, `select`, and `confirm`; each accepts `--run-dir`. Existing stage sentinels refuse accidental reruns or overwrite. A new output directory with the same seed reproduces this known cohort; it does not constitute another fresh confirmation.

From the CineMatch repository root, an exact replay uses a new directory:

```sh
.venv/bin/python -B -u experiments/run_app_ensemble.py prepare --run-dir artifacts/experiments/app-model-ensemble-20261119-replay
.venv/bin/python -B -u experiments/run_app_ensemble.py train --run-dir artifacts/experiments/app-model-ensemble-20261119-replay
.venv/bin/python -B -u experiments/run_app_ensemble.py select --run-dir artifacts/experiments/app-model-ensemble-20261119-replay
.venv/bin/python -B -u experiments/run_app_ensemble.py confirm --run-dir artifacts/experiments/app-model-ensemble-20261119-replay
```

Compact records in this folder include [policy](policy.json), [data manifest](data_manifest.json), [validation results](validation_results.json), [selection](selection.json), [final results](test_results.json), current [environment](environment.json), and source/data/model/selection hashes. Large arrays, cohort IDs, model artifacts, per-user outputs, sentinels and frozen source copies live in `artifacts/experiments/app-model-ensemble-20261119-v1` in this repository.

Forty-eight focused fusion/data/adapter/metric tests passed before execution. The independent audit **passed**: all training rows and profiles were reproduced from source data; all cohorts were disjoint; all seven per-user metrics for all five methods matched, with maximum aggregate discrepancy 1.11e-16; the primary interval was reproduced; sample native/manual/fusion rankings and all baseline rankings matched. See [audit.json](audit.json) and [audit_app_ensemble.py](../../audit_app_ensemble.py). The audit does not tune or select a model. Production model and app-bundle fingerprints matched after confirmation.

The earlier standalone experiment and its fifteen-input follow-up remain separate: [previous comparison](../app-model-comparison-20261119-v1/REPORT.md), [previous fifteen-input report](../app-model-comparison-20261119-v1/onboarding15_top5/REPORT.md). Their absolute scores should not be treated as changes caused by fusion, because this experiment has new users and refitted models.
