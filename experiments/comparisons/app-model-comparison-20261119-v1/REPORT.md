# Conditional neighbours versus Anton's updated rating model

19 November 2026. Run `app-model-comparison-20261119-v1`.

User-requested follow-up: [evaluate the top fifteen recommendations](depth15/REPORT.md), retaining the same five positive inputs and frozen models. This is a descriptive extension; the original primary top-five comparison below is preserved.

A second follow-up supplies [fifteen positive onboarding opinions and evaluates five recommendations](onboarding15_top5/REPORT.md), retaining the same frozen models and user cohort.

A separate later experiment tests [combining both model outputs](../app-model-ensemble-20261119-v1/REPORT.md), with new heldout users and refitted models. It does not alter the standalone comparison below.

**The conditional-neighbour method wins this offline comparison of recovering hidden likes.** With five positive onboarding opinions, its final NDCG@5 is **0.584141**, versus **0.380903** for Anton's updated rating model and **0.489307** for rating-count popularity. That is a relative improvement of **53.36% over Anton** and **19.38% over count popularity**. All three prespecified primary comparisons favour conditional neighbours after the multiple-comparison adjustment.

This supports keeping the selected method for the app while investigating its concentration and negative-feedback weaknesses. It does not establish superior live discovery or enjoyment. Anton exposes more distinct movies and recommends fewer movies with recorded low ratings in this test. No combined model was trained, and neither the app bundle nor the original selected model was changed.

## Scope and versions

Both algorithms were **refitted on the same 3,788-movie verified common catalogue**, using the same **21,431,215 ratings from 196,442 users**. This is a controlled method comparison, not a comparison of unchanged production artifacts on the full 8,937-movie app catalogue.

The selected conditional recipe remains positive co-occurrence, conditional similarity, support shrinkage 10, 100 stored neighbours per seed, and positive inference weights `max(verdict - 3, 0)`. All disclosed opinions are watched exclusions. Its original deployment artifact is preserved.

Anton is pinned to commit [`6e85958f503525b3020ec80fe613934e3c723d7b`](https://github.com/AntonDev29/CineMatch-AntonModel/tree/6e85958f503525b3020ec80fe613934e3c723d7b), which added `load_export`, `score_candidates` and `recommend` after our previous review. The benchmark calls that exact `score_candidates` implementation: content, SVD audience features, tags and film-average bias; 50 rounded neighbours per source; fixed validation-fitted weights; predicted scores clipped to 1–4; stable catalogue-order ties.

The code **does not enforce the ten-rating fallback**, despite exporting `personalFromRatings: 10`. Primary results measure actual code behavior. If the documented fallback were enforced at five opinions, Anton's ranking would equal the mean-rating baseline below, NDCG@5 **0.336564**. We did not silently repair or change the scorer.

## Final primary comparison: five positive opinions

These are the **2,000 fresh confirmation users**, not the 1,000 users used for Anton's weight fitting. NDCG@5 rewards putting known liked targets near the top. HitRate@5 means at least one hidden recorded like appears in the first five; it is not a measured chance that a person will choose or enjoy a new recommendation.

| Method | NDCG@5 | HitRate@5 | Precision@5 | Recall@10 | Distinct movies across all top fives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional neighbours | **0.584141** | **91.95%** | **56.56%** | **9.56%** | 49 |
| Rating-count popularity | 0.489307 | 87.75% | 47.58% | 7.14% | 8 |
| Anton's actual updated scorer | 0.380903 | 78.00% | 34.68% | 5.28% | **233** |
| Smoothed training mean rating | 0.336564 | 73.55% | 29.01% | 4.66% | 7 |

The baseline distinction matters: Anton's earlier report called average rating “popularity”; our established baseline ranks by rating count. Here Anton beats the average-rating baseline, while rating-count popularity beats Anton.

Paired user-bootstrap differences, 5,000 resamples:

| Conditional minus reference, NDCG@5 | Difference | Ordinary 95% interval | Interval adjusted for three primary comparisons |
| --- | ---: | --- | --- |
| Anton | +0.203238 | [0.191743, 0.214812] | [0.188736, 0.217592] |
| Rating count | +0.094834 | [0.087154, 0.102522] | [0.085519, 0.104056] |
| Mean rating | +0.247578 | [0.236775, 0.258192] | [0.234324, 0.260829] |

The adjusted intervals use confidence `1 - 0.05/3` for the three declared comparisons. Their lower bounds exceed the prespecified practical screening margin of 0.01 absolute NDCG. That margin is a research decision rule, not an established business-value threshold. These intervals describe user-sampling uncertainty for the frozen models; they do not include uncertainty across retraining seeds or different populations.

## Larger and mixed profiles

All four prespecified secondary checks also favour conditional neighbours. They are exploratory checks on the same confirmation users, not four additional independent trials.

| Disclosed opinions | Conditional NDCG@5 | Anton NDCG@5 | Count popularity | Mean rating | Conditional minus Anton, ordinary 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| 10 positive | 0.556643 | 0.375702 | 0.460880 | 0.312534 | +0.180941 [0.168840, 0.192851] |
| 20 positive | 0.488589 | 0.326752 | 0.398578 | 0.269611 | +0.161837 [0.150043, 0.173514] |
| 10 mixed | 0.581836 | 0.430871 | 0.488269 | 0.331747 | +0.150965 [0.137162, 0.164626] |
| 20 mixed | 0.554477 | 0.420705 | 0.455449 | 0.304862 | +0.133771 [0.119947, 0.147645] |

Mixed profiles include low and neutral opinions. They compare the raw model scorers; a mixed profile need not satisfy the frontend's five-positive readiness rule. Their result is not proof that negative feedback is useless: it compares two complete algorithms, not an isolated removal of negative evidence.

Do not interpret lower absolute scores at 20 positive inputs as deterioration from learning more. More known liked movies are then excluded and removed from the target set, making the remaining retrieval problem different. Compare methods within each row.

## Tradeoffs and diagnostic observations

**Exposure concentration remains a weakness.** Conditional neighbours show only 49 distinct films across 10,000 recommendation slots; its five most exposed movies occupy 69.0% of those slots. Anton shows 233 distinct films, though its top five still occupy 62.36%. Count popularity's top five occupy 96.15%. Distinct-film count measures catalogue exposure, not within-list diversity, novelty satisfaction or enjoyment.

**Anton recommends fewer explicitly low-rated targets.** With five positive inputs, 0.89% of Anton's top-five slots are movies the user had rated at most two stars, versus 2.09% for conditional neighbours and 2.39% for count popularity. Mean-rating ranking is lowest at 0.73%. These are descriptive secondary outcomes; unknown movies are not classified as disliked. A deployment decision prioritizing avoidance of known disliked films over recovery of liked films would require a different prespecified objective.

**Clipping creates ties in Anton's actual scorer.** For 27.55% of five-positive profiles, its highest prediction reaches the upper bound of four. Across all 2,000 five-positive profiles, an average of 39.795 movies share the highest score. Native stable tie handling then uses the common catalogue order. This is a diagnostic, not proof of the cause of its deficit. Testing unbounded ranking scores would be a separate model change and must not be tuned on this confirmation set.

Local Python scoring averaged about **0.027 ms/user** for conditional neighbours and **1.128 ms/user** for Anton on the five-opinion profiles, excluding model loading, network access and UI work. These are incidental host measurements, not iPhone latency or a controlled device performance study.

## Data boundaries and fairness

Fresh users were selected deterministically from the previous training population: **1,000 validation users and 2,000 confirmation users**. Every one of their ratings was removed before fitting either comparison model. The old 1,500 held-out users also remain excluded. The confirmation cohort was not scored until both candidates, adapters, metrics, source snapshots and policy were frozen. No model tuning followed development or confirmation results.

Eligibility required at least 35 recorded ratings and 21 positives on the common catalogue. This guarantees remaining positive targets after disclosing 20 likes, but selects relatively active MovieLens users. Five randomly sampled positives approximate onboarding; they do not reproduce self-selected favourites from real CineMatch users. These are newly reserved evaluation users from an existing dataset, not newly collected preferences. The algorithms had been developed previously using broader MovieLens data, so this cannot substitute for truly external validation.

Support opinions are an explicit app proxy: floor raw MovieLens ratings and clip to 1–5. Thus 4/4.5 map to Great (4), 5 maps to Fantastic (5), and 3.5 maps to Neutral (3). Anton receives the corresponding linear 1–4 scale, `1 + 0.75 × (verdict - 1)`. This mapping is an experiment assumption, not proof that historical stars have exactly the same meaning as app verdicts.

Anton retains its native fitting procedure: ten random raw historical validation ratings, at most 30 hidden rating targets per user, native MovieLens-to-1–4 conversion, least-squares fitting and four-decimal weights. Final fitted weights are content **0.1906**, audience **0.2884**, film deviation **0.7303**, tags **0.4951**, intercept **−0.0388**. Validation ranking values are development diagnostics because those users supplied fitting targets. Only confirmation results support generalization claims here.

Both models have the same training interactions, candidate set and watched exclusions. Hidden ranking targets are all remaining observed common-catalogue ratings; a rating of at least four is relevant. Unrated movies are unknown and receive no recovery credit. This rewards recovering preferences documented in the historical dataset and is affected by exposure/selection bias. There are no streaming-provider filters, daily-pick promotion or future-watch/temporal claims in this test.

Anton’s metadata vectors were rebuilt with its unchanged code on all **90,999 rated Letterboxd metadata records** available locally. Its audience vectors use only the shared training ratings. All 4,500 old/new held-out tag authors are excluded before tag vocabulary fitting, removing **27,578 catalogue tag rows**; **1,078,302 tag rows** remain. This strengthens held-out-user separation relative to the upstream unfiltered tag loader and is recorded as an adapter boundary.

## Identity correction and remaining integration issue

The source legacy mappings contained two Letterboxd films pointing to TMDB 768362 / MovieLens 282843. Cached independent metadata identifies this movie as **Missing (2023), 111 minutes**. Letterboxd 1001148 agrees. Letterboxd 1020663 is **The Missing (2023), 90 minutes**, with a different plot about a mouthless young man and an alien.

The benchmark rejects the incorrect 1020663 alias and retains 1001148, leaving 3,788 distinct movies. Exact evidence is preserved in `data_manifest.json` and `data_audit.json`. This comparison did not modify existing opinions or the app's legacy mapping bundle; that erroneous app alias remains a specific integration correction to make separately. The old 3,789 mapping-entry count must not be interpreted as 3,789 distinct common films.

## Verification and reproducibility

- **109 Python tests passed**, including 36 new tests for cohort separation, mapping correction, score parity, directed neighbours, clipping, exclusions, metrics and paired uncertainty.
- The independent data audit passed 16 checks, including exact reconstruction of all 21,431,215 training rows and tag exclusions.
- An independent evaluator recomputed seven ranking measures for every profile/method/user in both development and confirmation. Maximum aggregate discrepancy was **0.0**. It also checked native Anton and manually summed conditional top-ten rankings on ten primary profiles per split.
- The original selected NPZ and active frontend bundle hashes are unchanged; see `artifact_checks.json`.
- Source snapshots, candidate models, profiles, scores' resulting rankings, per-user metrics, immutable policy, training contract, freeze record, timestamps and one-time completion records are retained locally. Large artifacts remain under the repository's ignored artifact directory; compact results/provenance are outside that ignored directory and ready for version control. Changes are local and uncommitted.

Read the [final results](test_results.json), [development results](validation_results.json), [frozen policy](policy.json), [data audit](data_audit.json), [confirmation arithmetic audit](test_audit.json), [frozen hashes](frozen.json), [data manifest](data_manifest.json), [environment](environment.json) and [source attribution](../../vendor/anton_6e85958/README.md).

Implementation: [data preparation](../../prepare_app_comparison.py), [Anton adapter](../../anton_adapter.py), [runner](../../run_app_comparison.py), [metrics](../../app_comparison_metrics.py), [independent arithmetic audit](../../audit_app_comparison.py). Original large artifacts: [comparison run](../../../artifacts/experiments/app-model-comparison-20261119-v1).

To reproduce into a separate directory, from the `CineMatch` root:

```sh
.venv/bin/python experiments/prepare_app_comparison.py --output artifacts/experiments/app-model-comparison-replay/data
.venv/bin/python experiments/run_app_comparison.py train --run-dir artifacts/experiments/app-model-comparison-replay
.venv/bin/python experiments/run_app_comparison.py validate --run-dir artifacts/experiments/app-model-comparison-replay
.venv/bin/python experiments/run_app_comparison.py confirm --run-dir artifacts/experiments/app-model-comparison-replay
```

Use the recorded environment and unchanged sources (the original source snapshot is also in `frozen_source` within the run directory). Preparation refuses to overwrite a nonempty data directory. Training binds reusable components to their input/source hashes; confirmation refuses to run twice in one directory. A replay verifies reproducibility on this same known test set; it is not a new untouched confirmation cohort. Future model changes need new confirmation evidence.
