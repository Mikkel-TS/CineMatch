# Benchmarking CineMatch's app models

**Comparison now executed:** [final report and reproducible evidence](comparisons/app-model-comparison-20261119-v1/REPORT.md). The direct comparison pins the newer `6e85958` recommendation function, which does not apply the documented ten-rating fallback. It compares refitted standalone methods on the same 3,788-movie catalogue and 2,000 newly held-out evaluation users; no hybrid was trained. The design and earlier source reviews below remain as provenance.

19 November 2026. Source review and proposed evaluation design, not a new experiment or a decision to change the app. Initial inspection: [`fa9dc38bf56affd1df602470cfe7cb9077c6a229`](https://github.com/AntonDev29/CineMatch-AntonModel/tree/fa9dc38bf56affd1df602470cfe7cb9077c6a229). Rechecked after Mikkel reported an update: [`57efce37717baca62249629585fccd6082013efa`](https://github.com/AntonDev29/CineMatch-AntonModel/commit/57efce37717baca62249629585fccd6082013efa), committed at 11:21:17 UTC that day. No comparative scores were computed and no training or serving code was changed.

## Latest repository update

The new commit adds [`rating_model.py`](https://github.com/AntonDev29/CineMatch-AntonModel/blob/57efce37717baca62249629585fccd6082013efa/rating_model.py) as the main rating-model entry point, moves five earlier scripts into `Test/old`, and updates the report's file list. It consolidates the existing model, rating-error checks, catalogue ranking checks and export. `cinematch.py`, `movielens.py` and `export_for_app.py` are unchanged. No new numerical results or trained JSON artifacts are committed in this update.

The consolidated model combines Letterboxd content, MovieLens audience vectors, user tags and a smoothed film-average term. It uses deviations from the user's own mean rating, with blend weights fitted on validation users. Ratings below the user's mean can reduce predictions for similar movies; this is relative rating evidence, not a fixed penalty attached to the app's Bad/Horrible labels. Its expected rating scale is 1–4, requiring an explicit mapping from CineMatch's current verdicts for comparison or integration. Defaults remain 8,000 movies, 50 neighbours per source, and an exported instruction to use film-average ranking before ten ratings. The final export is `rating_model_test.json`; the older app bundle is unchanged. [Defaults](https://github.com/AntonDev29/CineMatch-AntonModel/blob/57efce37717baca62249629585fccd6082013efa/rating_model.py#L1-L55), [centred scoring](https://github.com/AntonDev29/CineMatch-AntonModel/blob/57efce37717baca62249629585fccd6082013efa/rating_model.py#L191-L257).

This changes the practical comparison target: **compare the new consolidated rating model directly with our selected conditional model and the fixed baselines first**. Keep the old Anton app scorer as an optional historical reference. Combining either with our model remains a separate, optional experiment; no combination is recommended for adoption without comparative evidence.

The current script includes full-catalogue ranking at 3, 5, 10, 15 and 30 random historical ratings, using AUC, precision at five and recall at twenty. The earlier archived-import problem no longer blocks this new entry point, although datasets and prebuilt indexes remain external requirements. It would now be inaccurate to describe the current workflow as only predicting ratings or ranking a small set of known-rated movies. Its target set is still capped at 30 withheld ratings per user, with the other recorded movies excluded. [Ranking implementation](https://github.com/AntonDev29/CineMatch-AntonModel/blob/57efce37717baca62249629585fccd6082013efa/rating_model.py#L297-L357).

One material benchmark/export mismatch remains: ranking uses dense similarities and refits weights separately for each profile size. The exported model uses weights fitted once at ten ratings on sparse 50-neighbour signals. The ranking benchmark also does not enforce the ten-rating fallback. Rating-error checks do compare sparse and dense scoring, but that does not establish full-catalogue ranking performance of the exported serving policy. Our direct comparison should score the exact proposed serving rule, including that fallback. [Benchmark fitting](https://github.com/AntonDev29/CineMatch-AntonModel/blob/57efce37717baca62249629585fccd6082013efa/rating_model.py#L430-L460), [export fitting](https://github.com/AntonDev29/CineMatch-AntonModel/blob/57efce37717baca62249629585fccd6082013efa/rating_model.py#L495-L511).

Evaluation audience vectors and film averages exclude validation/test users. Tag features still omit user/timestamp filtering, and final export loads a pre-existing audience checkpoint and computes film averages from all ratings. The independent confirmation requirements below therefore still apply. The prior five-identical-positive-opinions case also remains: centred preference signals become zero, while the exported policy would already be using its below-ten-rating fallback. [Evaluation setup](https://github.com/AntonDev29/CineMatch-AntonModel/blob/57efce37717baca62249629585fccd6082013efa/rating_model.py#L477-L492), [final export](https://github.com/AntonDev29/CineMatch-AntonModel/blob/57efce37717baca62249629585fccd6082013efa/rating_model.py#L416-L425).

## What we can currently conclude

The selected conditional-neighbour model beat count-based popularity and the two matrix-factorisation references on the [previous frozen benchmark](CAMPAIGN_REVIEW.md). That campaign did **not** compare Anton's app model. Its 27.66% relative NDCG@5 improvement is not evidence of superiority over Anton, nor evidence that a hybrid helps.

The local frontend now selects conditional neighbours for the active flows. Anton's older scorer and bundle remain in the repository. The latest Anton GitHub repository also contains a separate rating-prediction experiment; it must not be confused with the retained app bundle.

## Which Anton model?

| Candidate | Actual mechanism | Implication for comparison |
| --- | --- | --- |
| Retained Anton app bundle, version `2026-11-17` | Weighted content similarity plus a MovieLens audience component; average and strongest similarity to liked films, with a quality bonus | Already mixes content and collaborative information. Combining it with conditional neighbours is not simply adding content to a collaborative model. |
| Current Anton Python recommender | Content blocks include plot keywords, cast, crew, genre, era and other metadata; optionally adds audience vectors; ranks using mean/best similarity plus quality and popularity bonuses | The Python path is not identical to the deployed sparse Swift path. Freeze the intended implementation. |
| Anton's newer rating blend | Predicts ratings from deviations from a user's mean, content/audience/tag similarities and film-average bias | Separate candidate, not evidence about the old app scorer. Its report describes a ten-rating personalization threshold and an export not yet intended for the app. |
| Selected conditional model | Positive item co-occurrence, conditional edge weights, support shrinkage, weighted positive inputs | Learns which movies are liked together; does not currently subtract negative-neighbour scores. |

The retained Anton JSON contains 8,000 Letterboxd movies and 50 neighbours per movie. Its feature labels include MovieLens rating patterns, with positive audience contributions stored on 250,138 of 400,000 edges. This establishes that audience information is present; it does not recover its precise training split. Bundle SHA-256: `eacaa9f826d0726f20cac8deb9064f621b75d10f7afdd5bd753ea7f66e67b4a9`.

The Swift scorer accepts at most 25 distinct liked IDs, considers their stored neighbours, scores half the mean similarity plus half the strongest similarity, adds `0.02 × quality`, excludes watched movies and deduplicates normalized titles. The Python recommender also has a default popularity bonus and scores dense similarities. Neither should be substituted silently for the other in a benchmark. A content-only removal requires rebuilding similarities and neighbour selection; deleting explanation labels does not remove audience influence.

Sources: [Anton feature/scoring implementation](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/cinematch.py), [exporter](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/export_for_app.py), [rating-model report](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/notes/RATING_MODEL_REPORT.md), [local Swift scorer](../../peter/CineMatch/CineMatch/Core/ML/OnDeviceRecommender.swift), [active app wiring](../../peter/CineMatch/CineMatch/App/AppContainer.swift).

## Two questions, two catalogue policies

**Controlled ranking comparison:** which system best ranks the same movies from the same disclosed opinions? The verified legacy mappings contain 3,789 Letterboxd entries from the retained Anton bundle, representing **3,788 distinct TMDB movies** in the current 8,937-movie app catalogue. One canonical movie therefore has multiple legacy entries. Define a deterministic identity/deduplication policy before evaluation; never count that movie twice as a seed, recommendation or target. These counts describe catalogue overlap, not recovered user opinions. They apply to the retained legacy bundle; verify membership again for the new rating-model export before assigning it the same coverage count.

Use this verified intersection for an initial controlled comparison. Publish how many users, seeds and relevant targets this removes, and define user eligibility before examining model outcomes. Keep each deployed model's internal neighbour truncation as part of its implementation; do not silently rebuild top neighbours within the smaller intersection. A separate full-catalogue scoring experiment may investigate the effect of truncation.

**App comparison:** which complete system helps users on the full **8,937-movie catalogue**? The permitted catalogue, onboarding UI, streaming settings, watched exclusions and presentation stay the same. Rebuilding Anton with verified identities and sufficient metadata for that catalogue would provide the cleanest comparison. Otherwise declare its unsupported-input and missing-score fallback in advance, report coverage and empty/short lists, and include those failures in the result. Do not quietly remove difficult users or movies for only one variant. Success on the smaller intersection cannot establish full-catalogue superiority.

## Systems to compare

| System | Purpose |
| --- | --- |
| P: training rating-count popularity | Established simple baseline; identical watched and eligibility exclusions |
| C: selected conditional-neighbour method | Current app choice |
| A: consolidated Anton rating model | Updated primary alternative; freeze the exact proposed export and serving policy |
| L: retained Anton app scorer, optional | Historical reference; keep its version and training provenance distinct from A |
| H: optional validation-selected combination of C and A | A later experiment to test whether retaining both adds value |

Add a smoothed training mean-rating baseline because Anton's report uses average rating under the name popularity. Keep those two baselines separately named. Optional diagnostic variants remove audience/content components from a named Anton version. This design was updated before any comparison was executed; once evaluation starts, do not switch A between the newer rating blend and the older app scorer.

All systems receive the same disclosed opinions. Define the new rating model's 1–4 conversion and ten-rating fallback explicitly; preserve its native centred scoring. For the legacy scorer, converting binary likes into Great/Fantastic weights, adding dislike penalties, or changing its 25-like cap are separate experiments. Use the same candidate eligibility, canonical watched exclusions and output size. Keep presentation rules identical and record any remaining native title-deduplication behavior. A later controlled removal can isolate that behavior from scoring.

## Data and onboarding protocol

1. Split users into training, validation and untouched confirmation groups before fitting any rating-dependent component. Train graph edges, audience vectors, popularity counts and rating means only on training users. Validate feature provenance too: user tags and aggregates can expose held-out or future information.
2. Give each evaluation user the same five supported positive onboarding opinions, and hide their other labels from all recommenders. Disclosed negative/neutral opinions exclude watched movies; undisclosed future labels must not shape the profile. Define how MovieLens half-star values map to the app's verdicts and freeze that rule. Also test later profiles, for example 10 and 20 positive opinions, with identical disclosure across models.
3. Evaluate against the full permitted catalogue and held-out target set; do not give one model a small, easier set of previously rated candidates. Preserve native sparse output: missing ranks earn no recovery credit, and short lists receive no silent padding unless a declared serving fallback supplies it. Do not replace Anton with dense scoring merely to fill the list. Declare whether undisclosed rated movies remain recommendation targets and keep that convention identical.
4. Score withheld recorded likes, while retaining explicit low ratings as a separate harmful-recommendation measure. Unrated movies have unknown relevance. They receive no recovery credit, but should not be described as verified dislikes.
5. Report seed-selection sensitivity and participant exclusions. Randomly sampled positive seeds approximate onboarding, but cannot reproduce people selecting their own favourites. Use real app onboarding in the user study. Resampling the same person's seeds does not create independent users.

The old campaign used five random historical ratings, not five self-selected positives. Its final set has already been inspected. Do not tune a hybrid on it and claim another untouched final result. A new partition of users already included in a model's training is also not a holdout: refit benchmark copies excluding those users, or collect genuinely new app outcomes. Preserve the existing production artifacts while doing this.

Anton's normal audience builder reads all mapped ratings, so the retained bundle cannot be assumed safe for evaluation on our MovieLens holdouts. Rebuild it under the shared split, or establish its provenance first. For a temporal next-watch claim, enforce a global time cutoff on training interactions, tags and popularity too; merely splitting each user's history does not prevent learning from other users' future activity.

Sources: [audience builder](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/movielens.py#L141-L216), [tag feature construction](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/evaluate_content_blend.py#L54-L63). Historical recommendation data is affected by exposure and user self-selection; applying the same labels to every model does not remove that bias. See [Schnabel et al., 2016](https://proceedings.mlr.press/v48/schnabel16.html).

## What to measure

Choose **NDCG@5** as the primary offline measure for continuity: it rewards putting withheld liked films near the top. Also report top-one recovery for Tonight's Pick, HitRate@5 (at least one known positive), Recall@10, explicit-low-rating recommendations, and per-user differences. Report users without relevant targets separately and retain them for coverage measures.

Guardrails should include unique-film exposure and concentration, within-list diversity, supported-profile coverage, fallback frequency, empty/short lists after streaming filters, and measured device loading/scoring cost. Define allowable regressions in advance. Novelty or diversity alone is not proof of enjoyment; use them alongside relevance and user judgement. Report cold-start, mature-profile and mainstream/niche segments selected before seeing results, with uncertainty and sample counts.

Use paired comparisons: each model is evaluated for the same users. Bootstrap whole users, retaining repeated seeds/sessions within each user's group. Report confidence intervals for differences, not only separate model averages. Select a minimum worthwhile improvement before confirmation, and account for multiple planned claims. An interval crossing zero is inconclusive; failure to detect a difference is not proof of equivalence. If equivalence or noninferiority matters, predefine its margin and test that claim explicitly.

## Testing the combination

Executed on 19 November 2026: [fifteen-input/five-output ensemble experiment](comparisons/app-model-ensemble-20261119-v1/REPORT.md). Eight validation blends failed to beat conditional alone; the frozen best nonzero challenger also lost slightly on newly reserved confirmation users. The report preserves the exact protocol, complementary-output diagnostics and limitations. The following describes the design rationale.

The two raw scores have different meanings and scales. Start with a small, declared validation search over calibrated scores or rank fusion. For example, weighted reciprocal-rank fusion can combine each model's candidate ranking without treating its numeric scores as comparable. Define list depth, tie handling and missing-candidate contributions. Include exact standalone endpoints; do not let a fallback silently change what zero weight means. [Cormack et al., 2009](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/) provides a fusion method, not evidence that it will help CineMatch.

Freeze the chosen combination before confirmation. Compare it against both standalone models and the strongest validation baseline. A combination is supported when it improves the declared outcome by a useful amount, with uncertainty supporting the claim, while meeting coverage and cost requirements. Removing either model should worsen the outcome before claiming both are necessary for that outcome.

Inspect which users and relevant movies each model gets right when the other fails. Different lists alone do not demonstrate useful complementarity; an oracle that picks the best result after looking at hidden labels is only an upper bound, never a deployable benchmark result. Because A already includes audience information, compare H against conditional plus content-only to determine whether additional audience information or content actually contributes.

## User study and live app validation

For an initial study, participants provide the actual onboarding inputs once. Present a deduplicated pool from the competing systems with model identities hidden, randomized order and identical movie information. Ask whether each movie was already seen, whether the participant wants to watch it, and later whether a watched recommendation was enjoyable. Report blinded judgement as such; it is not a randomized estimate of long-term app performance.

For live confirmation, randomly assign users to stable variants, keeping UI, catalogue, onboarding and provider settings consistent. Pick a primary outcome such as the proportion of eligible recommendation sessions leading to selection of an unseen movie to watch; measure later enjoyment as a follow-up. Define denominators, attribution windows and minimum useful uplift first. Estimate sample needs from pilot rates and the target effect, with user-level clustering. More ratings from a handful of people do not substitute for independent participants. A small study with wide intervals should be reported as exploratory.

Current app logging records recommendation batches, list impressions and verdicts about already-seen movies. It does not yet record a controlled variant assignment or actual choice-to-watch/completion. List impressions are logged when items enter the displayed list, not when every card is confirmed visible. 'Another pick' is currently local state. Existing seen/liked response ratios are unsuitable as the primary discovery outcome: they can reward recommending familiar films and omit nonresponders.

Before a live benchmark, log variant and model versions, canonical movie IDs, profile/candidate snapshot references, ranking and displayed position, actual exposure, provider filters, choice/pass actions and attributable follow-up. Compare both the raw ranker and final app list, because streaming filtering and daily-pick promotion can change what the user sees. Keep explanations identical in the model-isolation study; test explanatory copy separately if it is part of the proposed product advantage.

Sources: [event model](../../peter/CineMatch/CineMatch/Core/Models/RecommendationEvent.swift), [recommendation view model](../../peter/CineMatch/CineMatch/Features/Recommendations/ViewModels/RecommendationsViewModel.swift), [active contract](../../peter/CineMatch/docs/ml_contract.md).

## Initial inspection of earlier Anton results

This section records the earlier scripts at `fa9dc38`; the new consolidated workflow and its remaining gaps are described above.

The rating experiments often rank at most 30 held-out movies that a user already rated. Their top-five liked fraction is not full-catalogue HitRate@5. An archived full-catalogue evaluator also exists, but uses a different catalogue, exclusion/target protocol and average-rating baseline. Its current imports refer to modules absent from the checked-in tree. We inspected the code and report; we did not reproduce its numerical results.

The report acknowledges that tags and the ten-rating cutoff were chosen after test inspection. Those choices need new confirmation. Its newer centered-rating blend also has a specific onboarding case to test: if all five positive verdicts are identical, all deviations from the profile mean are zero, leaving the film-average component. This does not apply to the old liked-films scorer.

Sources: [restricted candidate evaluation](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/evaluate_factor_model.py#L54-L98), [archived catalogue evaluator](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/Test/old/evaluate_catalogue_ranking.py), [acknowledged test reuse](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/notes/RATING_MODEL_REPORT.md#L180-L188), [centered scoring](https://github.com/AntonDev29/CineMatch-AntonModel/blob/fa9dc38bf56affd1df602470cfe7cb9077c6a229/tune_rating_blend.py#L46-L80).

## Decision the benchmark should support

Prefer a standalone model if the combination does not provide a credible, useful gain on the declared objective, or if added loading/runtime cost outweighs it. Prefer a hybrid only if frozen confirmation supports its extra value. If one model improves discovery or coverage while another improves retrospective recovery, report the tradeoff and make the product objective explicit; do not compress contradictory outcomes into an unexplained overall winner.

The eventual defensible claim is bounded: a named system improves a named outcome for a specified population, onboarding procedure and catalogue, by an estimated amount with uncertainty. None of the current evidence establishes that either model universally beats the other or that their combination is stronger.
