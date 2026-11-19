# CineMatch frontend integration

19 November 2026. Mikkel selected the campaign winner and explicitly requested its integration into the iOS app at `peter/CineMatch`. The frontend now wires Swipe, Search, onboarding and recommendations to the conditional-neighbour model and one catalogue of **8,937 verified TMDB movies**. The frozen experiment artifact and its learned graph remain unchanged. This is a local source integration; an iOS build/device run still requires Xcode.

Live verification checked all 9,014 unambiguous linked entries. The final policy requires MovieLens and TMDB to agree on IMDb identity: 8,937 passed, while 61 returned unavailable, 15 had conflicting IMDb IDs and one lacked sufficient confirmation. The three missing links and four duplicate-ID rows were excluded before requests. The [verification report](integration/tmdb_verification_report.json), [legacy mapping report](integration/legacy_mapping_verified.json), [frontend integration contract](../../peter/CineMatch/ml/conditional_knn/INTEGRATION.md) and [export manifest](../../peter/CineMatch/ml/conditional_knn/manifest.json) contain the reproducible evidence. The frontend bundle retains 9,021 graph nodes and 902,100 edges, with 3,789 conservative translations from the previous Letterboxd catalogue.

The selected artifact is `artifacts/experiments/cm-20261119-full-knn-conditional/model/model.npz`: positive conditional neighbours, 100 stored neighbours per movie, shrinkage 10, trained on the full training population. It contains 9,021 movies and 902,100 directed edges. Its final-holdout NDCG@5 was 0.485361 against popularity's 0.380212, a 27.66% relative gain. See [campaign review](CAMPAIGN_REVIEW.md) and [frozen final result](final-results/cinematch-systematic-20261119.json).

## What the app currently runs

Swipe, Search and Today all use `container.swipe`, now backed by the conditional-neighbour model, verified movie data service and supported-opinion adapter. Before integration this property loaded Anton's Letterboxd model. The hidden MF implementation remains available as historical code; replacing its bundle does not affect the active screens.

The campaign compared the selected model against MF and popularity. It did not establish an advantage over Anton's previous model in the app.

The active flow is:

```text
Swipe, Search and recommendation feedback
                  |
        Stored numeric movie verdicts
                  |
     New conditional-neighbour service
                  |
         Ranked movie candidates
                  |
  Existing streaming filter and daily pick
                  |
              Today screen
```

Supabase can continue handling login, opinions and recommendation events. TMDB can continue supplying posters and Danish streaming availability. Model scoring can run on the phone without a Python or ML server. Current recomputation still reads opinions from Supabase; local scoring alone does not make this flow fully offline.

Sources: [AppContainer](../../peter/CineMatch/CineMatch/App/AppContainer.swift), [MainTabView](../../peter/CineMatch/CineMatch/App/MainTabView.swift), [RecommendationService](../../peter/CineMatch/CineMatch/Core/Services/Protocols/RecommendationService.swift).

## Preserve the tested scoring rule

Do not load the new graph into the existing `OnDeviceRecommender`. That scorer caps likes at 25, combines mean and maximum similarity, adds a quality bonus, limits candidates to reached neighbours and removes duplicate titles. Those operations change the selected algorithm.

The new scorer must reproduce [score_neighbors](campaign_neighbors.py):

```text
score[candidate] = sum(max(rating[seed] - 3, 0)
                       * stored_weight[seed, candidate])
```

Keep all catalogue positions, including zero-score candidates. Use stored directed weights without symmetrising or renormalising them. If the complete score vector is zero, use training rating counts; this fallback happens before watched exclusions. Then exclude every seen movie, sort by descending score, and break ties in original MovieLens catalogue order. Preserve numeric precision and deterministic seed order through the export and Swift implementation. Do not add a popularity blend, quality bonus or title-based deduplication to the core scorer.

The existing verdict buttons already supply compatible values:

| Verdict | Saved value | Contribution to the model | Exclude as watched |
| --- | ---: | ---: | --- |
| Haven't seen | 0 | Not a rating | No |
| Horrible | 1 | 0 | Yes |
| Bad | 2 | 0 | Yes |
| Neutral | 3 | 0 | Yes |
| Great | 4 | 1 | Yes |
| Fantastic | 5 | 2 | Yes |

Pass numeric verdicts rather than only `likedMovieIDs`, which loses the Great/Fantastic distinction. Dislikes affect watched exclusion but do not subtract scores in this selected model. Keep the current five-positive onboarding gate for the first integration, counting only films recognised by the new model. The core scorer should still support and test the no-signal fallback. Half-star controls are not required.

Confirm migration `0006_neutral_verdict.sql` has been applied to the connected database before interpreting historical verdicts: it moves old Great=3 entries to Great=4. Its presence in the repository does not confirm deployment. Existing conversion of historical half-stars rounds them into verdicts; preserving their original fractional weights would require retaining the original numeric ratings.

Sources: [current scorer](../../peter/CineMatch/CineMatch/Core/ML/OnDeviceRecommender.swift), [Verdict](../../peter/CineMatch/CineMatch/Core/Models/Verdict.swift), [TasteSignal](../../peter/CineMatch/CineMatch/Core/Models/TasteSignal.swift). Parts of `docs/ml_contract.md` still describe obsolete verdict values and server behaviour; update that contract during implementation.

## How to investigate negative ratings

Bad ratings contain information worth preserving. Their zero contribution in the selected scorer is a limitation of that particular model, not a recommendation to discard dislikes. The graph was learned from people who liked pairs of films. Disliking one film does not necessarily imply disliking the films its fans liked, so negative evidence needs its own evaluation.

The campaign did test signed weights, where ratings below three subtract neighbour contributions. On the sample cosine graph, validation NDCG@5 fell from 0.463983 to 0.435832. The audit also found 16 users whose candidate scores contained negatives but no positives: unrelated zero-score films occupied all 80 top-five slots according to catalogue order. This tested rule failed; it does not establish that every use of dislikes fails, or that the result transfers to the selected full conditional graph. See [signed experiment](records/cm-20261119-knn-signed.json) and [screening audit](campaigns/cinematch-systematic-20261119-screening-audit.json).

A follow-up experiment should keep the selected model as its unchanged baseline, test a separately controlled negative penalty, and define sensible popularity-based behaviour when positive evidence is absent. Penalty strength must be chosen on validation data rather than assigning dislikes equal and opposite influence by intuition. Neutral and unseen answers should contribute no negative evidence. Measure both recovery of liked films and recommendations of films actually rated badly; unrated targets must remain unknown. If reusing co-like edges for penalties remains weak, a further candidate can learn associations from explicit low ratings, accounting for which films users actually rated.

This is a model change to validate separately from the frontend integration. Any replacement should preserve the original winner and use fresh confirmation data; the already inspected final holdout must not become a tuning set. No new negative-rating variant has been trained or evaluated as part of this integration overview.

## Make movie identity consistent

Use TMDB IDs for app-facing movies, stored opinions, posters and events. Keep the frozen MovieLens IDs and original graph positions internally, with an explicit mapping at the boundary. An integer alone cannot identify which ID system it belongs to.

The active Anton bundle contains 8,000 Letterboxd IDs and no populated TMDB IDs. The local `extra_data/id_mapping.csv` provides the following catalogue coverage:

| Mapping condition | Current app films |
| --- | ---: |
| Any MovieLens mapping | 5,700 |
| A unique mapping into the selected 9,021-film model | 3,347 |
| Of those, exact title and year | 3,275 |
| Of those, a year-offset match | 72 |

These are catalogue counts, not measured recovery of actual users' opinions. Even an exact title/year match needs identity checks; the 72 year-offset matches warrant particular review. `extra_data/merged_datasets.csv` also contains TMDB links and metadata, but should not be treated as an automatically verified identity registry.

The gap divides into two different cases: 2,353 films already map to MovieLens but fall outside the selected model, while 2,300 have no entry in the current mapping. Repairing links can recover some of the second group. It cannot add graph rows for the first group. The 9,021-film catalogue originally required at least ten ratings among the initial 10,000 sampled training users; full-data experiments deliberately retained it for comparable evaluation. See [data preparation](../src/cinematch/data.py) and [campaign catalogue checks](campaign_common.py).

The app should use the new model's full supported catalogue, not only its intersection with the old Letterboxd bundle. Use a shared movie registry keyed by TMDB movie ID, with explicit Letterboxd/MovieLens source IDs, mapping provenance and verification state. Model membership should be recorded separately from identity. New onboarding choices should come from supported movies. Broader Search can store opinions on unsupported movies if desired, but must not imply those opinions already affect this model.

Repair legacy links using title-convention normalization as candidate generation, then verify identity through available external IDs and metadata. Common differences such as `The Dark Knight` versus `Dark Knight, The` can hide real overlap. Preserve all original opinions while migrating verified links; show returning users additional supported onboarding choices only if their recovered positive inputs are insufficient. An expanded-catalogue model is a separate retraining and validation task, not an ID conversion.

The [reproducible identity audit](audit_frontend_identity.py) finds **645 additional unverified candidate links into the selected model** among the 2,300 unmapped films. It normalizes Unicode, punctuation and English article placement, requires the same year, and checks uniqueness against the entire MovieLens catalogue before testing model membership. Another 74 candidates fall outside the model, one title/year key is ambiguous, and 1,580 have no match under this limited rule. See the [audit summary and input checksums](integration/identity_audit_summary.json) and [candidate links](integration/identity_mapping_candidates.json). Every generated candidate is explicitly unverified and unaccepted; these files do not migrate users or change the accepted mapping.

The selected MovieLens catalogue has three missing/invalid TMDB links and four rows in duplicate-TMDB groups. Excluding every problematic row leaves 9,014 structurally unique mappings. That does not by itself verify movie identity or movie-only eligibility. Preserve all 9,021 internal graph positions and record serving eligibility separately while resolving these cases. Do not silently merge nodes or alter the frozen training artifact.

The implementation uses the existing `stars` TMDB storage namespace for new opinions and events. Its name is historical; model versions identify the scorer. The supported-opinion adapter translates legacy `swipe` and `search` Letterboxd records on read, preserves original records and timestamps, and selects the latest supported verdict per movie. Unresolved opinions remain in their source stores for later recovery. No existing integers are reinterpreted as a different ID system, and no bulk database migration is performed.

Daily picks now use `dailyPick.tmdb.knn.v1.<user>`. A one-time local transition preserves answered-day/streak history from the old key while leaving its Letterboxd movie references behind. Old storage remains intact, and a marker prevents old progress from returning after an explicit reset. Historical event identifiers stay associated with their original pipeline/model; new queued events capture their originating user.

## Restricting the app to model movies verified in TMDB

An app catalogue defined as the intersection of the selected model and verified TMDB movie records addresses unsupported new inputs without retraining. Start from the selected 9,021 entries, not the complete MovieLens dataset: a film can exist in MovieLens without belonging to this trained model.

The local `links.csv` gives the following coverage. These are positive integer identifiers, not results of live TMDB lookups.

| Scope | MovieLens entries | With a TMDB link | Missing or invalid link |
| --- | ---: | ---: | ---: |
| Selected trained model | 9,021 | 9,018 | 3 |
| Complete local MovieLens 32M catalogue | 87,585 | 87,461 | 124 |

The selected model's linked rows represent 9,016 distinct TMDB IDs because two pairs share an ID. Excluding all four rows in those pairs gives **9,014 unambiguous linked candidates**. This is a conservative starting set, not a live-verified final app count. A missing link does not prove a film is absent from TMDB. The complete dataset's link coverage is 99.86%; its entries are not automatically supported by this model. The [audit summary](integration/identity_audit_summary.json) now records both scopes and explicitly marks live verification as unperformed.

1. Join the selected model's MovieLens IDs to `links.csv` and exclude unresolved or ambiguous mappings from the first app catalogue.
2. Verify each remaining TMDB movie record exists and identifies the intended film. Record confirmed failures separately from temporary network or service failures. TMDB movie existence does not establish Danish streaming availability.
3. Export one eligible catalogue containing the public TMDB ID, internal MovieLens ID and original model position, plus display metadata. Keep all frozen graph positions and weights intact; eligibility is a separate filter.
4. Make Swipe, Search, onboarding and recommendation displays use that catalogue. Rank eligible output candidates from the full score vector rather than filtering only an already truncated list.
5. Migrate or filter previously saved opinions and check catalogue membership inside the recommendation service. Translate accepted TMDB IDs to original model positions there. UI restriction alone cannot protect against old saved records or stale app versions.

With these rules, every input accepted by the scorer is supported by the selected model, and every displayed recommendation has passed the chosen TMDB eligibility check. Film identities still need a lookup table because the app and model use different identifiers. Supporting the two duplicate pairs later requires an explicit policy for input assignment and watched exclusions, rather than duplicating one opinion across both graph nodes.

Reference contracts: [MovieLens link identifiers](https://files.grouplens.org/datasets/movielens/ml-32m-README.html#links-data-file-structure-linkscsv) and [TMDB movie details by ID](https://developer.themoviedb.org/reference/movie-details). The live sweep and frontend catalogue change are now complete, with the final results recorded at the beginning of this document. Production eligibility requires matching IMDb IDs; title/year agreement alone is insufficient.

## Implementation boundaries

The implemented components and their responsibilities are listed below. Exact file names and the current schema are in the frontend integration contract linked above.

| Area | Work |
| --- | --- |
| ML exporter | Export the frozen graph, original catalogue order, training counts, MovieLens/TMDB mapping, serving eligibility, titles and genres. Include format/model versions, source checksum and reference scoring cases. Measure the exported size; the 3.41 MB NPZ is not an iOS JSON size estimate. |
| `Core/ML` | Add a dedicated conditional-neighbour file type, cached store and scorer. Validate dimensions, index bounds, unique public identities and finite weights. Keep all-catalogue scoring separate from serving filters. |
| Local recommendation service | Read current numeric opinions, map recognised IDs, exclude all seen films and return candidates using the existing `RecommendationService` protocol. Use a distinct model version and record usable input counts. |
| Movie data service | Supply supported onboarding films, title lookup and full-catalogue Search. Implement Search explicitly: the default search method only searches whatever `allTitles()` returns, which can be an onboarding subset. |
| Persistence | Add the identity namespace/migration, preserve old opinions, account for unknown films and handle daily-pick references. A new pipeline value also requires updating database constraints and Swift enums. |
| `AppContainer` and `MainTabView` | Wire all three active screens to the new movie, signal and recommendation services. Do not switch only the hidden MF pipeline or only Today. |
| Explanations and documentation | Use “Based on your ratings,” or name a genuinely contributing seed. Do not inherit director/theme explanations from Anton's content model. Update the app's model contract and bundle documentation. |

The existing `MFModelStore`, `MFRecommendationService` and `MatrixFactorizationTests` offer useful patterns for loading, opinion mapping, versioned events and Python-to-Swift reference cases. Reuse the structure, not the matrix-factorisation scoring formula. Current cards do not display a match percentage; retain raw scores for ranking and logs.

The service currently returns 50 candidates. Today displays a daily pick plus ten results, expandable to twenty, and scans candidates in rank order for streaming matches. Keep that presentation layer initially, including separate model rank and displayed-position logging. If streaming filtering exhausts the candidate pool, request more candidates from the same ranking instead of changing model weights. Daily-pick promotion can deliberately change displayed order.

## Implementation and verification order

1. Produce the identity report and export contract. Decide how unresolved mappings are represented and how existing opinions are preserved before switching storage.
2. Export the selected model and Python reference cases without retraining or changing its frozen artifact. Cover mixed Great/Fantastic ratings, dislikes, unknown IDs, zero-signal fallback, watched exclusions and ties.
3. Implement the Swift scorer and compare ordered IDs and numeric scores with those reference cases. Validate the full frozen catalogue separately from the app's serving eligibility rules.
4. Add the data and recommendation adapters, then exercise a rerunnable identity migration against copied test records. Verify Search can find supported movies outside onboarding and that seen movies remain excluded.
5. Switch the active dependency wiring and test Swipe → saved opinion → Today → feedback → recomputation, including restart, daily picks and provider filtering. Measure bundle loading, memory and scoring on a target device.
6. Evaluate the resulting app experience against popularity and the existing app model with a new, appropriate comparison. Do not retune on the frozen final holdout. Five self-selected positive answers, a TMDB-eligible catalogue and streaming filters differ from the experiment's five random historical ratings and full catalogue, so the 27.66% result is not an app uplift prediction.

The verified export and Swift services are implemented. Search now explicitly promises title search, matching the available verified metadata; the previous director/actor prompt was removed. Full iOS build and device measurements remain separate from the portable model/service checks because this environment has Command Line Tools but no Xcode installation.
