# CineMatch campaign: what changed and what we learned

Review date: 19 November 2026. This document interprets **40 completed validation experiments** across seven model-change families and subsequent removal tests, followed by **one frozen-candidate final evaluation**. The family reviews and selection reasoning were written from validation evidence before the final result; the final section reports the separate one-time outcome. Correctness tests and the independent screening audit are verification work, not additional model experiments. All 40 recorded attempts completed; disappointing model results are retained as regressions or inconclusive findings, not mislabelled as execution failures.

The main finding is that **positive item-to-item co-occurrence is much more effective than the current explicit-rating matrix-factorisation model on this benchmark**. After selection was frozen, the conditional-neighbour model achieved **0.485361 final-holdout NDCG@5 versus popularity's 0.380212: a 27.66% relative improvement**, with a positive paired uncertainty interval. It also beat both MF references. This supports advancing this candidate to deliberate app-integration testing; it does not establish online satisfaction or authorize automatic deployment.

The complete attempt list and exact comparisons remain in [leaderboard.json](leaderboard.json) and [JOURNAL.md](JOURNAL.md). [Per-run reviews](campaigns/cinematch-systematic-20261119-reviews.json) contain an individual conclusion, decision and limitations for every attempt, together with the record hash before review merging. Original plans, checkpoints, source snapshots and adoption decisions remain authoritative; this review does not replace their provenance.

Mikkel subsequently selected the winning model and authorized its integration into the iOS app. The [frontend integration record](FRONTEND_INTEGRATION.md) documents the implemented local scorer, the 8,937 verified TMDB movies shared by all active flows, and compatibility with existing opinions in `peter/CineMatch`. The frozen experiment artifact remains unchanged; iOS build/device validation requires Xcode.

## The comparison held fixed

Every main-benchmark result uses the original 500 validation users, their exact five onboarding ratings, withheld targets, 9,021 candidate movies in the original order, watched-item exclusions, binary relevance threshold of four stars and established metric implementation. Two users have no relevant targets, leaving 498 in ranking averages. They remain included in coverage and rating-error calculations where applicable.

Sample runs use 1,500,603 ratings from 10,000 training users. Full-data runs use 30,495,721 ratings from 199,447 users, restricted to the same catalogue. Validation, historical test and reserved fresh-final users are excluded from training. Popularity controls fitted on the sample and full populations yield the same aggregate metrics on these validation profiles; both controls remain recorded separately.

This is retrospective recovery of recorded likes, not a future-watch experiment. Unrated movies have unknown relevance but earn no ranking credit. Popular films therefore have more opportunities to appear among observed positives. The result does not measure novelty satisfaction, streaming availability, self-selected onboarding favourites or performance outside the fixed catalogue.

## Leading validation results

| Model | NDCG@5 | Hit rate@5 | Recall@5 | Distinct movies in all top fives | Rating RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original sample MF | 0.151901 | 38.15% | 1.62% | 231 | 0.92470 |
| Existing full-user MF | 0.167518 | 40.96% | 1.72% | 441 | 0.92912 |
| Popularity, either training population | 0.379548 | 74.30% | 4.34% | 8 | Not applicable |
| Sample conditional neighbours | 0.488362 | 82.73% | 6.17% | 82 | Not applicable |
| Full conditional neighbours, shrinkage 10 | **0.494177** | 82.93% | 6.23% | 71 | Not applicable |
| Full conditional neighbours, no shrinkage | 0.494207 | 82.93% | 6.23% | 71 | Not applicable |
| Full cosine neighbours, K300/shrinkage 50 | 0.479044 | 83.33% | 6.17% | 484 | Not applicable |
| Full BPR32, seed 42, 1M pairs/epoch | 0.450809 | 82.93% | 5.93% | 57 | Not applicable |

The regularized full conditional model improves NDCG by **0.114629, or 30.20% relative**, over popularity. Its paired user-bootstrap 95% interval is [0.098119, 0.132605]. Differences from original MF and full-user MF are 0.342276 [0.314926, 0.370840] and 0.326659 [0.299377, 0.355504], respectively. These intervals describe sampling of the validation users. They do not correct for selecting among 40 attempts.

The no-shrinkage score is numerically highest by just 0.00002985. Hit rate, recall and distinct-title count are unchanged. That difference is too small to support a practical preference; retaining the previously established regularized parent avoids selecting on a negligible final decimal.

## What each family established

### 1. New-user profile fitting: useful sensitivity, insufficient improvement

Reducing MF fold-in regularization from 1 to 0.1 raises NDCG from 0.151901 to 0.205594. Reducing it again to 0.01 falls to 0.198521, while RMSE deteriorates to 1.12209. The 0.01-versus-0.1 interval spans zero. Five ratings provide too little evidence to conclude that ever-weaker regularization helps.

Removing the fitted user intercept at lambda 0.1 scores 0.204319, essentially unchanged from its 0.205594 parent. Replacing the global mean with the onboarding mean scores 0.148931 and removes the preceding gain. Neither refinement warrants selection. These tests concern particular fitting rules on the existing checkpoint, not every possible cold-start estimator.

### 2. Movie biases: a material weakness in the current MF ranking

Removing movie biases scores 0.233599; halving them scores 0.248111. Count-dependent shrinkage scores 0.281979 at strength 100 and 0.310111 at strength 1,000. The latter improvement over shrinkage 100 has a paired interval entirely above zero, but it still trails popularity and worsens rating RMSE.

This is evidence that the current checkpoint's treatment of movie priors is important to its ranking deficit. These transformations change both the residuals used to fit a new profile and the candidate scores, so the experiment does not isolate only the final ranking bias. It also does not establish an optimal shrinkage strength.

### 3. Popularity hybrids: rescue weak MF, add little beyond popularity

A 25% standardized-MF blend scores 0.375830, close to popularity. A 50% blend falls to 0.350007. A confidence-dependent blend scores 0.386272, and the full-user MF 25% blend scores 0.393960; both apparent improvements over popularity have intervals crossing zero.

Reranking the 100 most popular eligible movies with MF scores 0.306680. Restricting where the weak ranker can act helps relative to original MF, but still loses to popularity. Candidate generation remained a scoring stage: excluded-from-candidate titles received finite low scores, so evaluation membership was unchanged.

Once conditional neighbours work well, adding a 50% popularity blend lowers NDCG from 0.494177 to 0.488018, with parent interval [-0.009677, -0.002801]. The successful model already combines personal co-like evidence with a preference for broadly liked films. Extra popularity is not automatically an improvement.

### 4. MF retraining: better rating prediction is not the same objective

Eight factors scores 0.119530; 32 factors scores 0.138805. Neither improves the original MF. Training with count-aware bias regularization reaches 0.207051, still far below popularity.

Huber loss is particularly informative: rating RMSE improves to **0.91645**, while NDCG falls to **0.115829**. This is a concrete example of the mismatch between predicting observed star ratings accurately and recovering the best five recorded likes.

These screens used one seed, a fixed fold-in penalty of 1 and a maximum of 12 epochs, whereas the original policy allowed 20 epochs and selected among three penalties. The eight-factor screen reached its 12-epoch cap. Consequently, the results assess these configurations under the campaign policy; they do not prove that factor count alone caused every difference, or that MF as an entire model class cannot work.

### 5. Item neighbours: the strongest direction, with separable causes

The first positive-feedback cosine graph already scores **0.463983**, clearly above popularity. A combined change to K300 and shrinkage 50 scores 0.479735. The later isolates are necessary to interpret that result:

| Change from cosine K100/shrinkage 10 | NDCG@5 | Difference from parent | Paired 95% interval |
| --- | ---: | ---: | --- |
| Increase K alone to 300 | 0.470711 | +0.006729 | [-0.004991, 0.018999] |
| Increase shrinkage alone to 50 | 0.484459 | +0.020476 | [0.010032, 0.031311] |
| Increase both | 0.479735 | +0.015752 | [0.003936, 0.028299] |

The stronger evidence concerns support shrinkage on the sample cosine model, not a larger neighbourhood. A three-times-larger graph is not justified by the K-only result.

Conditional similarity estimates co-likes divided by the seed film's positive-user count, then applies support shrinkage. It scores 0.488362 on sample data and 0.494177 on full data. The full-minus-sample interval is [-0.000660, 0.013014], so most of the gain over MF and popularity was already available from the model change on the smaller training population. Full-data cosine K300/shrinkage 50 scores 0.479044, effectively unchanged from its sample parent.

Replacing co-likes with all co-ratings harms both tested settings: sample cosine falls from 0.463983 to 0.396532; full conditional falls from 0.494177 to 0.473514. The full comparison has a negative parent interval [-0.031192, -0.009240]. Retaining information about which movies were actually rated positively matters in these binary graph models.

The signed variant scores 0.435832, below positive-only cosine aggregation. The [independent screening audit](campaigns/cinematch-systematic-20261119-screening-audit.json) reproduced every audited per-user metric and top-five order and found a specific failure mechanism: **16 nonfallback users have negative candidate scores but no positive candidate score**. Unrelated zero-score movies then outrank the negative neighbours; all **80 top-five slots** for these users are resolved by the ordinary catalogue-order tie rule. The fallback fires only if the entire score vector is zero. The current signed rule should be rejected in this form. A separately recorded fallback repair could test whether dislikes help; this campaign did not evaluate that repair.

There is also an important threshold distinction. Training co-likes use ratings >=4. Default positive inference weights are `max(rating - 3, 0)`, so **3.5-star onboarding ratings contribute**; 177 of the 500 validation users have at least one such seed. The binary removal uses a weight of one only for ratings >=4. Its 0.490270 score is slightly below the weighted parent's 0.494177, with an interval spanning zero. That removal changes both rating intensity and inclusion of 3.5-star seeds; it does not isolate rating strength alone.

### 6. Pairwise ranking: credible alternative, with a compute confound

The ranking model learns item-context and target vectors from a user's positive-item pairs against sampled unobserved negatives. A new user is represented by a weighted average of disclosed item-context vectors. This provides an explicit method for users absent from training; it does not require retrieving a learned training-user vector.

Sample BPR16 scores 0.409495, removing target-item biases scores 0.403306, and BPR32 scores 0.426355. The 32-factor model is the strongest tested sample BPR configuration. It remains relatively concentrated: 33 distinct recommended titles compared with popularity's eight.

Full BPR32 was repeated under a 1M-pair-per-epoch budget:

| Training seed | Validation NDCG@5 |
| --- | ---: |
| 42 | 0.450809 |
| 7 | 0.443191 |
| 123 | 0.448373 |
| Mean | **0.447458** |
| Sample standard deviation across three seeds | **0.003891** |

All three exceed popularity on this validation cohort. Three seeds provide a useful descriptive range, not a precise distribution of training outcomes. The best seed should not be presented as typical performance.

The original sample-to-full BPR comparison changed **both population and sampled training budget**, from 300k to 1M pairs per epoch. The equal-budget removal resolves part of this ambiguity: full-data BPR at 300k scores **0.426368**, nearly the sample 300k result of **0.426355**. Reducing the full-data budget from 1M loses 0.024441 NDCG, with interval [-0.041427, -0.005937]. Compute contributes materially to the apparent full-data improvement. The campaign lacks a sample-data/1M run and repeated seeds for the budget removal, so it cannot completely decompose data and compute effects. A BPR epoch is a fixed sampled budget, not an exhaustive pass over the dataset.

### 7. Verified genres: insufficient alone, no demonstrated hybrid gain

Genre cosine scores **0.043478**, despite reaching **1,054 distinct movies**. A 25% genre/75% popularity blend recovers to 0.371208 but does not establish a gain over popularity. These results reject the tested coarse genre ranking rules, not richer metadata or all content-aware models. Only the supplied MovieLens genre labels were used; no plots, moods, actors or external features were invented.

## Does the leading model personalize, or just recover popularity?

The strongest evidence is the removal control. Setting the full conditional model's personal blend weight to zero reproduces matched popularity exactly: NDCG falls from **0.494177 to 0.379548**. The successful output is not merely the fallback list.

The full conditional model's lists share an average of **61.56%** of their titles with matched popularity, but only **4%** of users receive the identical ordered top five. It recommends 71 distinct titles overall. The sample-model audit also finds that the conditional model's gain persists among the 482 ranking users who do not receive fallback scores: 0.496137 versus 0.383710 for popularity on those same users. These observations support useful personalization of familiar movies.

The model is still strongly oriented toward heavily rated titles. Full-model recommended training support has a median of **93,117 ratings**, and no recommended slot has fewer than 100 training ratings. Do not claim that it solves long-tail discovery. Exploratory groups with zero/one, two/three, or four/five >=4-star onboarding seeds all show higher average NDCG than popularity, but those post-selection subgroup summaries are not independent confirmation.

## Practical cost and app implications

The regularized full conditional graph retains **902,100 edges**, at most 100 neighbours for each of 9,021 movies. Its recorded model artifact is **3,413,070 bytes compressed**; saved numeric arrays occupy **7,505,480 bytes**, excluding serialized information text and runtime object overhead. The full cosine K300 graph is **14,937,216 bytes compressed** and **21,939,080 bytes** in numeric arrays. Full BPR32 seed 42 is approximately **2.21 MB** on disk.

The conditional run records 8.12 seconds of worker time and approximately 1.51 GB peak process memory during the experiment. Those are local measurements for data loading, fitting and evaluation work, not a promise about another machine or app memory use. The scoring timer averages about **0.028 ms per validation user** for conditional neighbours and 0.068 ms for full BPR32. It excludes artifact loading, the later top-five ranking operation and app integration; it is not an end-to-end iPhone latency benchmark.

For five seeds, conditional inference adds up to roughly 500 saved neighbour contributions into an all-catalogue score vector, then excludes watched movies and ranks. This is a plausible on-device workload without an ML server. It still needs a deliberate export and loader contract, TMDB identity alignment, watched-item parity, reference-case tests and device measurements. The Python NPZ is an experiment artifact, not a drop-in replacement for the existing iOS model. Any catalogue filtering for movie-only or streaming availability changes the serving conditions and needs its own appropriate comparison.

## Decisions supported before final evaluation

1. Prefer the **regularized full conditional-neighbour candidate** for a separately frozen final evaluation. Keep its checkpoint, weights, catalogue, scoring and exclusions fixed before opening final users. Its no-shrinkage sibling's extra 0.00003 is not a material reason to change that choice.
2. Preserve cosine neighbours as a research alternative when broader catalogue coverage matters, and BPR32 as a compact stochastic alternative. Their practical strengths do not override the chosen primary-metric evidence.
3. Keep the current MF, full-user MF and both popularity controls in every comparison. Better than original MF and better than popularity are distinct claims.
4. Retain the unsuccessful experiments: Huber's better RMSE/worse ranking, harmful centring, signed zero-score ties, ineffective genre-only ranking, the weak larger-K isolate, and the BPR compute confound are useful design evidence.
5. Do not interpret a validation shortlist, a full-data run or a pointwise bootstrap interval as independent final confirmation. No app model is replaced by this review.

## Fresh-final evaluation: frozen candidate confirmed the gain

The separate workflow selected `cm-20261119-full-knn-conditional`, froze its source, model, scoring, catalogue and evaluation procedure, then evaluated that one candidate and the fixed references once. The final evaluation completed on 19 November 2026. The [final report](final-results/cinematch-systematic-20261119.json) is separate from the validation leaderboard and records the freeze checksum, exact protocol, artifact paths and paired comparisons. This review used that report; no additional candidate was evaluated and no final-holdout tuning was performed.

The fresh group contains 500 users, of whom 499 have relevant targets:

| Frozen final comparison | NDCG@5 | Hit rate@5 | Recall@5 | Distinct recommended movies |
| --- | ---: | ---: | ---: | ---: |
| Original sample MF | 0.143837 | 36.87% | 1.34% | 253 |
| Existing full-user MF | 0.159888 | 41.68% | 1.62% | 474 |
| Popularity, either training population | 0.380212 | 75.15% | 4.09% | 8 |
| **Frozen full conditional neighbours** | **0.485361** | **83.57%** | **5.62%** | **76** |

The candidate's NDCG improvement over popularity is **0.105149**, or **27.66% relative**, with paired 95% interval **[0.085936, 0.123522]**. Hit rate improves by **8.42 percentage points** and recall by **1.53 percentage points**. Its NDCG differences from original MF and full-user MF are **0.341524 [0.313343, 0.369709]** and **0.325473 [0.297254, 0.353659]**, respectively. Rating RMSE remains inapplicable to the neighbour and popularity ranking scores; the MF reference errors are recorded in the final report.

This is evidence that the selected model's advantage extends to a previously reserved user group under the established retrospective protocol. It does not remove that protocol's exposure bias, establish benefits under different onboarding questions, or measure real app satisfaction. Bootstrap uncertainty covers final users, not changing the training population. The final result must now be reported as frozen evidence; it must not drive another round of selection on the same holdout.

**Recommendation:** advance the regularized conditional-neighbour model as the preferred replacement candidate for controlled app integration and user-facing evaluation. Its 3.41 MB compressed graph and approximately 7.51 MB of numeric arrays make local inference plausible, while the replayed score contract and reference cases should guide a deliberate TMDB-aligned export. Preserve popularity as the no-signal fallback and continuing control. Keep the current app model unchanged until that separate integration and adoption work is explicitly undertaken.

The central trade-off is concentration in familiar, broadly liked movies. The candidate reaches more titles than popularity but fewer than the two MF models, and the validation audit found a strong focus on heavily rated films. Its supported advantage is better recovery and ordering of recorded likes, not a demonstrated solution to novelty, long-tail discovery or movie-only availability filtering.
