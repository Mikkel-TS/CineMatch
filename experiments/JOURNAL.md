# CineMatch systematic experiment journal

This extends the existing experiment records. Original checkpoints, datasets and records are preserved. App adoption is separate.

Every row links to a pre-execution record, exact executed source snapshot, artifacts and interpretation. Full-data validation is still exploratory; it is not final holdout confirmation.

| Run | Family | Data | Status | NDCG@5 | Δ original | Δ full-user | Δ popularity | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| [cm-20261119-bias-half](records/cm-20261119-bias-half.json) | movie_bias | sample | complete | 0.2481 | +0.0962 | +0.0806 | -0.1314 | retain bias-sensitivity finding; do not select |
| [cm-20261119-bias-shrink100](records/cm-20261119-bias-shrink100.json) | movie_bias | sample | complete | 0.2820 | +0.1301 | +0.1145 | -0.0976 | retain evidence-aware shrinkage as MF finding |
| [cm-20261119-bias-shrink1000](records/cm-20261119-bias-shrink1000.json) | movie_bias | sample | complete | 0.3101 | +0.1582 | +0.1426 | -0.0694 | retain strongest tested standalone MF adjustment; do not select |
| [cm-20261119-bias-zero](records/cm-20261119-bias-zero.json) | movie_bias | sample | complete | 0.2336 | +0.0817 | +0.0661 | -0.1459 | retain bias-sensitivity finding; do not select |
| [cm-20261119-blend-confidence](records/cm-20261119-blend-confidence.json) | popularity_hybrid | sample | complete | 0.3863 | +0.2344 | +0.2188 | +0.0067 | retain as inconclusive hybrid; do not select |
| [cm-20261119-blend25](records/cm-20261119-blend25.json) | popularity_hybrid | sample | complete | 0.3758 | +0.2239 | +0.2083 | -0.0037 | retain as near-popularity control; do not select |
| [cm-20261119-blend50](records/cm-20261119-blend50.json) | popularity_hybrid | sample | complete | 0.3500 | +0.1981 | +0.1825 | -0.0295 | reject tested stronger MF blend |
| [cm-20261119-bpr16-no-bias](records/cm-20261119-bpr16-no-bias.json) | pairwise_ranking | sample | complete | 0.4033 | +0.2514 | +0.2358 | +0.0238 | retain item bias for this BPR screen |
| [cm-20261119-bpr16](records/cm-20261119-bpr16.json) | pairwise_ranking | sample | complete | 0.4095 | +0.2576 | +0.2420 | +0.0299 | retain ranking-oriented alternative |
| [cm-20261119-bpr32](records/cm-20261119-bpr32.json) | pairwise_ranking | sample | complete | 0.4264 | +0.2745 | +0.2588 | +0.0468 | retain best sample BPR for repeated full-data checks |
| [cm-20261119-fit-centered](records/cm-20261119-fit-centered.json) | profile_fitting | sample | complete | 0.1489 | -0.0030 | -0.0186 | -0.2306 | reject tested centring rule |
| [cm-20261119-fit-lambda001](records/cm-20261119-fit-lambda001.json) | profile_fitting | sample | complete | 0.1985 | +0.0466 | +0.0310 | -0.1810 | reject further weakening for this candidate |
| [cm-20261119-fit-lambda01](records/cm-20261119-fit-lambda01.json) | profile_fitting | sample | complete | 0.2056 | +0.0537 | +0.0381 | -0.1740 | retain as MF diagnostic; do not select |
| [cm-20261119-fit-no-intercept](records/cm-20261119-fit-no-intercept.json) | profile_fitting | sample | complete | 0.2043 | +0.0524 | +0.0368 | -0.1752 | retain intercept; reject this refinement |
| [cm-20261119-full-blend25](records/cm-20261119-full-blend25.json) | popularity_hybrid | full | complete | 0.3940 | +0.2421 | +0.2264 | +0.0144 | retain as inconclusive full-data hybrid; do not select |
| [cm-20261119-full-bpr32-budget300k](records/cm-20261119-full-bpr32-budget300k.json) | pairwise_ranking | full | complete | 0.4264 | +0.2745 | +0.2588 | +0.0468 | retain compute-confound finding; do not claim data-only BPR gain |
| [cm-20261119-full-bpr32-seed123](records/cm-20261119-full-bpr32-seed123.json) | pairwise_ranking | full | complete | 0.4484 | +0.2965 | +0.2809 | +0.0688 | retain seed replication; do not choose best seed opportunistically |
| [cm-20261119-full-bpr32-seed42](records/cm-20261119-full-bpr32-seed42.json) | pairwise_ranking | full | complete | 0.4508 | +0.2989 | +0.2833 | +0.0713 | retain strongest tested stochastic alternative |
| [cm-20261119-full-bpr32-seed7](records/cm-20261119-full-bpr32-seed7.json) | pairwise_ranking | full | complete | 0.4432 | +0.2913 | +0.2757 | +0.0636 | retain seed replication; do not choose best seed opportunistically |
| [cm-20261119-full-knn-binary](records/cm-20261119-full-knn-binary.json) | removal_test | full | complete | 0.4903 | +0.3384 | +0.3228 | +0.1107 | retain simpler onboarding option as inconclusive; keep parent |
| [cm-20261119-full-knn-blend50](records/cm-20261119-full-knn-blend50.json) | popularity_hybrid | full | complete | 0.4880 | +0.3361 | +0.3205 | +0.1085 | reject extra popularity blend for this candidate |
| [cm-20261119-full-knn-conditional](records/cm-20261119-full-knn-conditional.json) | item_neighbors | full | complete | 0.4942 | +0.3423 | +0.3267 | +0.1146 | recommend freezing regularized conditional candidate for separate final evaluation |
| [cm-20261119-full-knn-cos300](records/cm-20261119-full-knn-cos300.json) | item_neighbors | full | complete | 0.4790 | +0.3271 | +0.3115 | +0.0995 | retain diversity-oriented alternative; do not choose over conditional on NDCG |
| [cm-20261119-full-knn-no-personal](records/cm-20261119-full-knn-no-personal.json) | removal_test | full | complete | 0.3795 | +0.2276 | +0.2120 | +0.0000 | retain successful personalization removal control |
| [cm-20261119-full-knn-no-shrink](records/cm-20261119-full-knn-no-shrink.json) | removal_test | full | complete | 0.4942 | +0.3423 | +0.3267 | +0.1147 | retain removal result; keep regularized parent |
| [cm-20261119-full-knn-observed](records/cm-20261119-full-knn-observed.json) | removal_test | full | complete | 0.4735 | +0.3216 | +0.3060 | +0.0940 | retain positive training threshold; reject co-rating replacement |
| [cm-20261119-genre-blend25](records/cm-20261119-genre-blend25.json) | metadata | sample | complete | 0.3712 | +0.2193 | +0.2037 | -0.0083 | retain inconclusive content hybrid; do not select |
| [cm-20261119-genre](records/cm-20261119-genre.json) | metadata | sample | complete | 0.0435 | -0.1084 | -0.1240 | -0.3361 | reject standalone genre cosine for this benchmark |
| [cm-20261119-knn-conditional](records/cm-20261119-knn-conditional.json) | item_neighbors | sample | complete | 0.4884 | +0.3365 | +0.3208 | +0.1088 | retain leading sample configuration |
| [cm-20261119-knn-cos100](records/cm-20261119-knn-cos100.json) | item_neighbors | sample | complete | 0.4640 | +0.3121 | +0.2965 | +0.0844 | retain successful item-to-item baseline |
| [cm-20261119-knn-cos300](records/cm-20261119-knn-cos300.json) | item_neighbors | sample | complete | 0.4797 | +0.3278 | +0.3122 | +0.1002 | retain combined configuration; use isolates for attribution |
| [cm-20261119-knn-k300-isolate](records/cm-20261119-knn-k300-isolate.json) | item_neighbors | sample | complete | 0.4707 | +0.3188 | +0.3032 | +0.0912 | do not prefer larger graph on this evidence |
| [cm-20261119-knn-observed](records/cm-20261119-knn-observed.json) | item_neighbors | sample | complete | 0.3965 | +0.2446 | +0.2290 | +0.0170 | reject replacing co-likes with co-ratings here |
| [cm-20261119-knn-shrink50-isolate](records/cm-20261119-knn-shrink50-isolate.json) | item_neighbors | sample | complete | 0.4845 | +0.3326 | +0.3169 | +0.1049 | retain stronger shrinkage for sample cosine models |
| [cm-20261119-knn-signed](records/cm-20261119-knn-signed.json) | item_neighbors | sample | complete | 0.4358 | +0.2839 | +0.2683 | +0.0563 | reject current signed rule; preserve failure diagnosis |
| [cm-20261119-mf-biasreg10](records/cm-20261119-mf-biasreg10.json) | mf_training | sample | complete | 0.2071 | +0.0552 | +0.0395 | -0.1725 | retain MF regularization finding; do not select |
| [cm-20261119-mf-huber](records/cm-20261119-mf-huber.json) | mf_training | sample | complete | 0.1158 | -0.0361 | -0.0517 | -0.2637 | reject Huber for the ranking objective tested |
| [cm-20261119-mf32](records/cm-20261119-mf32.json) | mf_training | sample | complete | 0.1388 | -0.0131 | -0.0287 | -0.2407 | reject tested larger MF model |
| [cm-20261119-mf8](records/cm-20261119-mf8.json) | mf_training | sample | complete | 0.1195 | -0.0324 | -0.0480 | -0.2600 | reject tested smaller MF model |
| [cm-20261119-rerank100](records/cm-20261119-rerank100.json) | popularity_hybrid | sample | complete | 0.3067 | +0.1548 | +0.1392 | -0.0729 | reject tested MF reranking rule |

## cm-20261119-bias-half

Does reducing unsupported movie-prior influence improve ranking?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"movie_bias_scale": 0.5}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-bias-half --run-id new-unique-id`.

Validation NDCG@5 0.248111; difference from established popularity -0.131437, paired 95% interval [-0.159057, -0.104992]. Halving movie bias improves ranking more than removing it, suggesting useful prior information should be reduced rather than discarded in this checkpoint.

The half-versus-zero comparison is descriptive; no dedicated paired contrast is included in this run record. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain bias-sensitivity finding; do not select.

## cm-20261119-bias-shrink100

Does reducing unsupported movie-prior influence improve ranking?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"bias_shrinkage_tau": 100}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-bias-shrink100 --run-id new-unique-id`.

Validation NDCG@5 0.281979; difference from established popularity -0.097569, paired 95% interval [-0.123421, -0.073355]. Count-dependent shrinkage gives a larger ranking improvement than the simple bias-scaling screens, while still losing to popularity.

This is a post-training transformation, not retraining under the same effective bias objective. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain evidence-aware shrinkage as MF finding.

## cm-20261119-bias-shrink1000

Does reducing unsupported movie-prior influence improve ranking?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"bias_shrinkage_tau": 1000}

Status: **complete**. Parent: `cm-20261119-bias-shrink100`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-bias-shrink1000 --run-id new-unique-id`.

Validation NDCG@5 0.310111; difference from established popularity -0.069437, paired 95% interval [-0.094823, -0.045311]. Stronger count shrinkage improves its shrink-100 parent with a positive paired interval, but remains below popularity and worsens RMSE.

Only two shrinkage strengths were tried; this does not establish an optimal penalty or general superiority outside this checkpoint. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain strongest tested standalone MF adjustment; do not select.

## cm-20261119-bias-zero

Does reducing unsupported movie-prior influence improve ranking?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"movie_bias_scale": 0}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-bias-zero --run-id new-unique-id`.

Validation NDCG@5 0.233599; difference from established popularity -0.145949, paired 95% interval [-0.174670, -0.117342]. Removing movie biases improves the original MF ranking markedly, but is worse than partial scaling or stronger count shrinkage and worsens RMSE.

Bias changes affect both support residuals and candidate scores. This cannot attribute the gain solely to suppressing candidate priors. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain bias-sensitivity finding; do not select.

## cm-20261119-blend-confidence

Can popularity stabilise sparse personalisation without erasing it?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"alpha": 0.5, "confidence": true}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-blend-confidence --run-id new-unique-id`.

Validation NDCG@5 0.386272; difference from established popularity +0.006725, paired 95% interval [-0.006742, +0.021288]. Support-dependent weighting produces a small observed gain over popularity, but the paired interval includes zero and the result is much weaker than item neighbours.

Confidence counts ratings >=4, whereas neighbour positive inference includes 3.5 ratings; these definitions are not interchangeable. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain as inconclusive hybrid; do not select.

## cm-20261119-blend25

Can popularity stabilise sparse personalisation without erasing it?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"alpha": 0.25}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-blend25 --run-id new-unique-id`.

Validation NDCG@5 0.375830; difference from established popularity -0.003717, paired 95% interval [-0.019857, +0.013332]. A popularity-heavy MF blend largely closes the original deficit but does not demonstrate an improvement over popularity.

The small negative difference from popularity has an interval spanning zero; this is inconclusive, not evidence of exact equivalence. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain as near-popularity control; do not select.

## cm-20261119-blend50

Can popularity stabilise sparse personalisation without erasing it?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"alpha": 0.5}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-blend50 --run-id new-unique-id`.

Validation NDCG@5 0.350007; difference from established popularity -0.029541, paired 95% interval [-0.051227, -0.008665]. Giving the original MF half of the standardized blend worsens ranking relative to popularity; adding more of this personal score does not help.

Blend weights apply after per-user standardization of scores and log counts, so they are not probabilities or literal fractions of recommended movies. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject tested stronger MF blend.

## cm-20261119-bpr16-no-bias

Can a pairwise item-context objective improve recommendations for unseen users?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"kind": "ranking", "factors": 16, "epochs": 8, "patience": 3, "pairs_per_epoch": 300000, "use_item_bias": false, "aggregation": "positive", "threads": 2}

Status: **complete**. Parent: `cm-20261119-bpr16`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-bpr16-no-bias --run-id new-unique-id`.

Validation NDCG@5 0.403306; difference from established popularity +0.023758, paired 95% interval [+0.013360, +0.034193]. Removing the learned target-item bias slightly reduces ranking relative to BPR16 and narrows the set of recommended titles.

The parent interval is only narrowly below zero after many comparisons; one seed does not establish a universal need for target-item biases. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain item bias for this BPR screen.

## cm-20261119-bpr16

Can a pairwise item-context objective improve recommendations for unseen users?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"kind": "ranking", "factors": 16, "epochs": 8, "patience": 3, "pairs_per_epoch": 300000, "use_item_bias": true, "aggregation": "positive", "threads": 2}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-bpr16 --run-id new-unique-id`.

Validation NDCG@5 0.409495; difference from established popularity +0.029947, paired 95% interval [+0.017044, +0.043439]. The new-user-compatible pairwise item-context model improves on popularity, although outputs remain concentrated in 16 titles across validation users.

Training uses sampled unobserved negatives, which are not confirmed dislikes; 300k pairs per epoch are a compute budget, not a pass over every rating. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain ranking-oriented alternative.

## cm-20261119-bpr32

Can a pairwise item-context objective improve recommendations for unseen users?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"kind": "ranking", "factors": 32, "epochs": 8, "patience": 3, "pairs_per_epoch": 300000, "use_item_bias": true, "aggregation": "positive", "threads": 2}

Status: **complete**. Parent: `cm-20261119-bpr16`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-bpr32 --run-id new-unique-id`.

Validation NDCG@5 0.426355; difference from established popularity +0.046807, paired 95% interval [+0.026313, +0.066151]. Increasing BPR factors from 16 to 32 improves the parent with a positive paired interval and broadens the recommended set.

One sample training seed; factor count also changes model capacity and optimization dynamics. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain best sample BPR for repeated full-data checks.

## cm-20261119-fit-centered

Does centring onboarding ratings reduce rating-scale differences?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"fold_in_regularization": 0.1, "user_center": true}

Status: **complete**. Parent: `cm-20261119-fit-lambda01`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-fit-centered --run-id new-unique-id`.

Validation NDCG@5 0.148931; difference from established popularity -0.230617, paired 95% interval [-0.257091, -0.204239]. Replacing the global mean with the five-support-rating mean loses the preceding ranking gain and ends slightly below the original MF.

This is this particular support-mean centring rule, combined with lambda 0.1; it does not rule out every centring scheme. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject tested centring rule.

## cm-20261119-fit-lambda001

Does very weak new-user regularisation help or overfit five ratings?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"fold_in_regularization": 0.01}

Status: **complete**. Parent: `cm-20261119-fit-lambda01`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-fit-lambda001 --run-id new-unique-id`.

Validation NDCG@5 0.198521; difference from established popularity -0.181027, paired 95% interval [-0.207820, -0.152705]. Lambda 0.01 regresses from 0.1 while RMSE worsens further; the parent comparison is inconclusive rather than decisive.

The parent interval includes zero. Increased coverage does not offset the lack of ranking improvement. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject further weakening for this candidate.

## cm-20261119-fit-lambda01

Does weaker new-user regularisation retain useful taste signal?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"fold_in_regularization": 0.1}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-fit-lambda01 --run-id new-unique-id`.

Validation NDCG@5 0.205594; difference from established popularity -0.173954, paired 95% interval [-0.199881, -0.146239]. Reducing fold-in regularization from 1 to 0.1 improves ranking over the original MF, but worsens rating RMSE substantially and remains far below popularity.

One fixed trained checkpoint and five random ratings per user; this is not evidence for weak regularization under other onboarding conditions. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain as MF diagnostic; do not select.

## cm-20261119-fit-no-intercept

Does removing the new-user intercept improve sparse fitting?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"fold_in_regularization": 0.1, "fit_user_bias": false}

Status: **complete**. Parent: `cm-20261119-fit-lambda01`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-fit-no-intercept --run-id new-unique-id`.

Validation NDCG@5 0.204319; difference from established popularity -0.175229, paired 95% interval [-0.201522, -0.146711]. Removing the user intercept makes little difference to the lambda-0.1 parent and increases RMSE; no useful gain was demonstrated.

Small observed ranking decline with paired interval crossing zero; do not claim the intercept is universally necessary. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain intercept; reject this refinement.

## cm-20261119-full-blend25

Does the same blend improve the full-user model?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: 25% standardized MF + 75% standardized log-count

Status: **complete**. Parent: `mf16-full-users-fixed-catalog-20261117`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-blend25 --run-id new-unique-id`.

Validation NDCG@5 0.393960; difference from established popularity +0.014412, paired 95% interval [-0.002246, +0.029740]. The full-user MF blend has a small observed advantage over popularity, with a paired interval spanning zero. More MF training data does not make this a clear winner.

The validation users have been reused for selection; a near-zero pointwise interval is not proof of reliable benefit. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain as inconclusive full-data hybrid; do not select.

## cm-20261119-full-bpr32-budget300k

How much of the full-data ranking gain comes from the larger sampling budget?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Use sample-run300k-pair budget with full-user population; compare1M-budget parent

Status: **complete**. Parent: `cm-20261119-full-bpr32-seed42`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-bpr32-budget300k --run-id new-unique-id`.

Validation NDCG@5 0.426368; difference from established popularity +0.046820, paired 95% interval [+0.030556, +0.063293]. At the sample 300k-pair budget, full-data BPR scores 0.426368, essentially the sample BPR32 score of 0.426355. Reducing the full-data 1M budget loses 0.024441 with a negative parent interval.

This supports a material role for training budget, but does not fully decompose data and compute: there is no matched sample-data 1M-budget run and only one seed for the budget removal. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain compute-confound finding; do not claim data-only BPR gain.

## cm-20261119-full-bpr32-seed123

Does pairwise ranking improve with full training data and remain stable across seeds?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Full eligible user population,1M uniformly sampled pairs per epoch; seed123

Status: **complete**. Parent: `cm-20261119-full-bpr32-seed42`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-bpr32-seed123 --run-id NEW_UNIQUE_ID`.

Validation NDCG@5 0.448373; difference from established popularity +0.068825, paired 95% interval [+0.048763, +0.087734]. Seed 123 also remains above popularity and lies between the other two seed outcomes. The promising BPR direction is not unique to seed 42.

The seeds share training data, validation users and selection policy; their results are not independent final tests. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain seed replication; do not choose best seed opportunistically.

## cm-20261119-full-bpr32-seed42

Does pairwise ranking improve with full training data and remain stable across seeds?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Full eligible user population,1M uniformly sampled pairs per epoch; seed42

Status: **complete**. Parent: `cm-20261119-bpr32`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-bpr32-seed42 --run-id new-unique-id`.

Validation NDCG@5 0.450809; difference from established popularity +0.071261, paired 95% interval [+0.050890, +0.090434]. Full-data BPR32 with a 1M-pair budget improves its sample parent, but remains below conditional neighbours in observed NDCG.

Both data population and pair budget increased. The equal-budget removal shows the initial gain cannot be credited to more data alone. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain strongest tested stochastic alternative.

## cm-20261119-full-bpr32-seed7

Does pairwise ranking improve with full training data and remain stable across seeds?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Full eligible user population,1M uniformly sampled pairs per epoch; seed7

Status: **complete**. Parent: `cm-20261119-full-bpr32-seed42`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-bpr32-seed7 --run-id NEW_UNIQUE_ID`.

Validation NDCG@5 0.443191; difference from established popularity +0.063643, paired 95% interval [+0.045103, +0.081650]. Seed 7 remains above popularity but scores below seed 42; this demonstrates variation even under the same full-data policy.

Three seeds give a small descriptive estimate of stochastic variability, not a reliable interval over all training outcomes. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain seed replication; do not choose best seed opportunistically.

## cm-20261119-full-knn-binary

Does the strength of onboarding ratings help beyond a binary like?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Reuse exact graph; binary >=4 seed weights instead of max(r-3,0)

Status: **complete**. Parent: `cm-20261119-full-knn-conditional`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-knn-binary --run-id new-unique-id`.

Validation NDCG@5 0.490270; difference from established popularity +0.110723, paired 95% interval [+0.094075, +0.128149]. Binary >=4 onboarding weights slightly lower NDCG while slightly raising hit rate; neither approach clearly dominates the primary-metric parent comparison.

This removes rating-strength differences AND removes contributions from 3.5-star seeds. It does not isolate rating intensity alone; the paired NDCG interval includes zero. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain simpler onboarding option as inconclusive; keep parent.

## cm-20261119-full-knn-blend50

Does extra popularity improve already successful conditional neighbours?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: 50% standardized neighbour scores +50% standardized log-count

Status: **complete**. Parent: `cm-20261119-full-knn-conditional`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-knn-blend50 --run-id new-unique-id`.

Validation NDCG@5 0.488018; difference from established popularity +0.108470, paired 95% interval [+0.092404, +0.125079]. Adding a 50% standardized popularity component lowers NDCG relative to conditional neighbours, with a negative paired interval. The successful graph already supplies a useful popularity-aware prior.

This is one calibrated blend weight on reused validation users, not a claim that no other blend could help. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject extra popularity blend for this candidate.

## cm-20261119-full-knn-conditional

Does the promising item-neighbour result persist with all eligible training users?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Training data expands to30,495,721 ratings; algorithm settings fixed

Status: **complete**. Parent: `cm-20261119-knn-conditional`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-knn-conditional --run-id new-unique-id`.

Validation NDCG@5 0.494177; difference from established popularity +0.114629, paired 95% interval [+0.098119, +0.132605]. The full-data conditional graph is a strong practical validation candidate: large gains over all fixed references, useful personalization, and a compact top-100 graph. Its small improvement over the sample conditional parent is inconclusive.

This remains validation-selected evidence. Do not attribute the whole gain to more data: the sample conditional model already achieves 0.488362. Final evaluation and app adoption are separate decisions. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: recommend freezing regularized conditional candidate for separate final evaluation.

## cm-20261119-full-knn-cos300

Does the promising item-neighbour result persist with all eligible training users?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Training data expands to30,495,721 ratings; algorithm settings fixed

Status: **complete**. Parent: `cm-20261119-knn-cos300`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-knn-cos300 --run-id new-unique-id`.

Validation NDCG@5 0.479044; difference from established popularity +0.099496, paired 95% interval [+0.070714, +0.126966]. Full-data cosine K300/shrink50 does not improve its sample counterpart, though it recovers likes well above popularity and reaches 484 titles.

The sample-to-full interval spans zero. Higher catalogue coverage and a larger graph are trade-offs, not evidence of higher user satisfaction. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain diversity-oriented alternative; do not choose over conditional on NDCG.

## cm-20261119-full-knn-no-personal

Does the chosen neighbour model add useful personalisation beyond popularity?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Remove the entire personal score; retain matched training-count ranking and same exclusions

Status: **complete**. Parent: `cm-20261119-full-knn-conditional`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-knn-no-personal --run-id new-unique-id`.

Validation NDCG@5 0.379548; difference from established popularity +0.000000, paired 95% interval [+0.000000, +0.000000]. Setting the personal weight to zero exactly reproduces matched popularity and loses 0.114629 NDCG. The conditional model gain is not merely its popularity fallback.

This removes the entire co-like score, including conditional popularity structure; the experiment does not separate every causal component of personalization. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain successful personalization removal control.

## cm-20261119-full-knn-no-shrink

Does support shrinkage contribute to the strongest full-data neighbour model?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Remove co-occurrence support shrinkage only

Status: **complete**. Parent: `cm-20261119-full-knn-conditional`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-knn-no-shrink --run-id new-unique-id`.

Validation NDCG@5 0.494207; difference from established popularity +0.114659, paired 95% interval [+0.098192, +0.132634]. Removing shrinkage changes NDCG by only +0.00002985 with identical hit rate, recall and distinct-title count. The tiny numerical maximum does not justify replacing the established regularized parent.

The paired interval touches zero and is driven by an extremely small ranking change; this is practically negligible. No general benefit from removing shrinkage is established. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain removal result; keep regularized parent.

## cm-20261119-full-knn-observed

Does co-liking rather than merely co-rating account for the gain?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Replace positive>=4 training events with all observed ratings; rest fixed

Status: **complete**. Parent: `cm-20261119-full-knn-conditional`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-full-knn-observed --run-id new-unique-id`.

Validation NDCG@5 0.473514; difference from established popularity +0.093966, paired 95% interval [+0.075409, +0.112452]. Replacing full-data co-likes with co-ratings lowers NDCG by 0.020663 with a negative paired interval, while still beating popularity. Positive rating information contributes useful signal.

This conclusion applies to the tested binary feedback and conditional normalization, not all possible rating-aware similarities. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain positive training threshold; reject co-rating replacement.

## cm-20261119-genre-blend25

Does genre taste improve a popularity prior?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: 25% standardized genre score + 75% standardized log-count

Status: **complete**. Parent: `cm-20261119-genre`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-genre-blend25 --run-id new-unique-id`.

Validation NDCG@5 0.371208; difference from established popularity -0.008340, paired 95% interval [-0.026997, +0.012608]. Adding popularity rescues much of the genre model deficit but the hybrid does not demonstrate a gain over popularity itself.

Improvement over a weak genre parent is insufficient evidence that genre personalization adds value; the popularity interval includes zero. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain inconclusive content hybrid; do not select.

## cm-20261119-genre

Do verified MovieLens genres help from sparse ratings?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Unit-normalized genre vectors and signed onboarding aggregation

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-genre --run-id new-unique-id`.

Validation NDCG@5 0.043478; difference from established popularity -0.336070, paired 95% interval [-0.364072, -0.307540]. Genre-only cosine is worse than both reference MF models and popularity despite recommending 1,054 distinct titles. Diversity alone is not recommendation quality.

Only verified coarse MovieLens genres were used; this result does not reject richer verified metadata or item models with content features. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject standalone genre cosine for this benchmark.

## cm-20261119-knn-conditional

Can item co-occurrence transfer five onboarding ratings into better rankings?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"kind": "neighbors", "feedback": "positive", "top_k": 100, "shrink": 10, "similarity": "conditional", "aggregation": "positive"}

Status: **complete**. Parent: `cm-20261119-knn-cos100`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-knn-conditional --run-id new-unique-id`.

Validation NDCG@5 0.488362; difference from established popularity +0.108814, paired 95% interval [+0.091618, +0.126482]. Conditional seed-to-candidate similarity improves the cosine-100 parent and strongly beats popularity; rankings differ across users despite favouring broadly liked titles.

The gain is not evidence of long-tail discovery: recommended titles are heavily rated. Conditional similarity changes normalization as well as row-neighbour selection. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain leading sample configuration.

## cm-20261119-knn-cos100

Can item co-occurrence transfer five onboarding ratings into better rankings?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"kind": "neighbors", "feedback": "positive", "top_k": 100, "shrink": 10, "similarity": "cosine", "aggregation": "positive"}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-knn-cos100 --run-id new-unique-id`.

Validation NDCG@5 0.463983; difference from established popularity +0.084435, paired 95% interval [+0.055387, +0.111369]. Positive co-occurrence cosine neighbours beat the original MF, full-user MF and popularity while returning many more distinct titles than popularity.

One fixed sample population; deterministic graph construction removes optimization-seed variation but not data-sample or onboarding uncertainty. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain successful item-to-item baseline.

## cm-20261119-knn-cos300

Can item co-occurrence transfer five onboarding ratings into better rankings?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"kind": "neighbors", "feedback": "positive", "top_k": 300, "shrink": 50, "similarity": "cosine", "aggregation": "positive"}

Status: **complete**. Parent: `cm-20261119-knn-cos100`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-knn-cos300 --run-id new-unique-id`.

Validation NDCG@5 0.479735; difference from established popularity +0.100187, paired 95% interval [+0.073770, +0.126828]. Increasing K to 300 and shrinkage to 50 improves the cosine-100 parent, but the two changes cannot be credited independently from this run.

Later isolates show a clearer gain from shrinkage than from K. Do not describe this as a neighbourhood-size-only improvement. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain combined configuration; use isolates for attribution.

## cm-20261119-knn-k300-isolate

Does neighbourhood size account for the combined cosine improvement?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Only K100→300; shrink remains10

Status: **complete**. Parent: `cm-20261119-knn-cos100`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-knn-k300-isolate --run-id new-unique-id`.

Validation NDCG@5 0.470711; difference from established popularity +0.091164, paired 95% interval [+0.063430, +0.118342]. Increasing K alone produces a small observed gain with an interval including zero, at roughly triple neighbour storage.

The comparison controls shrinkage but still uses the same validation users; lower confidence in a practical benefit than the joint screen implied. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: do not prefer larger graph on this evidence.

## cm-20261119-knn-observed

Can item co-occurrence transfer five onboarding ratings into better rankings?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"kind": "neighbors", "feedback": "observed", "top_k": 100, "shrink": 10, "similarity": "cosine", "aggregation": "positive"}

Status: **complete**. Parent: `cm-20261119-knn-cos100`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-knn-observed --run-id new-unique-id`.

Validation NDCG@5 0.396532; difference from established popularity +0.016985, paired 95% interval [-0.014476, +0.048111]. Using all observed ratings instead of ratings >=4 reduces ranking substantially relative to positive cosine neighbours; its advantage over popularity is inconclusive.

Observed ratings include dislikes and neutral feedback. This tests binary co-rating, not every implicit-feedback treatment. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject replacing co-likes with co-ratings here.

## cm-20261119-knn-shrink50-isolate

Does shrinkage account for the combined cosine improvement?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: Only shrink10→50; K remains100

Status: **complete**. Parent: `cm-20261119-knn-cos100`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-knn-shrink50-isolate --run-id new-unique-id`.

Validation NDCG@5 0.484459; difference from established popularity +0.104911, paired 95% interval [+0.078018, +0.131969]. Increasing support shrinkage alone improves the cosine-100 parent with a positive paired interval and exceeds the combined K300/shrink50 point estimate.

The isolate applies to sample-trained cosine similarity; it cannot establish that the same penalty helps full-data conditional similarity. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain stronger shrinkage for sample cosine models.

## cm-20261119-knn-signed

Can item co-occurrence transfer five onboarding ratings into better rankings?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"kind": "neighbors", "feedback": "positive", "top_k": 100, "shrink": 10, "similarity": "cosine", "aggregation": "signed"}

Status: **complete**. Parent: `cm-20261119-knn-cos100`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-knn-signed --run-id new-unique-id`.

Validation NDCG@5 0.435832; difference from established popularity +0.056284, paired 95% interval [+0.026602, +0.083395]. Signed rating weights worsen the positive parent. An independent audit found 16 nonfallback users with no positive candidate score: all 80 top-five slots were zero-score catalogue-order ties.

The fallback triggers only if every score is zero, so negative-only evidence can produce arbitrary zero-score recommendations. A repair would require a separate recorded experiment; this does not show that dislikes are inherently harmful. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject current signed rule; preserve failure diagnosis.

## cm-20261119-mf-biasreg10

Does evidence-aware bias regularisation protect low-support movies?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"factors": 16, "bias_count_regularization": 10.0}; fixed fold-in lambda1, maximum12 epochs

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-mf-biasreg10 --run-id new-unique-id`.

Validation NDCG@5 0.207051; difference from established popularity -0.172496, paired 95% interval [-0.199273, -0.148164]. Evidence-aware bias regularization during training improves ranking over the original MF but remains far below popularity.

This screen differs from the original checkpoint in both the intended regularizer and the campaign epoch/fold-in policy; only one training seed was run. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: retain MF regularization finding; do not select.

## cm-20261119-mf-huber

Does a robust rating loss improve ranking?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"factors": 16, "loss": "huber", "huber_delta": 1.0}; fixed fold-in lambda1, maximum12 epochs

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-mf-huber --run-id new-unique-id`.

Validation NDCG@5 0.115829; difference from established popularity -0.263719, paired 95% interval [-0.291635, -0.235380]. Huber achieves a lower rating RMSE while materially worsening NDCG. Better rating prediction did not translate into better top-five recovery here.

This is a single robust-loss strength and training seed under the campaign policy; it does not rule out all robust or ranking losses. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject Huber for the ranking objective tested.

## cm-20261119-mf32

Does a larger latent space improve new-user ranking?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"factors": 32}; fixed fold-in lambda1, maximum12 epochs

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-mf32 --run-id new-unique-id`.

Validation NDCG@5 0.138805; difference from established popularity -0.240743, paired 95% interval [-0.269195, -0.212890]. Thirty-two factors does not improve original MF ranking; the observed decline has an interval including zero.

The training-policy difference and a single random seed limit attribution to factor count alone. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject tested larger MF model.

## cm-20261119-mf8

Does a smaller latent profile generalise better from five ratings?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"factors": 8}; fixed fold-in lambda1, maximum12 epochs

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-mf8 --run-id new-unique-id`.

Validation NDCG@5 0.119530; difference from established popularity -0.260018, paired 95% interval [-0.288467, -0.232472]. Eight factors worsens ranking relative to the original checkpoint; lower model size does not compensate for this quality loss.

Training used a fixed fold-in lambda 1 and a 12-epoch cap rather than the original 20-epoch/three-penalty policy, so this is not a perfectly isolated factor-count comparison. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject tested smaller MF model.

## cm-20261119-rerank100

Can popularity stabilise sparse personalisation without erasing it?

Hypothesis: This change may improve recovery of held-out likes; no improvement is assumed.

Change: {"candidate_count": 100}

Status: **complete**. Parent: `mf16-sample-20261023`. Reproduce: `.venv/bin/python -B -u experiments/run_campaign.py --replay cm-20261119-rerank100 --run-id new-unique-id`.

Validation NDCG@5 0.306680; difference from established popularity -0.072868, paired 95% interval [-0.097834, -0.048483]. Reranking the 100 most popular eligible movies improves the original MF but remains well below recommending popularity directly.

All catalogue items remain in the benchmark with finite low scores outside the candidate set; the candidate generator changes scoring, not evaluation membership. All comparisons are exploratory on the same 500 validation users (498 with relevant targets). Paired user-bootstrap intervals do not account for choosing among 40 attempts, training-population variation, or alternative onboarding protocols. Unobserved relevance is unknown. No fresh-final-holdout evidence enters this review.

Next decision: reject tested MF reranking rule.

## Reserved final holdout — separate benchmark

One frozen candidate, `cm-20261119-full-knn-conditional`, was evaluated once against all fixed references. No tuning followed this evaluation. These numbers are not directly comparable to the validation table above.

| Method | NDCG@5 | Hit rate@5 | Recall@5 | Distinct movies |
| --- | ---: | ---: | ---: | ---: |
| candidate | 0.485361 | 83.57% | 5.62% | 76 |
| original_sample | 0.143837 | 36.87% | 1.34% | 253 |
| full_user | 0.159888 | 41.68% | 1.62% | 474 |
| established_popularity | 0.380212 | 75.15% | 4.09% | 8 |
| full_popularity | 0.380212 | 75.15% | 4.09% | 8 |

Candidate minus popularity NDCG: +0.105149; paired95% interval [0.08593586621705428, 0.12352216660464296]. App deployment remains unchanged.

[Frozen final result](final-results/cinematch-systematic-20261119.json). The one-shot sentinel and exact frozen evaluation source are in the linked artifact directory.

## app-model-comparison-20261119-v1 — separate direct comparison

Mikkel authorized comparing the selected conditional method with Anton's updated rating model. Source pinned to Anton commit `6e85958f503525b3020ec80fe613934e3c723d7b`; the newly added recommendation function is used unchanged. No hybrid was trained.

Both algorithms were refitted on 21,431,215 identical ratings from 196,442 users over 3,788 verified common movies. One thousand fresh validation users fit Anton's fixed blend weights; 2,000 further users were withheld for confirmation. All 4,500 old/new held-out users were excluded from comparison training and tag features. This benchmark has different catalogue/cohort/onboarding rules from the campaign above and must remain separate from its leaderboard.

Frozen confirmation with five positive opinions: conditional NDCG@5 **0.584141**, Anton **0.380903**, rating-count popularity **0.489307**, mean-rating baseline **0.336564**. Conditional-minus-Anton is +0.203238, paired 95% interval [0.191743, 0.214812], and remains positive after adjustment for the three primary comparisons. Conditional leads all four secondary profile checks. Anton exposes more distinct movies and fewer explicitly low-rated targets. This establishes an advantage in retrospective recovery of recorded likes, not online discovery or universal superiority.

All 109 Python tests passed; independent data and ranking audits passed. Source, models, policies, splits, per-user outcomes and final confirmation records are preserved. Production model/bundle unchanged. See the [complete comparison report](comparisons/app-model-comparison-20261119-v1/REPORT.md) and [final machine-readable results](comparisons/app-model-comparison-20261119-v1/test_results.json). Future changes must not tune on this confirmation set.

## app-model-comparison-20261119-v1/depth15 — requested evaluation cutoff extension

Mikkel requested fifteen instead of five after the NDCG@5 explanation. Evaluate the first fifteen recommendations while retaining the same five positive inputs, frozen model weights and 2,000 evaluation users. No retraining or hybrid. This is a descriptive follow-up on already inspected users, not fresh independent confirmation.

NDCG@15: conditional **0.514879**, rating-count popularity **0.411025**, Anton **0.313419**, mean rating **0.286619**. Corresponding HitRate@15: **98.45%, 96.30%, 90.40%, 87.05%**. All original top-ten rankings match exactly. Six new boundary tests and independent recomputation of every user's top-fifteen metrics passed. See the [top-fifteen report](comparisons/app-model-comparison-20261119-v1/depth15/REPORT.md), [results](comparisons/app-model-comparison-20261119-v1/depth15/results.json) and [audit](comparisons/app-model-comparison-20261119-v1/depth15/audit.json).

## app-model-comparison-20261119-v1/onboarding15_top5 — fifteen positive inputs

Mikkel explicitly requested fifteen positive onboarding opinions and five recommendations. Keep the same frozen models, 2,000 evaluation users and original randomized positive seed order; extend the original five inputs to fifteen. All fifteen are excluded as watched. Reconstruct all original raw observations and return the old twenty-opinion profile's last five entries to hidden targets. No retraining or ensemble.

NDCG@5: conditional **0.527139**, rating-count popularity **0.431625**, Anton **0.354245**, mean rating **0.290871**. HitRate@5: **88.65%, 83.10%, 73.45%, 65.95%**. Conditional remains highest within this setting. Cross-size score changes do not isolate the value of more onboarding because additional disclosed likes leave the target set. This is descriptive follow-up on known test users, not independent new confirmation.

Eight focused tests and independent all-user/profile/ranking audits passed. Original artifacts and app unchanged. See the [fifteen-input report](comparisons/app-model-comparison-20261119-v1/onboarding15_top5/REPORT.md), [results](comparisons/app-model-comparison-20261119-v1/onboarding15_top5/results.json) and [audit](comparisons/app-model-comparison-20261119-v1/onboarding15_top5/audit.json).

## app-model-ensemble-20261119-v1 — output complementarity experiment

Mikkel authorized testing whether the two models improve one another's output. Keep fifteen positive onboarding opinions and five recommendations, but reserve new users because previous confirmation results motivated the hypothesis. Exclude all 4,500 previous holdouts, then reserve 1,000 dedicated Anton calibration users, 1,000 blend-validation users and 2,000 fresh confirmation users. Refit both methods on 20,783,195 identical ratings from 192,442 users over the same 3,788 movies; tag features exclude all 8,500 old/new holdouts.

Test eight conditional-dominant reciprocal-rank blends, with average ranks for ties and watched exclusions before ranking. All blends lose to conditional on validation. Freeze the best nonzero challenger (97.5% conditional/2.5% Anton) under the prespecified policy and evaluate once on confirmation. NDCG@5: conditional **0.518893**, blend **0.517325**, count popularity **0.423478**, Anton **0.358378**, mean rating **0.289065**. Blend-minus-conditional **−0.001568**, paired 95% interval **[−0.002639,−0.000504]**. No measured ranking improvement; known-low rate remains 2.01%, with only 33 rather than 31 distinct exposed films.

Complementary recorded likes exist: Anton finds at least one top-five known like absent from conditional for 1,155/2,000 users. But the actual blend adds 46 and removes 61 known-like recommendation slots, improving 52 users' NDCG and worsening 91. This tested combination does not earn adoption on the stated objective. No claim that all conceivable hybrids must fail; no further tuning on confirmation. Production artifacts remain unchanged.

Forty-eight focused tests and the independent data/ranking audit passed. See [report](comparisons/app-model-ensemble-20261119-v1/REPORT.md), [frozen selection](comparisons/app-model-ensemble-20261119-v1/selection.json), [results](comparisons/app-model-ensemble-20261119-v1/test_results.json), and [independent audit](comparisons/app-model-ensemble-20261119-v1/audit.json). This is a new benchmark; do not compare its absolute scores to earlier cohorts as a pure model change.
