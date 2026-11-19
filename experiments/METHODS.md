# CineMatch campaign methods

This describes the implementations used by the systematic campaign. Detailed records contain each run's exact settings, source snapshot and artifacts. The original app model is unchanged. Results belong in the [journal](JOURNAL.md), [leaderboard](leaderboard.json) and per-run records; this document does not declare a final winner.

## Benchmark and evidence stages

All comparable campaign runs use the existing ordered catalogue of 9,021 MovieLens IDs and the same 500 validation users, five disclosed ratings per user, remaining target ratings, and watched-item exclusions. A target rating of at least four stars supplies binary relevance. Two validation users have no positive targets, so 498 enter the ranking averages. Every scorer receives only onboarding movie indices and ratings; it does not receive hidden targets or a training-user ID.

The original `top_items` function excludes the five disclosed movies and breaks score ties by ascending catalogue order. The campaign's final evaluation delegates to the original `evaluate_profiles` implementation for every user. NDCG@5 is primary; hit rate, recall, catalogue coverage and distinct recommended movies accompany it. Rating RMSE clips rating predictions to 0.5–5, averages squared errors within each user, then takes the square root of their equally weighted mean. Ranking-only scores have `rmse_macro: null`.

| Training scale | Population | Meaning |
| --- | --- | --- |
| Sample | 10,000 users; 1,500,603 ratings | Original sample within the fixed catalogue |
| Full | 199,447 users; 30,495,721 ratings | Established full-user training population within the same catalogue |

Both populations exclude the established validation, legacy-test and fresh-final-test identities. The campaign verifies those reservations using the full-data manifest and checks array hashes and movie order. Model development opens validation profiles only. The already-inspected legacy test remains historical evidence. The reserved fresh final holdout may be scored only after a candidate and procedure have been frozen; full-data **validation** runs are still exploratory, even where a proposal filename calls them “confirmation.”

Every completed run is compared with original sample MF, full-user MF, established sample-count popularity, full-data-count popularity, its training-matched popularity control, and its immediate parent where applicable. Popularity counts every recorded rating, not just likes. Identical aggregate scores do not justify omitting either popularity control.

The initial screen contains 27 model configurations across seven change families. The separate runs of 36 and later 58 automated correctness checks are test-suite results, not 94 model experiments. Tests exercise identities, score arithmetic, inference and failure handling on controlled fixtures. Replications, ablations, failed attempts and later final evaluation remain separately labelled records.

## Explicit-rating factorisation and profile fitting

The base score is `mu + b_user + b_movie + p_user dot q_movie`. The training mean `mu` is fixed. Training jointly learns user/movie vectors and biases from observed ratings using Adam. The default objective is mean squared error plus a per-observed-rating L2 penalty on the participating vectors and biases. No missing rating is replaced by zero.

For a new user, movie parameters stay fixed. The solver minimises the sum of squared onboarding residuals plus `lambda * ||p_user||² + lambda_bias * b_user²`. The original selected settings are 16 factors, `lambda=1`, and `lambda_bias=5`. Profile experiments vary lambda, omit the fitted user intercept, or substitute the onboarding mean for the training mean. The centred version adds its substituted mean back, retaining rating-scale scores. No-rating input returns `mu + b_movie`.

Movie-prior experiments apply `b'_i = scale * b_i * n_i/(n_i+tau)`; `tau=0` leaves shrinkage inactive. The transformed bias is used both when fitting onboarding residuals and scoring candidates. Removing a bias therefore refits the user profile; the experiment is not merely subtracting bias from an unchanged score vector.

Training variants change factor count, bias regularisation or loss. An extra inverse-count bias penalty adds `tau * mean(b_user²/n_user + b_movie²/n_movie)`. Holding other parameters fixed under MSE, a bias's normal-equation denominator becomes `n * (1 + lambda_bias_train) + tau`. This strengthens shrinkage of poorly supported entities. The Huber variant uses PyTorch's Huber loss with the recorded delta; its central quadratic term is `0.5 * error²`, whereas MSE uses `error²`. Its effect consequently includes altered loss-to-regularisation scaling, not only robustness to large errors.

Training selects the checkpoint by validation NDCG, with lower rating RMSE breaking ties, and retains the chosen fold-in penalty. All searched penalties and epoch histories are recorded. Changing factor count changes initialisation draws; holding the seed constant does not isolate every initial parameter.

## Popularity hybrids and candidate ranking

For score blending, `z(x)=(x-mean(x))/std(x)` is computed across the full catalogue before exclusions; a constant vector maps to zeros. The score is `alpha * z(personal_score) + (1-alpha) * z(log(1+training_count))`. Both components use the exact same movie order. A confidence variant replaces alpha with `alpha * n_positive/(n_positive+strength)`, where `n_positive` counts disclosed ratings at least four; it uses no hidden outcomes.

The candidate-ranking variant selects the `candidate_count` most-rated unwatched movies and orders those by the personal score. Other catalogue movies receive a finite score below every selected candidate. This changes the recommendation algorithm's candidate generation, not the evaluator's catalogue or relevance denominator: all 9,021 movies remain in the benchmark. It can therefore be compared on the main benchmark, while being explicitly identified as a candidate-generation experiment. These blends and candidate scores are not star predictions; RMSE is inapplicable.

## Item neighbours

For positive-feedback training, construct a binary user–movie matrix with an entry only when the user rated the movie at least four stars. The observed-feedback variant includes every recorded rating. Multiple contributions from one user to one movie are collapsed to one binary entry. Let `c_ij` be the number of training users with both entries and `n_i` the number with the seed entry.

- Cosine weight: `c_ij / sqrt(n_i*n_j)`.
- Conditional weight: `c_ij / n_i`, estimating co-occurrence conditional on the seed.
- General asymmetric weight, supported by the implementation: `c_ij / (n_i^alpha * n_j^(1-alpha))`.

Each weight is multiplied by `c_ij/(c_ij+shrink)`. Self-edges are removed; each seed retains its top K outgoing neighbours, with deterministic movie-ID tie breaking. Computation uses sparse row blocks instead of a dense user–movie matrix. The saved arrays retain the original movie order.

For new users, the score is `sum_seed weight(rating_seed) * similarity(seed,candidate)`. The default positive aggregation uses `max(rating-3,0)`, so **3.5-star onboarding ratings contribute even though training positives require four stars**. Signed aggregation uses `rating-3`; binary-positive aggregation, where configured, uses `1[rating>=4]`. No user-factor fitting is required.

If every accumulated candidate score is zero, return training rating-count popularity. Otherwise the existing catalogue-wide ranking applies, including zero-score ties. In the recorded signed screen, 16 users had no positive score but some negative scores: zero-valued untouched candidates then won by catalogue-order tie breaking. This limitation is preserved in the result; a changed fallback must be a new experiment. Conditional similarity naturally favours broadly co-liked candidates, so its gains should not be described as evidence of long-tail discovery.

The independent [screening audit](campaigns/cinematch-systematic-20261119-screening-audit.json) reproduces existing scores, verifies sparse weights and quantifies fallback use. It is descriptive verification, not an additional model configuration.

## Pairwise ranking with item contexts

The ranking model learns separate context vectors `c_i`, candidate vectors `q_i`, and optional candidate biases `b_i`. For each training triple, sample an eligible training user uniformly, then two distinct items that user rated at least four stars (anchor `a`, positive `p`), and one item `n` the user never rated. Every recorded item, including known dislikes, is excluded from negative sampling. Unobserved negatives are an approximation, not asserted dislikes.

The pairwise loss is `softplus(-(c_a dot (q_p-q_n) + b_p-b_n))`, plus per-pair L2 penalties on participating vectors and biases. The sampled pair budget defines an epoch; it is **not** an exhaustive pass over the training ratings. The screen uses 300,000 pairs per epoch; full-data follow-up proposals use 1,000,000. Each run records its actual eligible training users, positive interactions, pair budget, seed, selected epoch and history. A full-data model has access to the full reserved training population but does not consume all positive pairs.

For a new user, default weights are `rating-3` for ratings at least four and zero otherwise. Form `p = sum(weight_i*c_i)/sum(abs(weight_i))`, then score candidates with `p dot q_j + b_j`. Signed aggregation instead uses `rating-3` for every disclosed rating; binary aggregation uses a positive indicator. No informative input returns `log(1+training_count)`, which has the popularity ordering. Inference is compatible with the training item-context representation and requires neither a known training user nor the explicit-rating MF ridge solver. Scores are not predicted stars.

Stochastic follow-ups vary training seeds as separate runs. Report their distribution rather than treating a favourable seed as robust evidence. Epoch selection still uses the same validation group, and all such results remain exploratory until a final frozen evaluation.

## Genre metadata

The lightweight content model uses only the supplied MovieLens genre labels, aligned by movie ID and checked against the known genre vocabulary. It normalises each movie's binary genre vector to unit length. A rating-weighted sum of seed vectors forms a user profile, and scores are cosine similarities with that profile. The default weights are signed `rating-3`; an empty or zero profile falls back to training rating counts. The hybrid applies the same standardised popularity blend described above.

These labels do not provide mood, actor, plot, content-safety or reliable movie-versus-TV classifications. Genre scores are ranking scores, not ratings. No external text embeddings or unverified metadata are introduced by this family.

## Uncertainty, preservation and reproduction

Paired NDCG differences use the existing 2,000-replicate user bootstrap with seed 42. Report observed change, interval, trade-offs and next decision separately. These intervals condition on the chosen training sample, seed and validation experiment; they do not correct a search across many configurations. Greater coverage alone does not establish improved relevance. The retrospective benchmark rewards recovery of recorded likes and cannot credit every enjoyable unfamiliar title.

The runner writes a plan before execution and records the exact executed source, dependency versions, data hashes, parent, configuration and command. Per-user rows, top-five lists, examples, comparisons and runtime measurements accompany completed runs. Failed and interrupted runs remain visible. Existing run IDs and model artifacts are preserved, and deployment requires a separate decision. Reproduce a specific run using its record's command with a new unique run ID; replay uses its saved source snapshot.
