# CineMatch recommendation model

An explicit-rating matrix-factorisation model trained on MovieLens. It learns user and movie profiles plus rating biases. New users receive a regularized profile from their onboarding ratings while all trained movie parameters stay fixed.

The original MovieLens data and licence are documented in [Dataset/ml-32m/README.txt](Dataset/ml-32m/README.txt). Raw dataset files stay in `Dataset/ml-32m/`. Model code, tests, configuration, and generated artifacts live at the CineMatch repository root.

## Set up

Use Python 3.11 or later (Python 3.12 is a tested choice). Run all commands below from the CineMatch repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For the exact dependency versions used in the verified environment, install with `python -m pip install -r requirements.lock -e .` instead. The lock records Python 3.12/macOS arm64 dependencies; other platforms may resolve additional platform-specific packages.

The implementation runs on CPU. Start with the sample below before expanding to all ratings. Dependencies, prepared data, and model outputs are ignored by Git.

## First verified local run

On 23 October 2026, the pipeline was run locally on 1,500,603 training ratings from 10,000 users, with 9,021 eligible movies. Validation and test each used 500 different users with five onboarding ratings. The saved checkpoint is `artifacts/mf16/model.pt`; the selected epoch was 6, with fold-in regularization 1. The full test report is `artifacts/mf16/test.json`.

| Method | Test NDCG@5 | Catalogue coverage@5 |
| --- | ---: | ---: |
| Matrix factorisation with onboarding | 0.1328 | 2.90% |
| Regularized rating-bias baseline | 0.0619 | 0.067% |
| Rating-count popularity | 0.3721 | 0.078% |
| Trained movie prior without a personal vector | 0.0055 | 0.055% |

Onboarding improves this model over its own movie prior, but the model does **not** beat the popularity control on this evaluation. It is a functioning experimental starting point, not an established production winner. Further tuning should use validation data; this inspected test group should not be reused as a final untouched test. Ten automated tests passed, and both catalogue search and recommendations were verified on the saved model.

These artifacts already exist in the local workspace and are not tracked by Git. Use new output directories for further experiments. A separate `artifacts/diagnostic-cold-bias` run records an earlier initialization diagnostic and is not the recommended checkpoint.

## Prepare a reproducible experiment

```sh
cinematch prepare --data-dir Dataset/ml-32m --output artifacts/data \
  --train-users 10000 --validation-users 500 --test-users 500 \
  --support-count 5 --min-item-ratings 10 --seed 42
```

Preparation scans the whole ratings file in chunks and samples **users**, retaining their histories. Training, validation, and test users are disjoint. The candidate catalogue contains movies with at least ten ratings from the training users. This restriction makes the first experiment manageable; it does not solve recommendation for new or obscure movies.

Each evaluation user contributes exactly five randomly selected onboarding ratings. Remaining eligible ratings are targets. Users with fewer than six eligible ratings are excluded and counted in the manifest. Item counts, eligibility, ID mappings, and global rating averages come from training data only. Unknown ratings remain unknown; no zero-filled dense user/movie matrix is created.

The output directory contains training arrays, separate validation and test profiles, a catalogue, and a manifest with user IDs, configuration, counts, and file checksums. Existing directories are never overwritten; use a new output path for another split. To sample all remaining users, pass `--train-users 0`. Full-data runs need more memory and time; preparation retains selected ratings in memory after the chunked scan.

## Train and choose settings on validation data

```sh
cinematch train --data artifacts/data --output artifacts/mf16 \
  --factors 16 --epochs 20 --batch-size 4096 --learning-rate 0.001 \
  --regularization 0.01 --fold-in-grid 1 5 20 --patience 3
```

The prediction is `global mean + user bias + movie bias + dot(user profile, movie profile)`.

User/movie biases are first initialized from a regularized bias fit on training ratings. Training then jointly updates biases and vectors to minimize mean squared error on **observed** ratings plus a per-rating L2 penalty on the participating user/movie vectors and biases. Adam performs the updates. The training regularization parameter applies to this specific loss definition; it is not interchangeable with penalties from other libraries.

After each epoch, the pipeline fits new-user profiles from the five validation onboarding ratings. It evaluates the supplied fold-in penalties and retains the epoch/penalty with highest validation NDCG@5 (lower RMSE breaks ties). Early stopping occurs after three epochs without improvement. Training never loads test profiles.

Fold-in minimizes the **sum** of squared onboarding errors plus L2 penalties on the new user's vector and bias. The default bias penalty is 5. These parameters are separate from the training penalty and should be tuned on validation users. Strong regularization matters when there are only a few ratings.

Outputs include `model.pt`, `catalog.csv`, `config.json`, `history.json`, and `validation.json`. The checkpoint includes ID mappings, parameters, the selected fold-in penalty, versions, and the fingerprint of the prepared dataset. The best checkpoint may come from an earlier epoch than the final one.

To compare configurations, keep the same prepared data and use distinct run directories. For example:

```sh
cinematch train --data artifacts/data --output artifacts/mf32 --factors 32
```

Compare `validation.json` across runs, including three controls: the separate rating-bias baseline, a rating-count popularity ranking, and the trained model's own movie prior without a personal taste vector. A high score accompanied by very low catalogue coverage can reflect recommending the same well-known films to everyone. Do not automatically adopt matrix factorisation if it fails to improve on the controls. The first run verifies a working method, not the strongest possible model.

## Evaluate the selected model once on test users

After selecting a run using validation results:

```sh
cinematch evaluate --model artifacts/mf16/model.pt --data artifacts/data \
  --output artifacts/mf16/test.json
```

This compares the selected model with the three controls above. All use the same candidate catalogue and watched-item exclusions. Bias-based methods estimate a user intercept from the onboarding ratings; that intercept changes rating predictions but not ranking. The popularity control ranks by training rating count and has no meaningful star-rating RMSE. The report includes NDCG@5, Recall@5, Hit Rate@5, catalogue coverage, macro RMSE where applicable, and paired user-bootstrap intervals for the NDCG differences.

- A held-out rating of at least four stars counts as relevant.
- Ranking scores are unclipped; watched/onboarding movies are excluded. Ties use ascending movie ID.
- Users without relevant targets are excluded from ranking averages only and counted in the report.
- RMSE clips predicted ratings to 0.5–5 and gives each user equal weight.
- Unrated movies have **unknown relevance**. These metrics measure recovery of observed likes, not real-world satisfaction.
- This is retrospective reconstruction, not future prediction. Random disclosed ratings do not simulate self-selected favourites perfectly.
- After using test results to change a model, reserve fresh test users for a final claim. Do not repeatedly tune against the test report.

The eligible catalogue omits movies with weak training support. Record that limitation when reporting results. Report files cannot silently overwrite previous evaluations.

MovieLens also contains some TV/miniseries entries. Its provided metadata has no reliable media-type field, so this implementation does not guarantee feature-film-only results. Explicitly exclude verified non-movie IDs if the product requires that restriction.

## Find movies and recommend for a new user

First find eligible MovieLens IDs:

```sh
cinematch search --model artifacts/mf16/model.pt --query 'Toy Story'
```

Supply actual ratings (the following IDs/ratings are illustrative; check your model's catalogue):

```sh
cinematch recommend --model artifacts/mf16/model.pt \
  --rating 1=4.5 --rating 356=5 --rating 296=3 \
  --rating 318=5 --rating 593=2 --top-k 5
```

`--rating` may be repeated for any number of known movies. Supply half-star ratings from 0.5 through 5. Duplicate or unsupported movie IDs are rejected with an explanation. Add `--exclude 50 260` to exclude additional watched films. Omit ratings for a non-personalized movie-prior list. If fewer than five eligible unseen movies remain, the command returns those available.

The result contains movie IDs, titles, genres, and raw ranking scores. Scores are not probabilities or guaranteed star predictions. Genres are display metadata, not model inputs. No actor, mood, or tag-based explanations are invented.

For application integration, load `cinematch.recommend.Recommender` once with the checkpoint path and call `recommend([(movie_id, rating), ...])`. Store each app user's ratings in the app and pass their current ratings when recommendations are requested. A user ID from MovieLens is not required. The model does not persist app profiles or provide an API server.

## Verify changes

```sh
pytest -q
```

Tests cover disjoint users, training-only catalogue eligibility, reproducible preparation, hidden-target separation, data-integrity checks, new-user fitting, watched-item exclusion, ranking metrics, checkpoint reload, and the separation between training and final testing.

## Next comparisons

Evaluate item-to-item collaborative filtering and a cleaned-tag score blend against this model using the same protocol. These alternatives, a user interface, and full-data training are outside this first implementation.
