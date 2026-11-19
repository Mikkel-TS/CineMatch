# CineMatch experiment tracking

Each experiment answers a stated question and preserves the model, data definition, code, settings, results, and reason for the decision. Earlier runs stay available for comparison. A higher score is evidence to review; it does not automatically replace the app's model.

## Systematic campaign, 19 November 2026

The campaign extends these same `records/` and `artifacts/experiments/` locations. Start with:

- [Campaign review](CAMPAIGN_REVIEW.md): findings, regressions, trade-offs and decisions.
- [Experiment journal](JOURNAL.md): every attempted configuration, including status and immediate-parent comparisons.
- [Machine-readable leaderboard](leaderboard.json): all validation attempts and all fixed reference metrics. Every validation row is exploratory, including full-data confirmations.
- [Methods](METHODS.md): objectives, scoring, feedback definitions and deployment constraints.
- [Campaign plan](campaigns/cinematch-systematic-20261119.json), [full-data confirmations](campaigns/cinematch-systematic-20261119-confirmation.json), and [removal tests](campaigns/cinematch-systematic-20261119-removals.json).
- Final holdout results live separately under `final-results/`; they must never be mixed into the validation leaderboard.

The campaign ran 27 initial configurations across seven change families, followed by 13 controlled refinements, full-data confirmations and removal tests. Correctness tests use synthetic data and are not counted as model experiments. The earlier `diagnostic-cold-bias` artifact remains a historical initialization diagnostic (saved validation NDCG 0.36361, epoch 1, only seven distinct recommendations); its executed-source provenance is incomplete, so it is not a clean campaign comparison or the current model.

Use Python 3.12 and the campaign's exact environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/requirements.lock -e .
.venv/bin/python -B -m pytest -q
```

Do not recreate an existing environment merely to inspect results. SciPy was added for sparse item-neighbour computation; the original project dependency files were preserved. Exact dependency snapshots also accompany every run.

Replay an individual run with a fresh unique ID:

```sh
.venv/bin/python -B -u experiments/run_campaign.py \
  --replay cm-20261119-full-knn-conditional --run-id replay-conditional-example
```

Replay copies the original run's archived executed source, preserving its configuration, data identity and fixed-reference comparisons. Its record links back to the original. Training data and parent artifacts must still exist locally. Use a new ID for every attempt; results are never overwritten. For a new experiment, write a proposal JSON following the campaign plan schema and pass `--config path/to/proposal.json`. Defaults and selected settings are additionally preserved beside each model. During execution, `run.log`, the record, journal and leaderboard update automatically.

The candidate freeze and one-shot final evaluation are implemented in `final_evaluation.py`. The final-start sentinel prevents another evaluation, including after a failed or interrupted attempt. Reading results after final evaluation does not reopen development or permit selecting a different candidate against that holdout. App adoption remains a separate decision.

Tracking corrections made during implementation are recorded in affected run records, while their pre-execution `plan.json` files remain unchanged. They correct refinement-parent links and replay command dispatch; numerical results are preserved. Early snapshots predate additional input-hash and interruption checks; the final audit verifies that their recorded inputs remained unchanged.

## Records and artifacts

Keep this directory, its runner, and the small JSON records in Git. Large datasets and model files stay under the existing ignored `artifacts/` directory. They are currently local; a shared backup of those artifacts is necessary for the team to reproduce a run if this machine or its files are unavailable. An experiment record alone cannot restore model weights.

| Location | Contents |
| --- | --- |
| `records/<run-id>.json` | Question, parent run, decisions, settings, benchmark identity, status and results |
| `../artifacts/experiments/<run-id>/plan.json` | Original experiment plan, preserved before training |
| `../artifacts/experiments/<run-id>/source/` | Exact executed model source, runner, dependency files and Git diff; the revision is in the record |
| `../artifacts/experiments/<run-id>/data/` | Training arrays, frozen evaluation profiles, fresh final holdout and checksums |
| `../artifacts/experiments/<run-id>/model/` | Selected checkpoint, configuration, learning history and validation metrics |
| `../artifacts/experiments/<run-id>/comparison.json` | Paired validation comparisons against the parent model and popularity |
| `../artifacts/experiments/<run-id>/run.log` | Progress and errors |
| `../artifacts/experiments/<run-id>/result.json` | Final status and results; failed runs remain recorded |
| `../artifacts/experiments/<run-id>/review.json` | Interpretation of the result, uncertainty, verification and adoption status |

The historical sample run is recorded in [records/mf16-sample-20261023.json](records/mf16-sample-20261023.json). Its original code revision was not embedded in the checkpoint; the record explicitly preserves that uncertainty.

## First comparison

The [full-user experiment](records/mf16-full-users-fixed-catalog-20261117.json) completed using 30,495,721 ratings from 199,447 users. Early stopping selected epoch 3 after 6 epochs. Both models were compared on the same 500 validation users, of whom 498 have relevant targets.

| Validation measure | Original sample model | Full-user model | Popularity |
| --- | ---: | ---: | ---: |
| NDCG at 5 | 0.1519 | 0.1675 | 0.3795 |
| Hit rate at 5 | 38.15% | 40.96% | 74.30% |
| Distinct recommended movies | 231 | 441 | 8 |
| Rating RMSE | 0.9247 | 0.9291 | Not applicable |

Popularity fitted on the old and expanded training sets produced the same aggregate ranking metrics on this benchmark. The model's observed NDCG gain is 0.0156, or 10.3% relative, but the paired 95% bootstrap interval for that difference is [-0.0059, 0.0370]. It therefore remains uncertain whether the ranking improvement generalises beyond these validation users. Coverage increased and rating RMSE was slightly worse. More data alone did not close the gap to popularity. Adoption remains pending; neither test group was evaluated. Exact results and limits are preserved in the experiment record and comparison artifacts.

## Decisions for the full data experiment

On 17 November 2026, Mikkel requested full-data training and chose to keep the existing 9,021-movie catalogue for a clean comparison.

The experiment asks whether additional training users and their ratings improve recommendations for that fixed catalogue. It retains 16 factors, the rating-prediction objective, training settings, fold-in search, random seed and early-stopping policy from the sample run. It also retains the exact validation users, their five onboarding ratings, remaining targets, candidate order, exclusions and metric implementation.

The original preparation command cannot provide this comparison just by changing `--train-users`: its held-out users depend on the training-user count. The experiment runner instead copies the original evaluation profiles and explicitly excludes their users from training.

Implementation choices recorded separately from user decisions:

- Reserve 500 previously unused users for a later final test before expanding training. These users were absent from the old training sample too.
- Keep the original inspected test users excluded from training. Their previous results are historical development evidence.
- Select and compare models using validation data only. Neither test group is evaluated in this experiment.
- Recompute popularity using the expanded training ratings, and report both the old and new popularity controls.
- Keep the existing model unchanged until a separate adoption decision.

This run uses all available training users with ratings inside the fixed catalogue. Ratings outside that catalogue and ratings belonging to reserved evaluation users are excluded. One eligible training user has no ratings inside the catalogue and contributes no training examples.

More ratings also mean more updates per epoch. The same random seed can produce different initial movie vectors when the user table changes size. This experiment therefore compares data scale under the same training policy; it does not hold compute or every initial parameter fixed.

## How to assess a change

Before starting, record the question, expected benefit, parent run, intended change, fixed conditions and how success will be judged. Distinguish user-confirmed decisions from implementation choices and untested hypotheses.

Compare NDCG@5 first, alongside hit rate, recall, catalogue coverage and rating RMSE. Compare the candidate with both the older model and popularity using the same benchmark. Preserve per-user results and paired uncertainty estimates. One training seed and a bootstrap over evaluation users do not establish robustness across training seeds or datasets.

Afterwards, record whether the result improved, worsened or was inconclusive; what the evidence supports; and whether the decision is to adopt, reject or investigate further. Leave adoption pending until explicitly decided. A higher coverage number alone does not establish better recommendations.

Change one intended variable per experiment when practical. If several changes are combined, record them and avoid attributing the entire effect to one of them. Changing onboarding, candidate movies, relevance rules or evaluation users creates a new benchmark; rerun older models on that benchmark rather than comparing incompatible scores.

The fresh final holdout must remain untouched while selecting models. The existing `cinematch evaluate` command loads `test.npz`, which is the historical test group, not `fresh_test.npz`. Final evaluation of the fresh group needs an explicit later step after selecting the candidate.

## Running the full data experiment

From the repository root:

```sh
.venv/bin/python -B -u experiments/run_full_data.py
```

The default run ID is `mf16-full-users-fixed-catalog-20261117`. Existing run IDs are rejected. An exact rerun needs a new `--run-id`; the original record and artifacts remain preserved.

This runner automates recording for the full-user experiment. Other experiment types should follow the same record structure; calling `cinematch train` directly does not automatically add a record here. Future runs should reuse the established holdout groups rather than draw replacements.
