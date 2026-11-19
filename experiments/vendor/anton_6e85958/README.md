# Anton source snapshot

Source: [AntonDev29/CineMatch-AntonModel](https://github.com/AntonDev29/CineMatch-AntonModel/tree/6e85958f503525b3020ec80fe613934e3c723d7b), commit `6e85958f503525b3020ec80fe613934e3c723d7b` (`recommend_added`, 19 November 2026).

These three files are unmodified source snapshots used for the direct comparison. Their Git blob hashes were independently matched to the GitHub tree:

| File | Git blob SHA-1 |
| --- | --- |
| cinematch.py | ddd4bb503480de0a0db3e7e6a58c0c2ebb29f851 |
| movielens.py | ba13fe2b1ea1aa614c031b0527a6a048a5abb458 |
| rating_model.py | aa51877f4ec2ce086b2cb3c2546f24d8dafca541 |

The surrounding adapter supplies shared verified movie identities, training rows and exclusion of held-out tag authors. It calls the source scoring function unchanged. The function does not enforce `personalFromRatings`, despite that field in the export. The benchmark therefore names actual function behavior as primary and documents the declared fallback separately.

The upstream recommendation entry point defaults to training artifacts absent from this repository. The comparison builds isolated artifacts; it does not run the upstream CLI or overwrite the app bundle. No changes to the upstream repository were made.
