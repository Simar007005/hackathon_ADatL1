# Leaderboard

Every submission is evaluated on the test samples of `config/baseline.yml`: zero-bias run 396102 against the
four benchmark signals. Ranked by **mean AUC**; ties go to the model with fewer BOPs. ★ marks the
*Pareto front*: models that no other model beats on both mean AUC and BOPs.
See [How to submit](README.md#how-to-submit) to add your team.

| # | team | mean AUC | lowest AUC | highest AUC | reconstruction error | BOPs | submitted |
|---|---|---|---|---|---|---|---|
| 1 | [baseline-dense-qae](submissions/baseline-dense-qae) ★ | 0.8112 | 0.7339 | 0.9123 | 0.4934 | 470,811 |  |
| 2 | [baseline-dense-ae](submissions/baseline-dense-ae) | 0.8084 | 0.7357 | 0.9102 | 0.4867 | 2,575,131 |  |

*Last updated 2026-10-16 03:35 UTC.*
