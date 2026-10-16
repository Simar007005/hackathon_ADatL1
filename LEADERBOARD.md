# Leaderboard

Every submission is evaluated on the test samples of `config/baseline.yml`: zero-bias run 396102 against the
four benchmark signals. Ranked by **mean AUC**; ties go to the model with fewer BOPs. ★ marks the
*Pareto front*: models that no other model beats on both mean AUC and BOPs.
See [How to submit](README.md#how-to-submit) to add your team.

| # | team | mean AUC | lowest AUC | highest AUC | reconstruction error | BOPs | submitted |
|---|---|---|---|---|---|---|---|
| 1 | [signal_hunters](submissions/signal_hunters) ★ | 0.9022 | 0.8479 | 0.9668 | 0.4134 | 4,555 | [`df2f25b`](https://github.com/ekauffma/hackathon_ADatL1/commit/df2f25b04c71a3bbe19de1cb49a792ead41e08af) 2026-10-16 04:24 by [tm1307](https://github.com/tm1307) |
| 2 | [baseline-dense-qae](submissions/baseline-dense-qae) | 0.8112 | 0.7339 | 0.9123 | 0.4934 | 470,811 | [`d297f57`](https://github.com/ekauffma/hackathon_ADatL1/commit/d297f57a9ac2adc3d709d12a2aed9d6ab0f16b12) 2026-10-16 03:37 by [ekauffma](https://github.com/ekauffma) |
| 3 | [baseline-dense-ae](submissions/baseline-dense-ae) | 0.8084 | 0.7357 | 0.9102 | 0.4867 | 2,575,131 | [`d297f57`](https://github.com/ekauffma/hackathon_ADatL1/commit/d297f57a9ac2adc3d709d12a2aed9d6ab0f16b12) 2026-10-16 03:37 by [ekauffma](https://github.com/ekauffma) |

*Last updated 2026-10-16 05:52 UTC.*
