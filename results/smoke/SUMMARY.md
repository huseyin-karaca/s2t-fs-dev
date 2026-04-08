# Results — smoke

## Test WER per dataset

| Method | ami | macro |
|---|---:|---:|
| `fastt_sdt` | 0.2943 | 0.2943 |
| `fastt_xgb` | 0.2909 | 0.2909 |
| `oracle` | 0.1961 | 0.1961 |
| `selectkbest_xgb` | 0.2863 | 0.2863 |

## Primary criterion: FASTT vs best classical baseline (per dataset)

| Dataset | best classical | classical WER | best FASTT | FASTT WER | Δ rel |
|---|---|---:|---|---:|---:|
| ami | `selectkbest_xgb` | 0.2863 | `fastt_xgb` | 0.2909 | **-1.62%** ❌ |

## Wallclock per job

| Dataset | Method | n_train | trials | wallclock | tuning | refit fit | refit score |
|---|---|---:|---:|---:|---:|---:|---:|
| ami | `fastt_sdt` | 500 | 2 | 78.3s | 16.4s | 61.53s | 0.2779s |
| ami | `fastt_xgb` | 500 | 2 | 400.4s | 178.7s | 221.41s | 0.2034s |
| ami | `oracle` | 200 | 0 | 1.0s | 0.0s | 0.00s | 0.0001s |
| ami | `selectkbest_xgb` | 500 | 2 | 6.4s | 2.2s | 3.97s | 0.0347s |
