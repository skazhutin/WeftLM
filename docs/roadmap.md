# WeftLM implementation roadmap

Based on [the project specification](project_spec.md). Initialization is complete.
Each numbered row is one bounded implementation task and one feature commit.

## Execution and verification

Implement, test, review the diff, commit, push, verify the remote SHA and wait for
successful CI before starting the next task. Record the verified SHA in the
following commit; this avoids embedding a commit's own hash in its contents.
Physical-node and GPU verification are distinct from CPU CI. Pending hardware
work is not reported as completed by local-process tests.

Initial inputs: Q `[1,Hq,1,D]`, K/V `[1,Hkv,N,D]`, `Hq % Hkv == 0`.
All KV tokens are visible. Start in FP32, then FP16 with FP32 statistics.
Use an independent full reference, local `m/l/u`, and stable merging. Do not
tile KV for GQA. Start communication with MLX ring/TCP and all-gather.

Tolerance defaults: FP32 `rtol=1e-4, atol=1e-5`; FP16
`rtol=1e-2, atol=1e-3`, against the same quantized inputs. Investigate failures
rather than automatically relaxing tolerances.

Benchmark defaults: batch/query count 1, 32 query heads, 8 KV heads, dimension
128; lengths 4096, 16384, 65536, 131072, 262144; 10 warmups, 50 fresh evaluated
operations. Inputs are prepared outside timing; complete output is available
on every participant. Total distributed latency is the maximum per-rank
operation time. Component timings are separate diagnostic measurements.

## Required prototype

| Step | Deliverable and acceptance | Status / verified commit |
|---|---|---|
| 1 | Input contract and independent FP32 full attention; hand examples and invalid shapes | Verified `81d582f`; local checks and CI passed |
| 2 | Local m/l/u and stable merge; two equal shards match reference | Verified `57eb2f4`; local checks and CI passed |
| 3 | GQA/MQA, unequal and empty shards, concentrated attention; finite correct results | Verified `b3ce354`; local checks and CI passed |
| 4 | FP16 inputs with FP32 computation/statistics; quantized-input reference parity | Verified `97045a6`; local checks and CI passed |
| 5 | Bounded-block partial attention; parity with direct path and bounded temporaries | Verified `88c9cb8`; local checks and CI passed |
| 6 | Deterministic global Q/K/V fixture generation; identical data across partitions | Verified `a17a514`; local checks and CI passed |
| 7 | Optimized MLX full-attention control; FP32/FP16 correctness without KV tiling | Verified `45d86a5`; local checks and CI passed |
| 8 | Warmup/timing runner; 50 newly computed GPU-complete outputs after 10 warmups | Verified `0351ef7`; local checks and CI passed |
| 9 | CLI, raw JSONL, run metadata and CSV median/p95 summary; reproducible commands | Verified `944497b`; local checks and CI passed |
| 10 | Physical local-Mac sweep at the prescribed lengths; timings, error and memory | Verified `f9fa839`; local checks and CI passed |
| 11 | Ring collective probe on two local CPU processes; strict world size and timeout | Verified `a541c0b`; local checks and CI passed |
| 12 | Distributed context attention; all-gather m/l/u, full output on both ranks | Verified `097ccb9`; local checks and CI passed |
| 13 | Optimized head-split control; full gathered output and correct GQA head ownership | Verified `be64c13`; local checks and CI passed |
| 14 | Distributed benchmark runner; full operation latency and per-rank raw data | Verified `73b5735`; local checks and CI passed |
| 15 | Prepare two physical Macs: inventory, connection, access, identical checkout/lock | Waiting for second Mac preparation/access |
| 16 | Physical transport/correctness probe and measured communication latency | Hardware required |
| 17 | Three-mode physical campaign; single mode on each Mac and best single control | Hardware required |
| 18 | Evidence report: latency, memory, numerical error, bottleneck and next decision | Requires steps 15–17 |

The first completed research result is step 18, including negative findings.
Synthetic one-layer performance does not establish model throughput or quality.

## Conditional optimization

| Step | Deliverable and acceptance | Status |
|---|---|---|
| O1 | One optimization addressing a measured bottleneck; correctness and component evidence | Conditional |
| O2 | Repeat affected physical comparisons and record before/after evidence | Conditional |

Consider custom Metal only after a limitation of ordinary MLX operations is
measured. Investigate RDMA separately after hardware support is established.

## One real model

Start after a useful microexperiment or a concrete, actionable limitation.

| Step | Deliverable and acceptance | Status |
|---|---|---|
| M1 | Freeze one dense Qwen3 model, weights, configuration and compatible MLX-LM; single baseline | Conditional |
| M2 | Distributed append/reset KV cache, global positions, owner = position % world_size | Conditional |
| M3 | One-layer attention adapter preserving normalization, RoPE and output projection | Conditional |
| M4 | Full-model distributed decode, synchronized next token and stopping; logits parity | Conditional |
| M5 | Correct token-by-token prefill into the distributed cache, no full KV per rank | Conditional |
| M6 | Whole-model TTFT, prefill, decode throughput and total memory measurements | Conditional |
| M7 | Native MLX-LM tensor-parallel control, same model/input and verified outputs | Conditional |
| M8 | Single/TP/WeftLM physical campaign with raw measurements | Conditional |
| M9 | Model evidence report and integration decision | Conditional |

## Exo integration and later research

| Step | Deliverable and acceptance | Status |
|---|---|---|
| E1 | Freeze Exo revision, minimal integration point, contract and compute-group layout | Conditional |
| E2 | Explicitly enabled adapter/minimal patch with documented unsupported behavior | Conditional |
| E3 | Two-Mac Exo requests and applicable native PP/TP comparisons | Conditional |
| E4 | Reproducible technical-review material; sending it requires user instruction | Conditional |
| R1 | Multi-query/global-position causal masks, including fully masked local shards | Future decision |
| R2 | One distributed blocked-prefill scheme and correct final cache | Future decision |
| R3 | Independent physical prefill/TTFT/memory campaign | Future decision |
| R4 | One evidence-driven CP+TP or CP+PP combination on a sufficient physical cluster | Future decision |

A server, management UI, discovery, universal model support and automatic
scheduling are outside this roadmap. The second Mac will be prepared later.
