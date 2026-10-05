# Distributed prototype

The prototype uses MLX 0.32.3 ring/TCP collectives. It requires exactly two
participants; a singleton is an error. A fixed-size protocol fingerprint rejects
different arguments before shape-dependent collectives.

Run this local CPU communication diagnostic from the repository root:

```sh
uv run --locked mlx.launch --backend ring -n 2 --python .venv/bin/python -- \
  -m weftlm probe --warmup 3 --repeats 10
```

The default payload is 16,640 bytes: the FP32 m/l/u packet for 32 heads of
dimension 128. Each repeat constructs new rank/iteration values, evaluates the
collective, and verifies the final gathered values. Each rank prints raw
nanosecond samples and median/p95. Logical payload sizes are calculated;
network traffic is not measured. These two local processes test correctness
and integration, and do not represent two physical Macs.

The integration tests impose a 30-second launcher timeout and kill the process
group on expiry. Native collectives can block if a participant disappears;
manual runs should be supervised and interrupted if the peer fails. MLX's
launcher terminates peers on detected process failure.

Transport and launcher reference: [MLX distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html).

Check context partitioning, GQA and full output recovery on both participants:

```sh
uv run --locked mlx.launch --backend ring -n 2 --python .venv/bin/python -- \
  -m weftlm check-distributed --device cpu --length 17 --dtype float32
```

Rank r owns tokens `[N*r/2, N*(r+1)/2)` with integer division. Only its KV
slice is generated. Each rank computes bounded-block FP32 statistics, packs
m/l/u into one FP32 array, gathers the packets on the CPU communication stream
and merges them on the selected compute device. An empty local slice is legal.
This check builds full inputs outside the distributed operation for independent
reference validation; lengths above 4096 are rejected to keep it a small diagnostic.

For the optimized head-split control, add `--mode heads` to the check command.
Hkv must be divisible by two. Each participant owns half the KV heads and their
whole consecutive GQA query groups, over the full context. It calls native
`mx.fast.scaled_dot_product_attention`, then gathers output heads in rank order.
The full `[1,Hq,1,D]` result is available on both participants. This control
splits one attention operation; it is not full-model tensor parallelism.

## Benchmark protocol and artifacts

```sh
uv run --locked python -m weftlm bench --mode single --output results/reference-run
uv run --locked mlx.launch --backend ring -n 2 --python .venv/bin/python -- \
  -m weftlm bench --mode context --reference results/reference-run \
  --topology local-processes --link-description "loopback TCP" \
  --output results/context-debug
uv run --locked python -m weftlm summarize --input results/context-debug
```

For a small integration run use `--lengths 17 --warmup 1 --repeats 3` on all
commands. The defaults remain 10 warmups and 50 fresh completed operations at
4k/16k/64k/128k/256k, on GPU. Substitute `heads` for the optimized head control.
Different configs, mode, repeat counts, dependency version, package source or
reference fingerprint are rejected before the benchmark collectives.

Each participant stores `rank-r/MODE-N/` with:

- `metadata.json`: run ID, hardware, code/dependency identities, ownership,
  numerical error, total-operation MLX peak, lifetime process RSS, swap,
  declared topology and link description.
- `samples.jsonl`: raw duration of each participant, this rank's duration and
  the per-iteration maximum. Both ranks preserve the same global samples.
- `summary.csv`: median, nearest-rank p95, minimum and numerical pass status,
  calculated from the global samples. `summarize` validates the reduction.
- `diagnostics.json`: separate raw series for local compute, communication,
  merge/assembly and barrier; their summaries use the per-iteration maximum.

The timer includes local compute, CPU collective, merging and complete output
evaluation. An evaluated native all-sum barrier precedes every repeat outside
the total timer. Barrier cost is measured separately. Diagnostics run after
the main series and are explanatory; adding their medians does not reconstruct
total latency. All arrays supplied to timed operations are already evaluated.
The full output is available on both participants after every main iteration.

Context packets contain `Hq*(D+2)*4` bytes per rank; head packets contain
`(Hq/2)*D*input_itemsize` bytes. These are logical array sizes, excluding
transport overhead and protocol/barrier messages. No wire traffic is measured.
`--topology` and `--link-description` are declarations, not automatic hardware
or network discovery. An unspecified link cannot support a transport conclusion.

Verified standalone runs now save `reference.json`, produced by the independent
full reference outside the timer. A distributed length above 4096 requires this
artifact so no full KV is generated on its participants during validation.
Copy the identical artifact and metadata to both physical nodes. Geometry,
dtype, seed and fixture version must match; kernel block size may differ.
The earlier preliminary local campaign predates these reference sidecars and
must be rerun to produce them. If a standalone run cannot fit, a separate bounded
memory oracle is needed for capacity-only comparisons before proceeding.

CLI operation failures preserve `failure.json` with the actual reason and no
invented timings. Summarization includes those failures. Allocation failure,
system swap and a missing/killed participant must be recorded in the campaign;
partial output does not establish a completed distributed result. Native launcher
warnings and both rank artifacts must be inspected: its exit status alone is
insufficient evidence that both workers succeeded. A benchmark stops after a
failed configuration; reruns use a fresh output root.

## Physical-node preparation: pending

The second physical Mac has not been prepared. Steps 15–18 require its hardware
and access; local process tests do not satisfy them. Before measuring, record
both chip/model, RAM, OS, power state, connection/cable/interface and IPs. Keep
the comparison revision, Python, `uv.lock`, dimensions, dtype and seed identical.
Measure each Mac standalone with the same inputs and use the better single
median for each length, then compare both distributed modes.

Use the native MLX hostfile schema with actual SSH names and reachable link IPs:

```json
[
  {"ssh": "MAC_A_SSH", "ips": ["MAC_A_LINK_IP"]},
  {"ssh": "MAC_B_SSH", "ips": ["MAC_B_LINK_IP"]}
]
```

Keep that hostfile local. With the same absolute checkout path on both Macs,
set `WEFTLM_DIR` to that path and launch the small probe first:

```sh
uv run --locked mlx.launch --backend ring --hostfile hosts.json \
  --cwd "$WEFTLM_DIR" --python "$WEFTLM_DIR/.venv/bin/python" -- \
  -m weftlm probe
```

Then run both small `check-distributed` modes through that hostfile. Use
`--topology two-macs` and a precise `--link-description` for physical benchmarks.
Collect both `rank-r` directories without overwriting them. Treat runs with
memory pressure as exploratory; repeat controlled measurements before claims of
speed or capacity. OS/network/SSH changes are part of preparing the hardware,
and have not been made by this initialization.
