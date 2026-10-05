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
