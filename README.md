# WeftLM

Experimental context-parallel inference for long-context LLMs on Apple Silicon.

## Status

Early research prototype. We are currently benchmarking whether sequence/context parallelism can provide practical performance benefits for distributed LLM inference across multiple Macs.

## Goals

- Context-parallel attention across Apple Silicon devices
- Sharded KV-cache experiments
- MLX/JACCL distributed benchmarks
- Comparison with existing tensor/pipeline-parallel approaches
- Potential integration with EXO
