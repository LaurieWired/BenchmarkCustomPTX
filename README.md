# Custom PTX Instruction Benchmark

This repository contains a CUDA benchmark designed to demonstrate and measure the performance benefits of using specialized PTX cache control instructions on NVIDIA Hopper GPUs. Specifically, it benchmarks the `ld.global.nc.L1::no_allocate.L2::256B` instruction that was used by DeepSeek in their [DeepEP library](https://github.com/deepseek-ai/DeepEP).

## What This Benchmark Demonstrates

The benchmark simulates a common scenario in high-performance GPU computing:

1. First, we load a small dataset that should ideally stay in L1 cache
2. Next, we stream through a large dataset that could potentially evict the small data from L1 cache
3. Finally, we repeatedly access the small dataset again

The benchmark compares two approaches:
- **Standard Load**: Uses regular CUDA memory access operations
- **Specialized Load**: Uses the custom PTX instruction for streaming data

When the specialized instruction is used correctly, it prevents the streaming data from polluting the L1 cache, keeping the frequently accessed data in L1 cache and resulting in significantly better performance.

## Requirements

- CUDA 12.3 and above
- NVIDIA Hopper GPU (Ex. H100)

## Building and Running
```
nvcc -arch=sm_90 -o benchmark benchmark.cu
./benchmark
```
