# Sampling Algorithms

A comprehensive collection of sampling algorithms implemented in both **Rust** and **Python** for educational purposes and practical applications.

## Overview

This repository provides implementations of various sampling techniques commonly used in statistics, data analysis, and machine learning. The focus is on understanding the underlying mechanics of each algorithm through from-scratch implementations.

## Project Structure

```
sampling-algo/
├── src/                    # Rust implementation (main focus)
│   ├── lib.rs             # Library root
│   └── random_sampling/   # Random sampling algorithms
├── random-sampling/       # Python implementation (reference)
├── time-based-sampling/   # Python time-based sampling
├── examples/              # Usage examples
├── benches/               # Performance benchmarks
└── tests/                 # Integration tests
```

## Rust Implementation

The primary implementation is in Rust, designed for:
- **Educational clarity**: Each algorithm implemented from scratch
- **Performance**: Zero-copy operations where possible
- **Memory efficiency**: Careful ownership and borrowing
- **Type safety**: Generic implementations with proper error handling

### Implemented Algorithms

- **Simple Random Sampling**: Fisher-Yates shuffle based sampling
- **Stratified Sampling**: Proportional and equal allocation methods
- **Systematic Sampling**: Fixed-interval sampling
- **Reservoir Sampling**: Algorithm R for streaming data
- **Weighted Sampling**: A-Res algorithm for probability-proportional sampling
- **Time Series Sampling**: Temporal-aware sampling methods

### Quick Start

```bash
# Build the project
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Run examples
cargo run --example demo
```

### Usage Example

```rust
use sampling_algo::random_sampling::simple::simple_random_sample;

let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let sample = simple_random_sample(&data, 3, Some(42))?;
println!("Random sample: {:?}", sample);
```

## Python Implementation

The Python implementations serve as reference and comparison:
- Located in `random-sampling/` and `time-based-sampling/`
- Focus on algorithmic correctness
- Minimal use of external libraries

## Learning Objectives

This project is designed to help understand:
- **Algorithm mechanics**: How sampling algorithms work internally
- **Performance trade-offs**: Memory vs time complexity
- **Implementation challenges**: Edge cases and error handling
- **Language differences**: Rust vs Python approaches

## Performance Considerations

Each algorithm includes:
- Time complexity analysis
- Space complexity analysis
- Benchmark comparisons
- Memory usage patterns

## Contributing

This is an educational project. Contributions should focus on:
- Correctness of implementations
- Clear documentation
- Educational value
- Performance improvements

## License

MIT OR Apache-2.0