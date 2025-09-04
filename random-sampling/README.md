# Random Sampling

Random sampling is a fundamental technique in statistics and data analysis that allows us to select a subset of data points from a larger population. This module implements various random sampling algorithms particularly suited for time series data analysis.

## Overview

Random sampling serves several purposes:
- **Computational efficiency**: Working with smaller datasets when full datasets are too large
- **Statistical inference**: Making predictions about populations based on samples
- **Data exploration**: Getting quick insights from representative subsets
- **Memory constraints**: Processing data that doesn't fit entirely in memory

## Implemented Algorithms

### 1. Simple Random Sampling
Selects elements with equal probability without replacement.

**Time Complexity**: O(k) where k is sample size
**Space Complexity**: O(k)
**Use Case**: General-purpose sampling when all data points are equally important

### 2. Stratified Sampling
Divides population into strata and samples from each stratum proportionally.

**Time Complexity**: O(n + k) where n is population size
**Space Complexity**: O(k)
**Use Case**: When population has distinct subgroups that should be represented

### 3. Systematic Sampling
Selects every nth element after a random start.

**Time Complexity**: O(k)
**Space Complexity**: O(k)
**Use Case**: Time series data with periodic patterns

### 4. Reservoir Sampling
Maintains a fixed-size sample from a stream of unknown length.

**Time Complexity**: O(n) single pass
**Space Complexity**: O(k)
**Use Case**: Streaming data, large datasets that don't fit in memory

### 5. Weighted Random Sampling
Selects elements with probability proportional to their weights.

**Time Complexity**: O(n log k) using heap
**Space Complexity**: O(k)
**Use Case**: When data points have different importance levels

## Mathematical Foundation

### Simple Random Sampling
Each subset of size k has equal probability of selection:
```
P(sample) = k! * (n-k)! / n!
```

### Reservoir Sampling (Vitter's Algorithm)
For element i (i > k), replacement probability:
```
P(replace) = k/i
```

### Weighted Sampling
Selection probability for element i:
```
P(i) = w_i / Σw_j
```

## Usage Examples

```python
from random_sampling import (
    simple_random_sample,
    stratified_sample,
    systematic_sample,
    reservoir_sample,
    weighted_sample
)

# Simple random sampling
data = list(range(1000))
sample = simple_random_sample(data, sample_size=100)

# Stratified sampling
labels = ['A'] * 300 + ['B'] * 400 + ['C'] * 300
sample = stratified_sample(data, labels, sample_size=100)

# Reservoir sampling for streaming data
reservoir = reservoir_sample(data_stream, reservoir_size=50)
```

## Performance Characteristics

| Algorithm | Best Case | Average Case | Worst Case | Memory |
|-----------|-----------|--------------|------------|---------|
| Simple Random | O(k) | O(k) | O(k) | O(k) |
| Stratified | O(n+k) | O(n+k) | O(n+k) | O(k) |
| Systematic | O(k) | O(k) | O(k) | O(k) |
| Reservoir | O(n) | O(n) | O(n) | O(k) |
| Weighted | O(n log k) | O(n log k) | O(n log k) | O(k) |

## Applications in Time Series

- **Downsampling**: Reducing temporal resolution while preserving key patterns
- **Cross-validation**: Creating representative train/test splits
- **Anomaly detection**: Sampling normal periods for baseline establishment
- **Feature extraction**: Sampling windows for pattern recognition
- **Data compression**: Maintaining essential characteristics with fewer points

