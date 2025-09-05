# Time-Based Sampling

Time-based sampling algorithms are specialized techniques for sampling time series data where temporal relationships, patterns, and constraints are crucial. Unlike general random sampling, these methods preserve temporal structure and account for time-dependent characteristics.

## Overview

Time-based sampling addresses unique challenges in temporal data:
- **Temporal dependencies**: Adjacent time points are often correlated
- **Seasonal patterns**: Data may have periodic behaviors
- **Trend preservation**: Long-term trends should be maintained in samples
- **Event-driven sampling**: Important events may require special handling
- **Real-time constraints**: Streaming data with latency requirements

## Implemented Algorithms

### 1. Uniform Time Sampling
Samples at regular time intervals, preserving temporal spacing.

**Time Complexity**: O(n)
**Space Complexity**: O(k)
**Use Case**: Downsampling high-frequency data while maintaining temporal resolution

### 2. Adaptive Sampling
Dynamically adjusts sampling rate based on data characteristics or change detection.

**Time Complexity**: O(n)
**Space Complexity**: O(k)
**Use Case**: IoT sensors, financial data with varying volatility

### 3. Event-Based Sampling
Samples around significant events or anomalies in the time series.

**Time Complexity**: O(n log k)
**Space Complexity**: O(k)
**Use Case**: Anomaly detection, rare event analysis

### 4. Window-Based Sampling
Samples representative points from fixed or sliding time windows.

**Time Complexity**: O(n)
**Space Complexity**: O(w + k) where w is window size
**Use Case**: Aggregating sensor data, reducing storage requirements

### 5. Peak/Valley Sampling
Identifies and samples local extrema to capture important signal characteristics.

**Time Complexity**: O(n)
**Space Complexity**: O(k)
**Use Case**: Signal processing, trend analysis

### 6. Frequency-Based Sampling
Samples based on frequency domain characteristics or spectral analysis.

**Time Complexity**: O(n log n)
**Space Complexity**: O(k)
**Use Case**: Audio processing, vibration analysis

### 7. Multi-Resolution Sampling
Creates samples at multiple time scales simultaneously.

**Time Complexity**: O(n log n)
**Space Complexity**: O(k * levels)
**Use Case**: Hierarchical time series analysis, wavelet transforms

## Mathematical Foundation

### Uniform Time Sampling
For time series with timestamps T = {t₁, t₂, ..., tₙ}, sample every Δt:
```
sample_times = {t₁ + i*Δt | i ∈ ℕ, t₁ + i*Δt ≤ tₙ}
```

### Adaptive Sampling (Change Detection)
Sample when change metric exceeds threshold θ:
```
C(t) = |x(t) - x(t-1)| / σ
sample if C(t) > θ
```

### Event-Based Sampling (Z-Score)
Sample when anomaly score exceeds threshold:
```
Z(t) = |x(t) - μ| / σ
sample if Z(t) > k (typically k = 2 or 3)
```

### Window-Based Sampling
For window size w, sampling strategies:
- **Mean**: x̄ᵢ = (1/w) Σ x(t)
- **Max/Min**: extreme values in window
- **Median**: robust central tendency

## Temporal Considerations

### Nyquist-Shannon Criterion
For signal with maximum frequency fₘₐₓ, minimum sampling rate:
```
fₛ ≥ 2 * fₘₐₓ
```

### Aliasing Prevention
When downsampling, apply anti-aliasing filter before sampling to prevent frequency folding.

### Temporal Autocorrelation
Consider lag-k autocorrelation when determining sampling intervals:
```
ρ(k) = Cov(X(t), X(t+k)) / Var(X(t))
```

## Usage Examples

```python
from time_based_sampling import (
    uniform_time_sample,
    adaptive_sample,
    event_based_sample,
    window_sample,
    peak_valley_sample,
    frequency_sample
)
import pandas as pd
from datetime import datetime, timedelta

# Create sample time series
timestamps = pd.date_range('2024-01-01', periods=10000, freq='1min')
values = np.sin(np.arange(10000) * 0.1) + np.random.normal(0, 0.1, 10000)

# Uniform time sampling - every 10 minutes
uniform_sample = uniform_time_sample(timestamps, values, interval=timedelta(minutes=10))

# Adaptive sampling based on value changes
adaptive_sample = adaptive_sample(timestamps, values, threshold=0.5)

# Event-based sampling around anomalies
event_sample = event_based_sample(timestamps, values, z_threshold=2.0)

# Window-based sampling with hourly windows
window_sample = window_sample(timestamps, values, window_size=timedelta(hours=1))

# Peak/valley sampling
extrema_sample = peak_valley_sample(timestamps, values, min_distance=50)
```

## Performance Characteristics

| Algorithm | Time Complexity | Space Complexity | Temporal Preservation | Real-time |
|-----------|-----------------|------------------|---------------------|-----------|
| Uniform Time | O(n) | O(k) | Excellent | Yes |
| Adaptive | O(n) | O(k) | Good | Yes |
| Event-Based | O(n log k) | O(k) | Moderate | Yes |
| Window-Based | O(n) | O(w + k) | Good | Yes |
| Peak/Valley | O(n) | O(k) | Excellent | Yes |
| Frequency | O(n log n) | O(k) | Moderate | No |
| Multi-Resolution | O(n log n) | O(k * levels) | Excellent | No |

## Applications

### IoT and Sensor Networks
- **Smart meters**: Sample power consumption during peak usage
- **Environmental monitoring**: Adaptive sampling based on weather changes
- **Industrial sensors**: Event-based sampling for equipment monitoring

### Financial Time Series
- **High-frequency trading**: Uniform sampling for consistent analysis
- **Risk management**: Event sampling around market volatility
- **Algorithmic trading**: Multi-resolution for different time horizons

### Signal Processing
- **Audio compression**: Frequency-based sampling for perceptual coding
- **Biomedical signals**: Peak detection for heartbeat analysis
- **Communication systems**: Nyquist-rate sampling for digital transmission

### Scientific Computing
- **Climate modeling**: Multi-resolution sampling for different time scales
- **Astronomical data**: Event sampling for transient phenomena
- **Experimental physics**: Adaptive sampling for real-time experiments

## Quality Metrics

### Temporal Fidelity
Measure how well the sample preserves temporal characteristics:
```python
def temporal_fidelity(original_ts, sampled_ts):
    """Compare autocorrelation structures"""
    orig_acf = autocorrelation(original_ts)
    samp_acf = autocorrelation(sampled_ts)
    return correlation(orig_acf, samp_acf)
```

### Information Preservation
Quantify information retention in frequency domain:
```python
def information_preservation(original, sampled):
    """Compare power spectral densities"""
    orig_psd = power_spectral_density(original)
    samp_psd = power_spectral_density(sampled)
    return 1 - mean_squared_error(orig_psd, samp_psd)
```

### Compression Ratio
Efficiency metric for storage reduction:
```python
compression_ratio = len(original_data) / len(sampled_data)
```

## Best Practices

1. **Domain Knowledge**: Understand the physical process generating the data
2. **Frequency Analysis**: Analyze dominant frequencies before sampling
3. **Stationarity Testing**: Check if statistical properties change over time
4. **Validation**: Compare samples against ground truth when available
5. **Real-time Constraints**: Consider computational and memory limitations
6. **Quality Monitoring**: Continuously assess sampling effectiveness

## Integration with Time Series Analysis

Time-based sampling integrates with downstream analysis:
- **Forecasting**: Maintain seasonal patterns for ARIMA/LSTM models
- **Anomaly Detection**: Preserve rare events for training
- **Clustering**: Sample representatives from temporal clusters  
- **Feature Engineering**: Extract temporal features from samples
- **Visualization**: Create meaningful plots at appropriate resolutions