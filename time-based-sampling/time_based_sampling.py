"""
Time-Based Sampling Algorithms

This module implements various time-based sampling techniques specifically designed
for time series data where temporal relationships and patterns are crucial.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Union, Optional, Callable, Dict, Any
from dataclasses import dataclass
from scipy import signal
from scipy.stats import zscore
import warnings


@dataclass
class TimeSeriesPoint:
    """Represents a single point in a time series."""
    timestamp: Union[datetime, pd.Timestamp, float]
    value: float
    metadata: Optional[Dict[str, Any]] = None


def uniform_time_sample(
    timestamps: Union[List, pd.DatetimeIndex, np.ndarray],
    values: Union[List, np.ndarray],
    interval: Union[timedelta, float],
    start_time: Optional[Union[datetime, float]] = None,
    end_time: Optional[Union[datetime, float]] = None
) -> Tuple[List, List]:
    """
    Sample time series data at uniform time intervals.
    
    Args:
        timestamps: Time points for each observation
        values: Data values corresponding to timestamps
        interval: Time interval between samples
        start_time: Optional start time for sampling
        end_time: Optional end time for sampling
        
    Returns:
        Tuple of (sampled_timestamps, sampled_values)
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length")
    
    if isinstance(timestamps, pd.DatetimeIndex):
        timestamps = timestamps.to_list()
    
    # Convert to arrays for easier manipulation
    ts_array = np.array(timestamps)
    val_array = np.array(values)
    
    # Determine time bounds
    if start_time is None:
        start_time = ts_array[0]
    if end_time is None:
        end_time = ts_array[-1]
    
    sampled_timestamps = []
    sampled_values = []
    
    if isinstance(interval, timedelta):
        # Handle datetime timestamps
        current_time = start_time
        while current_time <= end_time:
            # Find closest timestamp
            if isinstance(ts_array[0], (datetime, pd.Timestamp)):
                time_diffs = np.abs([(t - current_time).total_seconds() for t in ts_array])
            else:
                time_diffs = np.abs(ts_array - current_time)
            
            closest_idx = np.argmin(time_diffs)
            sampled_timestamps.append(ts_array[closest_idx])
            sampled_values.append(val_array[closest_idx])
            
            current_time += interval
    else:
        # Handle numeric timestamps
        current_time = float(start_time)
        end_time_float = float(end_time)
        
        while current_time <= end_time_float:
            # Find closest timestamp
            time_diffs = np.abs(ts_array - current_time)
            closest_idx = np.argmin(time_diffs)
            
            sampled_timestamps.append(ts_array[closest_idx])
            sampled_values.append(val_array[closest_idx])
            
            current_time += interval
    
    return sampled_timestamps, sampled_values


def adaptive_sample(
    timestamps: Union[List, np.ndarray],
    values: Union[List, np.ndarray],
    threshold: float = 0.1,
    min_interval: Optional[Union[timedelta, float]] = None,
    max_interval: Optional[Union[timedelta, float]] = None,
    change_metric: str = "absolute"
) -> Tuple[List, List]:
    """
    Adaptively sample based on rate of change in the data.
    
    Args:
        timestamps: Time points for each observation
        values: Data values corresponding to timestamps
        threshold: Change threshold for triggering sampling
        min_interval: Minimum time between samples
        max_interval: Maximum time between samples
        change_metric: Type of change metric ("absolute", "relative", "z_score")
        
    Returns:
        Tuple of (sampled_timestamps, sampled_values)
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length")
    
    ts_array = np.array(timestamps)
    val_array = np.array(values)
    
    sampled_timestamps = [ts_array[0]]
    sampled_values = [val_array[0]]
    last_sampled_idx = 0
    
    for i in range(1, len(val_array)):
        current_time = ts_array[i]
        current_value = val_array[i]
        last_value = val_array[last_sampled_idx]
        
        # Calculate change metric
        if change_metric == "absolute":
            change = abs(current_value - last_value)
        elif change_metric == "relative":
            if last_value != 0:
                change = abs((current_value - last_value) / last_value)
            else:
                change = abs(current_value)
        elif change_metric == "z_score":
            # Use rolling window for z-score calculation
            window_size = min(50, i)
            window_values = val_array[max(0, i-window_size):i+1]
            if len(window_values) > 1:
                z_scores = np.abs(zscore(window_values))
                change = z_scores[-1]
            else:
                change = 0
        else:
            raise ValueError(f"Unknown change metric: {change_metric}")
        
        # Check time constraints
        time_since_last = current_time - sampled_timestamps[-1]
        
        min_time_met = True
        max_time_exceeded = False
        
        if min_interval is not None:
            if isinstance(time_since_last, timedelta):
                min_time_met = time_since_last >= min_interval
            else:
                min_time_met = time_since_last >= min_interval
        
        if max_interval is not None:
            if isinstance(time_since_last, timedelta):
                max_time_exceeded = time_since_last >= max_interval
            else:
                max_time_exceeded = time_since_last >= max_interval
        
        # Sample if change exceeds threshold or max interval exceeded
        if (change > threshold and min_time_met) or max_time_exceeded:
            sampled_timestamps.append(current_time)
            sampled_values.append(current_value)
            last_sampled_idx = i
    
    return sampled_timestamps, sampled_values


def event_based_sample(
    timestamps: Union[List, np.ndarray],
    values: Union[List, np.ndarray],
    z_threshold: float = 2.0,
    window_size: int = 100,
    context_points: int = 5
) -> Tuple[List, List]:
    """
    Sample around significant events or anomalies in the time series.
    
    Args:
        timestamps: Time points for each observation
        values: Data values corresponding to timestamps
        z_threshold: Z-score threshold for event detection
        window_size: Window size for calculating statistics
        context_points: Number of points to include around each event
        
    Returns:
        Tuple of (sampled_timestamps, sampled_values)
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length")
    
    ts_array = np.array(timestamps)
    val_array = np.array(values)
    
    # Calculate rolling z-scores
    z_scores = np.zeros_like(val_array)
    for i in range(len(val_array)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(val_array), i + window_size // 2)
        window_data = val_array[start_idx:end_idx]
        
        if len(window_data) > 1:
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            if std_val > 0:
                z_scores[i] = abs((val_array[i] - mean_val) / std_val)
    
    # Find event indices
    event_indices = np.where(z_scores > z_threshold)[0]
    
    # Collect samples with context
    sample_indices = set()
    for event_idx in event_indices:
        start = max(0, event_idx - context_points)
        end = min(len(val_array), event_idx + context_points + 1)
        sample_indices.update(range(start, end))
    
    # Sort and extract samples
    sorted_indices = sorted(sample_indices)
    sampled_timestamps = [ts_array[i] for i in sorted_indices]
    sampled_values = [val_array[i] for i in sorted_indices]
    
    return sampled_timestamps, sampled_values


def window_sample(
    timestamps: Union[List, np.ndarray],
    values: Union[List, np.ndarray],
    window_size: Union[timedelta, float, int],
    aggregation: str = "mean",
    overlap: float = 0.0
) -> Tuple[List, List]:
    """
    Sample representative points from time windows.
    
    Args:
        timestamps: Time points for each observation
        values: Data values corresponding to timestamps
        window_size: Size of each window
        aggregation: Aggregation method ("mean", "median", "max", "min", "first", "last")
        overlap: Overlap fraction between windows (0 to 1)
        
    Returns:
        Tuple of (sampled_timestamps, sampled_values)
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length")
    
    ts_array = np.array(timestamps)
    val_array = np.array(values)
    
    sampled_timestamps = []
    sampled_values = []
    
    if isinstance(window_size, int):
        # Fixed number of points per window
        step_size = int(window_size * (1 - overlap))
        for i in range(0, len(val_array), step_size):
            end_idx = min(i + window_size, len(val_array))
            window_times = ts_array[i:end_idx]
            window_values = val_array[i:end_idx]
            
            if len(window_values) > 0:
                sampled_timestamps.append(_aggregate_time(window_times, aggregation))
                sampled_values.append(_aggregate_values(window_values, aggregation))
    else:
        # Time-based windows
        start_time = ts_array[0]
        step_size = window_size * (1 - overlap)
        
        if isinstance(window_size, timedelta):
            current_time = start_time
            while current_time < ts_array[-1]:
                end_time = current_time + window_size
                
                # Find indices in window
                if isinstance(ts_array[0], (datetime, pd.Timestamp)):
                    mask = [(t >= current_time and t < end_time) for t in ts_array]
                else:
                    mask = (ts_array >= current_time) & (ts_array < end_time)
                
                window_indices = np.where(mask)[0]
                
                if len(window_indices) > 0:
                    window_times = ts_array[window_indices]
                    window_values = val_array[window_indices]
                    
                    sampled_timestamps.append(_aggregate_time(window_times, aggregation))
                    sampled_values.append(_aggregate_values(window_values, aggregation))
                
                current_time += step_size
        else:
            # Numeric time windows
            current_time = float(start_time)
            end_time_total = float(ts_array[-1])
            
            while current_time < end_time_total:
                end_time = current_time + float(window_size)
                
                mask = (ts_array >= current_time) & (ts_array < end_time)
                window_indices = np.where(mask)[0]
                
                if len(window_indices) > 0:
                    window_times = ts_array[window_indices]
                    window_values = val_array[window_indices]
                    
                    sampled_timestamps.append(_aggregate_time(window_times, aggregation))
                    sampled_values.append(_aggregate_values(window_values, aggregation))
                
                current_time += float(step_size)
    
    return sampled_timestamps, sampled_values


def peak_valley_sample(
    timestamps: Union[List, np.ndarray],
    values: Union[List, np.ndarray],
    min_distance: int = 10,
    prominence: Optional[float] = None,
    include_endpoints: bool = True
) -> Tuple[List, List]:
    """
    Sample local peaks and valleys in the time series.
    
    Args:
        timestamps: Time points for each observation
        values: Data values corresponding to timestamps
        min_distance: Minimum distance between peaks/valleys
        prominence: Required prominence for peak/valley detection
        include_endpoints: Whether to include start/end points
        
    Returns:
        Tuple of (sampled_timestamps, sampled_values)
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length")
    
    val_array = np.array(values)
    
    # Find peaks
    peak_kwargs = {"distance": min_distance}
    if prominence is not None:
        peak_kwargs["prominence"] = prominence
    
    peaks, _ = signal.find_peaks(val_array, **peak_kwargs)
    valleys, _ = signal.find_peaks(-val_array, **peak_kwargs)
    
    # Combine and sort indices
    extrema_indices = sorted(set(peaks) | set(valleys))
    
    # Add endpoints if requested
    if include_endpoints:
        if 0 not in extrema_indices:
            extrema_indices = [0] + extrema_indices
        if len(val_array) - 1 not in extrema_indices:
            extrema_indices.append(len(val_array) - 1)
        extrema_indices.sort()
    
    sampled_timestamps = [timestamps[i] for i in extrema_indices]
    sampled_values = [values[i] for i in extrema_indices]
    
    return sampled_timestamps, sampled_values


def frequency_sample(
    timestamps: Union[List, np.ndarray],
    values: Union[List, np.ndarray],
    target_frequencies: List[float],
    frequency_tolerance: float = 0.1,
    min_amplitude: Optional[float] = None
) -> Tuple[List, List]:
    """
    Sample based on frequency domain characteristics.
    
    Args:
        timestamps: Time points for each observation
        values: Data values corresponding to timestamps
        target_frequencies: Frequencies of interest for sampling
        frequency_tolerance: Tolerance around target frequencies
        min_amplitude: Minimum amplitude threshold in frequency domain
        
    Returns:
        Tuple of (sampled_timestamps, sampled_values)
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length")
    
    val_array = np.array(values)
    
    # Ensure uniform sampling for FFT
    ts_array = np.array(timestamps)
    if isinstance(ts_array[0], (datetime, pd.Timestamp)):
        # Convert to numeric for FFT
        ts_numeric = np.array([(t - ts_array[0]).total_seconds() for t in ts_array])
    else:
        ts_numeric = ts_array.astype(float)
    
    # Interpolate to uniform grid if necessary
    if not np.allclose(np.diff(ts_numeric), np.diff(ts_numeric)[0]):
        dt = np.median(np.diff(ts_numeric))
        uniform_times = np.arange(ts_numeric[0], ts_numeric[-1], dt)
        uniform_values = np.interp(uniform_times, ts_numeric, val_array)
    else:
        uniform_times = ts_numeric
        uniform_values = val_array
        dt = np.median(np.diff(ts_numeric))
    
    # Compute FFT
    fft_values = np.fft.fft(uniform_values)
    fft_freqs = np.fft.fftfreq(len(uniform_values), dt)
    
    # Find indices corresponding to target frequencies
    sample_indices = set()
    
    for target_freq in target_frequencies:
        # Find frequencies within tolerance
        freq_mask = np.abs(fft_freqs - target_freq) <= frequency_tolerance
        
        if min_amplitude is not None:
            # Filter by amplitude
            amplitude_mask = np.abs(fft_values) >= min_amplitude
            freq_mask = freq_mask & amplitude_mask
        
        target_indices = np.where(freq_mask)[0]
        
        # Map back to time domain
        for freq_idx in target_indices:
            # Find corresponding time indices
            phase = np.angle(fft_values[freq_idx])
            frequency = fft_freqs[freq_idx]
            
            # Sample at phase maxima
            if frequency != 0:
                period = 1.0 / abs(frequency)
                phase_time = -phase / (2 * np.pi * frequency)
                
                # Find sample times at this frequency
                sample_times = np.arange(
                    uniform_times[0] + phase_time,
                    uniform_times[-1],
                    period
                )
                
                for sample_time in sample_times:
                    # Find closest actual timestamp
                    time_diffs = np.abs(ts_numeric - sample_time)
                    closest_idx = np.argmin(time_diffs)
                    sample_indices.add(closest_idx)
    
    # Sort and extract samples
    sorted_indices = sorted(sample_indices)
    sampled_timestamps = [timestamps[i] for i in sorted_indices]
    sampled_values = [values[i] for i in sorted_indices]
    
    return sampled_timestamps, sampled_values


def multi_resolution_sample(
    timestamps: Union[List, np.ndarray],
    values: Union[List, np.ndarray],
    levels: int = 3,
    decimation_factor: int = 2,
    filter_type: str = "butter"
) -> Dict[int, Tuple[List, List]]:
    """
    Create multi-resolution samples using wavelet-like decomposition.
    
    Args:
        timestamps: Time points for each observation
        values: Data values corresponding to timestamps
        levels: Number of resolution levels
        decimation_factor: Factor for downsampling at each level
        filter_type: Anti-aliasing filter type ("butter", "bessel", "ellip")
        
    Returns:
        Dictionary mapping level to (timestamps, values) tuples
    """
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length")
    
    results = {}
    current_timestamps = list(timestamps)
    current_values = np.array(values, dtype=float)
    
    for level in range(levels):
        # Store current level
        results[level] = (current_timestamps.copy(), current_values.copy())
        
        if level < levels - 1:  # Don't process the last level
            # Apply anti-aliasing filter
            if len(current_values) > 2 * decimation_factor:
                try:
                    # Design low-pass filter
                    nyquist = 0.5
                    critical = 1.0 / decimation_factor / 2.0  # Normalized frequency
                    
                    if filter_type == "butter":
                        b, a = signal.butter(4, critical, btype='low')
                    elif filter_type == "bessel":
                        b, a = signal.bessel(4, critical, btype='low')
                    elif filter_type == "ellip":
                        b, a = signal.ellip(4, 1, 40, critical, btype='low')
                    else:
                        raise ValueError(f"Unknown filter type: {filter_type}")
                    
                    # Apply filter
                    filtered_values = signal.filtfilt(b, a, current_values)
                except Exception as e:
                    warnings.warn(f"Filtering failed at level {level}: {e}")
                    filtered_values = current_values
            else:
                filtered_values = current_values
            
            # Decimate
            decimated_indices = range(0, len(current_values), decimation_factor)
            current_timestamps = [current_timestamps[i] for i in decimated_indices]
            current_values = filtered_values[decimated_indices]
    
    return results


def _aggregate_time(timestamps, aggregation: str):
    """Helper function to aggregate timestamps."""
    if aggregation == "first":
        return timestamps[0]
    elif aggregation == "last":
        return timestamps[-1]
    elif aggregation == "mean":
        if isinstance(timestamps[0], (datetime, pd.Timestamp)):
            # Convert to numeric, take mean, convert back
            epoch = timestamps[0]
            offsets = [(t - epoch).total_seconds() for t in timestamps]
            mean_offset = np.mean(offsets)
            return epoch + timedelta(seconds=mean_offset)
        else:
            return np.mean(timestamps)
    elif aggregation == "median":
        if isinstance(timestamps[0], (datetime, pd.Timestamp)):
            epoch = timestamps[0]
            offsets = [(t - epoch).total_seconds() for t in timestamps]
            median_offset = np.median(offsets)
            return epoch + timedelta(seconds=median_offset)
        else:
            return np.median(timestamps)
    else:
        # For max/min, return the timestamp of the max/min value
        return timestamps[0]  # Default to first


def _aggregate_values(values, aggregation: str):
    """Helper function to aggregate values."""
    if aggregation == "mean":
        return np.mean(values)
    elif aggregation == "median":
        return np.median(values)
    elif aggregation == "max":
        return np.max(values)
    elif aggregation == "min":
        return np.min(values)
    elif aggregation == "first":
        return values[0]
    elif aggregation == "last":
        return values[-1]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def quality_metrics(
    original_timestamps: Union[List, np.ndarray],
    original_values: Union[List, np.ndarray],
    sampled_timestamps: Union[List, np.ndarray],
    sampled_values: Union[List, np.ndarray]
) -> Dict[str, float]:
    """
    Calculate quality metrics for time-based sampling.
    
    Args:
        original_timestamps: Original time points
        original_values: Original data values  
        sampled_timestamps: Sampled time points
        sampled_values: Sampled data values
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Compression ratio
    metrics['compression_ratio'] = len(original_values) / len(sampled_values)
    
    # Temporal coverage
    if isinstance(original_timestamps[0], (datetime, pd.Timestamp)):
        orig_span = (original_timestamps[-1] - original_timestamps[0]).total_seconds()
        samp_span = (sampled_timestamps[-1] - sampled_timestamps[0]).total_seconds()
    else:
        orig_span = original_timestamps[-1] - original_timestamps[0]
        samp_span = sampled_timestamps[-1] - sampled_timestamps[0]
    
    metrics['temporal_coverage'] = samp_span / orig_span if orig_span > 0 else 1.0
    
    # Value range preservation
    orig_range = np.max(original_values) - np.min(original_values)
    samp_range = np.max(sampled_values) - np.min(sampled_values)
    metrics['range_preservation'] = samp_range / orig_range if orig_range > 0 else 1.0
    
    # Mean preservation
    orig_mean = np.mean(original_values)
    samp_mean = np.mean(sampled_values)
    metrics['mean_error'] = abs(orig_mean - samp_mean) / abs(orig_mean) if orig_mean != 0 else 0.0
    
    return metrics


# Example usage and demonstrations
def demo_time_based_sampling():
    """Demonstrate time-based sampling algorithms."""
    
    # Generate sample time series data
    n_points = 1000
    timestamps = pd.date_range('2024-01-01', periods=n_points, freq='1min')
    
    # Create synthetic signal with trend, seasonality, and noise
    t = np.arange(n_points)
    trend = 0.01 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 100) + np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 0.5, n_points)
    anomalies = np.zeros(n_points)
    anomalies[[200, 500, 800]] = 10  # Add some anomalies
    
    values = trend + seasonal + noise + anomalies
    
    print("=== Time-Based Sampling Algorithm Demonstrations ===\n")
    
    # 1. Uniform time sampling
    print("1. Uniform Time Sampling (every 10 minutes):")
    uniform_ts, uniform_vals = uniform_time_sample(
        timestamps, values, interval=timedelta(minutes=10)
    )
    print(f"   Original points: {len(values)}")
    print(f"   Sampled points: {len(uniform_vals)}")
    print(f"   Compression ratio: {len(values) / len(uniform_vals):.1f}x")
    
    # 2. Adaptive sampling
    print("\n2. Adaptive Sampling (threshold=0.5):")
    adaptive_ts, adaptive_vals = adaptive_sample(
        timestamps, values, threshold=0.5, change_metric="absolute"
    )
    print(f"   Sampled points: {len(adaptive_vals)}")
    print(f"   Compression ratio: {len(values) / len(adaptive_vals):.1f}x")
    
    # 3. Event-based sampling
    print("\n3. Event-Based Sampling (z_threshold=2.0):")
    event_ts, event_vals = event_based_sample(
        timestamps, values, z_threshold=2.0, context_points=3
    )
    print(f"   Sampled points: {len(event_vals)}")
    print(f"   Compression ratio: {len(values) / len(event_vals):.1f}x")
    
    # 4. Window-based sampling
    print("\n4. Window-Based Sampling (1-hour windows, mean aggregation):")
    window_ts, window_vals = window_sample(
        timestamps, values, window_size=timedelta(hours=1), aggregation="mean"
    )
    print(f"   Sampled points: {len(window_vals)}")
    print(f"   Compression ratio: {len(values) / len(window_vals):.1f}x")
    
    # 5. Peak/valley sampling
    print("\n5. Peak/Valley Sampling:")
    peak_ts, peak_vals = peak_valley_sample(
        timestamps, values, min_distance=20, prominence=0.5
    )
    print(f"   Sampled points: {len(peak_vals)}")
    print(f"   Compression ratio: {len(values) / len(peak_vals):.1f}x")
    
    # 6. Multi-resolution sampling
    print("\n6. Multi-Resolution Sampling (3 levels):")
    multi_res = multi_resolution_sample(timestamps, values, levels=3)
    for level, (ts, vals) in multi_res.items():
        print(f"   Level {level}: {len(vals)} points")
    
    # 7. Quality metrics example
    print("\n7. Quality Metrics (Uniform vs Original):")
    metrics = quality_metrics(timestamps, values, uniform_ts, uniform_vals)
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.3f}")


if __name__ == "__main__":
    # Check for required dependencies
    try:
        import scipy
    except ImportError:
        print("Warning: scipy not installed. Some functions may not work.")
        print("Install with: pip install scipy")
    
    demo_time_based_sampling()