"""
Random Sampling Algorithms

This module implements various random sampling techniques for data analysis,
particularly suited for time series data processing.
"""

import random
import heapq
from typing import List, TypeVar, Iterator, Optional, Dict, Any, Union
from collections import defaultdict

T = TypeVar('T')


def simple_random_sample(data: List[T], sample_size: int, seed: Optional[int] = None) -> List[T]:
    """
    Perform simple random sampling without replacement.
    
    Args:
        data: The population to sample from
        sample_size: Number of elements to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled elements
        
    Raises:
        ValueError: If sample_size > len(data) or sample_size < 0
    """
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative")
    if sample_size > len(data):
        raise ValueError("Sample size cannot exceed population size")
    
    if seed is not None:
        random.seed(seed)
    
    return random.sample(data, sample_size)


def stratified_sample(
    data: List[T], 
    strata: List[str], 
    sample_size: int, 
    proportional: bool = True,
    seed: Optional[int] = None
) -> List[T]:
    """
    Perform stratified sampling.
    
    Args:
        data: The population to sample from
        strata: Stratum label for each element in data
        sample_size: Total number of elements to sample
        proportional: If True, sample proportionally to stratum size
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled elements
        
    Raises:
        ValueError: If len(data) != len(strata) or sample_size < 0
    """
    if len(data) != len(strata):
        raise ValueError("Data and strata must have the same length")
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative")
    
    if seed is not None:
        random.seed(seed)
    
    # Group data by strata
    strata_data: Dict[str, List[T]] = defaultdict(list)
    for item, stratum in zip(data, strata):
        strata_data[stratum].append(item)
    
    if not strata_data:
        return []
    
    sample = []
    
    if proportional:
        # Proportional allocation
        total_size = len(data)
        for stratum, stratum_data in strata_data.items():
            stratum_sample_size = int((len(stratum_data) / total_size) * sample_size)
            if stratum_sample_size > 0:
                sample.extend(random.sample(stratum_data, min(stratum_sample_size, len(stratum_data))))
    else:
        # Equal allocation
        strata_count = len(strata_data)
        per_stratum_size = sample_size // strata_count
        remaining = sample_size % strata_count
        
        for i, (stratum, stratum_data) in enumerate(strata_data.items()):
            stratum_sample_size = per_stratum_size + (1 if i < remaining else 0)
            if stratum_sample_size > 0:
                sample.extend(random.sample(stratum_data, min(stratum_sample_size, len(stratum_data))))
    
    return sample


def systematic_sample(
    data: List[T], 
    sample_size: int, 
    seed: Optional[int] = None
) -> List[T]:
    """
    Perform systematic sampling.
    
    Args:
        data: The population to sample from (assumed to be ordered)
        sample_size: Number of elements to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled elements
        
    Raises:
        ValueError: If sample_size > len(data) or sample_size < 0
    """
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative")
    if sample_size > len(data):
        raise ValueError("Sample size cannot exceed population size")
    if sample_size == 0:
        return []
    
    if seed is not None:
        random.seed(seed)
    
    n = len(data)
    k = n // sample_size  # Sampling interval
    start = random.randint(0, k - 1)  # Random starting point
    
    sample = []
    for i in range(sample_size):
        index = (start + i * k) % n
        sample.append(data[index])
    
    return sample


class ReservoirSampler:
    """
    Reservoir sampling for streaming data.
    
    Maintains a fixed-size sample from a stream of unknown length using
    Vitter's Algorithm R.
    """
    
    def __init__(self, reservoir_size: int, seed: Optional[int] = None):
        """
        Initialize the reservoir sampler.
        
        Args:
            reservoir_size: Size of the reservoir
            seed: Random seed for reproducibility
        """
        if reservoir_size <= 0:
            raise ValueError("Reservoir size must be positive")
            
        self.reservoir_size = reservoir_size
        self.reservoir: List[T] = []
        self.count = 0
        
        if seed is not None:
            random.seed(seed)
    
    def add(self, item: T) -> None:
        """
        Add an item to the stream.
        
        Args:
            item: Item to add to the stream
        """
        self.count += 1
        
        if len(self.reservoir) < self.reservoir_size:
            # Fill the reservoir
            self.reservoir.append(item)
        else:
            # Replace with probability k/count
            j = random.randint(1, self.count)
            if j <= self.reservoir_size:
                self.reservoir[j - 1] = item
    
    def get_sample(self) -> List[T]:
        """
        Get the current reservoir sample.
        
        Returns:
            Current sample in the reservoir
        """
        return self.reservoir.copy()


def reservoir_sample(data_stream: Iterator[T], reservoir_size: int, seed: Optional[int] = None) -> List[T]:
    """
    Perform reservoir sampling on a data stream.
    
    Args:
        data_stream: Iterator of data items
        reservoir_size: Size of the reservoir
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled elements
    """
    sampler = ReservoirSampler(reservoir_size, seed)
    
    for item in data_stream:
        sampler.add(item)
    
    return sampler.get_sample()


def weighted_sample(
    data: List[T], 
    weights: List[float], 
    sample_size: int, 
    replacement: bool = False,
    seed: Optional[int] = None
) -> List[T]:
    """
    Perform weighted random sampling.
    
    Args:
        data: The population to sample from
        weights: Weight for each element in data
        sample_size: Number of elements to sample
        replacement: Whether to sample with replacement
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled elements
        
    Raises:
        ValueError: If len(data) != len(weights) or invalid parameters
    """
    if len(data) != len(weights):
        raise ValueError("Data and weights must have the same length")
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative")
    if not replacement and sample_size > len(data):
        raise ValueError("Sample size cannot exceed population size when sampling without replacement")
    if any(w < 0 for w in weights):
        raise ValueError("All weights must be non-negative")
    if sum(weights) == 0:
        raise ValueError("At least one weight must be positive")
    
    if seed is not None:
        random.seed(seed)
    
    if replacement:
        # Simple weighted sampling with replacement
        return random.choices(data, weights=weights, k=sample_size)
    else:
        # Weighted sampling without replacement using A-Res algorithm
        if sample_size == 0:
            return []
        
        # Create heap with weighted keys
        heap = []
        for i, (item, weight) in enumerate(zip(data, weights)):
            if weight > 0:
                key = random.random() ** (1.0 / weight)
                heapq.heappush(heap, (-key, i, item))  # Negative for max-heap behavior
        
        # Extract top k elements
        sample = []
        for _ in range(min(sample_size, len(heap))):
            if heap:
                _, _, item = heapq.heappop(heap)
                sample.append(item)
        
        return sample


def time_series_sample(
    data: List[T], 
    sample_size: int, 
    method: str = "systematic",
    window_size: Optional[int] = None,
    seed: Optional[int] = None
) -> List[T]:
    """
    Sample from time series data with temporal awareness.
    
    Args:
        data: Time series data (ordered by time)
        sample_size: Number of elements to sample
        method: Sampling method ("systematic", "random", "windowed")
        window_size: Size of windows for windowed sampling
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled elements preserving temporal order
    """
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative")
    if sample_size > len(data):
        raise ValueError("Sample size cannot exceed data length")
    
    if seed is not None:
        random.seed(seed)
    
    if method == "systematic":
        return systematic_sample(data, sample_size, seed)
    elif method == "random":
        sample = simple_random_sample(data, sample_size, seed)
        # Sort by original indices to preserve temporal order
        indices = {id(item): i for i, item in enumerate(data)}
        return sorted(sample, key=lambda x: indices.get(id(x), 0))
    elif method == "windowed":
        if window_size is None:
            window_size = len(data) // sample_size
        
        if window_size <= 0:
            return simple_random_sample(data, sample_size, seed)
        
        sample = []
        for i in range(0, len(data), window_size):
            window = data[i:i + window_size]
            if window:
                sample.append(random.choice(window))
                if len(sample) >= sample_size:
                    break
        
        return sample[:sample_size]
    else:
        raise ValueError(f"Unknown sampling method: {method}")


# Example usage and testing functions
def demo_sampling_algorithms():
    """Demonstrate the sampling algorithms with example data."""
    
    # Generate example time series data
    time_series = list(range(1000))
    labels = ['A'] * 300 + ['B'] * 400 + ['C'] * 300
    weights = [i + 1 for i in range(1000)]  # Increasing weights
    
    print("=== Random Sampling Algorithm Demonstrations ===\n")
    
    # Simple random sampling
    print("1. Simple Random Sampling:")
    sample = simple_random_sample(time_series, 10, seed=42)
    print(f"   Sample: {sample}")
    
    # Stratified sampling
    print("\n2. Stratified Sampling:")
    sample = stratified_sample(time_series, labels, 15, seed=42)
    print(f"   Sample: {sorted(sample)}")
    
    # Systematic sampling
    print("\n3. Systematic Sampling:")
    sample = systematic_sample(time_series, 10, seed=42)
    print(f"   Sample: {sample}")
    
    # Reservoir sampling
    print("\n4. Reservoir Sampling:")
    sample = reservoir_sample(iter(time_series), 10, seed=42)
    print(f"   Sample: {sorted(sample)}")
    
    # Weighted sampling
    print("\n5. Weighted Sampling:")
    sample = weighted_sample(time_series[:50], weights[:50], 10, seed=42)
    print(f"   Sample: {sorted(sample)}")
    
    # Time series sampling
    print("\n6. Time Series Sampling:")
    sample = time_series_sample(time_series, 10, method="windowed", window_size=100, seed=42)
    print(f"   Sample: {sample}")


if __name__ == "__main__":
    demo_sampling_algorithms()