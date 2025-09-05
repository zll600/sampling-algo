//! Reservoir Sampling
//! 
//! Implementation of reservoir sampling algorithms for streaming data where the total
//! population size is unknown or too large to fit in memory. This is particularly
//! useful for processing large datasets, log files, or real-time data streams.
//! 
//! ## Algorithm (Vitter's Algorithm R)
//! 
//! 1. Fill reservoir with first k elements
//! 2. For each subsequent element i (i > k):
//!    - Generate random number j in [1, i]
//!    - If j <= k, replace reservoir[j-1] with element i
//! 3. Each element has equal probability k/n of being in final sample
//! 
//! ## Time Complexity: O(n) where n is the total number of elements processed
//! ## Space Complexity: O(k) where k is the reservoir size
//! 
//! ## Examples
//! 
//! ```rust
//! use sampling_algo::random_sampling::reservoir::{ReservoirSampler, reservoir_sample};
//! 
//! let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//! let sample = reservoir_sample(data.into_iter(), 3, Some(42)).unwrap();
//! println!("Reservoir sample: {:?}", sample);
//! ```

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::random_sampling::{SamplingError, SamplingResult};

/// Reservoir sampler for streaming data using Vitter's Algorithm R.
/// 
/// This struct maintains a fixed-size sample from a potentially unbounded
/// stream of data. Each element in the stream has an equal probability of
/// being included in the final sample.
/// 
/// # Examples
/// 
/// ```rust
/// use sampling_algo::random_sampling::reservoir::ReservoirSampler;
/// 
/// let mut sampler = ReservoirSampler::new(3, Some(42)).unwrap();
/// 
/// for i in 1..=10 {
///     sampler.add(i);
/// }
/// 
/// let sample = sampler.get_sample();
/// assert_eq!(sample.len(), 3);
/// ```
pub struct ReservoirSampler<T> {
    reservoir: Vec<T>,
    reservoir_size: usize,
    count: usize,
    rng: StdRng,
}

impl<T> ReservoirSampler<T> {
    /// Create a new reservoir sampler.
    /// 
    /// # Arguments
    /// 
    /// * `reservoir_size` - Maximum number of elements to keep in the reservoir
    /// * `seed` - Optional seed for reproducible random number generation
    /// 
    /// # Returns
    /// 
    /// Returns a new `ReservoirSampler` instance.
    /// 
    /// # Errors
    /// 
    /// * `SamplingError::InvalidParameters` - If reservoir_size is 0
    pub fn new(reservoir_size: usize, seed: Option<u64>) -> SamplingResult<Self> {
        if reservoir_size == 0 {
            return Err(SamplingError::InvalidParameters(
                "Reservoir size must be positive".to_string()
            ));
        }
        
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        Ok(ReservoirSampler {
            reservoir: Vec::with_capacity(reservoir_size),
            reservoir_size,
            count: 0,
            rng,
        })
    }
    
    /// Add an element to the stream.
    /// 
    /// This method processes each element according to the reservoir sampling
    /// algorithm. Elements are either added to fill the reservoir or replace
    /// existing elements with appropriate probability.
    /// 
    /// # Arguments
    /// 
    /// * `item` - The element to add to the stream
    pub fn add(&mut self, item: T) {
        self.count += 1;
        
        if self.reservoir.len() < self.reservoir_size {
            // Fill the reservoir with the first k elements
            self.reservoir.push(item);
        } else {
            // For element i (where i > k), replace with probability k/i
            let j = self.rng.gen_range(1..=self.count);
            if j <= self.reservoir_size {
                // Replace element at index j-1 (convert to 0-based indexing)
                self.reservoir[j - 1] = item;
            }
            // Otherwise, discard the item
        }
    }
    
    /// Get the current reservoir sample.
    /// 
    /// Returns a clone of the current reservoir contents. The sample
    /// represents a uniform random sample of all elements processed so far.
    /// 
    /// # Returns
    /// 
    /// A vector containing the current reservoir sample.
    pub fn get_sample(&self) -> Vec<T> 
    where 
        T: Clone 
    {
        self.reservoir.clone()
    }
    
    /// Get the current reservoir sample as a reference.
    /// 
    /// Returns a reference to the reservoir contents without cloning.
    /// Useful when you need to inspect the sample without taking ownership.
    pub fn get_sample_ref(&self) -> &[T] {
        &self.reservoir
    }
    
    /// Get the number of elements processed so far.
    pub fn count(&self) -> usize {
        self.count
    }
    
    /// Get the reservoir size.
    pub fn reservoir_size(&self) -> usize {
        self.reservoir_size
    }
    
    /// Check if the reservoir is full.
    pub fn is_full(&self) -> bool {
        self.reservoir.len() == self.reservoir_size
    }
    
    /// Reset the sampler to its initial state.
    pub fn reset(&mut self) {
        self.reservoir.clear();
        self.count = 0;
    }
}

/// Perform reservoir sampling on an iterator.
/// 
/// This is a convenience function that creates a reservoir sampler and
/// processes all elements from the provided iterator.
/// 
/// # Arguments
/// 
/// * `data_stream` - Iterator over elements to sample from
/// * `reservoir_size` - Number of elements to keep in the sample
/// * `seed` - Optional seed for reproducible results
/// 
/// # Returns
/// 
/// Returns a vector containing the sampled elements.
/// 
/// # Errors
/// 
/// * `SamplingError::InvalidParameters` - If reservoir_size is 0
/// 
/// # Examples
/// 
/// ```rust
/// use sampling_algo::random_sampling::reservoir::reservoir_sample;
/// 
/// let data: Vec<i32> = (1..=1000).collect();
/// let sample = reservoir_sample(data.into_iter(), 10, Some(42)).unwrap();
/// assert_eq!(sample.len(), 10);
/// ```
pub fn reservoir_sample<T, I>(
    data_stream: I, 
    reservoir_size: usize, 
    seed: Option<u64>
) -> SamplingResult<Vec<T>> 
where 
    I: IntoIterator<Item = T>,
    T: Clone,
{
    let mut sampler = ReservoirSampler::new(reservoir_size, seed)?;
    
    for item in data_stream {
        sampler.add(item);
    }
    
    Ok(sampler.get_sample())
}

/// Advanced reservoir sampling with weighted elements.
/// 
/// This variant allows elements to have different weights, affecting their
/// probability of being selected. Uses the A-ExpJ algorithm for weighted
/// reservoir sampling.
pub struct WeightedReservoirSampler<T> {
    reservoir: Vec<(T, f64)>, // (item, weight)
    reservoir_size: usize,
    rng: StdRng,
}

impl<T> WeightedReservoirSampler<T> {
    /// Create a new weighted reservoir sampler.
    pub fn new(reservoir_size: usize, seed: Option<u64>) -> SamplingResult<Self> {
        if reservoir_size == 0 {
            return Err(SamplingError::InvalidParameters(
                "Reservoir size must be positive".to_string()
            ));
        }
        
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        Ok(WeightedReservoirSampler {
            reservoir: Vec::with_capacity(reservoir_size),
            reservoir_size,
            rng,
        })
    }
    
    /// Add a weighted element to the stream.
    pub fn add(&mut self, item: T, weight: f64) {
        if weight <= 0.0 {
            return; // Skip items with non-positive weight
        }
        
        // Generate a key for this item: u^(1/w) where u is uniform(0,1)
        let u: f64 = self.rng.gen();
        let key = u.powf(1.0 / weight);
        
        if self.reservoir.len() < self.reservoir_size {
            // Fill the reservoir
            self.reservoir.push((item, key));
            // Sort by key to maintain the smallest keys
            self.reservoir.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        } else if key > self.reservoir[0].1 {
            // Replace the item with smallest key
            self.reservoir[0] = (item, key);
            // Re-sort to maintain order
            self.reservoir.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
    }
    
    /// Get the current weighted reservoir sample.
    pub fn get_sample(&self) -> Vec<T> 
    where 
        T: Clone 
    {
        self.reservoir.iter().map(|(item, _)| item.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_sampler_basic() {
        let mut sampler = ReservoirSampler::new(3, Some(42)).unwrap();
        
        for i in 1..=10 {
            sampler.add(i);
        }
        
        let sample = sampler.get_sample();
        assert_eq!(sample.len(), 3);
        assert_eq!(sampler.count(), 10);
        
        // All elements should be from the range 1..=10
        for &item in &sample {
            assert!(item >= 1 && item <= 10);
        }
    }

    #[test]
    fn test_reservoir_sampler_errors() {
        assert_eq!(
            ReservoirSampler::<i32>::new(0, Some(42)),
            Err(SamplingError::InvalidParameters("Reservoir size must be positive".to_string()))
        );
    }

    #[test]
    fn test_reservoir_sampler_fill_phase() {
        let mut sampler = ReservoirSampler::new(5, Some(42)).unwrap();
        
        // Add fewer elements than reservoir size
        for i in 1..=3 {
            sampler.add(i);
        }
        
        let sample = sampler.get_sample();
        assert_eq!(sample.len(), 3);
        assert_eq!(sample, vec![1, 2, 3]);
        assert!(!sampler.is_full());
    }

    #[test]
    fn test_reservoir_sampler_full() {
        let mut sampler = ReservoirSampler::new(3, Some(42)).unwrap();
        
        // Fill the reservoir exactly
        for i in 1..=3 {
            sampler.add(i);
        }
        
        assert!(sampler.is_full());
        assert_eq!(sampler.get_sample(), vec![1, 2, 3]);
    }

    #[test]
    fn test_reservoir_sample_function() {
        let data: Vec<i32> = (1..=100).collect();
        let sample = reservoir_sample(data.into_iter(), 5, Some(42)).unwrap();
        
        assert_eq!(sample.len(), 5);
        
        // All elements should be unique (since we're sampling without replacement)
        let mut sorted_sample = sample.clone();
        sorted_sample.sort();
        sorted_sample.dedup();
        assert_eq!(sorted_sample.len(), sample.len());
    }

    #[test]
    fn test_reservoir_sample_reproducible() {
        let data1: Vec<i32> = (1..=50).collect();
        let data2: Vec<i32> = (1..=50).collect();
        
        let sample1 = reservoir_sample(data1.into_iter(), 5, Some(42)).unwrap();
        let sample2 = reservoir_sample(data2.into_iter(), 5, Some(42)).unwrap();
        
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_reservoir_sampler_reset() {
        let mut sampler = ReservoirSampler::new(3, Some(42)).unwrap();
        
        for i in 1..=5 {
            sampler.add(i);
        }
        
        assert_eq!(sampler.count(), 5);
        assert_eq!(sampler.get_sample().len(), 3);
        
        sampler.reset();
        
        assert_eq!(sampler.count(), 0);
        assert_eq!(sampler.get_sample().len(), 0);
        assert!(!sampler.is_full());
    }

    #[test]
    fn test_reservoir_sample_empty_input() {
        let data: Vec<i32> = vec![];
        let sample = reservoir_sample(data.into_iter(), 3, Some(42)).unwrap();
        
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_weighted_reservoir_sampler() {
        let mut sampler = WeightedReservoirSampler::new(3, Some(42)).unwrap();
        
        // Add items with different weights
        sampler.add("high", 10.0);
        sampler.add("medium", 5.0);
        sampler.add("low", 1.0);
        sampler.add("very_low", 0.1);
        
        let sample = sampler.get_sample();
        assert_eq!(sample.len(), 3);
        
        // Higher weighted items should be more likely to appear
        // (though this is probabilistic, so we can't test deterministically)
    }

    #[test]
    fn test_weighted_reservoir_zero_weight() {
        let mut sampler = WeightedReservoirSampler::new(2, Some(42)).unwrap();
        
        sampler.add("valid", 1.0);
        sampler.add("zero_weight", 0.0);
        sampler.add("negative_weight", -1.0);
        
        let sample = sampler.get_sample();
        
        // Only the valid item should be in the sample
        assert_eq!(sample.len(), 1);
        assert_eq!(sample[0], "valid");
    }

    #[test]
    fn test_reservoir_sampler_different_types() {
        // Test with strings
        let mut string_sampler = ReservoirSampler::new(2, Some(42)).unwrap();
        string_sampler.add("hello".to_string());
        string_sampler.add("world".to_string());
        string_sampler.add("rust".to_string());
        
        let string_sample = string_sampler.get_sample();
        assert_eq!(string_sample.len(), 2);
        
        // Test with references
        let data = vec!["a", "b", "c", "d"];
        let sample = reservoir_sample(data.iter(), 2, Some(42)).unwrap();
        assert_eq!(sample.len(), 2);
    }
}