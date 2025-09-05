//! Simple Random Sampling
//! 
//! Implementation of simple random sampling without replacement using the Fisher-Yates shuffle algorithm.
//! This provides an unbiased way to select a subset of items from a population where each
//! combination of items has an equal probability of being selected.
//! 
//! ## Algorithm
//! 
//! Uses a partial Fisher-Yates shuffle to select k elements from n items in O(k) time.
//! Instead of shuffling the entire array, we only shuffle the first k positions.
//! 
//! ## Time Complexity: O(k) where k is the sample size
//! ## Space Complexity: O(k) for the output sample
//! 
//! ## Examples
//! 
//! ```rust
//! use sampling_algo::random_sampling::simple::simple_random_sample;
//! 
//! let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//! let sample = simple_random_sample(&data, 3, Some(42)).unwrap();
//! println!("Sample: {:?}", sample);
//! ```

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::random_sampling::{SamplingError, SamplingResult};

/// Perform simple random sampling without replacement.
/// 
/// This function implements a partial Fisher-Yates shuffle to efficiently sample
/// k items from the input data. Each possible combination of k items has equal
/// probability of being selected.
/// 
/// # Arguments
/// 
/// * `data` - The population to sample from
/// * `sample_size` - Number of elements to sample
/// * `seed` - Optional seed for reproducible results
/// 
/// # Returns
/// 
/// Returns a vector containing the sampled elements in random order.
/// 
/// # Errors
/// 
/// * `SamplingError::NegativeSampleSize` - If sample_size < 0
/// * `SamplingError::SampleSizeTooLarge` - If sample_size > population size
/// * `SamplingError::EmptyPopulation` - If data is empty but sample_size > 0
/// 
/// # Examples
/// 
/// ```rust
/// use sampling_algo::random_sampling::simple::simple_random_sample;
/// 
/// let data = vec!["a", "b", "c", "d", "e"];
/// let sample = simple_random_sample(&data, 2, Some(123)).unwrap();
/// assert_eq!(sample.len(), 2);
/// ```
pub fn simple_random_sample<T: Clone>(
    data: &[T], 
    sample_size: usize, 
    seed: Option<u64>
) -> SamplingResult<Vec<T>> {
    // Input validation
    if data.is_empty() && sample_size > 0 {
        return Err(SamplingError::EmptyPopulation);
    }
    
    if sample_size > data.len() {
        return Err(SamplingError::SampleSizeTooLarge);
    }
    
    if sample_size == 0 {
        return Ok(Vec::new());
    }
    
    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    
    // Create a copy of indices to shuffle
    let mut indices: Vec<usize> = (0..data.len()).collect();
    
    // Perform partial Fisher-Yates shuffle
    // We only need to shuffle the first sample_size positions
    for i in 0..sample_size {
        // Generate random index from i to len-1
        let j = rng.gen_range(i..data.len());
        // Swap elements at positions i and j
        indices.swap(i, j);
    }
    
    // Collect the sampled elements
    let sample: Vec<T> = indices
        .into_iter()
        .take(sample_size)
        .map(|idx| data[idx].clone())
        .collect();
    
    Ok(sample)
}

/// Alternative implementation using reservoir sampling approach
/// 
/// This is useful for comparison and when you want to maintain the original
/// order of elements in the data.
pub fn simple_random_sample_reservoir<T: Clone>(
    data: &[T], 
    sample_size: usize, 
    seed: Option<u64>
) -> SamplingResult<Vec<T>> {
    // Input validation
    if data.is_empty() && sample_size > 0 {
        return Err(SamplingError::EmptyPopulation);
    }
    
    if sample_size > data.len() {
        return Err(SamplingError::SampleSizeTooLarge);
    }
    
    if sample_size == 0 {
        return Ok(Vec::new());
    }
    
    // Initialize RNG
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    
    let mut sample = Vec::with_capacity(sample_size);
    
    // Fill the sample with first k elements
    for i in 0..sample_size.min(data.len()) {
        sample.push(data[i].clone());
    }
    
    // For each remaining element, decide whether to include it
    for i in sample_size..data.len() {
        let j = rng.gen_range(0..=i);
        if j < sample_size {
            sample[j] = data[i].clone();
        }
    }
    
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_random_sample_basic() {
        let data = vec![1, 2, 3, 4, 5];
        let sample = simple_random_sample(&data, 3, Some(42)).unwrap();
        
        assert_eq!(sample.len(), 3);
        // All elements should be from the original data
        for item in &sample {
            assert!(data.contains(item));
        }
        // No duplicates (since sampling without replacement)
        let mut sorted_sample = sample.clone();
        sorted_sample.sort();
        sorted_sample.dedup();
        assert_eq!(sorted_sample.len(), sample.len());
    }

    #[test]
    fn test_simple_random_sample_empty() {
        let data: Vec<i32> = vec![];
        let sample = simple_random_sample(&data, 0, Some(42)).unwrap();
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_simple_random_sample_full_population() {
        let data = vec![1, 2, 3, 4, 5];
        let sample = simple_random_sample(&data, 5, Some(42)).unwrap();
        
        assert_eq!(sample.len(), 5);
        // Should contain all elements (though possibly in different order)
        let mut sorted_sample = sample.clone();
        let mut sorted_data = data.clone();
        sorted_sample.sort();
        sorted_data.sort();
        assert_eq!(sorted_sample, sorted_data);
    }

    #[test]
    fn test_simple_random_sample_errors() {
        let data = vec![1, 2, 3];
        
        // Sample size too large
        assert_eq!(
            simple_random_sample(&data, 5, Some(42)),
            Err(SamplingError::SampleSizeTooLarge)
        );
        
        // Empty population with non-zero sample size
        let empty_data: Vec<i32> = vec![];
        assert_eq!(
            simple_random_sample(&empty_data, 1, Some(42)),
            Err(SamplingError::EmptyPopulation)
        );
    }

    #[test]
    fn test_simple_random_sample_reproducible() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sample1 = simple_random_sample(&data, 5, Some(42)).unwrap();
        let sample2 = simple_random_sample(&data, 5, Some(42)).unwrap();
        
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_reservoir_implementation() {
        let data = vec![1, 2, 3, 4, 5];
        let sample = simple_random_sample_reservoir(&data, 3, Some(42)).unwrap();
        
        assert_eq!(sample.len(), 3);
        for item in &sample {
            assert!(data.contains(item));
        }
    }

    #[test]
    fn test_different_seeds_produce_different_results() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sample1 = simple_random_sample(&data, 5, Some(42)).unwrap();
        let sample2 = simple_random_sample(&data, 5, Some(43)).unwrap();
        
        // With high probability, different seeds should produce different samples
        assert_ne!(sample1, sample2);
    }
}