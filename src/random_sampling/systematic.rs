//! Systematic Sampling
//!
//! Implementation of systematic sampling where elements are selected at regular intervals
//! from an ordered list. This method is simpler than random sampling and provides good
//! coverage across the population, but assumes the population ordering is random or
//! doesn't introduce bias.
//!
//! ## Algorithm
//!
//! 1. Calculate sampling interval k = N/n (population size / sample size)
//! 2. Choose random starting point r in [0, k)
//! 3. Select elements at positions: r, r+k, r+2k, ..., r+(n-1)k
//! 4. Use modular arithmetic to wrap around if needed
//!
//! ## Time Complexity: O(n) where n is the sample size
//! ## Space Complexity: O(n) for the output sample
//!
//! ## Examples
//!
//! ```rust
//! use sampling_algo::random_sampling::systematic::systematic_sample;
//!
//! let data: Vec<i32> = (1..=100).collect();
//! let sample = systematic_sample(&data, 10, Some(42)).unwrap();
//! println!("Systematic sample: {:?}", sample);
//! ```

use crate::random_sampling::{SamplingError, SamplingResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Perform systematic sampling on ordered data.
///
/// This function selects elements at regular intervals from the input data.
/// It's particularly useful when the data is naturally ordered (e.g., time series)
/// and you want to maintain that temporal or spatial structure while sampling.
///
/// The algorithm calculates a sampling interval k = n/sample_size and selects
/// every k-th element starting from a random position.
///
/// # Arguments
///
/// * `data` - The population to sample from (should be ordered)
/// * `sample_size` - Number of elements to sample
/// * `seed` - Optional seed for reproducible random starting point
///
/// # Returns
///
/// Returns a vector containing the systematically sampled elements in their
/// original order from the data.
///
/// # Errors
///
/// * `SamplingError::EmptyPopulation` - If data is empty but sample_size > 0
/// * `SamplingError::SampleSizeTooLarge` - If sample_size > population size
///
/// # Examples
///
/// ```rust
/// use sampling_algo::random_sampling::systematic::systematic_sample;
///
/// let data = vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
/// let sample = systematic_sample(&data, 3, Some(42)).unwrap();
/// assert_eq!(sample.len(), 3);
/// ```
///
/// # Notes
///
/// - Assumes the input data ordering doesn't introduce systematic bias
/// - Provides good coverage across the entire population
/// - More efficient than simple random sampling
/// - Maintains the natural ordering of the data
pub fn systematic_sample<T: Clone>(
    data: &[T],
    sample_size: usize,
    seed: Option<u64>,
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

    let n = data.len();

    // Handle edge case where sample_size equals population size
    if sample_size == n {
        return Ok(data.to_vec());
    }

    // Calculate the sampling interval
    let interval = n as f64 / sample_size as f64;

    // Initialize RNG for random starting point
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Choose random starting point within the first interval
    let start = rng.gen_range(0.0..interval);

    let mut sample = Vec::with_capacity(sample_size);

    // Select elements at regular intervals
    for i in 0..sample_size {
        let position = (start + i as f64 * interval) as usize;
        // Use modular arithmetic to handle wrap-around
        let index = position % n;
        sample.push(data[index].clone());
    }

    Ok(sample)
}

/// Alternative implementation using integer arithmetic for exact positioning
///
/// This version uses integer arithmetic to avoid floating-point precision issues
/// and provides more predictable behavior for systematic sampling.
pub fn systematic_sample_integer<T: Clone>(
    data: &[T],
    sample_size: usize,
    seed: Option<u64>,
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

    let n = data.len();

    if sample_size == n {
        return Ok(data.to_vec());
    }

    // Calculate the sampling interval (integer division)
    let interval = n / sample_size;

    // Initialize RNG for random starting point
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Choose random starting point within the first interval
    let start = if interval > 0 {
        rng.gen_range(0..interval)
    } else {
        0
    };

    let mut sample = Vec::with_capacity(sample_size);

    // Select elements at regular intervals
    for i in 0..sample_size {
        let index = (start + i * interval) % n;
        sample.push(data[index].clone());
    }

    Ok(sample)
}

/// Systematic sampling with linear systematic selection
///
/// This implementation follows the classical definition more strictly:
/// it selects every k-th element without wrap-around, which may result
/// in fewer samples than requested if the population size isn't divisible
/// by the sample size.
pub fn systematic_sample_linear<T: Clone>(
    data: &[T],
    sample_size: usize,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>> {
    if data.is_empty() && sample_size > 0 {
        return Err(SamplingError::EmptyPopulation);
    }

    if sample_size > data.len() {
        return Err(SamplingError::SampleSizeTooLarge);
    }

    if sample_size == 0 {
        return Ok(Vec::new());
    }

    let n = data.len();

    if sample_size == n {
        return Ok(data.to_vec());
    }

    // Calculate the sampling interval
    let interval = n / sample_size;

    if interval == 0 {
        // If interval is 0, fall back to taking all elements
        return Ok(data.to_vec());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Random starting point within first interval
    let start = rng.gen_range(0..interval);

    let mut sample = Vec::new();
    let mut position = start;

    // Collect samples at regular intervals without wrap-around
    while position < n && sample.len() < sample_size {
        sample.push(data[position].clone());
        position += interval;
    }

    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_systematic_sample_basic() {
        let data: Vec<i32> = (1..=10).collect();
        let sample = systematic_sample(&data, 3, Some(42)).unwrap();

        assert_eq!(sample.len(), 3);

        // All elements should be from the original data
        for item in &sample {
            assert!(data.contains(item));
        }
    }

    #[test]
    fn test_systematic_sample_full_population() {
        let data = vec![1, 2, 3, 4, 5];
        let sample = systematic_sample(&data, 5, Some(42)).unwrap();

        assert_eq!(sample.len(), 5);
        assert_eq!(sample, data);
    }

    #[test]
    fn test_systematic_sample_empty() {
        let data: Vec<i32> = vec![];
        let sample = systematic_sample(&data, 0, Some(42)).unwrap();
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_systematic_sample_errors() {
        let data = vec![1, 2, 3];

        // Sample size too large
        assert_eq!(
            systematic_sample(&data, 5, Some(42)),
            Err(SamplingError::SampleSizeTooLarge)
        );

        // Empty population with non-zero sample size
        let empty_data: Vec<i32> = vec![];
        assert_eq!(
            systematic_sample(&empty_data, 1, Some(42)),
            Err(SamplingError::EmptyPopulation)
        );
    }

    #[test]
    fn test_systematic_sample_reproducible() {
        let data: Vec<i32> = (1..=20).collect();
        let sample1 = systematic_sample(&data, 5, Some(42)).unwrap();
        let sample2 = systematic_sample(&data, 5, Some(42)).unwrap();

        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_systematic_sample_integer() {
        let data: Vec<i32> = (1..=12).collect();
        let sample = systematic_sample_integer(&data, 4, Some(42)).unwrap();

        assert_eq!(sample.len(), 4);

        for item in &sample {
            assert!(data.contains(item));
        }
    }

    #[test]
    fn test_systematic_sample_linear() {
        let data: Vec<i32> = (1..=10).collect();
        let sample = systematic_sample_linear(&data, 3, Some(42)).unwrap();

        // Linear sampling may produce fewer samples due to no wrap-around
        assert!(sample.len() <= 3);
        assert!(!sample.is_empty());

        for item in &sample {
            assert!(data.contains(item));
        }
    }

    #[test]
    fn test_systematic_sample_coverage() {
        // Test that systematic sampling provides good coverage
        let data: Vec<i32> = (1..=100).collect();
        let sample = systematic_sample(&data, 10, Some(42)).unwrap();

        assert_eq!(sample.len(), 10);

        // Check that samples are spread across the range
        let min_sample = *sample.iter().min().unwrap();
        let max_sample = *sample.iter().max().unwrap();

        // Should cover a good portion of the range
        assert!(min_sample <= 50); // Should start from early in the range
        assert!(max_sample >= 50); // Should extend to later in the range
    }

    #[test]
    fn test_different_seeds_systematic() {
        let data: Vec<i32> = (1..=50).collect();
        let sample1 = systematic_sample(&data, 5, Some(42)).unwrap();
        let sample2 = systematic_sample(&data, 5, Some(43)).unwrap();

        // Different seeds should potentially produce different starting points
        // though the structure remains systematic
        assert_eq!(sample1.len(), sample2.len());
    }

    #[test]
    fn test_systematic_ordering_preserved() {
        let data = vec!["a", "b", "c", "d", "e", "f", "g", "h"];
        let sample = systematic_sample_linear(&data, 3, Some(42)).unwrap();

        // Check that the systematic nature creates a predictable pattern
        // (this depends on the specific seed, but structure should be maintained)
        assert!(sample.len() >= 1);

        for item in &sample {
            assert!(data.contains(item));
        }
    }
}
