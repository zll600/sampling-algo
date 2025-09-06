//! Weighted Sampling
//!
//! Implementation of weighted random sampling algorithms where elements have different
//! probabilities of being selected based on their weights. This is useful when some
//! elements should be more likely to appear in the sample than others.
//!
//! ## Algorithms Implemented
//!
//! - **Weighted sampling with replacement**: Simple probability-proportional selection
//! - **A-Res algorithm**: Weighted sampling without replacement
//! - **Alias method**: Efficient weighted sampling for repeated sampling
//!
//! ## Time Complexity:
//! - With replacement: O(k) where k is sample size
//! - Without replacement (A-Res): O(n + k*log(k)) where n is population size
//!
//! ## Space Complexity: O(k) for output sample
//!
//! ## Examples
//!
//! ```rust
//! use sampling_algo::random_sampling::weighted::weighted_sample;
//!
//! let data = vec!["rare", "common", "very_common"];
//! let weights = vec![0.1, 1.0, 5.0];
//! let sample = weighted_sample(&data, &weights, 10, true, Some(42)).unwrap();
//! println!("Weighted sample: {:?}", sample);
//! ```

use crate::random_sampling::{SamplingError, SamplingResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Perform weighted random sampling.
///
/// Elements are selected with probability proportional to their weights.
/// Supports both sampling with and without replacement.
///
/// # Arguments
///
/// * `data` - The population to sample from
/// * `weights` - Weight for each element in data (must be non-negative)
/// * `sample_size` - Number of elements to sample
/// * `replacement` - Whether to sample with replacement
/// * `seed` - Optional seed for reproducible results
///
/// # Returns
///
/// Returns a vector containing the sampled elements.
///
/// # Errors
///
/// * `SamplingError::MismatchedInputs` - If data and weights have different lengths
/// * `SamplingError::SampleSizeTooLarge` - If sample_size > population size (without replacement)
/// * `SamplingError::InvalidParameters` - If all weights are zero or negative
/// * `SamplingError::EmptyPopulation` - If data is empty but sample_size > 0
///
/// # Examples
///
/// ```rust
/// use sampling_algo::random_sampling::weighted::weighted_sample;
///
/// let items = vec!["a", "b", "c"];
/// let weights = vec![1.0, 2.0, 3.0];
/// let sample = weighted_sample(&items, &weights, 5, true, Some(42)).unwrap();
/// assert_eq!(sample.len(), 5);
/// ```
pub fn weighted_sample<T: Clone + std::cmp::PartialEq>(
    data: &[T],
    weights: &[f64],
    sample_size: usize,
    replacement: bool,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>> {
    // Input validation
    if data.len() != weights.len() {
        return Err(SamplingError::MismatchedInputs);
    }

    if data.is_empty() && sample_size > 0 {
        return Err(SamplingError::EmptyPopulation);
    }

    if sample_size == 0 {
        return Ok(Vec::new());
    }

    if !replacement && sample_size > data.len() {
        return Err(SamplingError::SampleSizeTooLarge);
    }

    // Check that weights are valid
    if weights.iter().any(|&w| w < 0.0) {
        return Err(SamplingError::InvalidParameters(
            "All weights must be non-negative".to_string(),
        ));
    }

    let total_weight: f64 = weights.iter().sum();
    if total_weight <= 0.0 {
        return Err(SamplingError::InvalidParameters(
            "At least one weight must be positive".to_string(),
        ));
    }

    if replacement {
        weighted_sample_with_replacement(data, weights, sample_size, total_weight, seed)
    } else {
        weighted_sample_without_replacement(data, weights, sample_size, seed)
    }
}

/// Weighted sampling with replacement using cumulative distribution
/// @link https://en.wikipedia.org/wiki/Reservoir_sampling#Algorithm_A-Res
fn weighted_sample_with_replacement<T: Clone>(
    data: &[T],
    weights: &[f64],
    sample_size: usize,
    total_weight: f64,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Build cumulative distribution
    let mut cumulative_weights = Vec::with_capacity(weights.len());
    let mut cumulative_sum = 0.0;

    for &weight in weights {
        cumulative_sum += weight;
        cumulative_weights.push(cumulative_sum);
    }

    let mut sample = Vec::with_capacity(sample_size);

    for _ in 0..sample_size {
        let random_weight = rng.gen::<f64>() * total_weight;

        // Binary search for the selected element
        let index = match cumulative_weights.binary_search_by(|&w| {
            if w < random_weight {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }) {
            Ok(i) => i,
            Err(i) => i, // Insert position is the correct index
        };

        if index < data.len() {
            sample.push(data[index].clone());
        }
    }

    Ok(sample)
}

/// Weighted sampling without replacement using A-Res algorithm
fn weighted_sample_without_replacement<T: Clone + std::cmp::PartialEq>(
    data: &[T],
    weights: &[f64],
    sample_size: usize,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    #[derive(PartialEq)]
    struct WeightedItem<T> {
        item: T,
        key: f64,
        original_index: usize,
    }

    impl<T: std::cmp::PartialEq> Eq for WeightedItem<T> {}

    impl<T: std::cmp::PartialEq> PartialOrd for WeightedItem<T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            // Reverse ordering for max-heap behavior (we want largest keys first)
            other.key.partial_cmp(&self.key)
        }
    }

    impl<T: std::cmp::PartialEq> Ord for WeightedItem<T> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::with_capacity(sample_size.min(data.len()));

    // Generate keys for all items: key_i = random()^(1/weight_i)
    for (i, (item, &weight)) in data.iter().zip(weights.iter()).enumerate() {
        if weight > 0.0 {
            let u: f64 = rng.gen();
            let key = u.powf(1.0 / weight);

            let weighted_item = WeightedItem {
                item: item.clone(),
                key,
                original_index: i,
            };

            if heap.len() < sample_size {
                heap.push(weighted_item);
            } else if let Some(min_item) = heap.peek() {
                if key > min_item.key {
                    heap.pop();
                    heap.push(weighted_item);
                }
            }
        }
    }

    // Extract items from heap
    let mut sample = Vec::with_capacity(heap.len());
    while let Some(item) = heap.pop() {
        sample.push(item.item);
    }

    // Reverse to get items in the order they were selected (largest keys first)
    sample.reverse();

    Ok(sample)
}

/// Alias method for efficient repeated weighted sampling
///
/// This is useful when you need to perform many weighted samples from the same
/// distribution. The setup cost is O(n) but each sample is O(1).
#[derive(PartialEq, Debug)]
pub struct AliasMethod<T> {
    data: Vec<T>,
    prob: Vec<f64>,
    alias: Vec<usize>,
    rng: StdRng,
}

impl<T: Clone> AliasMethod<T> {
    /// Create a new alias method sampler.
    ///
    /// # Arguments
    ///
    /// * `data` - The items to sample from
    /// * `weights` - Weights for each item
    /// * `seed` - Optional seed for reproducible sampling
    pub fn new(data: Vec<T>, weights: &[f64], seed: Option<u64>) -> SamplingResult<Self> {
        if data.len() != weights.len() {
            return Err(SamplingError::MismatchedInputs);
        }

        if weights.iter().any(|&w| w < 0.0) {
            return Err(SamplingError::InvalidParameters(
                "All weights must be non-negative".to_string(),
            ));
        }

        let total_weight: f64 = weights.iter().sum();
        if total_weight <= 0.0 {
            return Err(SamplingError::InvalidParameters(
                "At least one weight must be positive".to_string(),
            ));
        }

        let n = data.len();
        let mut prob = vec![0.0; n];
        let mut alias = vec![0; n];

        // Normalize weights
        let normalized_weights: Vec<f64> = weights
            .iter()
            .map(|&w| w * n as f64 / total_weight)
            .collect();

        // Separate items into small and large based on normalized probability
        let mut small = Vec::new();
        let mut large = Vec::new();

        for (i, &p) in normalized_weights.iter().enumerate() {
            if p < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        // Build alias table
        while !small.is_empty() && !large.is_empty() {
            let small_idx = small.pop().unwrap();
            let large_idx = large.pop().unwrap();

            prob[small_idx] = normalized_weights[small_idx];
            alias[small_idx] = large_idx;

            let remaining = normalized_weights[large_idx] - (1.0 - normalized_weights[small_idx]);

            if remaining < 1.0 {
                small.push(large_idx);
            } else {
                large.push(large_idx);
            }
        }

        // Handle remaining items
        while !large.is_empty() {
            let idx = large.pop().unwrap();
            prob[idx] = 1.0;
        }

        while !small.is_empty() {
            let idx = small.pop().unwrap();
            prob[idx] = 1.0;
        }

        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Ok(AliasMethod {
            data,
            prob,
            alias,
            rng,
        })
    }

    /// Sample a single item using the alias method.
    pub fn sample(&mut self) -> &T {
        let n = self.data.len();
        let i = self.rng.gen_range(0..n);
        let coin_flip: f64 = self.rng.gen();

        if coin_flip < self.prob[i] {
            &self.data[i]
        } else {
            &self.data[self.alias[i]]
        }
    }

    /// Sample multiple items using the alias method.
    pub fn sample_multiple(&mut self, count: usize) -> Vec<T> {
        (0..count).map(|_| self.sample().clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_sample_with_replacement() {
        let data = vec!["a", "b", "c"];
        let weights = vec![1.0, 2.0, 3.0];

        let sample = weighted_sample(&data, &weights, 100, true, Some(42)).unwrap();
        assert_eq!(sample.len(), 100);

        // Count occurrences (higher weights should appear more often)
        let count_a = sample.iter().filter(|&&x| x == "a").count();
        let count_b = sample.iter().filter(|&&x| x == "b").count();
        let count_c = sample.iter().filter(|&&x| x == "c").count();

        // "c" should appear most often, "a" least often
        assert!(count_c > count_b);
        assert!(count_b > count_a);
    }

    #[test]
    fn test_weighted_sample_without_replacement() {
        let data = vec![1, 2, 3, 4, 5];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let sample = weighted_sample(&data, &weights, 3, false, Some(42)).unwrap();

        assert_eq!(sample.len(), 3);

        // All elements should be unique (no replacement)
        let mut sorted_sample = sample.clone();
        sorted_sample.sort();
        sorted_sample.dedup();
        assert_eq!(sorted_sample.len(), sample.len());

        // All elements should be from original data
        for &item in &sample {
            assert!(data.contains(&item));
        }
    }

    #[test]
    fn test_weighted_sample_errors() {
        let data = vec![1, 2, 3];
        let weights = vec![1.0, 2.0]; // Mismatched lengths

        assert_eq!(
            weighted_sample(&data, &weights, 2, true, Some(42)),
            Err(SamplingError::MismatchedInputs)
        );

        let weights = vec![1.0, 2.0, -1.0]; // Negative weight
        assert_eq!(
            weighted_sample(&data, &weights, 2, true, Some(42)),
            Err(SamplingError::InvalidParameters(
                "All weights must be non-negative".to_string()
            ))
        );

        let weights = vec![0.0, 0.0, 0.0]; // All zero weights
        assert_eq!(
            weighted_sample(&data, &weights, 2, true, Some(42)),
            Err(SamplingError::InvalidParameters(
                "At least one weight must be positive".to_string()
            ))
        );

        let weights = vec![1.0, 2.0, 3.0];
        assert_eq!(
            weighted_sample(&data, &weights, 5, false, Some(42)), // Sample size too large
            Err(SamplingError::SampleSizeTooLarge)
        );
    }

    #[test]
    fn test_weighted_sample_reproducible() {
        let data = vec!["x", "y", "z"];
        let weights = vec![1.0, 2.0, 3.0];

        let sample1 = weighted_sample(&data, &weights, 10, true, Some(42)).unwrap();
        let sample2 = weighted_sample(&data, &weights, 10, true, Some(42)).unwrap();

        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_alias_method() {
        let data = vec!["rare", "common", "very_common"];
        let weights = vec![0.1, 1.0, 5.0];

        let mut alias_sampler = AliasMethod::new(data, &weights, Some(42)).unwrap();

        let samples = alias_sampler.sample_multiple(1000);
        assert_eq!(samples.len(), 1000);

        let count_rare = samples.iter().filter(|&&x| x == "rare").count();
        let count_common = samples.iter().filter(|&&x| x == "common").count();
        let count_very_common = samples.iter().filter(|&&x| x == "very_common").count();

        // "very_common" should appear most often
        assert!(count_very_common > count_common);
        assert!(count_common > count_rare);
    }

    #[test]
    fn test_alias_method_errors() {
        let data = vec![1, 2, 3];
        let weights = vec![1.0, 2.0]; // Mismatched lengths

        assert_eq!(
            AliasMethod::new(data, &weights, Some(42)),
            Err(SamplingError::MismatchedInputs)
        );
    }

    #[test]
    fn test_weighted_sample_edge_cases() {
        // Single element
        let data = vec!["only"];
        let weights = vec![1.0];

        let sample = weighted_sample(&data, &weights, 3, true, Some(42)).unwrap();
        assert_eq!(sample.len(), 3);
        assert!(sample.iter().all(|&x| x == "only"));

        // Zero sample size
        let sample = weighted_sample(&data, &weights, 0, true, Some(42)).unwrap();
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_extreme_weights() {
        let data = vec!["tiny", "huge"];
        let weights = vec![0.001, 1000.0];

        let sample = weighted_sample(&data, &weights, 100, true, Some(42)).unwrap();

        let count_tiny = sample.iter().filter(|&&x| x == "tiny").count();
        let count_huge = sample.iter().filter(|&&x| x == "huge").count();

        // "huge" should dominate
        assert!(count_huge > count_tiny);
        assert!(count_huge > 90); // Should be close to 100
    }
}
