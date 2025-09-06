//! Stratified Sampling
//!
//! Implementation of stratified sampling that divides the population into homogeneous
//! subgroups (strata) and samples from each stratum. This ensures representation from
//! all subgroups and can reduce sampling variance compared to simple random sampling.
//!
//! ## Algorithm
//!
//! 1. Divide population into strata based on provided grouping
//! 2. Determine allocation (proportional or equal) for each stratum
//! 3. Apply simple random sampling within each stratum
//! 4. Combine samples from all strata
//!
//! ## Time Complexity: O(n + k*log(k)) where n is population size, k is sample size
//! ## Space Complexity: O(n + k) for stratum organization and output
//!
//! ## Examples
//!
//! ```rust
//! use sampling_algo::random_sampling::stratified::{stratified_sample, AllocationMethod};
//!
//! let data = vec![1, 2, 3, 4, 5, 6];
//! let strata = vec!["A", "A", "B", "B", "C", "C"];
//! let sample = stratified_sample(&data, &strata, 4, AllocationMethod::Proportional, Some(42)).unwrap();
//! println!("Stratified sample: {:?}", sample);
//! ```

use crate::random_sampling::simple::simple_random_sample;
use crate::random_sampling::{SamplingError, SamplingResult};
use std::collections::HashMap;

/// Method for allocating sample sizes to strata
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationMethod {
    /// Allocate proportionally to stratum size (maintains population proportions)
    Proportional,
    /// Allocate equally across all strata (uniform representation)
    Equal,
    /// Optimal allocation for minimizing variance (requires stratum variances)
    Optimal(Vec<f64>), // stratum standard deviations
}

/// Perform stratified sampling.
///
/// This function divides the population into strata and samples from each stratum
/// according to the specified allocation method. This ensures representation from
/// all subgroups and can provide more precise estimates than simple random sampling.
///
/// # Arguments
///
/// * `data` - The population to sample from
/// * `strata` - Stratum labels corresponding to each element in data
/// * `sample_size` - Total number of elements to sample across all strata
/// * `allocation` - Method for allocating samples to strata
/// * `seed` - Optional seed for reproducible results
///
/// # Returns
///
/// Returns a vector containing the sampled elements from all strata.
///
/// # Errors
///
/// * `SamplingError::MismatchedInputs` - If data and strata have different lengths
/// * `SamplingError::EmptyPopulation` - If data is empty but sample_size > 0
/// * `SamplingError::SampleSizeTooLarge` - If sample_size exceeds population size
/// * `SamplingError::InvalidParameters` - If optimal allocation parameters are invalid
///
/// # Examples
///
/// ```rust
/// use sampling_algo::random_sampling::stratified::{stratified_sample, AllocationMethod};
///
/// let data = vec![10, 20, 30, 40, 50, 60];
/// let strata = vec!["low", "low", "mid", "mid", "high", "high"];
/// let sample = stratified_sample(&data, &strata, 3, AllocationMethod::Equal, Some(42)).unwrap();
/// assert_eq!(sample.len(), 3);
/// ```
pub fn stratified_sample<T, S>(
    data: &[T],
    strata: &[S],
    sample_size: usize,
    allocation: AllocationMethod,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>>
where
    T: Clone,
    S: std::hash::Hash + Eq + Clone,
{
    // Input validation
    if data.len() != strata.len() {
        return Err(SamplingError::MismatchedInputs);
    }

    if data.is_empty() && sample_size > 0 {
        return Err(SamplingError::EmptyPopulation);
    }

    if sample_size > data.len() {
        return Err(SamplingError::SampleSizeTooLarge);
    }

    if sample_size == 0 {
        return Ok(Vec::new());
    }

    // Group data by strata
    let mut strata_map: HashMap<S, Vec<T>> = HashMap::new();
    for (item, stratum) in data.iter().zip(strata.iter()) {
        strata_map
            .entry(stratum.clone())
            .or_insert_with(Vec::new)
            .push(item.clone());
    }

    if strata_map.is_empty() {
        return Ok(Vec::new());
    }

    // Calculate allocation for each stratum
    let allocations = calculate_allocations(&strata_map, sample_size, &allocation)?;

    let mut final_sample = Vec::new();

    // Sample from each stratum
    for (stratum_key, stratum_data) in strata_map {
        if let Some(&stratum_allocation) = allocations.get(&stratum_key) {
            if stratum_allocation > 0 && !stratum_data.is_empty() {
                // Generate a new seed for this stratum to ensure randomness
                let stratum_seed = seed.map(|s| s.wrapping_add(hash_stratum(&stratum_key)));

                let stratum_sample = simple_random_sample(
                    &stratum_data,
                    stratum_allocation.min(stratum_data.len()),
                    stratum_seed,
                )?;

                final_sample.extend(stratum_sample);
            }
        }
    }

    // If we have fewer samples than requested due to small strata,
    // we could implement additional logic here, but for now we return what we have

    Ok(final_sample)
}

/// Calculate allocation sizes for each stratum based on the allocation method
fn calculate_allocations<S>(
    strata_map: &HashMap<S, Vec<impl Clone>>,
    sample_size: usize,
    allocation: &AllocationMethod,
) -> SamplingResult<HashMap<S, usize>>
where
    S: std::hash::Hash + Eq + Clone,
{
    let mut allocations = HashMap::new();
    let total_population = strata_map.values().map(|v| v.len()).sum::<usize>();

    match allocation {
        AllocationMethod::Proportional => {
            let mut total_allocated = 0;
            let strata_count = strata_map.len();
            let mut remaining_strata = strata_count;

            for (stratum, stratum_data) in strata_map {
                remaining_strata -= 1;

                if total_population == 0 {
                    allocations.insert(stratum.clone(), 0);
                    continue;
                }

                // Calculate proportional allocation
                let proportion = stratum_data.len() as f64 / total_population as f64;
                let mut stratum_allocation = if remaining_strata == 0 {
                    // For the last stratum, allocate remaining samples
                    sample_size - total_allocated
                } else {
                    (proportion * sample_size as f64).round() as usize
                };

                // Ensure we don't exceed the stratum size
                stratum_allocation = stratum_allocation.min(stratum_data.len());

                allocations.insert(stratum.clone(), stratum_allocation);
                total_allocated += stratum_allocation;
            }
        }

        AllocationMethod::Equal => {
            let strata_count = strata_map.len();
            let base_allocation = sample_size / strata_count;
            let remainder = sample_size % strata_count;

            let mut extra_assigned = 0;
            for (_i, (stratum, stratum_data)) in strata_map.iter().enumerate() {
                let mut stratum_allocation = base_allocation;

                // Distribute remainder among first few strata
                if extra_assigned < remainder {
                    stratum_allocation += 1;
                    extra_assigned += 1;
                }

                // Ensure we don't exceed the stratum size
                stratum_allocation = stratum_allocation.min(stratum_data.len());

                allocations.insert(stratum.clone(), stratum_allocation);
            }
        }

        AllocationMethod::Optimal(std_devs) => {
            if std_devs.len() != strata_map.len() {
                return Err(SamplingError::InvalidParameters(
                    "Number of standard deviations must match number of strata".to_string(),
                ));
            }

            // Neyman allocation: n_h = n * (N_h * σ_h) / Σ(N_i * σ_i)
            let strata_keys: Vec<_> = strata_map.keys().collect();
            let mut weighted_sizes = Vec::new();
            let mut total_weighted = 0.0;

            for (i, stratum) in strata_keys.iter().enumerate() {
                if let Some(stratum_data) = strata_map.get(*stratum) {
                    let weighted_size = stratum_data.len() as f64 * std_devs[i];
                    weighted_sizes.push(weighted_size);
                    total_weighted += weighted_size;
                }
            }

            if total_weighted == 0.0 {
                return Err(SamplingError::InvalidParameters(
                    "Total weighted size cannot be zero".to_string(),
                ));
            }

            for (i, stratum) in strata_keys.iter().enumerate() {
                if let Some(stratum_data) = strata_map.get(*stratum) {
                    let proportion = weighted_sizes[i] / total_weighted;
                    let mut stratum_allocation = (proportion * sample_size as f64).round() as usize;
                    stratum_allocation = stratum_allocation.min(stratum_data.len());

                    allocations.insert((*stratum).clone(), stratum_allocation);
                }
            }
        }
    }

    Ok(allocations)
}

/// Simple hash function for generating stratum-specific seeds
fn hash_stratum<S: std::hash::Hash>(stratum: &S) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    stratum.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratified_sample_proportional() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let strata = vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"];

        let sample =
            stratified_sample(&data, &strata, 6, AllocationMethod::Proportional, Some(42)).unwrap();

        assert_eq!(sample.len(), 6);

        // Check that all sampled elements are from original data
        for item in &sample {
            assert!(data.contains(item));
        }
    }

    #[test]
    fn test_stratified_sample_equal() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let strata = vec!["A", "A", "B", "B", "C", "C"];

        let sample =
            stratified_sample(&data, &strata, 3, AllocationMethod::Equal, Some(42)).unwrap();

        assert_eq!(sample.len(), 3);

        for item in &sample {
            assert!(data.contains(item));
        }
    }

    #[test]
    fn test_stratified_sample_optimal() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let strata = vec!["A", "A", "B", "B", "C", "C"];
        let std_devs = vec![1.0, 2.0, 1.5]; // Standard deviations for strata A, B, C

        let sample = stratified_sample(
            &data,
            &strata,
            4,
            AllocationMethod::Optimal(std_devs),
            Some(42),
        )
        .unwrap();

        assert!(sample.len() <= 4); // May be less due to rounding and constraints

        for item in &sample {
            assert!(data.contains(item));
        }
    }

    #[test]
    fn test_stratified_sample_errors() {
        let data = vec![1, 2, 3];
        let strata = vec!["A", "B"]; // Mismatched length

        assert_eq!(
            stratified_sample(&data, &strata, 2, AllocationMethod::Equal, Some(42)),
            Err(SamplingError::MismatchedInputs)
        );

        let data = vec![1, 2, 3];
        let strata = vec!["A", "B", "C"];

        assert_eq!(
            stratified_sample(&data, &strata, 5, AllocationMethod::Equal, Some(42)),
            Err(SamplingError::SampleSizeTooLarge)
        );
    }

    #[test]
    fn test_stratified_sample_empty() {
        let data: Vec<i32> = vec![];
        let strata: Vec<&str> = vec![];

        let sample =
            stratified_sample(&data, &strata, 0, AllocationMethod::Equal, Some(42)).unwrap();
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_stratified_sample_reproducible() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let strata = vec!["A", "A", "B", "B", "C", "C", "D", "D"];

        let mut sample1 =
            stratified_sample(&data, &strata, 4, AllocationMethod::Proportional, Some(42)).unwrap();
        let mut sample2 =
            stratified_sample(&data, &strata, 4, AllocationMethod::Proportional, Some(42)).unwrap();

        // Sort both samples before comparison since HashMap iteration order may vary
        sample1.sort();
        sample2.sort();

        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_unbalanced_strata() {
        let data = vec![1, 2, 3, 4, 5, 6, 7];
        let strata = vec!["A", "A", "A", "A", "A", "B", "C"]; // Unbalanced: 5 A's, 1 B, 1 C

        let sample =
            stratified_sample(&data, &strata, 3, AllocationMethod::Equal, Some(42)).unwrap();

        assert_eq!(sample.len(), 3);
        for item in &sample {
            assert!(data.contains(item));
        }
    }
}
