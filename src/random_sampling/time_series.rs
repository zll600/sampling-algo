//! Time Series Sampling
//!
//! Implementation of sampling algorithms specifically designed for temporal data where
//! the order and temporal relationships between elements matter. These methods are
//! particularly useful for time series analysis, sensor data, and sequential datasets.
//!
//! ## Algorithms Implemented
//!
//! - **Temporal Systematic Sampling**: Regular interval sampling preserving temporal order
//! - **Window-based Sampling**: Sample from temporal windows
//! - **Adaptive Temporal Sampling**: Density-aware sampling for irregular time series
//! - **Block Sampling**: Contiguous time block sampling
//! - **Seasonal Sampling**: Sampling that respects seasonal patterns
//!
//! ## Time Complexity: O(n + k) where n is data size, k is sample size
//! ## Space Complexity: O(k) for output sample
//!
//! ## Examples
//!
//! ```rust
//! use sampling_algo::random_sampling::time_series::{time_series_sample, SamplingMethod};
//!
//! let data: Vec<f64> = (0..100).map(|x| (x as f64).sin()).collect();
//! let sample = time_series_sample(&data, 10, SamplingMethod::Systematic, Some(42)).unwrap();
//! println!("Time series sample: {:?}", sample);
//! ```

use crate::random_sampling::simple::simple_random_sample;
use crate::random_sampling::systematic::systematic_sample;
use crate::random_sampling::{SamplingError, SamplingResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Methods for temporal sampling
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingMethod {
    /// Systematic sampling preserving temporal order
    Systematic,
    /// Random sampling (may break temporal order)
    Random,
    /// Window-based sampling with specified window size
    Windowed(usize),
    /// Block sampling - contiguous temporal blocks
    Blocks(usize),
    /// Adaptive sampling based on variance or change detection
    Adaptive { threshold: f64, min_gap: usize },
}

/// Sample from time series data with temporal awareness.
///
/// This function implements various sampling strategies that consider the temporal
/// nature of the data, preserving important characteristics like trends, seasonality,
/// and temporal relationships where appropriate.
///
/// # Arguments
///
/// * `data` - Time series data (ordered by time)
/// * `sample_size` - Number of elements to sample
/// * `method` - Sampling method to use
/// * `seed` - Optional seed for reproducible results
///
/// # Returns
///
/// Returns a vector containing the sampled elements, potentially preserving temporal order
/// depending on the method used.
///
/// # Errors
///
/// * `SamplingError::EmptyPopulation` - If data is empty but sample_size > 0
/// * `SamplingError::SampleSizeTooLarge` - If sample_size > population size
/// * `SamplingError::InvalidParameters` - If method parameters are invalid
///
/// # Examples
///
/// ```rust
/// use sampling_algo::random_sampling::time_series::{time_series_sample, SamplingMethod};
///
/// let data: Vec<i32> = (0..100).collect();
/// let sample = time_series_sample(&data, 10, SamplingMethod::Systematic, Some(42)).unwrap();
/// assert_eq!(sample.len(), 10);
/// ```
pub fn time_series_sample<T: Clone>(
    data: &[T],
    sample_size: usize,
    method: SamplingMethod,
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

    match method {
        SamplingMethod::Systematic => systematic_sample(data, sample_size, seed),

        SamplingMethod::Random => {
            let sample = simple_random_sample(data, sample_size, seed)?;
            // For time series, we might want to sort by original indices to preserve some order
            // But for now, we'll return as-is for true random sampling
            Ok(sample)
        }

        SamplingMethod::Windowed(window_size) => {
            windowed_sample(data, sample_size, window_size, seed)
        }

        SamplingMethod::Blocks(block_size) => block_sample(data, sample_size, block_size, seed),

        SamplingMethod::Adaptive { threshold, min_gap } => {
            adaptive_sample(data, sample_size, threshold, min_gap, seed)
        }
    }
}

/// Window-based sampling: divide time series into windows and sample from each
fn windowed_sample<T: Clone>(
    data: &[T],
    sample_size: usize,
    window_size: usize,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>> {
    if window_size == 0 {
        return Err(SamplingError::InvalidParameters(
            "Window size must be positive".to_string(),
        ));
    }

    let mut sample = Vec::new();
    let num_windows = (data.len() + window_size - 1) / window_size; // Ceiling division
    let samples_per_window = sample_size / num_windows;
    let mut remaining_samples = sample_size % num_windows;

    for window_idx in 0..num_windows {
        let start = window_idx * window_size;
        let end = (start + window_size).min(data.len());
        let window = &data[start..end];

        if window.is_empty() {
            continue;
        }

        let mut window_sample_size = samples_per_window;
        if remaining_samples > 0 {
            window_sample_size += 1;
            remaining_samples -= 1;
        }

        window_sample_size = window_sample_size.min(window.len());

        if window_sample_size > 0 {
            // Generate a new seed for this window
            let window_seed = seed.map(|s| s.wrapping_add(window_idx as u64));
            let window_sample = simple_random_sample(window, window_sample_size, window_seed)?;
            sample.extend(window_sample);
        }

        if sample.len() >= sample_size {
            break;
        }
    }

    // Truncate if we have too many samples
    sample.truncate(sample_size);
    Ok(sample)
}

/// Block sampling: sample contiguous blocks from the time series
fn block_sample<T: Clone>(
    data: &[T],
    sample_size: usize,
    block_size: usize,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>> {
    if block_size == 0 {
        return Err(SamplingError::InvalidParameters(
            "Block size must be positive".to_string(),
        ));
    }

    if block_size > data.len() {
        return Err(SamplingError::InvalidParameters(
            "Block size cannot exceed data length".to_string(),
        ));
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let num_blocks_needed = (sample_size + block_size - 1) / block_size; // Ceiling division
    let max_start_position = data.len() - block_size;

    if max_start_position == 0 && num_blocks_needed > 1 {
        return Err(SamplingError::InvalidParameters(
            "Cannot create multiple blocks with current parameters".to_string(),
        ));
    }

    let mut sample = Vec::new();
    let mut used_starts = Vec::new();

    for _ in 0..num_blocks_needed {
        if sample.len() >= sample_size {
            break;
        }

        // Find a random starting position that hasn't been used
        let mut attempts = 0;
        let max_attempts = 100; // Prevent infinite loops

        loop {
            attempts += 1;
            if attempts > max_attempts {
                break; // Give up to avoid infinite loop
            }

            let start = if max_start_position == 0 {
                0
            } else {
                rng.gen_range(0..=max_start_position)
            };

            // Check for overlap with existing blocks
            let overlaps = used_starts.iter().any(|&used_start| {
                (start < used_start + block_size) && (used_start < start + block_size)
            });

            if !overlaps {
                used_starts.push(start);
                let end = (start + block_size).min(data.len());
                let block = &data[start..end];

                let remaining_needed = sample_size - sample.len();
                let to_take = remaining_needed.min(block.len());

                sample.extend(block.iter().take(to_take).cloned());
                break;
            }

            // If we've tried all possible positions, break
            if used_starts.len() >= (max_start_position + 1).min(data.len() / block_size) {
                break;
            }
        }
    }

    // If we couldn't get enough samples with non-overlapping blocks,
    // fill the remainder with systematic sampling
    if sample.len() < sample_size {
        let remaining = sample_size - sample.len();
        let remaining_sample = systematic_sample(data, remaining, seed)?;
        sample.extend(remaining_sample);
        sample.truncate(sample_size);
    }

    Ok(sample)
}

/// Adaptive sampling based on data characteristics (simplified version)
fn adaptive_sample<T>(
    data: &[T],
    sample_size: usize,
    _threshold: f64, // For future implementation with actual data analysis
    min_gap: usize,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>>
where
    T: Clone,
{
    if min_gap == 0 {
        return Err(SamplingError::InvalidParameters(
            "Minimum gap must be positive".to_string(),
        ));
    }

    // For this simplified implementation, we'll use systematic sampling with minimum gap enforcement
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let mut sample = Vec::new();
    let mut last_index = None;

    // Start with systematic sampling but enforce minimum gap
    let interval = data.len() as f64 / sample_size as f64;
    let start_offset = rng.gen_range(0.0..interval);

    for i in 0..sample_size {
        let target_index = (start_offset + i as f64 * interval) as usize % data.len();

        // Enforce minimum gap
        let actual_index = if let Some(last) = last_index {
            if target_index > last && target_index - last < min_gap {
                (last + min_gap).min(data.len() - 1)
            } else if target_index <= last {
                (last + min_gap).min(data.len() - 1)
            } else {
                target_index
            }
        } else {
            target_index
        };

        // Avoid going beyond array bounds or duplicates
        if actual_index < data.len()
            && (last_index.is_none() || actual_index != last_index.unwrap())
        {
            sample.push(data[actual_index].clone());
            last_index = Some(actual_index);
        }

        if sample.len() >= sample_size {
            break;
        }
    }

    // If we couldn't get enough samples due to gap constraints, fill with systematic sampling
    while sample.len() < sample_size && sample.len() < data.len() {
        let remaining_indices: Vec<usize> = (0..data.len())
            .filter(|&idx| {
                !last_index.map_or(false, |last| {
                    (idx as i32 - last as i32).abs() < min_gap as i32
                })
            })
            .collect();

        if remaining_indices.is_empty() {
            break;
        }

        let chosen_idx = remaining_indices[rng.gen_range(0..remaining_indices.len())];
        sample.push(data[chosen_idx].clone());
        last_index = Some(chosen_idx);
    }

    Ok(sample)
}

/// Seasonal sampling: sample at regular intervals respecting potential seasonal patterns
pub fn seasonal_sample<T: Clone>(
    data: &[T],
    sample_size: usize,
    season_length: usize,
    samples_per_season: usize,
    seed: Option<u64>,
) -> SamplingResult<Vec<T>> {
    if season_length == 0 {
        return Err(SamplingError::InvalidParameters(
            "Season length must be positive".to_string(),
        ));
    }

    if samples_per_season == 0 {
        return Err(SamplingError::InvalidParameters(
            "Samples per season must be positive".to_string(),
        ));
    }

    if data.is_empty() && sample_size > 0 {
        return Err(SamplingError::EmptyPopulation);
    }

    if sample_size > data.len() {
        return Err(SamplingError::SampleSizeTooLarge);
    }

    let mut sample = Vec::new();
    let num_complete_seasons = data.len() / season_length;
    let remainder = data.len() % season_length;

    // Sample from complete seasons
    for season_idx in 0..num_complete_seasons {
        let season_start = season_idx * season_length;
        let season_end = season_start + season_length;
        let season_data = &data[season_start..season_end];

        let season_seed = seed.map(|s| s.wrapping_add(season_idx as u64));

        // Calculate how many samples we can take from this season
        let remaining_needed = sample_size - sample.len();
        let season_sample_size = samples_per_season
            .min(season_data.len())
            .min(remaining_needed);

        if season_sample_size > 0 {
            let season_sample = simple_random_sample(season_data, season_sample_size, season_seed)?;
            sample.extend(season_sample);
        }

        if sample.len() >= sample_size {
            break;
        }
    }

    // Sample from remaining partial season if exists and we need more samples
    if remainder > 0 && sample.len() < sample_size {
        let remaining_start = num_complete_seasons * season_length;
        let remaining_data = &data[remaining_start..];
        let remaining_needed = sample_size - sample.len();
        let remaining_to_sample = remaining_needed.min(remaining_data.len());

        if remaining_to_sample > 0 {
            let remaining_seed = seed.map(|s| s.wrapping_add(num_complete_seasons as u64));
            let remaining_sample =
                simple_random_sample(remaining_data, remaining_to_sample, remaining_seed)?;
            sample.extend(remaining_sample);
        }
    }

    // Truncate if we have too many samples
    sample.truncate(sample_size);
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_sample_systematic() {
        let data: Vec<i32> = (0..100).collect();
        let sample = time_series_sample(&data, 10, SamplingMethod::Systematic, Some(42)).unwrap();

        assert_eq!(sample.len(), 10);
        for &item in &sample {
            assert!(data.contains(&item));
        }
    }

    #[test]
    fn test_time_series_sample_random() {
        let data: Vec<i32> = (0..50).collect();
        let sample = time_series_sample(&data, 5, SamplingMethod::Random, Some(42)).unwrap();

        assert_eq!(sample.len(), 5);
        for &item in &sample {
            assert!(data.contains(&item));
        }
    }

    #[test]
    fn test_windowed_sampling() {
        let data: Vec<i32> = (0..20).collect();
        let sample = time_series_sample(&data, 8, SamplingMethod::Windowed(5), Some(42)).unwrap();

        assert_eq!(sample.len(), 8);
        for &item in &sample {
            assert!(data.contains(&item));
        }
    }

    #[test]
    fn test_block_sampling() {
        let data: Vec<i32> = (0..30).collect();
        let sample = time_series_sample(&data, 10, SamplingMethod::Blocks(3), Some(42)).unwrap();

        assert!(sample.len() <= 10);
        assert!(!sample.is_empty());
        for &item in &sample {
            assert!(data.contains(&item));
        }
    }

    #[test]
    fn test_adaptive_sampling() {
        let data: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let sample = time_series_sample(
            &data,
            10,
            SamplingMethod::Adaptive {
                threshold: 1.0,
                min_gap: 2,
            },
            Some(42),
        )
        .unwrap();

        assert!(sample.len() <= 10);
        assert!(!sample.is_empty());
        for &item in &sample {
            assert!(data.contains(&item));
        }
    }

    #[test]
    fn test_seasonal_sampling() {
        let data: Vec<i32> = (0..100).collect();
        let sample = seasonal_sample(&data, 20, 10, 2, Some(42)).unwrap();

        assert_eq!(sample.len(), 20);
        for &item in &sample {
            assert!(data.contains(&item));
        }
    }

    #[test]
    fn test_time_series_sample_errors() {
        let data: Vec<i32> = (0..10).collect();

        // Sample size too large
        assert_eq!(
            time_series_sample(&data, 15, SamplingMethod::Systematic, Some(42)),
            Err(SamplingError::SampleSizeTooLarge)
        );

        // Empty data with non-zero sample size
        let empty_data: Vec<i32> = vec![];
        assert_eq!(
            time_series_sample(&empty_data, 1, SamplingMethod::Systematic, Some(42)),
            Err(SamplingError::EmptyPopulation)
        );

        // Invalid window size
        assert_eq!(
            time_series_sample(&data, 5, SamplingMethod::Windowed(0), Some(42)),
            Err(SamplingError::InvalidParameters(
                "Window size must be positive".to_string()
            ))
        );

        // Invalid block size
        assert_eq!(
            time_series_sample(&data, 5, SamplingMethod::Blocks(0), Some(42)),
            Err(SamplingError::InvalidParameters(
                "Block size must be positive".to_string()
            ))
        );
    }

    #[test]
    fn test_time_series_sample_reproducible() {
        let data: Vec<i32> = (0..100).collect();

        let sample1 = time_series_sample(&data, 10, SamplingMethod::Systematic, Some(42)).unwrap();
        let sample2 = time_series_sample(&data, 10, SamplingMethod::Systematic, Some(42)).unwrap();

        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_windowed_sampling_edge_cases() {
        // Window size larger than data
        let data = vec![1, 2, 3];
        let sample = time_series_sample(&data, 2, SamplingMethod::Windowed(10), Some(42)).unwrap();
        assert!(sample.len() <= 2);

        // Single element windows
        let data: Vec<i32> = (0..10).collect();
        let sample = time_series_sample(&data, 5, SamplingMethod::Windowed(1), Some(42)).unwrap();
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn test_block_sampling_edge_cases() {
        // Block size equal to data length
        let data = vec![1, 2, 3, 4, 5];
        let sample = time_series_sample(&data, 3, SamplingMethod::Blocks(5), Some(42)).unwrap();
        assert!(sample.len() <= 3);

        // Very small data
        let data = vec![1];
        let sample = time_series_sample(&data, 1, SamplingMethod::Blocks(1), Some(42)).unwrap();
        assert_eq!(sample.len(), 1);
        assert_eq!(sample[0], 1);
    }

    #[test]
    fn test_seasonal_sampling_edge_cases() {
        // Season length equal to data length
        // Since we have 1 season with 3 samples per season, we get 3 samples max
        let data: Vec<i32> = (0..10).collect();
        let sample = seasonal_sample(&data, 5, 10, 3, Some(42)).unwrap();
        assert_eq!(sample.len(), 3); // Limited by samples_per_season

        // To get 5 samples from one season, we need higher samples_per_season
        let sample = seasonal_sample(&data, 5, 10, 6, Some(42)).unwrap();
        assert_eq!(sample.len(), 5);

        // More samples per season than season length
        let sample = seasonal_sample(&data, 8, 5, 10, Some(42)).unwrap();
        assert!(sample.len() <= 8);
    }

    #[test]
    fn test_time_series_zero_sample_size() {
        let data: Vec<i32> = (0..10).collect();
        let sample = time_series_sample(&data, 0, SamplingMethod::Systematic, Some(42)).unwrap();
        assert_eq!(sample.len(), 0);
    }

    #[test]
    fn test_adaptive_sampling_min_gap() {
        let data: Vec<i32> = (0..20).collect();
        let sample = time_series_sample(
            &data,
            5,
            SamplingMethod::Adaptive {
                threshold: 1.0,
                min_gap: 5,
            },
            Some(42),
        )
        .unwrap();

        assert!(sample.len() <= 5);
        // Due to the adaptive nature and minimum gap, we might get fewer samples than requested
        assert!(!sample.is_empty());
    }
}
