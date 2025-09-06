//! Demo of all sampling algorithms implemented in the crate.
//!
//! This example demonstrates the usage of all implemented sampling algorithms
//! with different types of data and use cases.

use sampling_algo::random_sampling::{
    reservoir::{reservoir_sample, ReservoirSampler},
    simple::simple_random_sample,
    stratified::{stratified_sample, AllocationMethod},
    systematic::systematic_sample,
    time_series::{seasonal_sample, time_series_sample, SamplingMethod},
    weighted::{weighted_sample, AliasMethod},
};

fn main() {
    println!("=== Sampling Algorithms Demo ===\n");

    // Generate example data
    let time_series_data: Vec<f64> = (0..100)
        .map(|x| (x as f64 * 0.1).sin() * 10.0 + (x as f64 * 0.05).cos() * 5.0)
        .collect();

    let categorical_data = vec![
        "Product A",
        "Product A",
        "Product A",
        "Product A",
        "Product A",
        "Product B",
        "Product B",
        "Product B",
        "Product B",
        "Product B",
        "Product C",
        "Product C",
        "Product C",
        "Product D",
        "Product D",
    ];

    let categories = vec![
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Clothing",
        "Clothing",
        "Clothing",
        "Books",
        "Books",
        "Books",
        "Home",
        "Home",
    ];

    let weights = vec![
        0.5, 2.0, 1.8, 0.8, 1.2, // Electronics (varying popularity)
        3.0, 2.5, 1.9, 2.2, 1.7, // Clothing (popular items)
        0.9, 1.1, 0.7, // Books (moderate popularity)
        1.5, 1.3, // Home (decent popularity)
    ];

    // 1. Simple Random Sampling
    println!("1. Simple Random Sampling");
    println!("   Selecting 5 products randomly from our catalog:");

    let simple_sample = simple_random_sample(&categorical_data, 5, Some(42)).unwrap();
    println!("   Sample: {:?}", simple_sample);
    println!("   Use case: Basic random selection for surveys or A/B testing\n");

    // 2. Stratified Sampling
    println!("2. Stratified Sampling");
    println!("   Ensuring representation from each product category:");

    let stratified_sample_prop = stratified_sample(
        &categorical_data,
        &categories,
        8,
        AllocationMethod::Proportional,
        Some(42),
    )
    .unwrap();

    let stratified_sample_equal = stratified_sample(
        &categorical_data,
        &categories,
        8,
        AllocationMethod::Equal,
        Some(43),
    )
    .unwrap();

    println!("   Proportional allocation: {:?}", stratified_sample_prop);
    println!("   Equal allocation: {:?}", stratified_sample_equal);
    println!("   Use case: Market research ensuring all segments are represented\n");

    // 3. Systematic Sampling
    println!("3. Systematic Sampling");
    println!("   Selecting every k-th time series data point:");

    let systematic_sample = systematic_sample(&time_series_data, 10, Some(42)).unwrap();
    println!("   Sample indices and values:");
    for (i, &value) in systematic_sample.iter().enumerate() {
        let original_idx = (i as f64 * time_series_data.len() as f64 / 10.0) as usize;
        println!("   Index {}: {:.2}", original_idx, value);
    }
    println!("   Use case: Quality control sampling, time series analysis\n");

    // 4. Reservoir Sampling
    println!("4. Reservoir Sampling");
    println!("   Sampling from a stream of unknown size:");

    // Simulate a data stream
    let stream_data: Vec<i32> = (1..=1000).collect();
    let reservoir_sample_result = reservoir_sample(stream_data.iter(), 7, Some(42)).unwrap();
    println!(
        "   Stream sample (from 1000 items): {:?}",
        reservoir_sample_result
    );

    // Interactive reservoir sampler
    let mut reservoir_sampler = ReservoirSampler::new(5, Some(43)).unwrap();

    // Simulate adding items one by one
    for i in 1..=50 {
        reservoir_sampler.add(format!("Item_{}", i));
        if i % 10 == 0 {
            let current_sample = reservoir_sampler.get_sample();
            println!("   After {} items: {:?}", i, current_sample);
        }
    }
    println!("   Use case: Processing large log files, streaming data analysis\n");

    // 5. Weighted Sampling
    println!("5. Weighted Sampling");
    println!("   Sampling based on product popularity (weights):");

    let weighted_sample_with =
        weighted_sample(&categorical_data, &weights, 8, true, Some(42)).unwrap();
    let weighted_sample_without =
        weighted_sample(&categorical_data, &weights, 5, false, Some(42)).unwrap();

    println!("   With replacement: {:?}", weighted_sample_with);
    println!("   Without replacement: {:?}", weighted_sample_without);

    // Count occurrences to show bias towards higher weights
    let mut counts = std::collections::HashMap::new();
    for item in &weighted_sample_with {
        *counts.entry(*item).or_insert(0) += 1;
    }
    println!("   Frequency count: {:?}", counts);

    // Alias Method for efficient repeated sampling
    println!("   \nUsing Alias Method for efficient repeated sampling:");
    let items = vec!["Common Item", "Rare Item", "Very Rare Item"];
    let item_weights = vec![10.0, 2.0, 0.5];

    let mut alias_sampler = AliasMethod::new(items.clone(), &item_weights, Some(44)).unwrap();
    let alias_samples = alias_sampler.sample_multiple(20);

    let mut alias_counts = std::collections::HashMap::new();
    for item in &alias_samples {
        *alias_counts.entry(*item).or_insert(0) += 1;
    }
    println!("   Alias method samples (20 draws): {:?}", alias_counts);
    println!("   Use case: Recommendation systems, importance sampling\n");

    // 6. Time Series Sampling
    println!("6. Time Series Sampling");
    println!("   Specialized sampling for temporal data:");

    let ts_systematic =
        time_series_sample(&time_series_data, 12, SamplingMethod::Systematic, Some(42)).unwrap();

    let ts_windowed = time_series_sample(
        &time_series_data,
        10,
        SamplingMethod::Windowed(20),
        Some(42),
    )
    .unwrap();

    let ts_blocks =
        time_series_sample(&time_series_data, 15, SamplingMethod::Blocks(5), Some(42)).unwrap();

    println!(
        "   Systematic sampling (12 points): [first 6] {:.2?}...",
        &ts_systematic[..6.min(ts_systematic.len())]
    );
    println!(
        "   Windowed sampling (10 points): [first 6] {:.2?}...",
        &ts_windowed[..6.min(ts_windowed.len())]
    );
    println!(
        "   Block sampling (15 points): [first 6] {:.2?}...",
        &ts_blocks[..6.min(ts_blocks.len())]
    );

    // Seasonal sampling
    let seasonal_data: Vec<i32> = (0..365).collect(); // Daily data for a year
    let seasonal_sample_result = seasonal_sample(&seasonal_data, 24, 30, 2, Some(42)).unwrap(); // 2 samples per month

    println!(
        "   Seasonal sampling (2 per month, 24 total): [first 12] {:?}...",
        &seasonal_sample_result[..12.min(seasonal_sample_result.len())]
    );
    println!("   Use case: Time series analysis, sensor data processing, financial data\n");

    // Summary
    println!("=== Summary ===");
    println!("✓ Simple Random: Unbiased, general-purpose sampling");
    println!("✓ Stratified: Ensures representation across subgroups");
    println!("✓ Systematic: Good coverage, efficient for ordered data");
    println!("✓ Reservoir: Perfect for streams and unknown-size datasets");
    println!("✓ Weighted: Probability-proportional sampling for biased selection");
    println!("✓ Time Series: Temporal-aware sampling preserving time structure");
    println!("\nAll algorithms implemented from scratch for educational purposes!");
    println!("Each algorithm has specific use cases and trade-offs in terms of");
    println!("bias, efficiency, and applicability to different data types.");
}
