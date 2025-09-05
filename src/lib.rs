//! # Sampling Algorithms
//! 
//! A comprehensive library of sampling algorithms implemented in Rust for educational purposes.
//! This library provides implementations of various sampling techniques commonly used in
//! statistics, data analysis, and machine learning.
//! 
//! ## Modules
//! 
//! - [`random_sampling`] - Classical random sampling algorithms
//! - [`time_based_sampling`] - Time-aware sampling for temporal data (planned)
//! 
//! ## Examples
//! 
//! ```rust
//! use sampling_algo::random_sampling::simple::simple_random_sample;
//! 
//! let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//! let sample = simple_random_sample(&data, 3, Some(42));
//! println!("Sample: {:?}", sample);
//! ```

pub mod random_sampling;

pub use random_sampling::*;