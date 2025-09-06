//! Random Sampling Algorithms
//!
//! This module contains implementations of various random sampling techniques:
//!
//! - **Simple Random Sampling**: Basic random sampling without replacement
//! - **Stratified Sampling**: Sampling that preserves population strata proportions
//! - **Systematic Sampling**: Sampling at regular intervals
//! - **Reservoir Sampling**: Streaming algorithm for unknown population size
//! - **Weighted Sampling**: Probability-proportional-to-size sampling
//! - **Time Series Sampling**: Temporal-aware sampling methods
//!
//! Each algorithm is implemented from scratch to demonstrate the underlying mechanics
//! without relying on high-level library abstractions.

pub mod reservoir;
pub mod simple;
pub mod stratified;
pub mod systematic;
// pub mod weighted;
// pub mod time_series;

// Re-export commonly used functions
pub use reservoir::*;
pub use simple::*;
pub use stratified::*;
pub use systematic::*;
// pub use weighted::*;
// pub use time_series::*;

/// Common error types for sampling algorithms
#[derive(Debug, PartialEq, Clone)]
pub enum SamplingError {
    /// Sample size is negative
    NegativeSampleSize,
    /// Sample size exceeds population size (for sampling without replacement)
    SampleSizeTooLarge,
    /// Invalid parameters provided
    InvalidParameters(String),
    /// Empty population provided
    EmptyPopulation,
    /// Mismatched input lengths
    MismatchedInputs,
}

impl std::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplingError::NegativeSampleSize => write!(f, "Sample size cannot be negative"),
            SamplingError::SampleSizeTooLarge => {
                write!(f, "Sample size cannot exceed population size")
            }
            SamplingError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            SamplingError::EmptyPopulation => write!(f, "Population cannot be empty"),
            SamplingError::MismatchedInputs => write!(f, "Input arrays have mismatched lengths"),
        }
    }
}

impl std::error::Error for SamplingError {}

pub type SamplingResult<T> = Result<T, SamplingError>;
