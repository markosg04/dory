//! Unified imports for recursion

#[cfg(feature = "recursion")]
pub use jolt_optimizations::ExponentiationSteps;

#[cfg(not(feature = "recursion"))]
#[derive(Debug, Clone, Default)]
pub struct ExponentiationSteps;

/// Type alias for GT steps
pub type RecursionOps = Option<Vec<ExponentiationSteps>>;
