//! Unified imports for recursion

#[cfg(feature = "recursion")]
pub use jolt_optimizations::ExponentiationSteps;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[cfg(not(feature = "recursion"))]
#[derive(Debug, Clone, Default, CanonicalSerialize, CanonicalDeserialize)]
/// Used for recursion poly tracking
pub struct ExponentiationSteps;

/// Used in proofs to store GT exponentiation results without the full witness data.
/// In the future may contain other types as well
#[cfg(feature = "recursion")]
#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTOffloadResult {
    /// The precomputed GT exponentiation result (Fq12)
    pub result: ark_bn254::Fq12,
}

/// Dummy version for non-recursion builds
#[cfg(not(feature = "recursion"))]
#[derive(Debug, Clone, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct GTOffloadResult;

/// Type alias for GT steps (full witness data)
pub type RecursionOps = Option<Vec<ExponentiationSteps>>;

/// Type alias for GT offload results (lightweight proof data)
pub type GTOffloadResults = Option<Vec<GTOffloadResult>>;
