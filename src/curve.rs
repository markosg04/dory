#![allow(missing_docs)]
use crate::arithmetic::*;
use crate::poly::Polynomial;
use ark_bn254::{g1, g2, Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::AdditiveGroup;
use ark_ec::{
    bn::{G1Prepared as BnG1Prepared, G2Prepared as BnG2Prepared},
    pairing::{MillerLoopOutput, Pairing as ArkPairing},
    scalar_mul::{glv::GLVConfig, wnaf::WnafContext},
    AffineRepr, CurveGroup,
};
use ark_ff::{Field as ArkField, One, PrimeField, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError};
use ark_serialize::{Read, Valid, Validate, Write};
use ark_std::rand::{rngs::StdRng, RngCore, SeedableRng};
use rayon::prelude::*;

/// Create a fixed RNG for deterministic tests
pub fn test_rng() -> StdRng {
    let seed = [
        1, 0, 0, 30, 23, 0, 0, 0, 200, 1, 0, 0, 210, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ];
    StdRng::from_seed(seed)
}

/* --------- Field trait for Fr --------------------------------------- */
impl Field for Fr {
    fn zero() -> Self {
        Zero::zero()
    }
    fn one() -> Self {
        One::one()
    }
    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }

    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }
    fn mul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
    fn inv(&self) -> Option<Self> {
        if Zero::is_zero(self) {
            None
        } else {
            Some(self.inverse().unwrap())
        }
    }
    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        Fr::rand(&mut rng)
    }

    fn from_u64(val: u64) -> Self {
        Fr::from(val)
    }

    fn from_i64(val: i64) -> Self {
        if val >= 0 {
            Fr::from(val as u64)
        } else {
            -Fr::from((-val) as u64)
        }
    }
}

/* --------- Group trait for G1Affine -------------------------------- */
impl Group for G1Affine {
    type Scalar = Fr;

    fn identity() -> Self {
        G1Affine::identity()
    }

    fn add(&self, rhs: &Self) -> Self {
        (self.into_group() + rhs.into_group()).into_affine()
    }

    fn neg(&self) -> Self {
        (-self.into_group()).into_affine()
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        self.mul_bigint((*k).into_bigint()).into_affine()
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        let mut rng = test_rng();
        G1Projective::rand(&mut rng).into_affine()
    }
}

/// G1Affine and G2Affine are the same up to alias from arkworks.
/// Hence, we have to use newType idiom here to avoid compiler conflicts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G2AffineWrapper(G2Affine);

// Implement operator traits for G2AffineWrapper
impl std::ops::Add for G2AffineWrapper {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        G2AffineWrapper((self.0.into_group() + rhs.0.into_group()).into_affine())
    }
}

impl std::ops::Add<&G2AffineWrapper> for G2AffineWrapper {
    type Output = Self;

    fn add(self, rhs: &G2AffineWrapper) -> Self::Output {
        G2AffineWrapper((self.0.into_group() + rhs.0.into_group()).into_affine())
    }
}

impl std::ops::Neg for G2AffineWrapper {
    type Output = Self;

    fn neg(self) -> Self::Output {
        G2AffineWrapper((-self.0.into_group()).into_affine())
    }
}

impl From<G2Affine> for G2AffineWrapper {
    fn from(value: G2Affine) -> Self {
        G2AffineWrapper(value)
    }
}

impl From<G2AffineWrapper> for G2Affine {
    fn from(value: G2AffineWrapper) -> Self {
        value.0
    }
}

// Implementations for ark-serialize
impl CanonicalSerialize for G2AffineWrapper {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }

    fn serialize_compressed<W: std::io::Write>(
        &self,
        writer: W,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.serialize_with_mode(writer, ark_serialize::Compress::Yes)
    }

    fn compressed_size(&self) -> usize {
        self.serialized_size(ark_serialize::Compress::Yes)
    }

    fn serialize_uncompressed<W: std::io::Write>(
        &self,
        writer: W,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.serialize_with_mode(writer, ark_serialize::Compress::No)
    }

    fn uncompressed_size(&self) -> usize {
        self.serialized_size(ark_serialize::Compress::No)
    }
}
impl CanonicalDeserialize for G2AffineWrapper {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        G2Affine::deserialize_with_mode(reader, compress, validate).map(G2AffineWrapper)
    }

    fn deserialize_compressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::Yes)
    }

    fn deserialize_compressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::Yes, Validate::No)
    }

    fn deserialize_uncompressed<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::Yes)
    }

    fn deserialize_uncompressed_unchecked<R: Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
    }
}
impl Valid for G2AffineWrapper {
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

/* --------- Group trait for G2AffineWrapper ------------------------ */
impl Group for G2AffineWrapper {
    type Scalar = Fr;

    fn identity() -> Self {
        G2AffineWrapper(G2Affine::identity())
    }

    fn add(&self, rhs: &Self) -> Self {
        G2AffineWrapper((self.0.into_group() + rhs.0.into_group()).into_affine())
    }

    fn neg(&self) -> Self {
        G2AffineWrapper((-self.0.into_group()).into_affine())
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        G2AffineWrapper(self.0.mul_bigint((*k).into_bigint()).into_affine())
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        G2AffineWrapper(G2Projective::rand(&mut rng).into_affine())
    }
}

/* --------- Group trait for Fq12 (GT) ------------------------------- */
impl Group for Fq12 {
    type Scalar = Fr;

    fn identity() -> Self {
        Self::one()
    }

    fn add(&self, rhs: &Self) -> Self {
        *self * *rhs // Multiplicative group
    }

    fn neg(&self) -> Self {
        if Zero::is_zero(self) {
            *self
        } else {
            self.inverse().unwrap()
        }
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        // We convert to BigInt representation suitable for powering
        let repr = (*k).into_bigint();
        self.pow(repr)
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // We use our own fixed RNG for testing
        let mut rng = test_rng();
        Self::rand(&mut rng)
    }
}

/* --------- lightweight Pairing wrapper ----------------------------- */
#[derive(Clone, Debug)]
pub struct ArkBn254Pairing;

impl Pairing for ArkBn254Pairing {
    type G1 = G1Affine;
    type G2 = G2AffineWrapper;
    type GT = Fq12;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        Bn254::pairing(*p, q.0).0
    }

    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        // Convert to the cached version format for consistency
        Self::multi_pair_cached(Some(ps), None, None, Some(qs), None, None)
    }

    fn multi_pair_cached(
        g1_points: Option<&[Self::G1]>,
        g1_count: Option<usize>,
        g1_cache: Option<&G1Cache>,
        g2_points: Option<&[Self::G2]>,
        g2_count: Option<usize>,
        g2_cache: Option<&G2Cache>,
    ) -> Self::GT {
        use crate::profiler::profile;
        
        profile("multi_pair_cached", || {
            match (g1_points, g1_count, g1_cache, g2_points, g2_count, g2_cache) {
                // Case 1: Both G1 and G2 use cached prepared values (fully optimized)
                (None, Some(g1_c), Some(g1_cache), None, Some(g2_c), Some(g2_cache)) => {
                    profile("multi_pair_cached::both_cached", || {
                        assert_eq!(g1_c, g2_c, "G1 and G2 counts must be equal");
                        if g1_c == 0 {
                            return Fq12::one();
                        }

                        // Extract prepared values by cloning - this is still faster than re-preparing from affine
                        let g1_prepared = profile("multi_pair_cached::clone_g1_prepared", || {
                            (0..g1_c)
                                .map(|i| g1_cache.get_prepared(i).expect("Index out of bounds in G1 cache").clone())
                                .collect::<Vec<_>>()
                        });
                        
                        let g2_prepared = profile("multi_pair_cached::clone_g2_prepared", || {
                            (0..g2_c)
                                .map(|i| g2_cache.get_prepared(i).expect("Index out of bounds in G2 cache").clone())
                                .collect::<Vec<_>>()
                        });

                        let ml_result = profile("multi_pair_cached::miller_loop", || {
                            Bn254::multi_miller_loop(g1_prepared, g2_prepared).0
                        });
                        
                        let pairing_result = profile("multi_pair_cached::final_exponentiation", || {
                            Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                                .expect("Final exponentiation should not fail")
                        });

                        pairing_result.0
                    })
                },
                
                // Case 2: G1 cached, G2 fresh points (partial optimization)
                (None, Some(g1_c), Some(g1_cache), Some(g2_points), _, _) => {
                    profile("multi_pair_cached::g1_cached_g2_fresh", || {
                        assert_eq!(g1_c, g2_points.len(), "G1 count must equal G2 points length");
                        if g1_c == 0 {
                            return Fq12::one();
                        }

                        // G1 from cache (clone), G2 fresh preparation
                        let g1_prepared = profile("multi_pair_cached::clone_g1_prepared_partial", || {
                            (0..g1_c)
                                .map(|i| g1_cache.get_prepared(i).expect("Index out of bounds in G1 cache").clone())
                                .collect::<Vec<_>>()
                        });
                        
                        let g2_prepared = profile("multi_pair_cached::prepare_g2_fresh_partial", || {
                            g2_points.par_iter().map(|q| BnG2Prepared::from(q.0)).collect::<Vec<_>>()
                        });

                        let ml_result = profile("multi_pair_cached::miller_loop_partial", || {
                            Bn254::multi_miller_loop(g1_prepared, g2_prepared).0
                        });
                        
                        let pairing_result = profile("multi_pair_cached::final_exponentiation_partial", || {
                            Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                                .expect("Final exponentiation should not fail")
                        });

                        pairing_result.0
                    })
                },
                
                // Case 3: G1 fresh points, G2 cached (partial optimization)
                (Some(g1_points), _, _, None, Some(g2_c), Some(g2_cache)) => {
                    profile("multi_pair_cached::g1_fresh_g2_cached", || {
                        assert_eq!(g1_points.len(), g2_c, "G1 points length must equal G2 count");
                        if g2_c == 0 {
                            return Fq12::one();
                        }

                        // G1 fresh preparation, G2 from cache (clone)
                        let g1_prepared = profile("multi_pair_cached::prepare_g1_fresh_partial", || {
                            g1_points.par_iter().map(|&g| BnG1Prepared::from(g)).collect::<Vec<_>>()
                        });
                        
                        let g2_prepared = profile("multi_pair_cached::clone_g2_prepared_partial", || {
                            (0..g2_c)
                                .map(|i| g2_cache.get_prepared(i).expect("Index out of bounds in G2 cache").clone())
                                .collect::<Vec<_>>()
                        });

                        let ml_result = profile("multi_pair_cached::miller_loop_partial", || {
                            Bn254::multi_miller_loop(g1_prepared, g2_prepared).0
                        });
                        
                        let pairing_result = profile("multi_pair_cached::final_exponentiation_partial", || {
                            Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                                .expect("Final exponentiation should not fail")
                        });

                        pairing_result.0
                    })
                },
                
                // Case 4: Both fresh points (no caching benefit)
                (Some(g1_points), _, _, Some(g2_points), _, _) => {
                    profile("multi_pair_cached::both_fresh", || {
                        assert_eq!(g1_points.len(), g2_points.len(), "G1 and G2 vectors must have equal length");
                        if g1_points.is_empty() {
                            return Fq12::one();
                        }

                        let left = profile("multi_pair_cached::prepare_g1_fresh", || {
                            g1_points.par_iter().map(|&g| BnG1Prepared::from(g)).collect::<Vec<_>>()
                        });
                        
                        let right = profile("multi_pair_cached::prepare_g2_fresh", || {
                            g2_points.par_iter().map(|q| BnG2Prepared::from(q.0)).collect::<Vec<_>>()
                        });

                        let ml_result = profile("multi_pair_cached::miller_loop_fresh", || {
                            Bn254::multi_miller_loop(left, right).0
                        });

                        let pairing_result = profile("multi_pair_cached::final_exponentiation_fresh", || {
                            Bn254::final_exponentiation(MillerLoopOutput(ml_result))
                                .expect("Final exponentiation should not fail")
                        });

                        pairing_result.0
                    })
                },
                
                _ => panic!("Invalid combination of parameters provided to multi_pair_cached")
            }
        })
    }
}


/* --------- Cache structures for optimized MSM operations ----------- */

/// Cache entry for a single G1 point containing precomputed values
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1CacheEntry {
    /// Original affine point
    pub affine: G1Affine,
    /// Projective version for faster group operations
    pub projective: G1Projective,
    /// Prepared version for pairing operations
    pub prepared: BnG1Prepared<ark_bn254::Config>,
    /// Precomputed small multiples [1g, 2g, 3g, ..., 8g]
    pub multiples: [G1Projective; 2],
}

/// Cache for multiple G1 points
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1Cache {
    /// Cached entries indexed by position
    pub entries: Vec<G1CacheEntry>,
}

impl G1Cache {
    /// Initialize cache from a vector of G1 affine points
    pub fn new(generators: &[G1Affine]) -> Self {
        let entries: Vec<G1CacheEntry> = generators
            .par_iter()
            .map(|&g| {
                let projective = g.into_group();
                let prepared = BnG1Prepared::from(g);

                // Compute small multiples
                let mut multiples = [G1Projective::zero(); 2];
                let mut acc = projective;
                for i in 0..2 {
                    multiples[i] = acc;
                    acc = acc + projective;
                }

                G1CacheEntry {
                    affine: g,
                    projective,
                    prepared,
                    multiples,
                }
            })
            .collect();

        Self { entries }
    }

    /// Save cache to file
    pub fn save_to_file(&self, path: &str) -> Result<(), SerializationError> {
        let mut file = std::fs::File::create(path).map_err(|e| SerializationError::IoError(e))?;
        self.serialize_compressed(&mut file)?;
        file.flush().map_err(|e| SerializationError::IoError(e))?;
        Ok(())
    }

    /// Load cache from file
    pub fn load_from_file(path: &str) -> Result<Self, SerializationError> {
        let file = std::fs::File::open(path).map_err(|e| SerializationError::IoError(e))?;
        Self::deserialize_compressed(file)
    }

    /// Get a cache entry by index
    pub fn get_entry(&self, index: usize) -> Option<&G1CacheEntry> {
        self.entries.get(index)
    }

    /// Get the projective version of a point by index
    pub fn get_projective(&self, index: usize) -> Option<&G1Projective> {
        self.entries.get(index).map(|e| &e.projective)
    }

    /// Get the prepared version of a point by index
    pub fn get_prepared(&self, index: usize) -> Option<&BnG1Prepared<ark_bn254::Config>> {
        self.entries.get(index).map(|e| &e.prepared)
    }

    /// Get a precomputed multiple (1-2) of a point by index
    pub fn get_multiple(&self, index: usize, multiple: usize) -> Option<&G1Projective> {
        if multiple == 0 || multiple > 2 {
            return None;
        }
        self.entries.get(index).map(|e| &e.multiples[multiple - 1])
    }

    /// Number of cached entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Cache entry for a single G2 point containing precomputed values
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2CacheEntry {
    /// Original affine point
    pub affine: G2Affine,
    /// Projective version for faster group operations
    pub projective: G2Projective,
    /// Prepared version for pairing operations
    pub prepared: BnG2Prepared<ark_bn254::Config>,
    /// Precomputed small multiples [1g, 2g, 3g, ..., 8g]
    pub multiples: [G2Projective; 2],
}

/// Cache for multiple G2 points
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2Cache {
    /// Cached entries indexed by position
    pub entries: Vec<G2CacheEntry>,
}

impl G2Cache {
    /// Initialize cache from a vector of G2 affine points
    pub fn new(generators: &[G2Affine]) -> Self {
        let entries: Vec<G2CacheEntry> = generators
            .par_iter()
            .map(|&g| {
                let projective = g.into_group();
                let prepared = BnG2Prepared::from(g);

                // Compute small multiples
                let mut multiples = [G2Projective::zero(); 2];
                let mut acc = projective;
                for i in 0..2 {
                    multiples[i] = acc;
                    acc = acc + projective;
                }

                G2CacheEntry {
                    affine: g,
                    projective,
                    prepared,
                    multiples,
                }
            })
            .collect();

        Self { entries }
    }

    /// Initialize cache from a vector of G2AffineWrapper points
    pub fn new_from_wrappers(generators: &[G2AffineWrapper]) -> Self {
        let native_generators: Vec<G2Affine> = generators.iter().map(|w| w.0).collect();
        Self::new(&native_generators)
    }

    /// Save cache to file
    pub fn save_to_file(&self, path: &str) -> Result<(), SerializationError> {
        let mut file = std::fs::File::create(path).map_err(|e| SerializationError::IoError(e))?;
        self.serialize_compressed(&mut file)?;
        file.flush().map_err(|e| SerializationError::IoError(e))?;
        Ok(())
    }

    /// Load cache from file
    pub fn load_from_file(path: &str) -> Result<Self, SerializationError> {
        let file = std::fs::File::open(path).map_err(|e| SerializationError::IoError(e))?;
        Self::deserialize_compressed(file)
    }

    /// Get a cache entry by index
    pub fn get_entry(&self, index: usize) -> Option<&G2CacheEntry> {
        self.entries.get(index)
    }

    /// Get the projective version of a point by index
    pub fn get_projective(&self, index: usize) -> Option<&G2Projective> {
        self.entries.get(index).map(|e| &e.projective)
    }

    /// Get the prepared version of a point by index
    pub fn get_prepared(&self, index: usize) -> Option<&BnG2Prepared<ark_bn254::Config>> {
        self.entries.get(index).map(|e| &e.prepared)
    }

    /// Get a precomputed multiple (1-2) of a point by index
    pub fn get_multiple(&self, index: usize, multiple: usize) -> Option<&G2Projective> {
        if multiple == 0 || multiple > 2 {
            return None;
        }
        self.entries.get(index).map(|e| &e.multiples[multiple - 1])
    }

    /// Number of cached entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// Optimized MSM implementation using ark-ec's VariableBaseMSM for G1
pub struct OptimizedMsmG1;

impl MultiScalarMul<G1Affine> for OptimizedMsmG1 {
    fn msm(bases: &[G1Affine], scalars: &[Fr]) -> G1Affine {
        if bases.is_empty() {
            return G1Affine::identity();
        }

        use ark_ec::VariableBaseMSM;

        G1Projective::msm(bases, scalars)
            .unwrap_or_else(|_| G1Projective::zero())
            .into_affine()
    }

    fn fixed_base_msm(base: &G1Affine, scalars: &[Fr]) -> G1Affine {
        if scalars.is_empty() {
            return G1Affine::identity();
        }

        // Sum scalars first, then use regular scalar multiplication for single result
        let sum_scalar = scalars.iter().fold(<Fr as Field>::zero(), |acc, s| acc + s);
        base.scale(&sum_scalar)
    }

    fn fixed_base_vector_msm(base: &G1Affine, scalars: &[Fr]) -> Vec<G1Affine> {
        if scalars.is_empty() {
            return vec![];
        }

        // Use arkworks FixedBase for efficient batch computation
        use ark_ec::scalar_mul::fixed_base::FixedBase;

        let scalar_bits = Fr::MODULUS_BIT_SIZE as usize;
        let window_size = FixedBase::get_mul_window_size(scalars.len());
        let base_projective = base.into_group();

        // Calculate the correct outer count for the windowed multiplication
        let outerc = (scalar_bits + window_size - 1) / window_size;

        // Create the precomputed table with correct dimensions
        let table = FixedBase::get_window_table(scalar_bits, window_size, base_projective);

        // Compute each scalar multiplication using the precomputed table
        scalars
            .iter()
            .map(|scalar| {
                FixedBase::windowed_mul::<G1Projective>(outerc, window_size, &table, scalar)
                    .into_affine()
            })
            .collect()
    }

    fn fixed_scalar_variable_with_add(bases: &[G1Affine], vs: &mut [G1Affine], scalar: &Fr) {
        let n = bases.len();
        assert_eq!(n, vs.len(), "bases and vs must have same length");
        if n == 0 {
            return;
        }

        // Use GLV + wNAF for BN254 G1 if available, otherwise use wNAF
        let use_glv = true; // GLV is beneficial for larger computations
        let window_size = if n > 1024 { 5 } else { 4 };

        // Precompute GLV scalar decomposition once
        let glv_decomp = if use_glv {
            Some(g1::Config::scalar_decomposition(*scalar))
        } else {
            None
        };

        // Optimize chunk size based on cache hierarchy
        // L1 cache: ~32KB, L2: ~256KB, L3: ~8MB per core
        // Each G1 point is ~96 bytes, so we want chunks that fit in L2
        const L2_CACHE_POINTS: usize = 2048; // ~192KB of G1 points
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((n + num_threads - 1) / num_threads).min(L2_CACHE_POINTS);

        // Process in parallel with optimized scalar multiplication
        vs.par_chunks_mut(chunk_size)
            .zip(bases.par_chunks(chunk_size))
            .for_each(|(v_chunk, base_chunk)| {
                v_chunk
                    .iter_mut()
                    .zip(base_chunk.iter())
                    .for_each(|(v, base)| {
                        // Convert to projective
                        let base_proj = base.into_group();
                        let v_proj = v.into_group();

                        // Compute scalar * base using precomputed decomposition
                        let scaled = if let Some(((sgn_k1, k1), (sgn_k2, k2))) = glv_decomp {
                            // Use precomputed GLV decomposition for direct multiplication
                            let mut b1 = base_proj;
                            let mut b2 = g1::Config::endomorphism(&base_proj);

                            if !sgn_k1 {
                                b1 = -b1;
                            }
                            if !sgn_k2 {
                                b2 = -b2;
                            }

                            let b1b2 = b1 + b2;

                            let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
                            let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

                            let mut res = G1Projective::zero();
                            let mut skip_zeros = true;
                            for pair in iter_k1.zip(iter_k2) {
                                if skip_zeros && pair == (false, false) {
                                    skip_zeros = false;
                                    continue;
                                }
                                res.double_in_place();
                                match pair {
                                    (true, false) => res += b1,
                                    (false, true) => res += b2,
                                    (true, true) => res += b1b2,
                                    (false, false) => {}
                                }
                            }
                            res
                        } else {
                            // Use wNAF multiplication
                            let wnaf_context = WnafContext::new(window_size);
                            wnaf_context.mul(base_proj, scalar)
                        };

                        // Update v[i] = v[i] + scaled
                        *v = (v_proj + scaled).into_affine();
                    });
            });
    }

    fn fixed_scalar_scale_with_add(vs: &mut [G1Affine], addends: &[G1Affine], scalar: &Fr) {
        let n = vs.len();
        assert_eq!(n, addends.len(), "vs and addends must have same length");
        if n == 0 {
            return;
        }

        // Use GLV + wNAF for BN254 G1 if available, otherwise use wNAF
        let use_glv = true; // GLV is beneficial for larger computations
        let window_size = if n > 1024 { 5 } else { 4 };

        // Precompute GLV scalar decomposition once
        let glv_decomp = if use_glv {
            Some(g1::Config::scalar_decomposition(*scalar))
        } else {
            None
        };

        // Optimize chunk size based on cache hierarchy
        const L2_CACHE_POINTS: usize = 2048; // ~192KB of G1 points
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((n + num_threads - 1) / num_threads).min(L2_CACHE_POINTS);

        // Process in parallel with optimized scalar multiplication
        vs.par_chunks_mut(chunk_size)
            .zip(addends.par_chunks(chunk_size))
            .for_each(|(v_chunk, addend_chunk)| {
                v_chunk
                    .iter_mut()
                    .zip(addend_chunk.iter())
                    .for_each(|(v, addend)| {
                        // Convert to projective
                        let v_proj = v.into_group();
                        let addend_proj = addend.into_group();

                        // Compute scalar * v using precomputed decomposition
                        let scaled = if let Some(((sgn_k1, k1), (sgn_k2, k2))) = glv_decomp {
                            // Use precomputed GLV decomposition for direct multiplication
                            let mut b1 = v_proj;
                            let mut b2 = g1::Config::endomorphism(&v_proj);

                            if !sgn_k1 {
                                b1 = -b1;
                            }
                            if !sgn_k2 {
                                b2 = -b2;
                            }

                            let b1b2 = b1 + b2;

                            let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
                            let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

                            let mut res = G1Projective::zero();
                            let mut skip_zeros = true;
                            for pair in iter_k1.zip(iter_k2) {
                                if skip_zeros && pair == (false, false) {
                                    skip_zeros = false;
                                    continue;
                                }
                                res.double_in_place();
                                match pair {
                                    (true, false) => res += b1,
                                    (false, true) => res += b2,
                                    (true, true) => res += b1b2,
                                    (false, false) => {}
                                }
                            }
                            res
                        } else {
                            // Use wNAF multiplication
                            let wnaf_context = WnafContext::new(window_size);
                            wnaf_context.mul(v_proj, scalar)
                        };

                        // Update v[i] = scalar * v[i] + addend[i]
                        *v = (scaled + addend_proj).into_affine();
                    });
            });
    }
}

// Optimized MSM implementation using ark-ec's VariableBaseMSM for G2
pub struct OptimizedMsmG2;

impl MultiScalarMul<G2AffineWrapper> for OptimizedMsmG2 {
    fn msm(bases: &[G2AffineWrapper], scalars: &[Fr]) -> G2AffineWrapper {
        if bases.is_empty() {
            return G2AffineWrapper::identity();
        }

        // Convert wrappers to native G2Affine
        let native_bases: Vec<G2Affine> = bases.iter().map(|b| b.0).collect();

        // Use ark-ec's optimized MSM
        use ark_ec::VariableBaseMSM;

        let result = G2Projective::msm(&native_bases, scalars)
            .unwrap_or_else(|_| G2Projective::zero())
            .into_affine();

        G2AffineWrapper(result)
    }

    fn fixed_base_msm(base: &G2AffineWrapper, scalars: &[Fr]) -> G2AffineWrapper {
        if scalars.is_empty() {
            return G2AffineWrapper::identity();
        }

        // Sum scalars first, then use regular scalar multiplication for single result
        let sum_scalar = scalars.iter().fold(<Fr as Field>::zero(), |acc, s| acc + s);
        base.scale(&sum_scalar)
    }

    fn fixed_base_vector_msm(base: &G2AffineWrapper, scalars: &[Fr]) -> Vec<G2AffineWrapper> {
        if scalars.is_empty() {
            return vec![];
        }

        // Use arkworks FixedBase for efficient batch computation
        use ark_ec::scalar_mul::fixed_base::FixedBase;

        let scalar_bits = Fr::MODULUS_BIT_SIZE as usize;
        let window_size = FixedBase::get_mul_window_size(scalars.len());
        let base_projective = base.0.into_group();

        // Calculate the correct outer count for the windowed multiplication
        let outerc = (scalar_bits + window_size - 1) / window_size;

        // Create the precomputed table with correct dimensions
        let table = FixedBase::get_window_table(scalar_bits, window_size, base_projective);

        // Compute each scalar multiplication using the precomputed table
        scalars
            .iter()
            .map(|scalar| {
                let result =
                    FixedBase::windowed_mul::<G2Projective>(outerc, window_size, &table, scalar)
                        .into_affine();
                G2AffineWrapper(result)
            })
            .collect()
    }

    fn fixed_scalar_variable_with_add(
        bases: &[G2AffineWrapper],
        vs: &mut [G2AffineWrapper],
        scalar: &Fr,
    ) {
        let n = bases.len();
        assert_eq!(n, vs.len(), "bases and vs must have same length");
        if n == 0 {
            return;
        }

        // Use GLV + wNAF for BN254 G2 if available, otherwise use wNAF
        let use_glv = true; // GLV is beneficial for larger computations
        let window_size = if n > 1024 { 5 } else { 4 };

        // Precompute GLV scalar decomposition once
        let glv_decomp = if use_glv {
            Some(g2::Config::scalar_decomposition(*scalar))
        } else {
            None
        };

        // Optimize chunk size based on cache hierarchy
        // L1 cache: ~32KB, L2: ~256KB, L3: ~8MB per core
        // Each G2 point is ~192 bytes, so we want chunks that fit in L2
        const L2_CACHE_POINTS: usize = 1024; // ~192KB of G2 points
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((n + num_threads - 1) / num_threads).min(L2_CACHE_POINTS);

        // Process in parallel with optimized scalar multiplication
        vs.par_chunks_mut(chunk_size)
            .zip(bases.par_chunks(chunk_size))
            .for_each(|(v_chunk, base_chunk)| {
                v_chunk
                    .iter_mut()
                    .zip(base_chunk.iter())
                    .for_each(|(v, base)| {
                        // Convert to projective
                        let base_proj = base.0.into_group();
                        let v_proj = v.0.into_group();

                        // Compute scalar * base using precomputed decomposition
                        let scaled = if let Some(((sgn_k1, k1), (sgn_k2, k2))) = glv_decomp {
                            // Use precomputed GLV decomposition for direct multiplication
                            let mut b1 = base_proj;
                            let mut b2 = g2::Config::endomorphism(&base_proj);

                            if !sgn_k1 {
                                b1 = -b1;
                            }
                            if !sgn_k2 {
                                b2 = -b2;
                            }

                            let b1b2 = b1 + b2;

                            let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
                            let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

                            let mut res = G2Projective::zero();
                            let mut skip_zeros = true;
                            for pair in iter_k1.zip(iter_k2) {
                                if skip_zeros && pair == (false, false) {
                                    skip_zeros = false;
                                    continue;
                                }
                                res.double_in_place();
                                match pair {
                                    (true, false) => res += b1,
                                    (false, true) => res += b2,
                                    (true, true) => res += b1b2,
                                    (false, false) => {}
                                }
                            }
                            res
                        } else {
                            // Use wNAF multiplication
                            let wnaf_context = WnafContext::new(window_size);
                            wnaf_context.mul(base_proj, scalar)
                        };

                        // Update v[i] = v[i] + scaled
                        *v = G2AffineWrapper((v_proj + scaled).into_affine());
                    });
            });
    }

    fn fixed_scalar_scale_with_add(
        vs: &mut [G2AffineWrapper],
        addends: &[G2AffineWrapper],
        scalar: &Fr,
    ) {
        let n = vs.len();
        assert_eq!(n, addends.len(), "vs and addends must have same length");
        if n == 0 {
            return;
        }

        // Use GLV + wNAF for BN254 G2 if available, otherwise use wNAF
        let use_glv = true; // GLV is beneficial for larger computations
        let window_size = if n > 1024 { 5 } else { 4 };

        // Precompute GLV scalar decomposition once
        let glv_decomp = if use_glv {
            Some(g2::Config::scalar_decomposition(*scalar))
        } else {
            None
        };

        // Optimize chunk size based on cache hierarchy
        const L2_CACHE_POINTS: usize = 1024; // ~192KB of G2 points
        let num_threads = rayon::current_num_threads();
        let chunk_size = ((n + num_threads - 1) / num_threads).min(L2_CACHE_POINTS);

        // Process in parallel with optimized scalar multiplication
        vs.par_chunks_mut(chunk_size)
            .zip(addends.par_chunks(chunk_size))
            .for_each(|(v_chunk, addend_chunk)| {
                v_chunk
                    .iter_mut()
                    .zip(addend_chunk.iter())
                    .for_each(|(v, addend)| {
                        // Convert to projective
                        let v_proj = v.0.into_group();
                        let addend_proj = addend.0.into_group();

                        // Compute scalar * v using precomputed decomposition
                        let scaled = if let Some(((sgn_k1, k1), (sgn_k2, k2))) = glv_decomp {
                            // Use precomputed GLV decomposition for direct multiplication
                            let mut b1 = v_proj;
                            let mut b2 = g2::Config::endomorphism(&v_proj);

                            if !sgn_k1 {
                                b1 = -b1;
                            }
                            if !sgn_k2 {
                                b2 = -b2;
                            }

                            let b1b2 = b1 + b2;

                            let iter_k1 = ark_ff::BitIteratorBE::new(k1.into_bigint());
                            let iter_k2 = ark_ff::BitIteratorBE::new(k2.into_bigint());

                            let mut res = G2Projective::zero();
                            let mut skip_zeros = true;
                            for pair in iter_k1.zip(iter_k2) {
                                if skip_zeros && pair == (false, false) {
                                    skip_zeros = false;
                                    continue;
                                }
                                res.double_in_place();
                                match pair {
                                    (true, false) => res += b1,
                                    (false, true) => res += b2,
                                    (true, true) => res += b1b2,
                                    (false, false) => {}
                                }
                            }
                            res
                        } else {
                            // Use wNAF multiplication
                            let wnaf_context = WnafContext::new(window_size);
                            wnaf_context.mul(v_proj, scalar)
                        };

                        // Update v[i] = scalar * v[i] + addend[i]
                        *v = G2AffineWrapper((scaled + addend_proj).into_affine());
                    });
            });
    }
}

// Implementation of MultiScalarMul for GT (Fq12) - fallback to dummy since no native MSM
pub struct DummyMsm<G: Group> {
    _phantom: std::marker::PhantomData<G>,
}

impl<G: Group> MultiScalarMul<G> for DummyMsm<G> {
    fn msm(bases: &[G], scalars: &[G::Scalar]) -> G {
        assert_eq!(
            bases.len(),
            scalars.len(),
            "msm requires equal length inputs"
        );
        if bases.is_empty() {
            return G::identity();
        }

        bases
            .iter()
            .zip(scalars.iter())
            .fold(G::identity(), |acc, (base, scalar)| {
                acc.add(&base.scale(scalar))
            })
    }

    fn fixed_base_msm(base: &G, scalars: &[G::Scalar]) -> G {
        if scalars.is_empty() {
            return G::identity();
        }

        // Sum scalars first, then scale once for efficiency
        let sum_scalar = scalars.iter().fold(G::Scalar::zero(), |acc, s| acc.add(s));
        base.scale(&sum_scalar)
    }
}

/// Standard polynomial with field elements for testing
#[derive(Clone, Debug, PartialEq)]
pub struct StandardPolynomial<'a, F: Field> {
    pub coeffs: &'a [F],
}

impl<'a, F: Field> StandardPolynomial<'a, F> {
    pub fn new(coeffs: &'a [F]) -> Self {
        Self { coeffs }
    }
}

// Implement Polynomial trait for StandardPolynomial
impl<'a, F: Field, G1: Group<Scalar = F>> Polynomial<F, G1> for StandardPolynomial<'a, F> {
    fn get(&self, index: usize) -> F {
        if index < self.coeffs.len() {
            self.coeffs[index]
        } else {
            F::zero()
        }
    }

    fn len(&self) -> usize {
        self.coeffs.len()
    }
}
