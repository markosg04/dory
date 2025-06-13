#![allow(missing_docs)]
use crate::arithmetic::*;
use crate::poly::Polynomial;
use ark_bn254::{g1, g2, Bn254, Fq12, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::AdditiveGroup;
use ark_ec::{
    pairing::Pairing as ArkPairing,
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
#[derive(Clone, Debug, PartialEq, Eq)]
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
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        // Extract G1 and G2 elements separately
        let g1_elements: Vec<G1Affine> = ps.iter().copied().collect();
        let g2_elements: Vec<G2Affine> = qs.iter().map(|q| q.0).collect();

        // Use the optimized multi-pairing from arkworks (takes two separate iterators)
        Bn254::multi_pairing(g1_elements, g2_elements).0
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

    fn fixed_scalar_scale_with_add(vs: &mut [G2AffineWrapper], addends: &[G2AffineWrapper], scalar: &Fr) {
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
