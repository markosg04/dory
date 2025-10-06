//! Dory follows an interactive model. Hence, a "proof" consists of some messages
//! between P and V. We use Prover and Verifier "builders" to manage these messages
//! and the fiat-shamir challenges throughout the implementation.
use crate::transcript::Transcript;
use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    arithmetic::{Field, Group},
    messages::{
        FirstReduceChallenge, FirstReduceMessage, FoldScalarsChallenge, ScalarProductChallenge,
        ScalarProductMessage, SecondReduceChallenge, SecondReduceMessage, VMVMessage,
    },
    toy_transcript::ToyTranscript,
};

#[cfg(feature = "recursion")]
use crate::recursion_prelude::ExponentiationSteps;
#[cfg(feature = "recursion")]
use crate::recursion_prelude::GTOffloadResult;

/// A serializable proof struct that contains all the messages exchanged
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProof<G1, G2, GT>
where
    G1: Group,
    G2: Group,
    GT: Group,
{
    /// First prover messages for each round
    pub first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    /// Second prover messages for each round
    pub second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    /// Final scalar product message
    pub final_message: Option<ScalarProductMessage<G1, G2>>,
    /// Vector-matrix-vector message (for PCS)
    pub vmv_message: Option<VMVMessage<G1, GT>>,
    /// GT offload results for recursion (lightweight, only contains Fq12 results)
    #[cfg(feature = "recursion")]
    pub gt_offload_results: Option<Vec<GTOffloadResult>>,
}

/// Trait that defines the structure of the Dory proof.
///
/// A type implementing this trait acts as both the transcript and the proof serializer.
/// This is because these two concepts are closely related, and likely should use the same
/// underlying serialization.
pub trait ProofBuilder {
    /// G1 x G2 -> GT
    type Pairing;
    /// The $\mathbb{G}_1$ group
    type G1: Group;
    /// The $\mathbb{G}_2$ group
    type G2: Group;
    /// The target group, $\mathbb{G}_T$
    type GT: Group;
    /// The scalar field, $\mathbb{F}$, of the groups
    type Scalar: Field;

    /// Append a [`FirstReduceMessage`] to the proof and transcript and return a [`FirstReduceChallenge`] drawn from the transcript.
    #[must_use]
    fn append_first_reduce_message(
        self,
        message: FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (FirstReduceChallenge<Self::Scalar>, Self);
    /// Append a [`SecondReduceMessage`] to the proof and transcript and return a [`SecondReduceChallenge`] drawn from the transcript.
    #[must_use]
    fn append_second_reduce_message(
        self,
        message: SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (SecondReduceChallenge<Self::Scalar>, Self);
    /// Draw a [`FoldScalarsChallenge`] from the transcript.
    #[must_use]
    fn challenge_fold_scalars(self) -> (FoldScalarsChallenge<Self::Scalar>, Self);
    /// Append a [`ScalarProductMessage`] to the proof and transcript.
    /// The optional scalars are used for recursion tracking.
    #[must_use]
    fn append_scalar_product_message(
        self,
        message: ScalarProductMessage<Self::G1, Self::G2>,
        s1_final: Option<Self::Scalar>,
        s2_final: Option<Self::Scalar>,
    ) -> Self;
    #[must_use]
    /// Append a [`VMVMessage`] to the proof and transcript.
    fn append_vmv_message(self, message: VMVMessage<Self::G1, Self::GT>) -> Self;

    /// Draw a [`ScalarProductChallenge`] from the transcript.
    #[must_use]
    fn challenge_scalar_product_scalars(self) -> (ScalarProductChallenge<Self::Scalar>, Self);

    /// Finalize the proof for recursion by computing GT exponentiation steps
    /// This should be called after all rounds are complete
    #[cfg(feature = "recursion")]
    fn finalize_for_recursion<E>(
        self,
        _setup: &crate::setup::ProverSetup<E>,
        _initial_nu: usize,
        _initial_d1: Option<Self::GT>,
        _initial_e1: Self::G1,
        _initial_e2: Self::G2,
    ) -> Self
    where
        E: crate::arithmetic::Pairing<GT = Self::GT, G1 = Self::G1, G2 = Self::G2>,
        Self::GT: crate::arithmetic::Group + Clone,
        Self::G1: crate::arithmetic::Group + Clone,
        Self::G2: crate::arithmetic::Group + Clone,
        Self: Sized,
    {
        // requires a concrete implementation due to types coming from a16z arkworks fork
        self
    }
}

/// Concrete ProofBuilder to collect messages and perform transcript tasks
#[derive(Clone, Debug)]
pub struct DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    /// First prover message for round i
    pub first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    /// First reduce challenges for each round (beta, beta_inverse)
    #[cfg(feature = "recursion")]
    pub first_challenges: Vec<FirstReduceChallenge<Scalar>>,
    /// Second reduce challenges for each round (alpha, alpha_inverse)
    #[cfg(feature = "recursion")]
    pub second_challenges: Vec<SecondReduceChallenge<Scalar>>,
    /// Second prover message for round i
    pub second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    /// Last Scalar product message at end of protocol
    pub final_message: Option<ScalarProductMessage<G1, G2>>,

    /// vector-matrix-vector message, used to transform general dory into PCS
    pub vmv_message: Option<VMVMessage<G1, GT>>,
    /// GT exponentiation steps for recursion (full witness data)
    #[cfg(feature = "recursion")]
    pub gt_exponentiation_steps: Option<Vec<ExponentiationSteps>>,
    /// GT offload results for recursion (lightweight proof data)
    #[cfg(feature = "recursion")]
    pub gt_offload_results: Option<Vec<GTOffloadResult>>,
    /// Delta values from setup for round 1 left (recursion feature)
    #[cfg(feature = "recursion")]
    pub setup_delta_1l: Option<Vec<GT>>,
    /// Delta values from setup for round 1 right (recursion feature)
    #[cfg(feature = "recursion")]
    pub setup_delta_1r: Option<Vec<GT>>,
    /// Delta values from setup for round 2 left (recursion feature)
    #[cfg(feature = "recursion")]
    pub setup_delta_2l: Option<Vec<GT>>,
    /// Delta values from setup for round 2 right (recursion feature)
    #[cfg(feature = "recursion")]
    pub setup_delta_2r: Option<Vec<GT>>,
    /// Fold scalars challenge for final phase
    #[cfg(feature = "recursion")]
    pub fold_scalars_challenge: Option<FoldScalarsChallenge<Scalar>>,
    /// Scalar product challenge for final verification
    #[cfg(feature = "recursion")]
    pub scalar_product_challenge: Option<ScalarProductChallenge<Scalar>>,
    /// Setup HT value for pairing computation
    #[cfg(feature = "recursion")]
    pub setup_ht: Option<GT>,
    /// Setup H1 generator
    #[cfg(feature = "recursion")]
    pub setup_h1: Option<G1>,
    /// Setup H2 generator
    #[cfg(feature = "recursion")]
    pub setup_h2: Option<G2>,
    /// Setup G1 generator at position 0
    #[cfg(feature = "recursion")]
    pub setup_g1_0: Option<G1>,
    /// Setup G2 generator at position 0
    #[cfg(feature = "recursion")]
    pub setup_g2_0: Option<G2>,
    /// Final s1 scalar value
    #[cfg(feature = "recursion")]
    pub s1_final: Option<Scalar>,
    /// Final s2 scalar value
    #[cfg(feature = "recursion")]
    pub s2_final: Option<Scalar>,
    /// Fiat shamir
    pub transcript: T,
    /// Phantom
    pub _phantom: PhantomData<(G1, G2, GT, Scalar)>,
}

impl<G1, G2, GT, Scalar, T> DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    /// Constructor from new transcript and setup
    #[cfg(feature = "recursion")]
    pub fn new<E>(transcript: T, setup: &crate::setup::ProverSetup<E>) -> Self
    where
        E: crate::arithmetic::Pairing<GT = GT, G1 = G1, G2 = G2>,
        GT: Clone,
        G1: Clone,
        G2: Clone,
    {
        Self {
            first_messages: Vec::new(),
            first_challenges: Vec::new(),
            second_challenges: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            gt_exponentiation_steps: Some(Vec::new()),
            gt_offload_results: Some(Vec::new()),
            setup_delta_1l: setup.delta_1l.clone(),
            setup_delta_1r: setup.delta_1r.clone(),
            setup_delta_2l: setup.delta_2l.clone(),
            setup_delta_2r: setup.delta_2r.clone(),
            fold_scalars_challenge: None,
            scalar_product_challenge: None,
            setup_ht: Some(setup.ht().clone()),
            setup_h1: Some(setup.h1().clone()),
            setup_h2: Some(setup.h2().clone()),
            setup_g1_0: Some(setup.g1_vec()[0].clone()),
            setup_g2_0: Some(setup.g2_vec()[0].clone()),
            s1_final: None,
            s2_final: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Constructor from new transcript (without setup values for non-recursion mode)
    #[cfg(not(feature = "recursion"))]
    pub fn new(transcript: T) -> Self {
        Self {
            first_messages: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Constructor to create with ToyTranscript for testing
    #[cfg(feature = "recursion")]
    pub fn new_with_toy_transcript<E>(
        domain: &[u8],
        setup: &crate::setup::ProverSetup<E>,
    ) -> DoryProofBuilder<G1, G2, GT, Scalar, ToyTranscript>
    where
        ToyTranscript: Transcript<Scalar = Scalar>,
        E: crate::arithmetic::Pairing<GT = GT, G1 = G1, G2 = G2>,
        GT: Clone,
        G1: Clone,
        G2: Clone,
    {
        let transcript = ToyTranscript::new(domain);
        DoryProofBuilder {
            first_messages: Vec::new(),
            first_challenges: Vec::new(),
            second_challenges: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            gt_exponentiation_steps: Some(Vec::new()),
            gt_offload_results: Some(Vec::new()),
            setup_delta_1l: setup.delta_1l.clone(),
            setup_delta_1r: setup.delta_1r.clone(),
            setup_delta_2l: setup.delta_2l.clone(),
            setup_delta_2r: setup.delta_2r.clone(),
            fold_scalars_challenge: None,
            scalar_product_challenge: None,
            setup_ht: Some(setup.ht().clone()),
            setup_h1: Some(setup.h1().clone()),
            setup_h2: Some(setup.h2().clone()),
            setup_g1_0: Some(setup.g1_vec()[0].clone()),
            setup_g2_0: Some(setup.g2_vec()[0].clone()),
            s1_final: None,
            s2_final: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Constructor to create with ToyTranscript for testing (non-recursion)
    #[cfg(not(feature = "recursion"))]
    pub fn new_with_toy_transcript(
        domain: &[u8],
    ) -> DoryProofBuilder<G1, G2, GT, Scalar, ToyTranscript>
    where
        ToyTranscript: Transcript<Scalar = Scalar>,
    {
        let transcript = ToyTranscript::new(domain);
        DoryProofBuilder {
            first_messages: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Build a serializable Dory proof
    pub fn build(&self) -> DoryProof<G1, G2, GT> {
        DoryProof {
            first_messages: self.first_messages.clone(),
            second_messages: self.second_messages.clone(),
            final_message: self.final_message.clone(),
            vmv_message: self.vmv_message.clone(),
            #[cfg(feature = "recursion")]
            gt_offload_results: self.gt_offload_results.clone(),
        }
    }

    /// Build a Dory proof and return full exponentiation steps separately
    /// This is used when recursion feature is enabled to preserve the full witness data.
    /// The proof contains only lightweight GTOffloadResult, while full ExponentiationSteps
    /// are returned separately for witness generation in recursive SNARKs.
    #[cfg(feature = "recursion")]
    pub fn build_with_full_steps(
        self,
    ) -> (DoryProof<G1, G2, GT>, Option<Vec<ExponentiationSteps>>) {
        // Build the proof with lightweight results
        let proof = self.build();

        // Return full exponentiation steps separately for witness generation
        (proof, self.gt_exponentiation_steps)
    }

    /// Create a DoryProofBuilder from a DoryProof and a fresh transcript
    pub fn from_proof(proof: DoryProof<G1, G2, GT>, transcript: T) -> Self {
        Self {
            first_messages: proof.first_messages,
            #[cfg(feature = "recursion")]
            first_challenges: Vec::new(), // Challenges are not stored in proof, need to be regenerated
            #[cfg(feature = "recursion")]
            second_challenges: Vec::new(),
            second_messages: proof.second_messages,
            final_message: proof.final_message,
            vmv_message: proof.vmv_message,
            #[cfg(feature = "recursion")]
            gt_exponentiation_steps: None, // Full steps not in proof
            #[cfg(feature = "recursion")]
            gt_offload_results: proof.gt_offload_results,
            #[cfg(feature = "recursion")]
            setup_delta_1l: Some(Vec::new()),
            #[cfg(feature = "recursion")]
            setup_delta_1r: Some(Vec::new()),
            #[cfg(feature = "recursion")]
            setup_delta_2l: Some(Vec::new()),
            #[cfg(feature = "recursion")]
            setup_delta_2r: Some(Vec::new()),
            #[cfg(feature = "recursion")]
            fold_scalars_challenge: None,
            #[cfg(feature = "recursion")]
            scalar_product_challenge: None,
            #[cfg(feature = "recursion")]
            setup_ht: None,
            #[cfg(feature = "recursion")]
            setup_h1: None,
            #[cfg(feature = "recursion")]
            setup_h2: None,
            #[cfg(feature = "recursion")]
            setup_g1_0: None,
            #[cfg(feature = "recursion")]
            setup_g2_0: None,
            #[cfg(feature = "recursion")]
            s1_final: None,
            #[cfg(feature = "recursion")]
            s2_final: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Create a DoryProofBuilder from a DoryProof with a default transcript
    pub fn from_proof_no_transcript(proof: DoryProof<G1, G2, GT>) -> Self
    where
        T: Default,
    {
        DoryProofBuilder {
            first_messages: proof.first_messages,
            #[cfg(feature = "recursion")]
            first_challenges: Vec::new(),
            #[cfg(feature = "recursion")]
            second_challenges: Vec::new(),
            second_messages: proof.second_messages,
            final_message: proof.final_message,
            vmv_message: proof.vmv_message,
            #[cfg(feature = "recursion")]
            gt_exponentiation_steps: None, // Full steps not in proof
            #[cfg(feature = "recursion")]
            gt_offload_results: proof.gt_offload_results,
            #[cfg(feature = "recursion")]
            setup_delta_1l: Some(Vec::new()),
            #[cfg(feature = "recursion")]
            setup_delta_1r: Some(Vec::new()),
            #[cfg(feature = "recursion")]
            setup_delta_2l: Some(Vec::new()),
            #[cfg(feature = "recursion")]
            setup_delta_2r: Some(Vec::new()),
            #[cfg(feature = "recursion")]
            fold_scalars_challenge: None,
            #[cfg(feature = "recursion")]
            scalar_product_challenge: None,
            #[cfg(feature = "recursion")]
            setup_ht: None,
            #[cfg(feature = "recursion")]
            setup_h1: None,
            #[cfg(feature = "recursion")]
            setup_h2: None,
            #[cfg(feature = "recursion")]
            setup_g1_0: None,
            #[cfg(feature = "recursion")]
            setup_g2_0: None,
            #[cfg(feature = "recursion")]
            s1_final: None,
            #[cfg(feature = "recursion")]
            s2_final: None,
            transcript: T::default(),
            _phantom: PhantomData,
        }
    }
}

impl<G1Arg, G2Arg, GTArg, ScalarArg, T> ProofBuilder
    for DoryProofBuilder<G1Arg, G2Arg, GTArg, ScalarArg, T>
where
    G1Arg: Group<Scalar = ScalarArg>,
    G2Arg: Group<Scalar = ScalarArg>,
    GTArg: Group<Scalar = ScalarArg> + std::fmt::Debug,
    ScalarArg: Field,
    T: Transcript<Scalar = ScalarArg>,
{
    type G1 = G1Arg;
    type G2 = G2Arg;
    type GT = GTArg;
    type Scalar = ScalarArg;
    type Pairing = ();

    fn append_first_reduce_message(
        mut self,
        message: FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (FirstReduceChallenge<Self::Scalar>, Self) {
        self.transcript.append_group(b"d1_left", &message.d1_left);
        self.transcript.append_group(b"d1_right", &message.d1_right);
        self.transcript.append_group(b"d2_left", &message.d2_left);
        self.transcript.append_group(b"d2_right", &message.d2_right);
        self.transcript.append_group(b"e1_beta", &message.e1_beta);
        self.transcript.append_group(b"e2_beta", &message.e2_beta);

        let beta = self.transcript.challenge_scalar(b"first_reduce_beta");
        let beta_inverse = beta.inv().expect("Inverse for beta must exist");
        let challenge = FirstReduceChallenge { beta, beta_inverse };

        self.first_messages.push(message);
        #[cfg(feature = "recursion")]
        self.first_challenges.push(challenge.clone());
        (challenge, self)
    }

    fn append_second_reduce_message(
        mut self,
        message: SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (SecondReduceChallenge<Self::Scalar>, Self) {
        self.transcript.append_group(b"c_plus", &message.c_plus);
        self.transcript.append_group(b"c_minus", &message.c_minus);
        self.transcript.append_group(b"e1_plus", &message.e1_plus);
        self.transcript.append_group(b"e1_minus", &message.e1_minus);
        self.transcript.append_group(b"e2_plus", &message.e2_plus);
        self.transcript.append_group(b"e2_minus", &message.e2_minus);

        let alpha = self.transcript.challenge_scalar(b"second_reduce_alpha");
        let alpha_inverse = alpha.inv().expect("Inverse for alpha must exist");
        let challenge = SecondReduceChallenge {
            alpha,
            alpha_inverse,
        };

        #[cfg(feature = "recursion")]
        self.second_challenges.push(challenge.clone());

        self.second_messages.push(message);
        (challenge, self)
    }

    fn append_scalar_product_message(
        mut self,
        message: ScalarProductMessage<Self::G1, Self::G2>,
        #[allow(unused_variables)] s1_final: Option<Self::Scalar>,
        #[allow(unused_variables)] s2_final: Option<Self::Scalar>,
    ) -> Self {
        self.transcript.append_group(b"e1", &message.e1);
        self.transcript.append_group(b"e2", &message.e2);
        #[cfg(feature = "recursion")]
        {
            self.s1_final = s1_final;
            self.s2_final = s2_final;
        }
        self.final_message = Some(message);
        self
    }

    fn append_vmv_message(mut self, message: VMVMessage<Self::G1, Self::GT>) -> Self {
        self.transcript.append_group(b"c_eval_vmv", &message.c);
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        self.vmv_message = Some(message);
        self
    }

    fn challenge_fold_scalars(mut self) -> (FoldScalarsChallenge<Self::Scalar>, Self) {
        let gamma = self.transcript.challenge_scalar(b"fold_scalars_gamma");
        let gamma_inverse = gamma.inv().expect("Inverse for gamma must exist");
        let challenge: FoldScalarsChallenge<ScalarArg> = FoldScalarsChallenge {
            gamma,
            gamma_inverse,
        };
        #[cfg(feature = "recursion")]
        {
            self.fold_scalars_challenge = Some(challenge.clone());
        }
        (challenge, self)
    }

    fn challenge_scalar_product_scalars(mut self) -> (ScalarProductChallenge<Self::Scalar>, Self) {
        let d = self.transcript.challenge_scalar(b"scalar_product_d");
        let d_inv = d.inv().unwrap();
        let challenge = ScalarProductChallenge {
            d,
            d_inverse: d_inv,
        };
        #[cfg(feature = "recursion")]
        {
            self.scalar_product_challenge = Some(challenge.clone());
        }
        (challenge, self)
    }

    #[cfg(feature = "recursion")]
    fn finalize_for_recursion<E>(
        mut self,
        setup: &crate::setup::ProverSetup<E>,
        initial_nu: usize,
        initial_d1: Option<Self::GT>,
        initial_e1: Self::G1,
        initial_e2: Self::G2,
    ) -> Self
    where
        E: crate::arithmetic::Pairing<GT = Self::GT, G1 = Self::G1, G2 = Self::G2>,
        Self::GT: crate::arithmetic::Group + Clone,
        Self::G1: crate::arithmetic::Group + Clone,
        Self::G2: crate::arithmetic::Group + Clone,
    {
        DoryProofBuilder::finalize_for_recursion(
            &mut self, setup, initial_nu, initial_d1, initial_e1, initial_e2,
        );
        self
    }
}

/// Verification analogue of `ProofBuilder`.
pub trait VerificationBuilder {
    /// G1
    type G1: Group;
    /// G2
    type G2: Group;
    /// GT
    type GT: Group;
    /// F_r
    type Scalar: Field;

    /// Number of rounds (nu)
    fn rounds(&mut self) -> usize;

    /// Returns the messages for round[idx]
    fn take_round(
        &mut self,
        idx: usize,
    ) -> (
        FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
        SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    );

    /// Getter for first msg
    fn first_message(&mut self, idx: usize) -> &FirstReduceMessage<Self::G1, Self::G2, Self::GT>;

    /// Getter for second msg
    fn second_message(&mut self, idx: usize) -> &SecondReduceMessage<Self::G1, Self::G2, Self::GT>;

    /// Consume a FirstReduceMessage, append it to the transcript,
    /// and return β, β⁻¹.
    fn process_first_reduce_message(
        &mut self,
        msg: &FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> FirstReduceChallenge<Self::Scalar>;

    /// Consume a SecondReduceMessage, append, and return α, α⁻¹.
    fn process_second_reduce_message(
        &mut self,
        msg: &SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> SecondReduceChallenge<Self::Scalar>;

    /// Derive γ, γ⁻¹ after all rounds are ingested.
    fn challenge_fold_scalars(&mut self) -> FoldScalarsChallenge<Self::Scalar>;

    /// Derive d, d^-1 after all rounds are ingested.
    fn challenge_scalar_product_scalars(&mut self) -> ScalarProductChallenge<Self::Scalar>;

    /// Provide the final scalar-product message that the prover sent and append it to the transcript.
    fn process_scalar_product_message(&mut self) -> &ScalarProductMessage<Self::G1, Self::G2>;

    /// Process a [`VMVMessage`].
    fn process_vmv_message(&mut self) -> VMVMessage<Self::G1, Self::GT>;
}

/// Concrete Dory verify builder
pub struct DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    transcript: T,
    first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    scalar_msg: ScalarProductMessage<G1, G2>,
    vmv_msg: Option<VMVMessage<G1, GT>>,

    _phantom: PhantomData<(G1, G2, GT, Scalar)>,
}

impl<G1, G2, GT, Scalar, T> DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    /// Build from a serializable `DoryProof` and a fresh transcript.
    /// This is useful when you have a serialized proof that you want to verify.
    pub fn new_from_dory_proof(proof: DoryProof<G1, G2, GT>, transcript: T) -> Self {
        // Extract messages from the proof
        let first_messages = proof.first_messages;
        let second_messages = proof.second_messages;
        let scalar_msg = proof
            .final_message
            .expect("DoryProof must have a final (scalar product) message");
        let vmv_msg = proof.vmv_message;

        Self {
            transcript,
            first_messages,
            second_messages,
            scalar_msg,
            vmv_msg,
            _phantom: PhantomData,
        }
    }

    /// Build from a *proof* (any concrete `DoryProofBuilder`) and a fresh transcript.
    /// The caller is responsible for providing a fresh transcript with the correct domain.
    pub fn new_from_proof(proof: DoryProofBuilder<G1, G2, GT, Scalar, T>, transcript: T) -> Self {
        // destructure
        let DoryProofBuilder {
            first_messages,
            second_messages,
            final_message,
            vmv_message,
            ..
        } = proof;

        let scalar_msg = final_message.expect("proof must contain the scalar-product message");

        Self {
            transcript,
            first_messages,
            second_messages,
            scalar_msg,
            vmv_msg: vmv_message,
            _phantom: PhantomData,
        }
    }
}

impl<G1, G2, GT, Scalar, T> VerificationBuilder for DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    type G1 = G1;
    type G2 = G2;
    type GT = GT;
    type Scalar = Scalar;

    fn rounds(&mut self) -> usize {
        self.first_messages.len()
    }

    fn take_round(
        &mut self,
        idx: usize,
    ) -> (
        FirstReduceMessage<G1, G2, GT>,
        SecondReduceMessage<G1, G2, GT>,
    ) {
        let m1 = self.first_messages[idx].clone();
        let m2 = self.second_messages[idx].clone();
        (m1, m2)
    }

    fn first_message(&mut self, idx: usize) -> &FirstReduceMessage<G1, G2, GT> {
        &self.first_messages[idx]
    }
    fn second_message(&mut self, idx: usize) -> &SecondReduceMessage<G1, G2, GT> {
        &self.second_messages[idx]
    }

    fn process_first_reduce_message(
        &mut self,
        m: &FirstReduceMessage<G1, G2, GT>,
    ) -> FirstReduceChallenge<Scalar> {
        self.transcript.append_group(b"d1_left", &m.d1_left);
        self.transcript.append_group(b"d1_right", &m.d1_right);
        self.transcript.append_group(b"d2_left", &m.d2_left);
        self.transcript.append_group(b"d2_right", &m.d2_right);
        self.transcript.append_group(b"e1_beta", &m.e1_beta);
        self.transcript.append_group(b"e2_beta", &m.e2_beta);

        let beta = self.transcript.challenge_scalar(b"first_reduce_beta");
        let beta_inv = beta.inv().unwrap();
        FirstReduceChallenge {
            beta,
            beta_inverse: beta_inv,
        }
    }

    fn process_second_reduce_message(
        &mut self,
        m: &SecondReduceMessage<G1, G2, GT>,
    ) -> SecondReduceChallenge<Scalar> {
        self.transcript.append_group(b"c_plus", &m.c_plus);
        self.transcript.append_group(b"c_minus", &m.c_minus);
        self.transcript.append_group(b"e1_plus", &m.e1_plus);
        self.transcript.append_group(b"e1_minus", &m.e1_minus);
        self.transcript.append_group(b"e2_plus", &m.e2_plus);
        self.transcript.append_group(b"e2_minus", &m.e2_minus);

        let alpha = self.transcript.challenge_scalar(b"second_reduce_alpha");
        let alpha_inv = alpha.inv().unwrap();
        SecondReduceChallenge {
            alpha,
            alpha_inverse: alpha_inv,
        }
    }

    fn challenge_fold_scalars(&mut self) -> FoldScalarsChallenge<Scalar> {
        let gamma = self.transcript.challenge_scalar(b"fold_scalars_gamma");
        let gamma_inv = gamma.inv().unwrap();
        FoldScalarsChallenge {
            gamma,
            gamma_inverse: gamma_inv,
        }
    }

    fn challenge_scalar_product_scalars(&mut self) -> ScalarProductChallenge<Self::Scalar> {
        let d = self.transcript.challenge_scalar(b"scalar_product_d");
        let d_inv = d.inv().unwrap();
        ScalarProductChallenge {
            d,
            d_inverse: d_inv,
        }
    }

    fn process_scalar_product_message(&mut self) -> &ScalarProductMessage<G1, G2> {
        self.transcript.append_group(b"e1", &self.scalar_msg.e1);
        self.transcript.append_group(b"e2", &self.scalar_msg.e2);
        &self.scalar_msg
    }

    fn process_vmv_message(&mut self) -> VMVMessage<G1, GT> {
        let message = self
            .vmv_msg
            .as_ref()
            .expect("VMV message must be present in verify builder");
        self.transcript.append_group(b"c_eval_vmv", &message.c);
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        message.clone()
    }
}

impl<G1, G2, GT, Scalar, T> DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    /// Finalize the proof for recursion case by computing GT exponentiation steps to help verifier
    /// This must be called after all rounds are complete but before building the proof
    #[cfg(feature = "recursion")]
    pub fn finalize_for_recursion<E>(
        &mut self,
        _setup: &crate::setup::ProverSetup<E>,
        initial_nu: usize,
        initial_d1: Option<GT>,
        initial_e1: G1,
        initial_e2: G2,
    ) where
        E: crate::arithmetic::Pairing<GT = GT, G1 = G1, G2 = G2>,
        GT: crate::arithmetic::Group + Clone,
        G1: crate::arithmetic::Group + Clone,
        G2: crate::arithmetic::Group + Clone,
    {
        if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
            gt_steps.clear();
        }
        if let Some(ref mut gt_results) = self.gt_offload_results {
            gt_results.clear();
        }

        if self.setup_delta_1l.as_ref().map_or(true, |v| v.is_empty()) {
            tracing::warn!("No setup delta values available for recursion");
            return;
        }

        let num_rounds = self.first_messages.len();
        if num_rounds != self.second_messages.len()
            || num_rounds != self.first_challenges.len()
            || num_rounds != self.second_challenges.len()
        {
            tracing::warn!("Message/challenge count mismatch");
            return;
        }

        let mut nu = initial_nu;
        tracing::debug!(
            "finalize_for_recursion starting with initial_nu={}, num_rounds={}",
            initial_nu,
            num_rounds
        );

        // We need to mimic the verifier calculations of d_1, d_2, e_1, e_2 in **same order**
        let mut d_1 = initial_d1;
        let mut d_2 = self.vmv_message.as_ref().map(|vmv| vmv.d2.clone());
        let mut e_1 = Some(initial_e1);
        let mut e_2 = Some(initial_e2);

        for round_idx in 0..num_rounds {
            let first_msg = &self.first_messages[round_idx];
            let second_msg = &self.second_messages[round_idx];
            let beta = self.first_challenges[round_idx].beta.clone();
            let beta_inv = self.first_challenges[round_idx].beta_inverse.clone();
            let alpha = self.second_challenges[round_idx].alpha.clone();
            let alpha_inv = self.second_challenges[round_idx].alpha_inverse.clone();

            tracing::debug!("Processing round {} with nu={}", round_idx, nu);

            // 1. Operations from dory_reduce_verify_update_c
            // The verifier does: d_2.scale(&beta), d_1.scale(&beta_inv), c_plus.scale(&alpha), c_minus.scale(&alpha_inv)

            if let (Some(d1_val), Some(d2_val)) = (&d_1, &d_2) {
                let (_, steps_d2) = d2_val.scale_with_steps(&beta);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps_d2.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps_d2.result,
                    });
                }

                let (_, steps_d1) = d1_val.scale_with_steps(&beta_inv);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps_d1.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps_d1.result,
                    });
                }
            }

            // 2: c_plus.scale(&alpha) and c_minus.scale(&alpha_inv)
            let (_, steps_c_plus) = second_msg.c_plus.scale_with_steps(&alpha);
            if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                gt_steps.push(steps_c_plus.clone());
            }
            if let Some(ref mut gt_results) = self.gt_offload_results {
                gt_results.push(GTOffloadResult {
                    result: steps_c_plus.result,
                });
            }

            let (_, steps_c_minus) = second_msg.c_minus.scale_with_steps(&alpha_inv);
            if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                gt_steps.push(steps_c_minus.clone());
            }
            if let Some(ref mut gt_results) = self.gt_offload_results {
                gt_results.push(GTOffloadResult {
                    result: steps_c_minus.result,
                });
            }

            // 3. Operations from dory_reduce_verify_update_ds
            // The verifier does D1 operations (including deltas) first, then D2 operations

            // D1 operations:
            // d1_left.scale(&alpha)
            let (_, steps_d1l) = first_msg.d1_left.scale_with_steps(&alpha);
            if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                gt_steps.push(steps_d1l.clone());
            }
            if let Some(ref mut gt_results) = self.gt_offload_results {
                gt_results.push(GTOffloadResult {
                    result: steps_d1l.result,
                });
            }

            // Then the delta operations for D1 using the current nu value
            if self.setup_delta_1l.as_ref().map_or(false, |v| nu < v.len()) {
                // delta_1l.scale(&alpha_beta)
                let alpha_beta = alpha.mul(&beta);
                let (_, steps) =
                    self.setup_delta_1l.as_ref().unwrap()[nu].scale_with_steps(&alpha_beta);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps.result,
                    });
                }

                // delta_1r.scale(&beta)
                let (_, steps) = self.setup_delta_1r.as_ref().unwrap()[nu].scale_with_steps(&beta);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps.result,
                    });
                }
            }

            // D2 operations:
            // d2_left.scale(&alpha_inv)
            let (_, steps_d2l) = first_msg.d2_left.scale_with_steps(&alpha_inv);
            if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                gt_steps.push(steps_d2l.clone());
            }
            if let Some(ref mut gt_results) = self.gt_offload_results {
                gt_results.push(GTOffloadResult {
                    result: steps_d2l.result,
                });
            }

            // Then the delta operations for D2
            if self.setup_delta_2l.as_ref().map_or(false, |v| nu < v.len()) {
                // delta_2l.scale(&alpha_inv_beta_inv)
                let alpha_inv_beta_inv = alpha_inv.mul(&beta_inv);
                let (_, steps) =
                    self.setup_delta_2l.as_ref().unwrap()[nu].scale_with_steps(&alpha_inv_beta_inv);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps.result,
                    });
                }

                // delta_2r.scale(&beta_inv)
                let (_, steps) =
                    self.setup_delta_2r.as_ref().unwrap()[nu].scale_with_steps(&beta_inv);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps.result,
                    });
                }
            } else if self
                .setup_delta_1l
                .as_ref()
                .map_or(false, |v| nu >= v.len())
            {
                tracing::warn!(
                    "nu={} >= setup_delta_1l.len()={}",
                    nu,
                    self.setup_delta_1l.as_ref().unwrap().len()
                );
            }

            // Update d_1 and d_2 for next round (simulate verifier's update)
            if let (Some(d1_val), Some(d2_val)) = (&mut d_1, &mut d_2) {
                // d_1' = α·d1_left + d1_right + α·β·Δ1L + β·Δ1R
                let mut new_d1 = first_msg.d1_left.scale(&alpha);
                new_d1 = new_d1.add(&first_msg.d1_right);

                if self.setup_delta_1l.as_ref().map_or(false, |v| nu < v.len()) {
                    let alpha_beta = alpha.mul(&beta);
                    new_d1 =
                        new_d1.add(&self.setup_delta_1l.as_ref().unwrap()[nu].scale(&alpha_beta));
                    new_d1 = new_d1.add(&self.setup_delta_1r.as_ref().unwrap()[nu].scale(&beta));
                }
                *d1_val = new_d1;

                // d_2' = α⁻¹·d2_left + d2_right + α⁻¹·β⁻¹·Δ2L + β⁻¹·Δ2R
                let mut new_d2 = first_msg.d2_left.scale(&alpha_inv);
                new_d2 = new_d2.add(&first_msg.d2_right);

                if self.setup_delta_2l.as_ref().map_or(false, |v| nu < v.len()) {
                    let alpha_inv_beta_inv = alpha_inv.mul(&beta_inv);
                    new_d2 = new_d2
                        .add(&self.setup_delta_2l.as_ref().unwrap()[nu].scale(&alpha_inv_beta_inv));
                    new_d2 =
                        new_d2.add(&self.setup_delta_2r.as_ref().unwrap()[nu].scale(&beta_inv));
                }
                *d2_val = new_d2;
            }

            // Update e_1 and e_2 for next round (simulate verifier's update)
            // E_1' <- E_1 + β·E_1beta + α·E_1plus + α⁻¹·E_1minus
            if let (Some(e1_val), Some(e2_val)) = (&e_1, &e_2) {
                let mut new_e1 = e1_val.clone();
                new_e1 = new_e1.add(&first_msg.e1_beta.scale(&beta));
                new_e1 = new_e1.add(&second_msg.e1_plus.scale(&alpha));
                new_e1 = new_e1.add(&second_msg.e1_minus.scale(&alpha_inv));
                e_1 = Some(new_e1);

                // E_2' <- E_2 + β⁻¹·E_2beta + α·E_2plus + α⁻¹·E_2minus
                let mut new_e2 = e2_val.clone();
                new_e2 = new_e2.add(&first_msg.e2_beta.scale(&beta_inv));
                new_e2 = new_e2.add(&second_msg.e2_plus.scale(&alpha));
                new_e2 = new_e2.add(&second_msg.e2_minus.scale(&alpha_inv));
                e_2 = Some(new_e2);
            }

            nu = nu.saturating_sub(1);
        }

        // After all rounds are complete (nu = 0), compute operations for apply_fold_scalars and verify_final_pairing
        // These operations happen after the rounds, during the final verification phase

        if let (
            Some(gamma_challenge),
            Some(d_challenge),
            Some(s1_final),
            Some(s2_final),
            Some(_scalar_msg),
            Some(ht),
        ) = (
            &self.fold_scalars_challenge,
            &self.scalar_product_challenge,
            &self.s1_final,
            &self.s2_final,
            &self.final_message,
            &self.setup_ht,
        ) {
            let gamma = gamma_challenge.gamma.clone();
            let gamma_inv = gamma_challenge.gamma_inverse.clone();
            let d = d_challenge.d.clone();
            let d_inv = d_challenge.d_inverse.clone();

            // Operations from apply_fold_scalars:

            // 1. ht.scale(&s1_final.mul(&s2_final))
            let s_product = s1_final.mul(&s2_final);
            let (_, steps) = ht.scale_with_steps(&s_product);
            if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                gt_steps.push(steps.clone());
            }
            if let Some(ref mut gt_results) = self.gt_offload_results {
                gt_results.push(GTOffloadResult {
                    result: steps.result,
                });
            }

            // 2. pairing(h1, e2).scale(&gamma) - use tracked e_2
            if let (Some(h1), Some(h2), Some(e1_val), Some(e2_val)) =
                (&self.setup_h1, &self.setup_h2, &e_1, &e_2)
            {
                let pairing_h1_e2 = E::pair(h1, e2_val);
                let (_, steps) = pairing_h1_e2.scale_with_steps(&gamma);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps.result,
                    });
                }

                // 3. pairing(e1, h2).scale(&gamma_inv) - use tracked e_1
                let pairing_e1_h2 = E::pair(e1_val, h2);
                let (_, steps) = pairing_e1_h2.scale_with_steps(&gamma_inv);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps.result,
                    });
                }
            }

            // Operations from verify_final_pairing:
            // We need the d_1 and d_2 AFTER apply_fold_scalars updates them
            // apply_fold_scalars adds terms to d_1 and d_2:
            // d_1 = d_1 + e(H₁, g2_0 * s1_final * gamma)
            // d_2 = d_2 + e(g1_0 * s2_final * gamma_inv, H₂)

            if let (Some(mut final_d1), Some(mut final_d2)) = (d_1, d_2) {
                // Compute the updates that apply_fold_scalars makes to d_1 and d_2
                if let (Some(h1), Some(h2), Some(g1_0), Some(g2_0)) = (
                    &self.setup_h1,
                    &self.setup_h2,
                    &self.setup_g1_0,
                    &self.setup_g2_0,
                ) {
                    // Update d_1: add e(H₁, g2_0 * s1_final * gamma)
                    let scalar_for_g2_in_d1 = s1_final.mul(&gamma);
                    let g2_0_scaled = g2_0.scale(&scalar_for_g2_in_d1);
                    let pairing_h1_g2 = E::pair(h1, &g2_0_scaled);
                    final_d1 = final_d1.add(&pairing_h1_g2);

                    // Update d_2: add e(g1_0 * s2_final * gamma_inv, H₂)
                    let scalar_for_g1_in_d2 = s2_final.mul(&gamma_inv);
                    let g1_0_scaled = g1_0.scale(&scalar_for_g1_in_d2);
                    let pairing_g1_h2 = E::pair(&g1_0_scaled, h2);
                    final_d2 = final_d2.add(&pairing_g1_h2);
                }

                // Operations from verify_final_pairing on the UPDATED d_1 and d_2:
                // 4. d_2.scale(&d)
                // 5. d_1.scale(&d_inv)
                let (_, steps) = final_d2.scale_with_steps(&d);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps.result,
                    });
                }

                let (_, steps) = final_d1.scale_with_steps(&d_inv);
                if let Some(ref mut gt_steps) = self.gt_exponentiation_steps {
                    gt_steps.push(steps.clone());
                }
                if let Some(ref mut gt_results) = self.gt_offload_results {
                    gt_results.push(GTOffloadResult {
                        result: steps.result,
                    });
                }
            }
        }

        tracing::debug!(
            "finalize_for_recursion complete, tracked {} GT operations",
            self.gt_exponentiation_steps.as_ref().map_or(0, |v| v.len())
        );
    }

    /// Print statistics about the proof structure
    pub fn print_proof_stats(&self) {
        tracing::debug!("=== PROOF STATISTICS ===");
        tracing::debug!("Number of rounds: {}", self.first_messages.len());
        tracing::debug!("First reduce messages: {}", self.first_messages.len());
        tracing::debug!("Second reduce messages: {}", self.second_messages.len());
        tracing::debug!("Has final message: {}", self.final_message.is_some());
        tracing::debug!("Has VMV message: {}", self.vmv_message.is_some());

        // Calculate total proof elements
        let total_g1_elements = self.first_messages.iter().map(|_m| 1).sum::<usize>() + // e1_beta per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // e1_plus + e1_minus per round
                               if self.final_message.is_some() { 1 } else { 0 } + // final e1
                               if self.vmv_message.is_some() { 1 } else { 0 }; // vmv e1

        let total_g2_elements = self.first_messages.iter().map(|_m| 1).sum::<usize>() + // e2_beta per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // e2_plus + e2_minus per round
                               if self.final_message.is_some() { 1 } else { 0 }; // final e2

        let total_gt_elements = self.first_messages.iter().map(|_m| 4).sum::<usize>() + // d1_left/right + d2_left/right per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // c_plus + c_minus per round
                               if self.vmv_message.is_some() { 2 } else { 0 }; // vmv c + d2

        tracing::debug!("Total G1 elements in proof: {}", total_g1_elements);
        tracing::debug!("Total G2 elements in proof: {}", total_g2_elements);
        tracing::debug!("Total GT elements in proof: {}", total_gt_elements);
        tracing::debug!(
            "Total proof elements: {}",
            total_g1_elements + total_g2_elements + total_gt_elements
        );
        tracing::debug!("========================");
    }
}
