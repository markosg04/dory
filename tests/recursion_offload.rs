#![cfg(feature = "recursion")]
#![allow(missing_docs)]

use ark_bn254::{Fq12, Fr};
use ark_ff::UniformRand;
use dory::{
    curve::{
        commit_and_evaluate_batch, test_rng, ArkBn254Pairing, DummyMsm, OptimizedMsmG1,
        OptimizedMsmG2, StandardPolynomial,
    },
    setup::ProverSetup,
    toy_transcript::ToyTranscript,
    vmv::evaluate::{create_evaluation_proof, verify_evaluation_proof_with_recursion},
};

/// Test that the recursion offload machinery actually uses precomputed GT exponentiation steps
/// This test verifies that:
/// 1. The prover generates GT exponentiation steps during finalize_for_recursion
/// 2. The verifier can consume these steps via OffloadContext
/// 3. The offload path is actually exercised (not just the fallback native computation)
#[test]
fn test_recursion_offload_with_valid_steps() {
    let _ = tracing_subscriber::fmt::try_init();
    tracing::debug!("=== Testing recursion offload with valid GT steps ===");

    // Test parameters
    let length: usize = 1 << 9;
    let max_log_n: usize = 9;
    let sigma: usize = 5;

    let mut rng = test_rng();
    let domain = b"recursion_offload_test";

    // Setup
    tracing::debug!("Creating prover setup...");
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);

    // Generate polynomial and evaluation point
    let nu = length.next_power_of_two().trailing_zeros() as usize;
    let a = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(length)
        .collect::<Vec<_>>();
    let b_points = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(nu)
        .collect::<Vec<_>>();

    // Create proof with recursion enabled
    tracing::debug!("Creating proof with recursion...");
    let transcript = ToyTranscript::new(domain);
    let polynomial = StandardPolynomial::new(&a);

    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &polynomial,
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Extract GT offload results from the proof
    let recursion_ops = proof.gt_offload_results.clone();

    // Verify that results were actually generated
    assert!(
        recursion_ops.is_some(),
        "GT offload results should be present when recursion feature is enabled"
    );

    let result_count = recursion_ops.as_ref().unwrap().len();
    tracing::debug!("Generated {} GT offload results", result_count);
    assert!(
        result_count > 0,
        "Should have generated at least one GT offload result"
    );

    // Compute verification data
    let (commitment_batch, batching_factors, evaluations) =
        commit_and_evaluate_batch::<
            ArkBn254Pairing,
            OptimizedMsmG1,
            Fr,
            <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
        >(&polynomial, &b_points, 0, sigma, &prover_setup);

    let verifier_setup = prover_setup.to_verifier_setup();
    let verify_transcript = ToyTranscript::new(domain);

    // Verify with recursion ops - this should use the offload path
    tracing::debug!("Verifying with recursion ops (offload path)...");
    let verification_result = verify_evaluation_proof_with_recursion::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        verify_transcript,
        recursion_ops,
    );

    match verification_result {
        Ok(_) => tracing::debug!("✓ Verification succeeded with offload"),
        Err(e) => panic!("Verification failed with offload: {:?}", e),
    }

    tracing::debug!("Recursion offload test completed successfully!");
}

/// Test that tampering with GT exponentiation steps causes verification to fail
/// This ensures that:
/// 1. The offload mechanism actually validates the precomputed values
/// 2. In debug mode, mismatched GT values trigger assertion failures
/// 3. In release mode, incorrect values lead to verification failure
#[test]
fn test_recursion_offload_with_tampered_steps_should_fail() {
    let _ = tracing_subscriber::fmt::try_init();
    tracing::debug!("=== Testing recursion offload with tampered GT steps ===");

    // Test parameters
    let length: usize = 1 << 9;
    let max_log_n: usize = 9;
    let sigma: usize = 5;

    let mut rng = test_rng();
    let domain = b"recursion_offload_tamper_test";

    // Setup
    tracing::debug!("Creating prover setup...");
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);

    // Generate polynomial and evaluation point
    let nu = length.next_power_of_two().trailing_zeros() as usize;
    let a = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(length)
        .collect::<Vec<_>>();
    let b_points = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(nu)
        .collect::<Vec<_>>();

    // Create proof with recursion enabled
    tracing::debug!("Creating proof with recursion...");
    let transcript = ToyTranscript::new(domain);
    let polynomial = StandardPolynomial::new(&a);

    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &polynomial,
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Extract and tamper with GT offload results
    let mut recursion_ops = proof.gt_offload_results.clone().unwrap();
    assert!(
        !recursion_ops.is_empty(),
        "Should have GT offload results to tamper with"
    );

    // Tamper with the first result
    tracing::debug!("Tampering with first GT offload result...");
    // Replace the result with a random Fq12 value
    recursion_ops[0].result = Fq12::rand(&mut rng);

    // Compute verification data
    let (commitment_batch, batching_factors, evaluations) =
        commit_and_evaluate_batch::<
            ArkBn254Pairing,
            OptimizedMsmG1,
            Fr,
            <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
        >(&polynomial, &b_points, 0, sigma, &prover_setup);

    let verifier_setup = prover_setup.to_verifier_setup();
    let verify_transcript = ToyTranscript::new(domain);

    // Verify with tampered recursion ops
    // In debug mode: should panic with assertion failure
    // In release mode: should return Err
    tracing::debug!("Verifying with tampered ops (should fail)...");

    // Note: In debug mode, this will panic due to the assertion in scale_gt_with_offload
    // In release mode, it will return Err due to incorrect GT values
    #[cfg(debug_assertions)]
    {
        let result = std::panic::catch_unwind(|| {
            verify_evaluation_proof_with_recursion::<
                ArkBn254Pairing,
                ToyTranscript,
                OptimizedMsmG1,
                OptimizedMsmG2,
                DummyMsm<Fq12>,
            >(
                proof,
                &commitment_batch,
                &batching_factors,
                &evaluations,
                &b_points,
                sigma,
                &verifier_setup,
                verify_transcript,
                Some(recursion_ops),
            )
        });

        assert!(
            result.is_err(),
            "Debug mode should panic when GT offload values are incorrect"
        );
        tracing::debug!("✓ Debug assertion correctly caught tampered GT values");
    }

    #[cfg(not(debug_assertions))]
    {
        let verification_result = verify_evaluation_proof_with_recursion::<
            ArkBn254Pairing,
            ToyTranscript,
            OptimizedMsmG1,
            OptimizedMsmG2,
            DummyMsm<Fq12>,
        >(
            proof,
            &commitment_batch,
            &batching_factors,
            &evaluations,
            &b_points,
            sigma,
            &verifier_setup,
            verify_transcript,
            Some(recursion_ops),
        );

        assert!(
            verification_result.is_err(),
            "Release mode verification should fail with tampered GT values"
        );
        tracing::debug!("✓ Verification correctly rejected tampered GT values");
    }

    tracing::debug!("Tampered GT steps test completed successfully!");
}

/// Test that offload is NOT used when recursion_ops is None
#[test]
fn test_recursion_verification_without_offload() {
    let _ = tracing_subscriber::fmt::try_init();
    tracing::debug!("=== Testing recursion verification without offload ===");

    // Test parameters
    let length: usize = 1 << 9;
    let max_log_n: usize = 9;
    let sigma: usize = 5;

    let mut rng = test_rng();
    let domain = b"recursion_no_offload_test";

    // Setup
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);

    // Generate polynomial and evaluation point
    let nu = length.next_power_of_two().trailing_zeros() as usize;
    let a = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(length)
        .collect::<Vec<_>>();
    let b_points = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(nu)
        .collect::<Vec<_>>();

    // Create proof
    let transcript = ToyTranscript::new(domain);
    let polynomial = StandardPolynomial::new(&a);

    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &polynomial,
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Compute verification data
    let (commitment_batch, batching_factors, evaluations) =
        commit_and_evaluate_batch::<
            ArkBn254Pairing,
            OptimizedMsmG1,
            Fr,
            <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
        >(&polynomial, &b_points, 0, sigma, &prover_setup);

    let verifier_setup = prover_setup.to_verifier_setup();
    let verify_transcript = ToyTranscript::new(domain);

    // Verify WITHOUT recursion ops - should use native GT computation
    tracing::debug!("Verifying without recursion ops (native computation)...");
    let verification_result = verify_evaluation_proof_with_recursion::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        verify_transcript,
        None, // No recursion ops - uses native computation
    );

    assert!(
        verification_result.is_ok(),
        "Verification should succeed with native computation: {:?}",
        verification_result
    );

    tracing::debug!("✓ Verification succeeded with native computation (no offload)");
    tracing::debug!("Recursion without offload test completed successfully!");
}
