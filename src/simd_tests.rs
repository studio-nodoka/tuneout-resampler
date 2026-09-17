use super::*;

#[test]
fn phase_batch_matches_fallback_across_reset_chunks_and_drain() {
    for (from, to) in crate::test_cases::pairs() {
        let design = select(
            from,
            to,
            FilterLength::Custom(65),
            FilterLengthPolicy::Generic,
        )
        .unwrap();
        let mut batched = Fir::with_design(2, from, to, design);
        let mut fallback = Fir::with_design(2, from, to, design);
        let frames = 3 * design.taps + batched.additional_batch_input_frames() + 17;
        let output_bound =
            batched.phase_batch_backend.periods() * batched.clock.step_den as usize * 2;
        fallback.phase_batch_backend = PhaseBatchBackend::Disabled;
        let mut state = 0x68d1_3f29_abc0_5713u64;
        let input: Vec<f64> = (0..frames * 2)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as i64 as f64 / i64::MAX as f64 * 0.45
            })
            .collect();
        for chunk in [64, 257, 1024, 4096] {
            batched.reset();
            fallback.reset();
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            for samples in input.chunks(chunk * 2) {
                actual.extend(batched.process_f64(samples, false));
                expected.extend(fallback.process_f64(samples, false));
                assert!(actual.len() <= expected.len());
                assert!(expected.len() - actual.len() <= output_bound);
            }
            actual.extend(batched.process_f64(&[], true));
            expected.extend(fallback.process_f64(&[], true));
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(&expected) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            assert!(batched.process_f64(&[], true).is_empty());
        }
    }
}

fn convolve_batch(samples: [&[f64]; 4], coeffs: [&[f64]; 4]) -> [[f64; 2]; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        Fir::convolve_stereo_batch_sse2(samples, coeffs)
    }
    #[cfg(target_arch = "aarch64")]
    {
        Fir::convolve_stereo_batch_neon(samples, coeffs)
    }
}

fn assert_batch_bits(samples: [&[f64]; 4], coeffs: [&[f64]; 4]) {
    let expected = Fir::convolve_stereo_batch_scalar(samples, coeffs);
    let actual = convolve_batch(samples, coeffs);
    for frame in 0..4 {
        for ch in 0..2 {
            assert_eq!(actual[frame][ch].to_bits(), expected[frame][ch].to_bits());
        }
    }
}

#[test]
fn stereo_simd_preserves_accumulation_order_and_separate_rounding() {
    // A reassociated sum can produce 2 instead of 1 in the first row.
    let cancellation = [1e16, -1e16, 1., -1., -1e16, 1e16, 1., -1.];
    let ones = [1.; 4];
    assert_batch_bits([&cancellation; 4], [&ones; 4]);
    let actual = convolve_batch([&cancellation; 4], [&ones; 4]);
    assert_eq!(actual[0], [1., -1.]);

    // Fusing the second multiply/add produces -2^-54 instead of zero.
    let delta = 2.0f64.powi(-27);
    let input = [-1., -1., 1. + delta, 1. + delta];
    let coeffs = [1., 1. - delta];
    assert_batch_bits([&input; 4], [&coeffs; 4]);
    let actual = convolve_batch([&input; 4], [&coeffs; 4]);
    assert_eq!(actual[0].map(f64::to_bits), [0.0f64.to_bits(); 2]);
}

#[test]
fn stereo_simd_handles_empty_short_and_unaligned_rows() {
    let samples: Vec<f64> = (0..66).map(|i| (i as f64 - 31.) / 67.).collect();
    let coeffs: Vec<f64> = (0..33).map(|i| (17. - i as f64) / 37.).collect();
    for taps in 0..=31 {
        for offset in 0..2 {
            for shorter in 0..4 {
                let mut inputs = [&samples[offset..offset + taps * 2]; 4];
                let mut weights = [&coeffs[offset..offset + taps]; 4];
                inputs[shorter] = &samples[offset..offset + taps];
                assert_batch_bits(inputs, weights);
                weights[(shorter + 1) % 4] = &coeffs[..taps / 2];
                assert_batch_bits(inputs, weights);
            }
        }
    }
}

#[test]
fn stereo_simd_preserves_subnormal_and_signed_zero_results() {
    let tiny = f64::from_bits(1);
    let samples = [
        -0.,
        0.,
        tiny,
        -tiny,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        tiny,
        tiny,
    ];
    let coeffs = [1., 1., 0.5, -0.5];
    assert_batch_bits([&samples; 4], [&coeffs; 4]);
}

#[test]
fn stereo_simd_matches_long_random_dot_products() {
    let mut state = 0x19a26b3d9e50a401u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as i64 as f64) / i64::MAX as f64
    };
    for taps in [1, 3, 7, 16, 17, 511, 2047, 4095, 65535] {
        let samples: [Vec<f64>; 4] =
            std::array::from_fn(|_| (0..taps * 2 + 1).map(|_| next()).collect());
        let coeffs: [Vec<f64>; 4] =
            std::array::from_fn(|_| (0..taps + 1).map(|_| next()).collect());
        assert_batch_bits(
            std::array::from_fn(|i| &samples[i][1..]),
            std::array::from_fn(|i| &coeffs[i][1..]),
        );
    }
}
