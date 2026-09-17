use super::*;

fn equal_bits(a: [[f64; 2]; 4], b: [[f64; 2]; 4]) {
    for frame in 0..4 {
        for ch in 0..2 {
            assert_eq!(
                a[frame][ch].to_bits(),
                b[frame][ch].to_bits(),
                "frame {frame}, channel {ch}"
            );
        }
    }
}

fn check_kernels(samples: [&[f64]; 4], lower: [&[f64]; 4], upper: [&[f64]; 4], blends: [f64; 4]) {
    let expected = Fir::convolve_interpolated_stereo_scalar(samples, lower, upper, blends);
    #[cfg(target_arch = "x86_64")]
    {
        equal_bits(
            expected,
            Fir::convolve_interpolated_stereo_sse2(samples, lower, upper, blends),
        );
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: AVX is checked immediately above.
            equal_bits(expected, unsafe {
                Fir::convolve_interpolated_stereo_avx(samples, lower, upper, blends)
            });
        }
    }
    #[cfg(target_arch = "aarch64")]
    equal_bits(
        expected,
        Fir::convolve_interpolated_stereo_neon(samples, lower, upper, blends),
    );
}

#[test]
fn interpolation_and_accumulation_keep_separate_rounding() {
    let delta = 2.0f64.powi(-27);
    // Fusing coefficient interpolation would produce -2^-54 rather than zero.
    check_kernels([&[1., 1.]; 4], [&[-1.]; 4], [&[delta]; 4], [1. - delta; 4]);
    let output = Fir::convolve_interpolated_stereo_scalar(
        [&[1., 1.]; 4],
        [&[-1.]; 4],
        [&[delta]; 4],
        [1. - delta; 4],
    );
    assert_eq!(output[0].map(f64::to_bits), [0.0f64.to_bits(); 2]);
    // Fusing the dot product's second multiply/add would also produce -2^-54.
    check_kernels(
        [&[-1., -1., 1. + delta, 1. + delta]; 4],
        [&[1., 1. - delta]; 4],
        [&[1., 1. - delta]; 4],
        [0.3; 4],
    );
    check_kernels(
        [&[1e16, -1e16, 1., -1., -1e16, 1e16, 1., -1.]; 4],
        [&[1.; 4]; 4],
        [&[1.; 4]; 4],
        [0.7; 4],
    );
}

#[test]
fn interpolated_kernels_cover_unaligned_short_long_and_subnormal_inputs() {
    let mut state = 0x9e3c_2168_a4b1_570du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as i64 as f64 / i64::MAX as f64
    };
    for taps in [0, 1, 2, 3, 7, 16, 17, 513, 1151, 8183, 16387] {
        let samples: [Vec<f64>; 4] =
            std::array::from_fn(|_| (0..taps * 2 + 1).map(|_| next()).collect());
        let lower: [Vec<f64>; 4] = std::array::from_fn(|_| (0..taps + 1).map(|_| next()).collect());
        let upper: [Vec<f64>; 4] = std::array::from_fn(|_| (0..taps + 1).map(|_| next()).collect());
        for offset in 0..=1 {
            for blends in [
                [0., 0.25, 0.75, 1.],
                [f64::from_bits(1), 0.123456789, 0.5, 1. - f64::EPSILON],
            ] {
                check_kernels(
                    std::array::from_fn(|i| &samples[i][offset..offset + taps * 2]),
                    std::array::from_fn(|i| &lower[i][offset..offset + taps]),
                    std::array::from_fn(|i| &upper[i][offset..offset + taps]),
                    blends,
                );
            }
        }
    }
    let tiny = f64::from_bits(1);
    check_kernels(
        [&[-0., 0., tiny, -tiny, f64::MIN_POSITIVE, -f64::MIN_POSITIVE]; 4],
        [&[1., 0.5, 1.]; 4],
        [&[1., -0.5, 0.]; 4],
        [0., 0.25, 0.5, 1.],
    );
}

#[test]
fn an_incomplete_interpolated_batch_does_not_advance_the_clock() {
    let mut r = Fir::new(2, 48_000, 44_101);
    r.buffer = vec![0.125; (r.taps + 2) * 2];
    r.clock.source_index = r.half_taps;
    let position = r.current_position();
    assert!(r
        .try_process_interpolated_stereo_batch((r.taps + 2) as i64, false, None)
        .is_none());
    assert_eq!(position, r.current_position());
    r.buffer.extend_from_slice(&[0.125; 4]);
    assert!(r
        .try_process_interpolated_stereo_batch((r.taps + 4) as i64, false, None)
        .is_some());
}
