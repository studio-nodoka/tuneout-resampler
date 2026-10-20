use super::*;

pub(crate) fn available() -> Vec<Kernel> {
    [
        Some(Kernel(Backend::Scalar)),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        std::is_x86_feature_detected!("sse2").then_some(Kernel(Backend::Sse2)),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        std::is_x86_feature_detected!("avx").then_some(Kernel(Backend::Avx)),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        std::is_x86_feature_detected!("avx512f").then_some(Kernel(Backend::Avx512)),
        #[cfg(target_arch = "aarch64")]
        std::arch::is_aarch64_feature_detected!("neon").then_some(Kernel(Backend::Neon)),
    ]
    .into_iter()
    .flatten()
    .collect()
}

pub(crate) fn coarse_available() -> Vec<Kernel> {
    [Kernel(Backend::Scalar), Kernel::coarse_detect()]
        .into_iter()
        .chain(available().into_iter().skip(1))
        .collect()
}

fn assert_complex_bits(actual: &[Complex], expected: &[Complex], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            [a.re.to_bits(), a.im.to_bits()],
            [e.re.to_bits(), e.im.to_bits()],
            "{context}: index {i}, {a:?} != {e:?}"
        );
    }
}

fn next(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as i64 as f64 / i64::MAX as f64
}

#[test]
fn complex_vector_layout_and_dispatch() {
    assert_eq!(
        std::mem::size_of::<Complex>(),
        2 * std::mem::size_of::<f64>()
    );
    assert_eq!(std::mem::align_of::<Complex>(), std::mem::align_of::<f64>());
    assert_eq!(std::mem::offset_of!(Complex, re), 0);
    assert_eq!(
        std::mem::offset_of!(Complex, im),
        std::mem::size_of::<f64>()
    );
    assert!(available().contains(&Kernel::detect()));
    eprintln!(
        "Executed kernels: {:?}; selected {:?}",
        available(),
        Kernel::detect()
    );
}

#[test]
fn simd_loads_and_stores_require_only_f64_alignment() {
    #[repr(C, align(64))]
    struct Offset {
        prefix: f64,
        values: [Complex; 67],
    }
    let mut row = Offset {
        prefix: 0.0,
        values: [Complex::default(); 67],
    };
    let mut state = Offset {
        prefix: 0.0,
        values: [Complex::default(); 67],
    };
    for i in 0..67 {
        row.values[i] = Complex::new((i as f64 - 31.0) / 100.0, (47.0 - i as f64) / 100.0);
        state.values[i] = Complex::new((i as f64 - 17.0) / 71.0, (37.0 - i as f64) / 73.0);
    }
    assert_eq!(row.values.as_ptr() as usize % 16, 8);
    assert_eq!(state.values.as_ptr() as usize % 16, 8);
    let initial = state.values;
    for kernel in available() {
        for length in 0..=67 {
            state.values = initial;
            let mut expected = initial;
            update_scalar::<true>(&row.values[..length], &mut expected[..length], 0.314159);
            kernel.causal(&row.values[..length], &mut state.values[..length], 0.314159);
            assert_complex_bits(&state.values, &expected, &format!("unaligned {kernel:?}"));
            assert_eq!(
                kernel
                    .dot(&row.values[..length], &state.values[..length])
                    .to_bits(),
                dot_scalar(&row.values[..length], &state.values[..length]).to_bits()
            );
        }
    }
}

#[test]
fn recurrence_simd_preserves_fma_rounding_signed_zero_and_subnormals() {
    let tiny = f64::from_bits(1);
    let delta = 2.0f64.powi(-27);
    let values = [
        -0.0,
        0.0,
        tiny,
        -tiny,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1.0,
        -1.0,
        1.0 + delta,
    ];
    for kernel in available() {
        for &sample in &values {
            for length in 0..=37 {
                let mut state = Vec::new();
                let mut poles = Vec::new();
                for i in 0..length {
                    state.push(Complex::new(
                        values[i % values.len()],
                        values[(i * 5 + 1) % values.len()],
                    ));
                    poles.push(Complex::new(
                        values[(i * 3 + 2) % values.len()],
                        values[(i * 7 + 3) % values.len()],
                    ));
                }
                for causal in [false, true] {
                    let mut expected = state.clone();
                    let mut actual = state.clone();
                    if causal {
                        update_scalar::<true>(&poles, &mut expected, sample);
                        kernel.causal(&poles, &mut actual, sample);
                    } else {
                        update_scalar::<false>(&poles, &mut expected, sample);
                        kernel.future(&poles, &mut actual, sample);
                    }
                    assert_complex_bits(
                        &actual,
                        &expected,
                        &format!("{kernel:?}, causal {causal}, sample {sample:?}"),
                    );
                }
            }
        }
        let poles = [Complex::new(1.0 - delta, 1.0); 8];
        let mut state = [Complex::new(1.0 + delta, 1.0); 8];
        kernel.causal(&poles, &mut state, 0.0);
        for s in state {
            assert_eq!(s.re.to_bits(), 0.0f64.to_bits());
        }
    }
}

#[test]
fn recurrence_simd_matches_arbitrary_state_lengths_and_rounds() {
    let mut random = 0x19a2_6b3d_9e50_a401;
    for kernel in available() {
        for length in (0..=65).chain([127, 256, 257]) {
            for offset in 0..2 {
                let poles: Vec<_> = (0..length + offset)
                    .map(|_| Complex::new(next(&mut random) * 0.4, next(&mut random) * 0.4))
                    .collect();
                let initial: Vec<_> = (0..length + offset)
                    .map(|_| Complex::new(next(&mut random) * 1e80, next(&mut random) * 1e-80))
                    .collect();
                for causal in [false, true] {
                    let mut expected = initial.clone();
                    let mut actual = initial.clone();
                    for _ in 0..17 {
                        let sample = next(&mut random);
                        if causal {
                            update_scalar::<true>(
                                &poles[offset..],
                                &mut expected[offset..],
                                sample,
                            );
                            kernel.causal(&poles[offset..], &mut actual[offset..], sample);
                        } else {
                            update_scalar::<false>(
                                &poles[offset..],
                                &mut expected[offset..],
                                sample,
                            );
                            kernel.future(&poles[offset..], &mut actual[offset..], sample);
                        }
                        assert_complex_bits(
                            &actual,
                            &expected,
                            &format!("{kernel:?}, causal {causal}, length {length}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn dot_simd_preserves_four_lane_reduction_tails_and_rounding() {
    let tiny = f64::from_bits(1);
    let delta = 2.0f64.powi(-27);
    let special = [
        -0.0,
        0.0,
        tiny,
        -tiny,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1e16,
        -1e16,
        1.0,
        -1.0,
        1.0 + delta,
    ];
    let mut random = 0x68d1_3f29_abc0_5713;
    for kernel in available() {
        for length in (0..=65).chain([127, 255, 256, 257, 4095, 4096, 4097]) {
            for offset in 0..2 {
                for adverse in [false, true] {
                    let row: Vec<_> = (0..length + offset)
                        .map(|i| {
                            if adverse {
                                Complex::new(
                                    special[i % special.len()],
                                    special[(i * 3 + 1) % special.len()],
                                )
                            } else {
                                Complex::new(next(&mut random), next(&mut random))
                            }
                        })
                        .collect();
                    let states: Vec<_> = (0..length + offset)
                        .map(|i| {
                            if adverse {
                                Complex::new(
                                    special[(i * 5 + 2) % special.len()],
                                    special[(i * 7 + 3) % special.len()],
                                )
                            } else {
                                Complex::new(next(&mut random), next(&mut random))
                            }
                        })
                        .collect();
                    assert_eq!(
                        kernel.dot(&row[offset..], &states[offset..]).to_bits(),
                        dot_scalar(&row[offset..], &states[offset..]).to_bits(),
                        "{kernel:?}, length {length}, offset {offset}, adverse {adverse}"
                    );
                }
            }
        }
        let row = [Complex::new(1.0 - delta, 1.0); 8];
        let states = [Complex::new(1.0 + delta, 1.0); 8];
        assert_eq!(kernel.dot(&row, &states).to_bits(), 0.0f64.to_bits());
    }
}
