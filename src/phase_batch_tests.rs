use super::*;
use crate::test_assertions::assert_bits_eq;

fn backends() -> Vec<PhaseBatchBackend> {
    let mut result = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            result.push(PhaseBatchBackend::Avx);
        }
        if std::is_x86_feature_detected!("avx512f") {
            result.push(PhaseBatchBackend::Avx512);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        result.push(PhaseBatchBackend::Neon);
    }
    result
}

fn narrow_backends() -> Vec<PhaseBatchBackend> {
    backends()
        .into_iter()
        .filter(|b| b.periods() == 8)
        .collect()
}

fn rates() -> impl Iterator<Item = (u32, u32)> {
    // Coprime ratios exercise every two- and four-phase group remainder.
    (1..=9).flat_map(|num| {
        (1..=9)
            .filter(move |&den| gcd(num, den) == 1)
            .map(move |den| (num * 48_000, den * 48_000))
    })
}

fn signal(frames: usize) -> Vec<f64> {
    let mut state = 0xb483_9ea2_163d_725fu64;
    (0..frames * 2)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if i % 17 == 0 {
                [1e16, -1e16, -0.0, f64::from_bits(1), -f64::MIN_POSITIVE][i % 5]
            } else {
                state as i64 as f64 / i64::MAX as f64 * 0.45
            }
        })
        .collect()
}

fn kernel(from: u32, to: u32, taps: usize, backend: PhaseBatchBackend) -> Fir {
    let phases = (to / gcd(from, to)) as usize;
    let mut fir = Fir::with_design(
        2,
        from,
        to,
        Design {
            phases,
            taps: 3,
            beta: 20.0,
            rolloff: 1.0,
        },
    );
    fir.taps = taps;
    fir.half_taps = (taps / 2) as i64;
    fir.coeffs = std::sync::Arc::new(
        (0..(phases + 1) * taps)
            .map(|i| ((i * 7 % 71) as f64 - 35.0) / 64.0)
            .collect(),
    );
    fir.phase_batch_backend = backend;
    fir
}

fn check_stream(
    actual: &mut Fir,
    expected: &mut Fir,
    frames: usize,
    pattern: &[usize],
    eof_with_input: bool,
) {
    actual.reset();
    expected.reset();
    let input = signal(frames);
    let num = actual.clock.step_num as usize;
    let den = actual.clock.step_den as usize;
    let mut out = Vec::new();
    let mut reference = Vec::new();
    let mut offset = 0;
    for block in pattern.iter().cycle() {
        if offset == frames {
            break;
        }
        let end = (offset + block).min(frames);
        let eof = eof_with_input && end == frames;
        out.extend(actual.process_f64(&input[offset * 2..end * 2], eof));
        reference.extend(expected.process_f64(&input[offset * 2..end * 2], eof));
        assert!(out.len() <= reference.len());
        assert!(reference.len() - out.len() <= actual.phase_batch_backend.periods() * den * 2);
        offset = end;
    }
    out.extend(actual.process_f64(&[], true));
    reference.extend(expected.process_f64(&[], true));
    assert_bits_eq!(&out, &reference);
    assert_eq!(out.len(), (frames * den).div_ceil(num) * 2);
    assert!(actual.process_f64(&[], true).is_empty());
}

#[test]
fn tiles_cover_all_group_tails_phases_origins_and_tap_lengths() {
    for backend in backends() {
        let cycles = backend.periods();
        for (from, to) in rates() {
            for taps in [1, 3, 5, 7, 9, 17, 65] {
                let mut actual = kernel(from, to, taps, backend);
                let mut expected = kernel(from, to, taps, PhaseBatchBackend::Disabled);
                let num = actual.clock.step_num as usize;
                let den = actual.clock.step_den as usize;
                let frames = taps + (cycles + 1) * num + 13;
                for phase in 0..den {
                    let start = [0, 10_000, 1_i64 << 50][phase % 3];
                    for fir in [&mut actual, &mut expected] {
                        fir.buffer = signal(frames);
                        fir.buffer_start_frame = start;
                        fir.clock.source_index = start + fir.half_taps + (phase % 5) as i64;
                        fir.clock.phase_numer = phase as u32;
                    }
                    let mut output = vec![123.0, -456.0];
                    assert_eq!(
                        actual.try_process_phase_tile(start + frames as i64, false, &mut output),
                        Some(cycles * den * 2)
                    );
                    assert_eq!(&output[..2], &[123.0, -456.0]);
                    let mut reference = Vec::new();
                    for _ in 0..cycles * den / 4 {
                        reference.extend(
                            expected
                                .try_process_stereo_batch(start + frames as i64)
                                .unwrap()
                                .into_iter()
                                .flatten(),
                        );
                    }
                    assert_bits_eq!(&output[2..], &reference);
                    assert_eq!(actual.current_position(), expected.current_position());
                    assert!(actual.phase_scratch.len() <= (taps + num) * cycles * 2 + 8);
                }
            }
        }
    }
}

#[test]
fn tiles_wait_for_exact_input_extent_without_mutating_output_or_clock() {
    for backend in backends() {
        let cycles = backend.periods();
        for (from, to) in rates() {
            let mut actual = kernel(from, to, 17, backend);
            let num = actual.clock.step_num as usize;
            let den = actual.clock.step_den as usize;
            for phase in 0..den {
                for start in [0, 10_000, 1_i64 << 50] {
                    let origin = start + actual.half_taps + 3;
                    actual.buffer_start_frame = start;
                    actual.clock.source_index = origin;
                    actual.clock.phase_numer = phase as u32;
                    let last_center = origin + ((phase + (cycles * den - 1) * num) / den) as i64;
                    let end = last_center - actual.half_taps + actual.taps as i64;
                    actual.buffer = signal((end - start) as usize);
                    let position = actual.current_position();
                    let mut output = vec![123.0, -456.0];
                    let capacity = output.capacity();
                    assert_eq!(
                        actual.try_process_phase_tile(end - 1, false, &mut output),
                        Some(0)
                    );
                    assert_eq!(
                        actual.try_process_phase_tile(end - 1, true, &mut output),
                        None
                    );
                    assert_eq!(actual.current_position(), position);
                    assert_eq!(output, [123.0, -456.0]);
                    assert_eq!(output.capacity(), capacity);
                    assert_eq!(
                        actual.try_process_phase_tile(end, false, &mut output),
                        Some(cycles * den * 2)
                    );
                    assert_eq!(&output[..2], &[123.0, -456.0]);
                    assert_eq!(actual.clock.source_index, origin + (cycles * num) as i64);
                    assert_eq!(actual.clock.phase_numer, phase as u32);
                }
            }
        }
    }
}

#[test]
fn streams_preserve_bits_lengths_reset_and_final_drain_for_every_backend() {
    for backend in backends() {
        for (from, to) in rates() {
            let mut actual = kernel(from, to, 65, backend);
            let mut expected = kernel(from, to, 65, PhaseBatchBackend::Disabled);
            let tile = backend.periods() * actual.clock.step_num as usize;
            let half = actual.half_taps as usize;
            for frames in [
                0,
                1,
                half - 1,
                half,
                half + 1,
                actual.taps,
                actual.taps + tile - 1,
                actual.taps + tile,
                actual.taps * 3 + tile + 7,
            ] {
                for pattern in [&[1][..], &[1024], &[3, 17, 113, 257], &[65536]] {
                    for eof in [false, true] {
                        check_stream(&mut actual, &mut expected, frames, pattern, eof);
                    }
                }
            }
            assert!(!actual.phase_scratch.is_empty());
        }
    }
}

#[test]
fn production_lengths_match_unbatched_streaming_for_every_backend() {
    for backend in backends() {
        for (from, to) in rates() {
            for policy in [FilterLengthPolicy::Generic, FilterLengthPolicy::Fixed] {
                for length in [
                    FilterLength::Standard,
                    FilterLength::Long,
                    FilterLength::ExtraLong,
                ] {
                    let design = select(from, to, length, policy).unwrap();
                    let mut actual = Fir::with_design(2, from, to, design);
                    let mut expected = Fir::with_design(2, from, to, design);
                    actual.phase_batch_backend = backend;
                    expected.phase_batch_backend = PhaseBatchBackend::Disabled;
                    let frames =
                        actual.taps * 2 + backend.periods() * actual.clock.step_num as usize + 7;
                    for eof in [false, true] {
                        check_stream(&mut actual, &mut expected, frames, &[17, 257, 4096], eof);
                    }
                    assert!(!actual.phase_scratch.is_empty());
                }
            }
        }
    }
}

#[test]
fn production_dispatch_uses_common_geometry_latency_and_memory_limits() {
    for (from, to) in crate::test_cases::pairs() {
        let design = select(
            from,
            to,
            FilterLength::Standard,
            FilterLengthPolicy::Generic,
        )
        .unwrap();
        let backend = Fir::phase_batch_backend_for_rates(from, to, design);
        if backend != PhaseBatchBackend::Disabled {
            assert!(backends().contains(&backend));
            assert_eq!(design.phases, (to / gcd(from, to)) as usize);
            assert!(design.phases >= backend.periods() / 4 && design.phases >= 2);
            let num = u64::from(from / gcd(from, to));
            let cycles = backend.periods() as u64;
            assert!(num * cycles * 1000 <= u64::from(from) * MAX_BATCH_MILLISECONDS);
            assert!((design.taps as u64 + num) * cycles * 16 + 64 <= MAX_BATCH_SCRATCH_BYTES);
        }
    }
    let selected = Fir::select_phase_batch_backend();
    for (from, to) in rates() {
        let design = select(
            from,
            to,
            FilterLength::Standard,
            FilterLengthPolicy::Generic,
        )
        .unwrap();
        let expected = if design.phases < 2 {
            PhaseBatchBackend::Disabled
        } else {
            #[cfg(target_arch = "x86_64")]
            if selected == PhaseBatchBackend::Avx512 && design.phases < 4 {
                PhaseBatchBackend::Avx
            } else {
                selected
            }
            #[cfg(not(target_arch = "x86_64"))]
            selected
        };
        assert_eq!(
            Fir::phase_batch_backend_for_rates(from, to, design),
            expected
        );
        for channels in [1, 6] {
            assert_eq!(
                Fir::with_design(channels, from, to, design).phase_batch_backend,
                PhaseBatchBackend::Disabled
            );
        }
    }
    for divisor in [1, 124] {
        let (from, to) = (5 * divisor, 4 * divisor);
        let design = select(
            from,
            to,
            FilterLength::Standard,
            FilterLengthPolicy::Generic,
        )
        .unwrap();
        assert_eq!(
            Fir::phase_batch_backend_for_rates(from, to, design),
            PhaseBatchBackend::Disabled
        );
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx") {
        for (divisor, expected) in [
            (125, PhaseBatchBackend::Avx),
            (249, PhaseBatchBackend::Avx),
            (250, PhaseBatchBackend::Avx512),
        ] {
            let (from, to) = (5 * divisor, 4 * divisor);
            let design = select(
                from,
                to,
                FilterLength::Standard,
                FilterLengthPolicy::Generic,
            )
            .unwrap();
            assert_eq!(
                Fir::phase_batch_backend_for_rates(from, to, design),
                expected
            );
        }
        let (from, to) = (4 * 250_000, 5 * 250_000);
        let mut design = select(
            from,
            to,
            FilterLength::Standard,
            FilterLengthPolicy::Generic,
        )
        .unwrap();
        design.taps = 65_535;
        assert_eq!(
            Fir::phase_batch_backend_for_rates(from, to, design),
            PhaseBatchBackend::Avx
        );
    }
}

#[test]
fn narrow_dot_products_preserve_rounding_cancellation_and_subnormals() {
    for backend in narrow_backends() {
        let delta = 2.0f64.powi(-27);
        let cases = [
            (
                vec![1e16, -1e16, 1., -1., -1e16, 1e16, 1., -1.],
                vec![1.; 8],
            ),
            (vec![-1., 1. + delta], vec![1., 1. - delta]),
            (
                vec![
                    -0.,
                    0.,
                    f64::from_bits(1),
                    -f64::from_bits(1),
                    f64::MIN_POSITIVE,
                ],
                vec![1., -1., 1., 0.5, 0.5],
            ),
            (vec![], vec![]),
        ];
        for (values, weights) in cases {
            // Offsetting each row also tests unaligned loads. Every lane has a
            // separate dot product, including signs that expose lane mix-ups.
            let mut storage = vec![0.0; 1 + values.len() * 16];
            for (tap, value) in values.iter().enumerate() {
                for lane in 0..16 {
                    storage[1 + tap * 16 + lane] = if lane % 2 == 0 { *value } else { -*value };
                }
            }
            let samples = &storage[1..];
            let sums = unsafe {
                match backend {
                    #[cfg(target_arch = "x86_64")]
                    PhaseBatchBackend::Avx => {
                        Fir::convolve_phase_pair_avx([samples; 2], [&weights; 2])
                    }
                    #[cfg(target_arch = "aarch64")]
                    PhaseBatchBackend::Neon => {
                        Fir::convolve_phase_pair_neon([samples; 2], [&weights; 2])
                    }
                    _ => unreachable!(),
                }
            };
            let mut reference = [0.0; 16];
            for tap in 0..weights.len() {
                for lane in 0..16 {
                    reference[lane] += samples[tap * 16 + lane] * weights[tap];
                }
            }
            assert_bits_eq!(&sums[0], &reference);
            assert_bits_eq!(&sums[1], &reference);
        }
    }
}
