use tuneout_resampler::{Error, Resampler};
#[path = "support/cases.rs"]
mod cases;

fn signal(frames: usize, channels: usize) -> Vec<f64> {
    (0..frames * channels)
        .map(|i| {
            let bits = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (bits >> 32) as i32 as f64 / 2147483648.0 * 0.5
        })
        .collect()
}

#[path = "support/assertions.rs"]
mod assertions;
use assertions::assert_bits_eq;

#[test]
fn reusable_output_preserves_error_atomicity_flush_and_bits() {
    for (from, to) in cases::pairs() {
        let input = signal(257, 2);
        let mut expected = Resampler::new(from, to, 2).unwrap();
        let mut actual = Resampler::new(from, to, 2).unwrap();
        let mut output = vec![9.0; 11];
        let capacity = output.capacity();
        assert!(matches!(
            actual.process_f64_into(&[0.1], true, &mut output),
            Err(Error::IncompleteFrame { .. })
        ));
        assert_eq!(output, [9.0; 11]);
        assert_eq!(output.capacity(), capacity);
        for chunk in input.chunks(257 * 2) {
            actual.process_f64_into(chunk, false, &mut output).unwrap();
            assert_bits_eq!(&output, &expected.process_f64(chunk, false).unwrap());
        }
        actual.process_f64_into(&[], true, &mut output).unwrap();
        assert_bits_eq!(&output, &expected.process_f64(&[], true).unwrap());
        let tail = output.clone();
        assert_eq!(
            actual.process_f64_into(&[0.1, 0.2], false, &mut output),
            Err(Error::StreamFinished)
        );
        assert_bits_eq!(&output, &tail);
        actual.process_f64_into(&[], true, &mut output).unwrap();
        assert!(output.is_empty());
        actual.reset();
        expected.reset();
        actual.process_f64_into(&input, true, &mut output).unwrap();
        assert_bits_eq!(&output, &expected.process_f64(&input, true).unwrap());
    }
}

#[test]
fn reported_batch_buffering_bounds_the_default_stereo_delay() {
    for (from, to) in cases::pairs() {
        let mut stereo = Resampler::new(from, to, 2).unwrap();
        let mut mono = Resampler::new(from, to, 1).unwrap();
        let info = stereo.filter_info();
        assert_eq!(mono.filter_info().additional_batch_input_frames, 0);
        assert!(info.additional_batch_input_frames as u64 * 1000 <= u64::from(from) * 64);
        // Exercise running output even when the filter exceeds a small block,
        // without over-testing high upsampling ratios with a fixed long input.
        let frames = 2 * info.lookahead_input_frames + 2 * info.additional_batch_input_frames + 17;
        let input = signal(frames, 1);
        let mut stereo_frames = 0;
        let mut mono_frames = 0;
        let bound = (info.additional_batch_input_frames as u64 * u64::from(to))
            .div_ceil(u64::from(from)) as usize;
        for chunk in input.chunks(17) {
            let paired: Vec<f64> = chunk.iter().flat_map(|&x| [x, x]).collect();
            stereo_frames += stereo.process_f64(&paired, false).unwrap().len() / 2;
            mono_frames += mono.process_f64(chunk, false).unwrap().len();
            assert!(stereo_frames <= mono_frames);
            assert!(mono_frames - stereo_frames <= bound);
        }
        assert!(
            mono_frames > 0,
            "{from}->{to}: no running output was checked"
        );
    }
}

#[test]
fn invalid_configuration_and_partial_frames_are_rejected_without_consuming_input() {
    assert!(matches!(
        Resampler::new(0, 48_000, 2),
        Err(Error::ZeroSampleRate)
    ));
    assert!(matches!(
        Resampler::new(44_100, 0, 2),
        Err(Error::ZeroSampleRate)
    ));
    assert!(matches!(
        Resampler::new(44_100, 48_000, 0),
        Err(Error::ZeroChannels)
    ));
    assert!(matches!(
        Resampler::new(44_100, 48_000, usize::MAX),
        Err(Error::SizeOverflow)
    ));
    let input = signal(257, 2);
    let mut a = Resampler::new(44_100, 48_000, 2).unwrap();
    let mut b = Resampler::new(44_100, 48_000, 2).unwrap();
    assert!(matches!(
        a.process_f64(&[0.1], true),
        Err(Error::IncompleteFrame { .. })
    ));
    assert_bits_eq!(
        &a.process_f64(&input, true).unwrap(),
        &b.process_f64(&input, true).unwrap()
    );
}

#[test]
fn final_flush_is_idempotent_and_reset_starts_a_new_stream() {
    for (from, to) in cases::pairs() {
        let mut r = Resampler::new(from, to, 2).unwrap();
        let input = signal(257, 2);
        let bank = r.coefficients().as_ptr();
        let first = r.process_f64(&input, true).unwrap();
        assert!(r.process_f64(&[], true).unwrap().is_empty());
        assert!(r.process_f32(&[], false).unwrap().is_empty());
        assert_eq!(
            r.process_f64(&[0.0, 0.0], false),
            Err(Error::StreamFinished)
        );
        r.reset();
        assert_eq!(bank, r.coefficients().as_ptr());
        assert_bits_eq!(&first, &r.process_f64(&input, true).unwrap());
    }
}

#[test]
fn equal_rate_bypass_preserves_bits_and_uses_no_coefficients() {
    let mut r = Resampler::new(48_000, 48_000, 2).unwrap();
    let input = [0.0, -0.0, 0.5 + 2.0_f64.powi(-40), -2.0];
    assert_bits_eq!(&r.process_f64(&input, true).unwrap(), &input);
    assert!(r.filter_info().bypass);
    assert_eq!(r.filter_info().coefficient_bytes, 0);
    assert!(r.coefficients().is_empty());
}

#[test]
fn irregular_blocks_preserve_timing_samples_and_channel_isolation() {
    for (from, to) in cases::pairs() {
        for channels in [1, 2, 6, 8] {
            // Short support exercises output between calls as well as final
            // padding. Production coefficients are checked in separate tests.
            let length = tuneout_resampler::FilterLength::Custom(65);
            let input = signal(257, channels);
            let mut r = Resampler::with_filter_length(from, to, channels, length).unwrap();
            let expected = r.process_f64(&input, true).unwrap();
            assert_eq!(
                expected.len() / channels,
                (257_u64 * u64::from(to)).div_ceil(u64::from(from)) as usize
            );
            r.reset();
            let mut actual = Vec::new();
            let mut remaining = input.as_slice();
            for frames in [1, 17, 113, 64].into_iter().cycle() {
                if remaining.is_empty() {
                    break;
                }
                let (chunk, tail) = remaining.split_at((frames * channels).min(remaining.len()));
                actual.extend(r.process_f64(chunk, false).unwrap());
                remaining = tail;
            }
            assert!(
                !actual.is_empty(),
                "{from}->{to}: no running output was checked"
            );
            actual.extend(r.process_f64(&[], true).unwrap());
            assert_bits_eq!(&actual, &expected, "{from}->{to}, {channels} channels");
            let mono_input: Vec<f64> = input.chunks_exact(channels).map(|x| x[0]).collect();
            let mut mono = Resampler::with_filter_length(from, to, 1, length).unwrap();
            let mono_output = mono.process_f64(&mono_input, true).unwrap();
            assert!(actual
                .chunks_exact(channels)
                .zip(mono_output)
                .all(|(a, b)| a[0].to_bits() == b.to_bits()));
        }
    }
}

#[test]
fn f32_output_matches_f64_convolution_of_rounded_input() {
    for (from, to) in cases::pairs() {
        let input: Vec<f32> = signal(257, 2).into_iter().map(|x| x as f32).collect();
        let wide: Vec<f64> = input.iter().copied().map(f64::from).collect();
        let mut r = Resampler::new(from, to, 2).unwrap();
        let expected: Vec<u32> = r
            .process_f64(&wide, true)
            .unwrap()
            .into_iter()
            .map(|x| (x as f32).to_bits())
            .collect();
        r.reset();
        let mut actual = Vec::new();
        for chunk in input.chunks(67 * 2) {
            actual.extend(r.process_f32(chunk, false).unwrap());
        }
        actual.extend(r.process_f32(&[], true).unwrap());
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected
        );
    }
}

fn magnitude(coefficients: &[f64], rate: u32, frequency: f64) -> f64 {
    let omega = 2.0 * std::f64::consts::PI * frequency / f64::from(rate);
    let (mut real, mut imaginary) = (0.0, 0.0);
    for (index, &coefficient) in coefficients.iter().enumerate() {
        let (sine, cosine) = (omega * index as f64).sin_cos();
        real += coefficient * cosine;
        imaginary -= coefficient * sine;
    }
    real.hypot(imaginary)
}

#[test]
fn rate_matrix_retains_bandwidth_and_rejects_aliases() {
    for (from, to) in cases::pairs().into_iter().filter(|(from, to)| from != to) {
        let r = Resampler::new(from, to, 1).unwrap();
        let info = r.filter_info();
        let lower_rate = f64::from(from.min(to));
        for phase in [0, info.phases / 2, info.phases - 1] {
            let row =
                &r.coefficients()[phase * info.taps_per_phase..(phase + 1) * info.taps_per_phase];
            let passband = 20.0 * magnitude(row, from, lower_rate * 0.48).log10();
            assert!(
                passband.abs() <= 0.01,
                "{from}->{to} phase {phase}: {passband}"
            );
            // Only probe frequencies represented below the input Nyquist limit.
            // For upsampling or nearly equal rates, this stopband lies outside it.
            if lower_rate * 1.04 <= f64::from(from) {
                let stopband = 20.0 * magnitude(row, from, lower_rate * 0.52).log10();
                assert!(stopband <= -120.0, "{from}->{to} phase {phase}: {stopband}");
            }
        }
    }
}

#[test]
fn extreme_rate_construction_is_bounded_and_downsampling_finishes() {
    for (from, to) in [
        (u32::MAX, 1),
        (u32::MAX, u32::MAX - 4),
        (1, 768_000),
        (768_001, 8_000),
    ] {
        let r = Resampler::new(from, to, 1).unwrap();
        let info = r.filter_info();
        assert!(info.taps_per_phase <= 65_535);
        assert!(info.coefficient_bytes <= 64 * 1024 * 1024);
    }
    let mut r = Resampler::new(u32::MAX, 1, 1).unwrap();
    let input = vec![0.25; r.filter_info().taps_per_phase];
    let mut output = r.process_f64(&input, false).unwrap();
    output.extend(r.process_f64(&input, false).unwrap());
    output.extend(r.process_f64(&[], true).unwrap());
    assert_eq!(output.len(), 1);
    assert!((output[0] - 0.25).abs() < 1e-12);
}

#[test]
fn all_declared_pairs_have_exact_lengths_dc_gain_and_bounded_storage() {
    let pairs = cases::pairs();
    let mut count = 0;
    for (from, to) in pairs {
        let mut r = Resampler::new(from, to, 2).unwrap();
        let info = r.filter_info();
        assert!(info.coefficient_bytes <= 64 * 1024 * 1024);
        if !info.bypass {
            assert_eq!(info.taps_per_phase % 2, 1);
            assert!(info.lookahead_input_frames as f64 / f64::from(from) <= 0.080);
            for row in r.coefficients().chunks_exact(info.taps_per_phase) {
                assert!(row.iter().all(|v| v.is_finite()));
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            }
        }
        let output = r.process_f64(&[0.25; 66], true).unwrap();
        assert_eq!(
            output.len() / 2,
            (33_u64 * u64::from(to)).div_ceil(u64::from(from)) as usize
        );
        assert!(output.iter().all(|v| (v - 0.25).abs() < 1e-12));
        count += 1;
    }
    assert_eq!(count, 357);
}
