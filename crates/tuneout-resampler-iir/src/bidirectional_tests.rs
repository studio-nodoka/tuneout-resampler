use crate::{BufferedResampler, Error, Startup};
use std::f64::consts::PI;

fn bits(samples: &[f64]) -> Vec<u64> {
    samples.iter().map(|x| x.to_bits()).collect()
}

fn noise(frames: usize, channels: usize) -> Vec<f64> {
    (0..frames * channels)
        .map(|i| ((i * 1237 % 1013) as f64 - 506.0) / 1013.0)
        .collect()
}

fn length(frames: usize, input: u32, output: u32) -> usize {
    (frames as u64 * u64::from(output)).div_ceil(u64::from(input)) as usize
}

// Independent convolution of the continuous model, without recursive states,
// phase rows, guards, or the output scheduler.
use crate::bidirectional_design::model;
fn direct_reference(
    input: &[f64],
    converter: &crate::bidirectional::BufferedResampler,
) -> Vec<f64> {
    let info = converter.filter_info();
    let lower = info.input_rate.min(info.output_rate) as f64;
    let step = lower / info.input_rate as f64;
    (0..length(input.len(), info.input_rate, info.output_rate))
        .map(|out| {
            let t = out as f64 * lower / info.output_rate as f64;
            input
                .iter()
                .enumerate()
                .map(|(n, x)| {
                    let dt = (t - n as f64 * step).abs();
                    let h: f64 = model::MODES
                        .iter()
                        .map(|m| {
                            let angle = (m[2] + m[3]) * dt;
                            ((m[4] + m[5]) * angle.cos() - (m[6] + m[7]) * angle.sin())
                                * ((m[0] + m[1]) * dt).exp()
                        })
                        .sum();
                    x * step * h
                })
                .sum()
        })
        .collect()
}

#[test]
fn normalized_core_matches_independent_sampled_convolution() {
    for (input_rate, output_rate) in [(2, 3), (3, 2), (7, 1), (320, 147)] {
        let mut converter = crate::bidirectional::BufferedResampler::with_startup(
            input_rate,
            output_rate,
            1,
            Startup::Zero,
        )
        .unwrap();
        let mut input = noise(193, 1);
        input[0] += 1.0;
        input[96] -= 1.0;
        let expected = direct_reference(&input, &converter);
        let actual = converter.process_f64(&input, true).unwrap();
        assert_eq!(actual.len(), expected.len());
        let error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(error < 1e-11, "{input_rate}->{output_rate}: {error:e}");
    }
}

#[test]
fn buffered_fixed_schedule_is_independent_of_chunks_and_track_cuts() {
    for (input_rate, output_rate) in [(96_000, 44_100), (44_100, 48_000), (48_000, 96_000)] {
        for startup in [Startup::Predict, Startup::Zero] {
            let mut converter =
                BufferedResampler::with_startup(input_rate, output_rate, 2, startup).unwrap();
            let info = converter.filter_info();
            // Include ordinary streaming blocks even for long symmetric guards,
            // then an odd EOS remainder. A short record would only test EOS.
            let frames = info.startup_lookahead_frames + 2 * info.block_input_frames + 137;
            let input = noise(frames, 2);
            let expected = converter.process_f64(&input, true).unwrap();
            assert_eq!(
                expected.len(),
                length(input.len() / 2, input_rate, output_rate) * 2
            );
            assert!(expected.iter().all(|x| x.is_finite()));
            for block_frames in [1, 137, 4095, 4096, 5003] {
                converter.reset();
                let mut actual = Vec::new();
                let mut output = vec![f64::NAN; 3];
                for block in input.chunks(block_frames * 2) {
                    converter
                        .process_f64_into(block, false, &mut output)
                        .unwrap();
                    actual.extend_from_slice(&output);
                }
                assert!(
                    actual.len()
                        >= length(2 * info.block_input_frames, input_rate, output_rate) * 2,
                    "ordinary streaming must emit multiple blocks before EOS"
                );
                assert!(actual.len() < expected.len(), "EOS must retain a remainder");
                converter.process_f64_into(&[], true, &mut output).unwrap();
                actual.extend_from_slice(&output);
                assert_eq!(
                    bits(&actual),
                    bits(&expected),
                    "{input_rate}->{output_rate}, {startup:?}, block={block_frames}"
                );
                assert!(converter.process_f64(&[], true).unwrap().is_empty());
                assert_eq!(
                    converter.process_f64(&[0.0, 0.0], false).unwrap_err(),
                    Error::StreamFinished
                );
            }
            converter.reset();
            let mut tracks = Vec::new();
            let cuts = [0, 3288, 5701, 14_002, frames];
            for pair in cuts.windows(2) {
                tracks.extend(
                    converter
                        .process_f64(&input[pair[0] * 2..pair[1] * 2], false)
                        .unwrap(),
                );
            }
            tracks.extend(converter.process_f64(&[], true).unwrap());
            assert_eq!(bits(&tracks), bits(&expected));
            // These track lengths land between output samples. Maintaining one
            // clock and state is required, not restarting a converter per cut.
            if !output_rate.is_multiple_of(input_rate) {
                assert_ne!(
                    (3288_u64 * u64::from(output_rate)) % u64::from(input_rate),
                    0
                );
            }
        }
    }
}

#[test]
fn buffered_eos_short_records_bypass_and_validation() {
    assert!(matches!(
        BufferedResampler::new(0, 48_000, 1),
        Err(Error::ZeroSampleRate)
    ));
    assert!(matches!(
        BufferedResampler::new(48_000, 0, 1),
        Err(Error::ZeroSampleRate)
    ));
    for channels in [0, 65, usize::MAX] {
        assert!(matches!(
            BufferedResampler::new(48_000, 44_100, channels),
            Err(Error::InvalidChannels)
        ));
    }
    assert!(BufferedResampler::new(48_000, 44_099, 1).is_ok());
    let input = [-0.0, 0.0, f64::MIN_POSITIVE, f64::from_bits(1), 2.0, -2.0];
    let mut bypass = BufferedResampler::new(48_000, 48_000, 2).unwrap();
    assert_eq!(
        bits(&bypass.process_f64(&input, true).unwrap()),
        bits(&input)
    );
    for (input_rate, output_rate) in [(96_000, 44_100), (44_100, 48_000), (1, 96), (64, 1)] {
        for startup in [Startup::Predict, Startup::Zero] {
            let mut converter =
                BufferedResampler::with_startup(input_rate, output_rate, 2, startup).unwrap();
            for frames in [0, 1, 2, 3, 7, 17] {
                converter.reset();
                let input = noise(frames, 2);
                let output = converter.process_f64(&input, true).unwrap();
                assert_eq!(output.len(), length(frames, input_rate, output_rate) * 2);
                assert!(output.iter().all(|x| x.is_finite()));
                assert!(converter.process_f64(&[], true).unwrap().is_empty());
            }
        }
    }
    let mut a = BufferedResampler::new(96_000, 44_100, 2).unwrap();
    assert_eq!(
        a.process_f64(&[1.0], true).unwrap_err(),
        Error::IncompleteFrame
    );
    let mut b = BufferedResampler::new(96_000, 44_100, 2).unwrap();
    let input = noise(500, 2);
    assert_eq!(
        bits(&a.process_f64(&input, true).unwrap()),
        bits(&b.process_f64(&input, true).unwrap())
    );
}

#[test]
fn buffered_multitone_preserves_waveform_phase_and_rejects_aliases() {
    for (input_rate, output_rate) in [
        (96_000, 44_100),
        (44_100, 48_000),
        (48_000, 96_000),
        (192_000, 44_100),
    ] {
        let lower_nyquist = f64::from(input_rate.min(output_rate)) * 0.5;
        let frequencies =
            [0.013, 0.137, 0.503, 0.923, model::PASSBAND].map(|f| (f * lower_nyquist).floor());
        let at = |frame: usize, rate: u32| {
            frequencies
                .iter()
                .enumerate()
                .map(|(index, hz)| {
                    0.12 * (2.0 * PI * hz * frame as f64 / f64::from(rate) + 0.23 * index as f64)
                        .cos()
                })
                .sum::<f64>()
        };
        let input: Vec<_> = (0..input_rate as usize)
            .map(|frame| at(frame, input_rate))
            .collect();
        let mut converter = BufferedResampler::new(input_rate, output_rate, 1).unwrap();
        let output = converter.process_f64(&input, true).unwrap();
        // Analytic sinusoids provide an independent timestamp/phase reference.
        // The interior crosses multiple fixed scheduling blocks and is remote
        // from the unknown-history convention at both file boundaries.
        let start = output_rate as usize / 5;
        let end = 4 * output_rate as usize / 5;
        let error = (start..end)
            .map(|frame| (output[frame] - at(frame, output_rate)).abs())
            .fold(0.0, f64::max);
        assert!(
            error < 0.6 * (10.0_f64.powf(1e-6 / 20.0) - 1.0) + 1e-9,
            "{input_rate}->{output_rate}: passband waveform error {error:e}"
        );
        if input_rate > output_rate {
            let stop_edge = lower_nyquist * model::STOPBAND;
            let band = f64::from(input_rate) * 0.5 - stop_edge;
            let frequencies = [0.0, 0.031, 0.271, 0.913].map(|f| stop_edge + f * band);
            let input: Vec<_> = (0..input_rate as usize)
                .map(|frame| {
                    frequencies
                        .iter()
                        .enumerate()
                        .map(|(index, hz)| {
                            0.2 * (2.0 * PI * hz * frame as f64 / f64::from(input_rate)
                                + 0.41 * index as f64)
                                .cos()
                        })
                        .sum()
                })
                .collect();
            converter.reset();
            let output = converter.process_f64(&input, true).unwrap();
            let peak = output[start..end]
                .iter()
                .map(|v| v.abs())
                .fold(0.0, f64::max);
            assert!(
                peak < 1e-9,
                "{input_rate}->{output_rate}: folded alias peak {peak:e}"
            );
        }
    }
}

#[test]
fn buffered_channel_isolation_dc_and_zero_state_linearity() {
    let mono = noise(9001, 1);
    let input: Vec<_> = mono.iter().flat_map(|&x| [0.0, x, 0.0, -2.0 * x]).collect();
    let mut multichannel =
        BufferedResampler::with_startup(96_000, 44_100, 4, Startup::Zero).unwrap();
    let mut single = BufferedResampler::with_startup(96_000, 44_100, 1, Startup::Zero).unwrap();
    let actual = multichannel.process_f64(&input, true).unwrap();
    let expected = single.process_f64(&mono, true).unwrap();
    for (frame, expected) in actual.chunks_exact(4).zip(expected) {
        assert_eq!(frame[0], 0.0);
        assert_eq!(frame[2], 0.0);
        assert_eq!(frame[1].to_bits(), expected.to_bits());
        assert_eq!(frame[3], -2.0 * expected);
    }
    for frames in [1, 17, 256, 9001] {
        let mut converter = BufferedResampler::new(96_000, 44_100, 2).unwrap();
        let input: Vec<_> = (0..frames).flat_map(|_| [0.75, -0.25]).collect();
        let output = converter.process_f64(&input, true).unwrap();
        for frame in output.chunks_exact(2) {
            assert!((frame[0] - 0.75).abs() < 2e-9);
            assert!((frame[1] + 0.25).abs() < 2e-9);
        }
    }
}

#[test]
fn buffered_reported_lookahead_controls_first_output() {
    let mut converter = BufferedResampler::new(96_000, 44_100, 1).unwrap();
    let info = converter.filter_info();
    assert_eq!(info.block_input_frames, 4096);
    assert_eq!(
        info.startup_lookahead_frames,
        info.block_input_frames + info.guard_input_frames
    );
    assert!(info.guard_input_frames > 0);
    assert!(info.guard_error_bound.is_finite() && info.guard_error_bound <= 1e-14);
    let before = vec![0.0; info.startup_lookahead_frames - 1];
    assert!(converter.process_f64(&before, false).unwrap().is_empty());
    let first = converter.process_f64(&[0.0], false).unwrap();
    assert_eq!(first.len(), length(info.block_input_frames, 96_000, 44_100));
    assert!(first.iter().all(|&x| x == 0.0));
    let tail = converter.process_f64(&[], true).unwrap();
    assert_eq!(
        first.len() + tail.len(),
        length(info.startup_lookahead_frames, 96_000, 44_100)
    );
    assert!(tail.iter().all(|&x| x == 0.0));
}

#[test]
fn buffered_exact_input_nyquist_outside_transition_is_rejected() {
    for (input_rate, output_rate) in [
        (96_000, 44_100),
        (44_100, 48_000),
        (48_000, 96_000),
        (192_000, 44_100),
    ] {
        if f64::from(input_rate) < f64::from(input_rate.min(output_rate)) * model::STOPBAND {
            continue;
        }
        // Exact +/-1 samples avoid trigonometric argument-reduction leakage.
        // At input Nyquist, the positive/negative frequency components can
        // coincide after interpolation; a single-frequency response bound
        // misses the resulting coherent sum.
        let input: Vec<_> = (0..input_rate as usize)
            .map(|frame| if frame % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let mut converter = BufferedResampler::new(input_rate, output_rate, 1).unwrap();
        let output = converter.process_f64(&input, true).unwrap();
        assert_eq!(output.len(), output_rate as usize);
        // The middle half crosses scheduling blocks and excludes file-edge
        // continuation effects. This remains the existing -180 dBFS gate.
        let peak = output[output_rate as usize / 4..3 * output_rate as usize / 4]
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0, f64::max);
        assert!(
            peak < 1e-9,
            "{input_rate}->{output_rate}: input Nyquist peak {peak:e}"
        );
    }
}
