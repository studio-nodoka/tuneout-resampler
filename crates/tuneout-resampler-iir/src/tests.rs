use super::*;
use std::f64::consts::PI;

const PAIRS: &[(u32, u32)] = &[
    (96000, 44100),
    (44100, 96000),
    (48000, 44100),
    (44100, 48000),
    (48000, 96000),
    (96000, 48000),
    (48000, 32000),
    (32000, 48000),
    (44100, 44100),
    (8000, 768000),
    (768000, 8000),
    (4096, 4095),
];

fn signal(frames: usize, channels: usize) -> Vec<f64> {
    (0..frames * channels)
        .map(|i| ((i * 1237 % 1013) as f64 - 506.0) / 1013.0)
        .collect()
}

fn bits(data: &[f64]) -> Vec<u64> {
    data.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn chunks_lengths_reset_and_finalization() {
    for &(input_rate, output_rate) in PAIRS {
        for channels in [1, 2, 6] {
            let input = signal(1031, channels);
            let mut converter = Resampler::new(input_rate, output_rate, channels).unwrap();
            let whole = converter.process_f64(&input, true).unwrap();
            let count = (1031_u64 * u64::from(output_rate)).div_ceil(u64::from(input_rate));
            assert_eq!(whole.len(), count as usize * channels);
            assert!(whole.iter().all(|x| x.is_finite()));
            for block in [1, 7, 64, 257] {
                converter.reset();
                let mut chunked = Vec::new();
                for chunk in input.chunks(block * channels) {
                    chunked.extend(converter.process_f64(chunk, false).unwrap());
                }
                chunked.extend(converter.process_f64(&[], true).unwrap());
                assert_eq!(
                    bits(&whole),
                    bits(&chunked),
                    "{input_rate}->{output_rate}, {channels}, {block}"
                );
                assert!(converter.process_f64(&[], true).unwrap().is_empty());
                assert_eq!(
                    converter
                        .process_f64(&input[..channels], false)
                        .unwrap_err(),
                    Error::StreamFinished
                );
            }
        }
    }
}

#[test]
fn validation_and_bypass_preserve_bits_and_state() {
    assert!(matches!(
        Resampler::new(0, 48000, 1),
        Err(Error::ZeroSampleRate)
    ));
    assert!(matches!(
        Resampler::new(48000, 0, 1),
        Err(Error::ZeroSampleRate)
    ));
    for channels in [0, 65, usize::MAX] {
        assert!(matches!(
            Resampler::new(48000, 44100, channels),
            Err(Error::InvalidChannels)
        ));
    }
    assert!(matches!(
        Resampler::new(48000, 44099, 1),
        Err(Error::RatioTooLarge { .. })
    ));
    let input = [-0.0, 0.0, f64::MIN_POSITIVE, f64::from_bits(1), 2.0, -2.0];
    let mut bypass = Resampler::new(48000, 48000, 2).unwrap();
    assert_eq!(
        bits(&input),
        bits(&bypass.process_f64(&input, true).unwrap())
    );
    let mut a = Resampler::new(96000, 44100, 2).unwrap();
    assert_eq!(
        a.process_f64(&[1.0], true).unwrap_err(),
        Error::IncompleteFrame
    );
    let input = signal(100, 2);
    let mut b = Resampler::new(96000, 44100, 2).unwrap();
    assert_eq!(
        bits(&a.process_f64(&input, true).unwrap()),
        bits(&b.process_f64(&input, true).unwrap())
    );
}

#[test]
fn empty_and_short_records_have_exact_lengths() {
    for &(input_rate, output_rate) in PAIRS {
        for frames in 0..17 {
            let mut converter = Resampler::new(input_rate, output_rate, 1).unwrap();
            let output = converter.process_f64(&signal(frames, 1), true).unwrap();
            assert_eq!(
                output.len(),
                (frames as u64 * u64::from(output_rate)).div_ceil(u64::from(input_rate)) as usize
            );
        }
    }
}

#[test]
fn independent_channels_and_f32_rounding() {
    let mut converter = Resampler::new(96000, 44100, 6).unwrap();
    let mono: Vec<f32> = signal(4000, 1).into_iter().map(|x| x as f32).collect();
    let input: Vec<f64> = mono
        .iter()
        .flat_map(|&v| [0.0, f64::from(v), 0.0, 0.0, 0.0, 0.0])
        .collect();
    let multichannel = converter.process_f64(&input, true).unwrap();
    let mut one = Resampler::new(96000, 44100, 1).unwrap();
    let narrow = one.process_f32(&mono, true).unwrap();
    for (frame, &expected) in multichannel.chunks_exact(6).zip(&narrow) {
        assert_eq!((frame[1] as f32).to_bits(), expected.to_bits());
        for c in [0, 2, 3, 4, 5] {
            assert_eq!(frame[c], 0.0);
        }
    }
}

// An independent realization of the original high-rate filter. It actually
// inserts zeros, runs cascaded real all-pass sections, and drops samples. This
// deliberately does not use the runtime's residues, phase rows or recurrence.
fn direct_reference(input: &[f64], converter: &Resampler) -> Vec<f64> {
    let d = converter.design.as_ref().unwrap();
    let alpha = d.warp;
    type Section = ([f64; 3], [f64; 3], [f64; 2]);
    let mut branches: [Vec<Section>; 2] = [Vec::new(), Vec::new()];
    for (i, &a) in d.coefficients.iter().enumerate() {
        // (a + t^2)/(1 + a*t^2), t=(q-alpha)/(1-alpha*q).
        let denominator = [
            1.0 + a * alpha * alpha,
            -2.0 * alpha * (1.0 + a),
            alpha * alpha + a,
        ];
        let numerator = [a + alpha * alpha, denominator[1], 1.0 + a * alpha * alpha];
        branches[i % 2].push((
            numerator.map(|x| x / denominator[0]),
            denominator.map(|x| x / denominator[0]),
            [0.0; 2],
        ));
    }
    branches[1].push(([-alpha, 1.0, 0.0], [1.0, -alpha, 0.0], [0.0; 2]));
    let mut output = Vec::new();
    for (n, &sample) in input.iter().enumerate() {
        for phase in 0..d.interpolation {
            let x = if phase == 0 {
                sample * f64::from(d.interpolation)
            } else {
                0.0
            };
            let mut y = 0.0;
            for sections in &mut branches {
                let mut v = x;
                for (b, a, state) in sections {
                    let out = b[0] * v + state[0];
                    state[0] = b[1] * v - a[1] * out + state[1];
                    state[1] = b[2] * v - a[2] * out;
                    v = out;
                }
                y += v * 0.5;
            }
            if (n * d.interpolation as usize + phase as usize).is_multiple_of(d.decimation as usize)
            {
                output.push(y);
            }
        }
    }
    output
}

#[test]
fn recursive_polyphase_matches_direct_allpass_reference() {
    for (a, b) in [(2, 3), (3, 2), (147, 160), (320, 147), (1, 7), (7, 1)] {
        let mut converter = Resampler::with_startup(a, b, 1, Startup::Zero).unwrap();
        let input = signal(1024, 1);
        let expected = direct_reference(&input, &converter);
        let output = converter.process_f64(&input, true).unwrap();
        assert_eq!(output.len(), expected.len());
        let error = output
            .iter()
            .zip(expected)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(error < 2e-9, "{a}->{b}: {error:e}");
    }
}

#[test]
fn dense_analytic_shape_has_strict_nyquist_rejection() {
    for maximum in [2, 3, 7, 147, 160, 320, 4096] {
        let converter = Resampler::new(maximum, 1, 1).unwrap();
        let mut pass_error: f64 = 0.0;
        let mut stop_peak: f64 = 0.0;
        for i in 0..=8192 {
            let f = i as f64 / 8192.0;
            let (re, im) = converter.frequency_response(0.96 * f);
            pass_error = pass_error.max((re.hypot(im) - 1.0).abs());
            for freq in [1.0 + f, 1.0 + (f64::from(maximum) - 1.0) * f] {
                let (re, im) = converter.frequency_response(freq);
                stop_peak = stop_peak.max(re.hypot(im));
            }
        }
        assert!(
            pass_error < 1e-8,
            "factor {maximum}: pass error {pass_error:e}"
        );
        assert!(
            stop_peak < 1e-9,
            "factor {maximum}: rejection {} dB",
            -20.0 * stop_peak.log10()
        );
    }
}

fn measure_tone(input_rate: u32, output_rate: u32, frequency: f64) -> (f64, f64) {
    let n = input_rate as usize;
    let input: Vec<_> = (0..n)
        .map(|i| (2.0 * PI * frequency * i as f64 / f64::from(input_rate)).cos())
        .collect();
    let mut converter = Resampler::new(input_rate, output_rate, 1).unwrap();
    let output = converter.process_f64(&input, true).unwrap();
    let output = &output[output_rate as usize / 2..];
    let rms = (output.iter().map(|x| x * x).sum::<f64>() / output.len() as f64).sqrt();
    let peak = output.iter().map(|x| x.abs()).fold(0.0, f64::max);
    (rms, peak)
}

#[test]
fn actual_stream_rejects_aliases_not_just_the_designed_curve() {
    for frequency in [22050.0, 22100.0, 23000.0, 30000.0, 47000.0] {
        let (rms, peak) = measure_tone(96000, 44100, frequency);
        assert!(peak < 1e-9, "{frequency}: peak {peak:e}, rms {rms:e}");
    }
    let (rms, _) = measure_tone(96000, 44100, 20000.0);
    assert!((rms - 0.5_f64.sqrt()).abs() < 1e-8, "passband {rms}");
}

#[test]
fn dc_linearity_and_silence_decay() {
    let mut converter = Resampler::new(96000, 44100, 1).unwrap();
    let mut input = vec![1.0; 48000];
    input.extend(vec![0.0; 48000]);
    let output = converter.process_f64(&input, true).unwrap();
    let dc_error = output[20000..22000]
        .iter()
        .map(|x| (x - 1.0).abs())
        .fold(0.0, f64::max);
    let tail = output[43000..].iter().map(|x| x.abs()).fold(0.0, f64::max);
    assert!(dc_error < 1e-10, "DC {dc_error:e}");
    assert!(tail < 1e-12, "tail {tail:e}");
    converter.reset();
    assert!(converter
        .process_f64(&vec![0.0; 10000], true)
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
    converter.reset();
    let a = signal(1000, 1);
    let x = converter.process_f64(&a, true).unwrap();
    converter.reset();
    let scaled: Vec<_> = a.iter().map(|x| x * 0.123).collect();
    let y = converter.process_f64(&scaled, true).unwrap();
    let error = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x * 0.123 - y).abs())
        .fold(0.0, f64::max);
    // The modal realization has cancelling terms. Bound its actual f64 error
    // at -200 dBFS; the prototype target is not an arithmetic guarantee.
    assert!(error < 1e-10, "linearity roundoff {error:e}");
}

#[test]
fn predictive_startup_rejects_stopband_tones_at_the_boundary() {
    for input_rate in [96000, 192000] {
        for hz in [
            22050.0, 22100.0, 22123.95, 23000.0, 28000.0, 40000.0, 47990.0,
        ] {
            for phase in [0.0, 0.7, 1.9] {
                let input: Vec<_> = (0..4096)
                    .map(|i| 0.8 * (2.0 * PI * hz * i as f64 / f64::from(input_rate) + phase).cos())
                    .collect();
                let mut zero =
                    Resampler::with_startup(input_rate, 44100, 1, Startup::Zero).unwrap();
                let mut predicted = Resampler::new(input_rate, 44100, 1).unwrap();
                let peak = |x: Vec<f64>| x.into_iter().map(f64::abs).fold(0.0, f64::max);
                let zero_peak = peak(zero.process_f64(&input, true).unwrap());
                let actual = peak(predicted.process_f64(&input, true).unwrap());
                assert!(
                    actual < 1e-7 && actual < zero_peak * 1e-5,
                    "{hz} Hz, phase {phase}: predicted {actual:e}, zero {zero_peak:e}"
                );
            }
        }
    }
}

#[test]
fn startup_buffering_short_records_and_adverse_inputs() {
    let mut src = Resampler::new(96000, 44100, 2).unwrap();
    assert_eq!(src.filter_info().startup_lookahead_frames, 256);
    assert!(src.process_f64(&vec![1.0; 510], false).unwrap().is_empty());
    let out = src.process_f64(&[1.0, 1.0], false).unwrap();
    assert_eq!(
        out.len(),
        2 * ((256_usize * 44100).div_ceil(96000) - src.filter_info().alignment_advance_frames)
    );
    assert!(out.iter().all(|v| (v - 1.0).abs() < 1e-8));
    src.reset();
    assert!(src.process_f64(&[0.5, -0.5], false).unwrap().is_empty());
    assert_eq!(
        src.process_f64(&[0.0], false).unwrap_err(),
        Error::IncompleteFrame
    );
    let short = src.process_f64(&[], true).unwrap();
    assert_eq!(short.len(), 2);
    assert!((short[0] - 0.5).abs() < 1e-8 && (short[1] + 0.5).abs() < 1e-8);
    for kind in 0..6 {
        let input: Vec<_> = (0..4096)
            .map(|i| match kind {
                0 => {
                    if i == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                1 => {
                    if i < 256 {
                        i as f64 / 255.0
                    } else {
                        0.0
                    }
                }
                2 => {
                    if i % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                3 => {
                    if i < 255 {
                        0.0
                    } else {
                        1.0
                    }
                }
                4 => (i as f64 * i as f64 * 0.0001).cos(),
                _ => ((i * 1171 % 1013) as f64 - 506.0) / 506.0,
            })
            .collect();
        for scale in [1.0, 1e-150] {
            let mut src = Resampler::new(96000, 44100, 1).unwrap();
            let x: Vec<_> = input.iter().map(|v| v * scale).collect();
            let y = src.process_f64(&x, true).unwrap();
            assert!(
                y.iter().all(|v| v.is_finite() && v.abs() < 8.0 * scale),
                "case {kind}"
            );
        }
    }
    let mut zero = Resampler::with_startup(96000, 44100, 1, Startup::Zero).unwrap();
    assert_eq!(zero.filter_info().startup_lookahead_frames, 0);
    assert_eq!(zero.process_f64(&[1.0], false).unwrap().len(), 1);
}

// Mixed signals exercise the boundary contract without fitting a frequency,
// shifting either output, or comparing against an independently reset track.
fn mixed_track_signal(frames: usize, input_rate: u32) -> Vec<f64> {
    let mut seed = 0x3f71_a2e6_31ba_628d_u64;
    (0..frames)
        .flat_map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let noise = (seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let t = i as f64 / f64::from(input_rate);
            let transient = if [250, 259, 3280, 3287, 4101].contains(&i) {
                0.2
            } else {
                0.0
            };
            [
                0.19 * (2.0 * PI * 631.0 * t).sin()
                    + 0.11 * (2.0 * PI * (1643.0 * t + 25000.0 * t * t)).cos()
                    + noise * 0.07
                    + transient,
                0.13 * (2.0 * PI * 4337.0 * t + 0.7).cos()
                    + 0.08 * (2.0 * PI * (907.0 * t + 37000.0 * t * t)).sin()
                    - noise * 0.09
                    - transient,
            ]
        })
        .collect()
}

#[test]
fn continuous_track_boundaries_keep_history_clock_and_pending_audio() {
    let cuts = [0, 1, 17, 255, 256, 257, 1023, 3287, 4097, 8193];
    for (input_rate, output_rate) in [
        (96000, 44100),
        (44100, 48000),
        (48000, 96000),
        (192000, 44100),
        (48000, 32000),
        (32000, 48000),
    ] {
        for timing in [Timing::Causal, Timing::DcAligned] {
            let input = mixed_track_signal(*cuts.last().unwrap(), input_rate);
            let mut converter =
                Resampler::with_options(input_rate, output_rate, 2, Startup::Predict, timing)
                    .unwrap();
            let whole = converter.process_f64(&input, true).unwrap();
            let expected_frames =
                (8193_u64 * u64::from(output_rate)).div_ceil(u64::from(input_rate));
            assert_eq!(whole.len(), 2 * expected_frames as usize);
            assert!(whole.iter().all(|x| x.is_finite()));
            converter.reset();
            let mut joined = Vec::new();
            for cut in cuts.windows(2) {
                // Track boundaries do not terminate the stream. Some tracks
                // end while the initial predictor is still buffering input.
                joined.extend(
                    converter
                        .process_f64(&input[2 * cut[0]..2 * cut[1]], false)
                        .unwrap(),
                );
                joined.extend(converter.process_f64(&[], false).unwrap());
            }
            joined.extend(converter.process_f64(&[], true).unwrap());
            assert_eq!(
                bits(&joined),
                bits(&whole),
                "continuous tracks {input_rate}->{output_rate}, {timing:?}"
            );
            assert!(converter.process_f64(&[], true).unwrap().is_empty());
        }
    }
}

#[test]
fn continuous_track_retains_the_actual_preceding_impulse_tail() {
    let mut left = vec![0.0; 3287];
    *left.last_mut().unwrap() = 0.8;
    let right = vec![0.0; 2048];
    let mut converter =
        Resampler::with_options(96000, 44100, 1, Startup::Zero, Timing::Causal).unwrap();
    let whole = converter
        .process_f64(&[left.as_slice(), right.as_slice()].concat(), true)
        .unwrap();
    converter.reset();
    let mut joined = converter.process_f64(&left, false).unwrap();
    let tail = converter.process_f64(&right, true).unwrap();
    assert!(tail.iter().map(|x| x.abs()).fold(0.0_f64, f64::max) > 0.1);
    joined.extend(tail);
    assert_eq!(bits(&joined), bits(&whole));

    // The identical right-hand file could also follow silence. Its samples
    // cannot identify the true left-hand history; preserving that history is
    // necessary for exact continuity, independent of startup prediction.
    converter.reset();
    assert!(converter
        .process_f64(&right, true)
        .unwrap()
        .iter()
        .all(|&x| x == 0.0));
}

#[test]
fn dc_aligned_zero_startup_matches_independent_delay_realization() {
    for (input_rate, output_rate) in [
        (1, 4096),
        (4096, 1),
        (4095, 4096),
        (4096, 4095),
        (96000, 44100),
    ] {
        let input = [0.8, 0.0, 0.0, 0.0];
        let mut aligned =
            Resampler::with_options(input_rate, output_rate, 1, Startup::Zero, Timing::DcAligned)
                .unwrap();
        let info = aligned.filter_info();
        assert_eq!(info.startup_lookahead_frames, 0);
        let (coefficients, warp) = aligned.prototype();
        let dc_delay = (1.0 + warp) / (1.0 - warp)
            * (0.5
                + coefficients
                    .iter()
                    .map(|a| (1.0 - a) / (1.0 + a))
                    .sum::<f64>())
            / f64::from(info.decimation);
        let advance = dc_delay.ceil() as usize;
        assert_eq!(info.alignment_advance_frames, advance);
        let fraction = advance as f64 - dc_delay;
        let coefficient = (1.0 - fraction) / (1.0 + fraction);
        let count =
            (input.len() as u64 * u64::from(output_rate)).div_ceil(u64::from(input_rate)) as usize;
        let required =
            ((count + advance - 1) as u64 * u64::from(input_rate)) / u64::from(output_rate) + 1;
        let mut extended = input.to_vec();
        extended.resize(required as usize, 0.0);
        let mut causal =
            Resampler::with_startup(input_rate, output_rate, 1, Startup::Zero).unwrap();
        let raw = causal.process_f64(&extended, true).unwrap();
        // A different first-order realization checks sign, fractional delay,
        // initial state, and integer advance independently of Alignment.
        let mut state = 0.0;
        let expected: Vec<_> = raw
            .iter()
            .map(|&sample| {
                if coefficient == 1.0 {
                    sample
                } else {
                    let value = coefficient * sample + state;
                    state = (1.0 - coefficient * coefficient) * sample - coefficient * state;
                    value
                }
            })
            .skip(advance)
            .take(count)
            .collect();
        // With four input samples ending in three zeros the one-order Burg
        // endpoint model predicts exactly zero. This matches the explicitly
        // extended reference, without sharing the extension implementation.
        let actual = aligned.process_f64(&input, true).unwrap();
        assert_eq!(actual.len(), count);
        assert_eq!(actual.len(), expected.len());
        let peak_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            peak_error < 1e-12,
            "{input_rate}->{output_rate}: alignment reference error {peak_error:e}"
        );
    }
}

#[test]
fn dc_aligned_empty_eos_and_short_chunks_preserve_contract() {
    for (input_rate, output_rate) in [
        (96000, 44100),
        (44100, 48000),
        (4096, 1),
        (1, 4096),
        (4095, 4096),
        (4096, 4095),
        (48000, 48000),
    ] {
        let lengths: &[usize] = if output_rate == 4096 && input_rate == 1 {
            &[0, 1, 2, 3, 7, 17]
        } else {
            &[0, 1, 2, 3, 15, 255, 256, 257]
        };
        for startup in [Startup::Zero, Startup::Predict] {
            let mut converter =
                Resampler::with_options(input_rate, output_rate, 2, startup, Timing::DcAligned)
                    .unwrap();
            for &frames in lengths {
                let input = mixed_track_signal(frames, input_rate);
                converter.reset();
                let whole = converter.process_f64(&input, true).unwrap();
                let count = (frames as u64 * u64::from(output_rate)).div_ceil(u64::from(input_rate))
                    as usize;
                assert_eq!(whole.len(), 2 * count);
                assert!(whole.iter().all(|x| x.is_finite()));
                converter.reset();
                assert!(converter.process_f64(&[], false).unwrap().is_empty());
                let mut chunked = Vec::new();
                for frame in input.chunks_exact(2) {
                    chunked.extend(converter.process_f64(frame, false).unwrap());
                    chunked.extend(converter.process_f64(&[], false).unwrap());
                }
                chunked.extend(converter.process_f64(&[], true).unwrap());
                assert_eq!(
                    bits(&chunked),
                    bits(&whole),
                    "short EOS {input_rate}->{output_rate}, {startup:?}, {frames} frames"
                );
                let mut reused_output = vec![f64::NAN; 7];
                converter
                    .process_f64_into(&[], true, &mut reused_output)
                    .unwrap();
                assert!(reused_output.is_empty());
                assert_eq!(
                    converter.process_f64(&[], false).unwrap_err(),
                    Error::StreamFinished
                );
            }
        }
    }
}
