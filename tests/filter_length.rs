use tuneout_resampler::{Error, FilterLength, Resampler};
#[path = "support/cases.rs"]
mod cases;

#[path = "support/assertions.rs"]
mod assertions;
use assertions::assert_bits_eq;

#[test]
fn presets_and_equivalent_custom_lengths_share_the_same_filter_for_every_pair() {
    for (from, to) in cases::pairs() {
        let mut previous = 0;
        for length in [
            FilterLength::Standard,
            FilterLength::Long,
            FilterLength::ExtraLong,
        ] {
            let preset = Resampler::with_filter_length(from, to, 2, length).unwrap();
            let info = preset.filter_info();
            if info.bypass {
                assert!(preset.coefficients().is_empty());
                continue;
            }
            let taps = info.taps_per_phase;
            assert!(taps >= previous && taps % 2 == 1);
            assert_eq!(info.lookahead_input_frames, taps / 2);
            assert_eq!(info.coefficient_bytes, (info.phases + 1) * taps * 8);
            let custom =
                Resampler::with_filter_length(from, to, 1, FilterLength::Custom(taps)).unwrap();
            assert_bits_eq!(custom.coefficients(), preset.coefficients());
            previous = taps;
        }
    }
    assert_eq!(FilterLength::default(), FilterLength::Standard);
}

#[test]
fn configured_lengths_preserve_streaming_channels_reset_and_f32_for_every_pair() {
    let frames = 257;
    let input: Vec<f64> = (0..frames * 2)
        .map(|i| {
            let bits = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f64::from(((bits >> 32) as i32 as f64 / 4294967296.0) as f32)
        })
        .collect();
    for (from, to) in cases::pairs() {
        for length in [
            FilterLength::Long,
            FilterLength::ExtraLong,
            FilterLength::Custom(3),
            FilterLength::Custom(65),
        ] {
            let mut converter = Resampler::with_filter_length(from, to, 2, length).unwrap();
            let info = converter.filter_info();
            let bank = converter.coefficients().as_ptr();
            let expected = converter.process_f64(&input, true).unwrap();
            assert_eq!(
                expected.len() / 2,
                (frames as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize
            );
            assert!(expected.iter().all(|value| value.is_finite()));
            let mono_input: Vec<f64> = input.iter().step_by(2).copied().collect();
            let mut mono = Resampler::with_filter_length(from, to, 1, length).unwrap();
            let mono_output = mono.process_f64(&mono_input, true).unwrap();
            assert!(expected
                .as_chunks::<2>()
                .0
                .iter()
                .zip(mono_output)
                .all(|(stereo, mono)| stereo[0].to_bits() == mono.to_bits()));
            converter.reset();
            assert_eq!(converter.coefficients().as_ptr(), bank);
            assert_eq!(converter.filter_info(), info);
            let mut actual = Vec::new();
            let mut remaining = input.as_slice();
            for block in [1, 17, 113, 257].into_iter().cycle() {
                if remaining.is_empty() {
                    break;
                }
                let (chunk, tail) = remaining.split_at((block * 2).min(remaining.len()));
                actual.extend(converter.process_f64(chunk, false).unwrap());
                remaining = tail;
            }
            actual.extend(converter.process_f64(&[], true).unwrap());
            assert_bits_eq!(&actual, &expected, "{from}->{to}, {length:?}");
            assert!(converter.process_f64(&[], true).unwrap().is_empty());
            converter.reset();
            let mut narrowed = Vec::new();
            for chunk in input.chunks(67 * 2) {
                let chunk: Vec<f32> = chunk.iter().map(|&x| x as f32).collect();
                narrowed.extend(converter.process_f32(&chunk, false).unwrap());
            }
            narrowed.extend(converter.process_f32(&[], true).unwrap());
            assert_eq!(
                narrowed.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|&x| (x as f32).to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn custom_lengths_are_validated_for_every_pair_including_bypass() {
    for (from, to) in cases::pairs() {
        for requested in [0, 1, 2, 4, usize::MAX] {
            assert!(
                matches!(Resampler::with_filter_length(from, to, 2, FilterLength::Custom(requested)),
                Err(Error::InvalidFilterLength { requested: actual, .. }) if actual == requested)
            );
        }
        if from == to {
            for length in [
                FilterLength::Standard,
                FilterLength::Long,
                FilterLength::ExtraLong,
                FilterLength::Custom(3),
            ] {
                let mut converter = Resampler::with_filter_length(from, to, 1, length).unwrap();
                let input = [0.0, -0.0, 0.5 + 2.0_f64.powi(-40), -2.0];
                assert_bits_eq!(&converter.process_f64(&input, true).unwrap(), &input);
                assert!(converter.filter_info().bypass);
                assert_eq!(converter.filter_info().taps_per_phase, 0);
                assert!(converter.coefficients().is_empty());
            }
        }
    }
}
