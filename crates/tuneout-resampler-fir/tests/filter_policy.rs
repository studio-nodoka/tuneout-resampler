use tuneout_resampler::{Error, FilterLength, FilterLengthPolicy, Resampler};

#[path = "support/assertions.rs"]
mod assertions;
use assertions::assert_bits_eq;

#[test]
fn existing_constructors_keep_the_generic_policy() {
    assert_eq!(FilterLengthPolicy::default(), FilterLengthPolicy::Generic);
    for (from, to) in [
        (48_000, 44_100),
        (44_100, 48_000),
        (48_000, 48_001),
        (48_000, 48_000),
    ] {
        let default = Resampler::new(from, to, 2).unwrap();
        for length in [
            FilterLength::Standard,
            FilterLength::Long,
            FilterLength::ExtraLong,
        ] {
            let implicit = Resampler::with_filter_length(from, to, 2, length).unwrap();
            let explicit = Resampler::with_filter_length_policy(
                from,
                to,
                2,
                length,
                FilterLengthPolicy::Generic,
            )
            .unwrap();
            assert_eq!(implicit.filter_length_policy(), FilterLengthPolicy::Generic);
            assert_eq!(implicit.filter_info(), explicit.filter_info());
            assert_bits_eq!(implicit.coefficients(), explicit.coefficients());
            if length == FilterLength::Standard {
                assert_eq!(default.filter_info(), explicit.filter_info());
                assert_bits_eq!(default.coefficients(), explicit.coefficients());
            }
        }
    }
}

#[test]
fn policy_changes_presets_but_custom_banks_and_reset_stay_shared() {
    // Exact, interpolated, decimating, capped and bypass configurations.
    for (from, to) in [
        (48_000, 44_100),
        (44_100, 48_000),
        (48_000, 48_001),
        (48_000, 24_000),
        (1, 2),
        (48_000, 48_000),
    ] {
        for length in [
            FilterLength::Standard,
            FilterLength::Long,
            FilterLength::ExtraLong,
        ] {
            let mut fixed = Resampler::with_filter_length_policy(
                from,
                to,
                2,
                length,
                FilterLengthPolicy::Fixed,
            )
            .unwrap();
            let info = fixed.filter_info();
            let taps = if info.bypass { 3 } else { info.taps_per_phase };
            let bank = fixed.coefficients().as_ptr();
            let input: Vec<f64> = (0..8192)
                .map(|i| f64::from(((i * 31 % 127) - 63) as f32 / 256.0))
                .collect();
            let expected = fixed.process_f64(&input, true).unwrap();
            for policy in [FilterLengthPolicy::Generic, FilterLengthPolicy::Fixed] {
                let mut custom = Resampler::with_filter_length_policy(
                    from,
                    to,
                    2,
                    FilterLength::Custom(taps),
                    policy,
                )
                .unwrap();
                assert_eq!(custom.filter_info(), info);
                assert_bits_eq!(custom.coefficients(), fixed.coefficients());
                if !info.bypass {
                    assert_eq!(custom.coefficients().as_ptr(), bank);
                }
                let mut output = Vec::new();
                for chunk in input.chunks(37 * 2) {
                    output.extend(custom.process_f64(chunk, false).unwrap());
                }
                output.extend(custom.process_f64(&[], true).unwrap());
                assert_bits_eq!(&output, &expected, "{from}->{to}, {length:?}, {policy:?}");
            }
            fixed.reset();
            assert_eq!(fixed.filter_length_policy(), FilterLengthPolicy::Fixed);
            assert_eq!(fixed.filter_info(), info);
            assert_eq!(fixed.coefficients().as_ptr(), bank);
            let input32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
            let output = fixed.process_f32(&input32, true).unwrap();
            assert_eq!(
                output.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|&x| (x as f32).to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn both_policies_reject_invalid_configuration_and_preserve_bypass_bits() {
    for policy in [FilterLengthPolicy::Generic, FilterLengthPolicy::Fixed] {
        let create = |from, to, channels, length| {
            Resampler::with_filter_length_policy(from, to, channels, length, policy)
        };
        for (from, to) in [(0, 48_000), (48_000, 0), (0, 0)] {
            assert!(matches!(
                create(from, to, 2, FilterLength::Standard),
                Err(Error::ZeroSampleRate)
            ));
        }
        assert!(matches!(
            create(48_000, 44_100, 0, FilterLength::Standard),
            Err(Error::ZeroChannels)
        ));
        assert!(matches!(
            create(48_000, 44_100, usize::MAX, FilterLength::Standard),
            Err(Error::SizeOverflow)
        ));
        for (from, to) in [(48_000, 44_100), (48_000, 48_001), (48_000, 48_000)] {
            for requested in [0, 1, 2, 4, usize::MAX] {
                assert!(
                    matches!(create(from, to, 2, FilterLength::Custom(requested)), Err(Error::InvalidFilterLength { requested: actual, .. }) if actual == requested)
                );
            }
        }
        for length in [
            FilterLength::Standard,
            FilterLength::Long,
            FilterLength::ExtraLong,
            FilterLength::Custom(3),
        ] {
            let mut bypass = create(48_000, 48_000, 1, length).unwrap();
            let input = [0.0, -0.0, 0.5 + 2.0_f64.powi(-40), -2.0];
            assert_bits_eq!(&bypass.process_f64(&input, true).unwrap(), &input);
            assert!(bypass.coefficients().is_empty());
            assert!(bypass.filter_info().bypass);
            assert_eq!(bypass.filter_length_policy(), policy);
        }
    }
}
