use tuneout_resampler::Resampler;
#[path = "support/cases.rs"]
mod cases;

#[test]
fn coefficients_match_independent_high_precision_reference() {
    let mut pairs = std::collections::BTreeSet::new();
    let mut previous = (0, 0);
    let mut converter = None;
    let mut checked = 0;
    for line in include_str!("fixtures/reference-coefficients.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 7);
        let pair = (fields[0].parse().unwrap(), fields[1].parse().unwrap());
        if pair != previous {
            pairs.insert(pair);
            converter = Some(Resampler::new(pair.0, pair.1, 1).unwrap());
            previous = pair;
        }
        let r = converter.as_ref().unwrap();
        let info = r.filter_info();
        assert!(!info.uses_precise_c1_bank);
        let phases: usize = fields[2].parse().unwrap();
        let taps: usize = fields[3].parse().unwrap();
        assert_eq!((info.phases, info.taps_per_phase), (phases, taps));
        let phase: usize = fields[4].parse().unwrap();
        let tap: usize = fields[5].parse().unwrap();
        let expected = f64::from_bits(u64::from_str_radix(fields[6], 16).unwrap());
        let actual = r.coefficients()[phase * taps + tap];
        if expected == 0.0 {
            // The real-valued reference has no sign for exact zero.
            assert_eq!(actual, 0.0, "{line}");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{line}");
        }
        checked += 1;
    }
    let expected: std::collections::BTreeSet<_> = cases::pairs()
        .into_iter()
        .filter(|(from, to)| from != to)
        .chain([(1, 768_000), (u32::MAX, u32::MAX - 4)])
        .collect();
    assert_eq!(pairs, expected);
    assert_eq!(checked, 15_818);
}

#[test]
fn reflected_phases_and_interpolation_endpoint_agree() {
    for (from, to) in cases::pairs() {
        let r = Resampler::new(from, to, 1).unwrap();
        let info = r.filter_info();
        let bank = r.coefficients();
        let taps = info.taps_per_phase;
        if info.bypass {
            continue;
        }
        for phase in 1..info.phases {
            let row = &bank[phase * taps..(phase + 1) * taps];
            let opposite = &bank[(info.phases - phase) * taps..(info.phases - phase + 1) * taps];
            assert_eq!(row[0], 0.0);
            for tap in 1..taps {
                assert_eq!(row[tap], opposite[taps - tap]);
            }
        }
        assert_eq!(bank[info.phases * taps], 0.0);
        assert_eq!(&bank[info.phases * taps + 1..], &bank[..taps - 1]);
    }
}

#[test]
fn unity_cutoff_retains_exact_integer_sinc_zeros() {
    for (from, to) in cases::pairs().into_iter().filter(|(from, to)| from < to) {
        let r = Resampler::new(from, to, 1).unwrap();
        let taps = r.filter_info().taps_per_phase;
        for (tap, &coefficient) in r.coefficients()[..taps].iter().enumerate() {
            assert_eq!(coefficient, if tap == taps / 2 { 1.0 } else { 0.0 });
        }
    }
}
