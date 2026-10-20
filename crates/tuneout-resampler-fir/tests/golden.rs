//! Portable fingerprints pin coefficient and stream rounding on every target.
use tuneout_resampler::FilterLengthPolicy;
#[path = "support/fingerprints.rs"]
mod fingerprints;

#[track_caller]
fn assert_fingerprints(fixture: &str, header: &str, count: usize, actual: Vec<String>) {
    let mut expected = fixture.lines();
    assert_eq!(expected.next(), Some(header));
    let expected: Vec<_> = expected.collect();
    assert_eq!(expected.len(), count);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual, expected, "{header}");
    }
}

#[test]
fn complete_rate_matrix_matches_uniform_precision_fingerprints() {
    assert_fingerprints(
        include_str!("fixtures/uniform-precision.csv"),
        fingerprints::HEADER,
        357,
        fingerprints::rows(FilterLengthPolicy::Generic),
    );
}

#[test]
fn every_preset_matches_reference_banks_across_the_complete_rate_matrix() {
    assert_fingerprints(
        include_str!("fixtures/preset-banks.csv"),
        fingerprints::PRESET_HEADER,
        357 * 3,
        fingerprints::preset_rows(FilterLengthPolicy::Generic),
    );
}

#[test]
fn fixed_standard_matches_reference_coefficients_and_output_for_every_pair() {
    assert_fingerprints(
        include_str!("fixtures/fixed-precision.csv"),
        fingerprints::HEADER,
        357,
        fingerprints::rows(FilterLengthPolicy::Fixed),
    );
}

#[test]
fn fixed_presets_match_reference_banks_for_every_pair() {
    assert_fingerprints(
        include_str!("fixtures/fixed-preset-banks.csv"),
        fingerprints::PRESET_HEADER,
        357 * 3,
        fingerprints::preset_rows(FilterLengthPolicy::Fixed),
    );
}
