//! Select filter support and construct phase-major C1 windowed-sinc coefficients.

use crate::{Error, FilterLength, FilterLengthPolicy};

const MAX_EXACT_PHASES: usize = 2_048;
const INTERPOLATED_PHASES: usize = 1_024;
const MAX_COEFFICIENT_VALUES: usize = 8_388_608; // 64 MiB of f64 coefficients.
const MAX_TAPS: usize = 65_535;
// Lookahead is half the support.
const MAX_SUPPORT_MILLISECONDS: u64 = 160;
// The default support policy for every rate pair. Downsampling widens the kernel in
// proportion to the bandwidth reduction. Upsampling uses the same base span.
const STANDARD_SUPPORT: u64 = 1_664;

#[derive(Clone, Copy)]
pub(crate) struct Design {
    pub(crate) phases: usize,
    pub(crate) taps: usize,
    pub(crate) beta: f64,
    pub(crate) rolloff: f64,
}

pub(crate) fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

pub(crate) fn select(
    from_rate: u32,
    to_rate: u32,
    filter_length: FilterLength,
    policy: FilterLengthPolicy,
) -> Result<Design, Error> {
    let denominator = (to_rate / gcd(from_rate, to_rate)) as usize;
    // The reduced output-rate denominator counts distinct fractional positions.
    let exact = denominator <= MAX_EXACT_PHASES;
    let phases = if exact {
        denominator
    } else {
        INTERPOLATED_PHASES
    };
    // Include the extra interpolation row in the storage budget, and round
    // the smallest resource limit down to an odd tap count.
    let limit = ((MAX_COEFFICIENT_VALUES / (phases + 1))
        .min(MAX_TAPS)
        .min((u64::from(from_rate) * MAX_SUPPORT_MILLISECONDS / 1000) as usize)
        .max(3)
        - 1)
        | 1;
    let standard_taps = match policy {
        FilterLengthPolicy::Generic => {
            let support = (STANDARD_SUPPORT * u64::from(from_rate))
                .div_ceil(u64::from(from_rate.min(to_rate)));
            (support.min(limit as u64) as usize) | 1
        }
        FilterLengthPolicy::Fixed => fixed_standard_taps(from_rate, to_rate, exact).min(limit),
    };
    // Capping before scaling gives the same result as capping afterwards for
    // these multipliers >= 1, and bounds the integer arithmetic on 32-bit hosts.
    let taps = match filter_length {
        FilterLength::Standard => standard_taps,
        FilterLength::Long => ((standard_taps * 7).div_ceil(4) | 1).min(limit),
        FilterLength::ExtraLong => ((standard_taps * 2) | 1).min(limit),
        FilterLength::Custom(taps) => {
            if taps < 3 || taps.is_multiple_of(2) || taps > limit {
                return Err(Error::InvalidFilterLength {
                    requested: taps,
                    maximum: limit,
                });
            }
            taps
        }
    };
    Ok(Design {
        phases,
        taps,
        beta: 20.0,
        rolloff: 1.0,
    })
}

// Fixed minima and their established floating-point rounding select support only.
// Coefficient generation and SIMD stay common to both policies.
fn fixed_standard_taps(from_rate: u32, to_rate: u32, exact: bool) -> usize {
    let ratio = (f64::from(to_rate) / f64::from(from_rate)).min(1.0);
    let minimum = if !exact {
        0
    } else if to_rate == 44_100 && from_rate > to_rate {
        if from_rate >= 176_400 {
            2_815
        } else if from_rate >= 96_000 {
            3_583
        } else if from_rate >= 88_200 {
            1_919
        } else {
            1_663
        }
    } else if from_rate < to_rate {
        if from_rate <= 44_100 {
            1_151
        } else if from_rate <= 48_000 {
            959
        } else {
            767
        }
    } else if ratio <= 0.70 {
        767
    } else if ratio < 0.85 {
        639
    } else {
        0
    };
    minimum.max((512.0 / ratio).ceil() as usize | 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{precision::coefficients, test_assertions::assert_bits_eq};

    #[test]
    fn fixed_thresholds_retain_standard_lengths() {
        // Established Fixed lengths, including the
        // non-monotonic minima, interpolation switch and decimation fallback.
        for (from, to, taps) in [
            (44_075, 88_150, 1_151),
            (44_100, 88_200, 1_151),
            (44_125, 88_250, 959),
            (48_000, 96_000, 959),
            (48_025, 96_050, 767),
            (44_125, 44_100, 1_663),
            (88_175, 44_100, 1_663),
            (88_200, 44_100, 1_919),
            (88_225, 44_100, 1_919),
            (95_975, 44_100, 1_919),
            (96_000, 44_100, 3_583),
            (96_025, 44_100, 3_583),
            (176_375, 44_100, 3_583),
            (176_400, 44_100, 2_815),
            (176_425, 44_100, 2_815),
            (96_001, 44_100, 1_115),
            (100_000, 69_900, 767),
            (100_000, 70_000, 767),
            (100_000, 70_100, 731),
            (100_000, 84_900, 639),
            (100_000, 85_000, 603),
            (100_000, 85_100, 603),
            (65_504_000, 65_536_000, 767),
            (65_536_000, 65_568_000, 513),
        ] {
            let design =
                select(from, to, FilterLength::Standard, FilterLengthPolicy::Fixed).unwrap();
            assert_eq!(design.taps, taps, "{from}->{to}");
        }
    }

    #[test]
    fn scaling_both_rates_preserves_the_filter_when_resource_limits_do_not_bind() {
        for numerator in 1..=9 {
            for denominator in 1..=9 {
                if gcd(numerator, denominator) != 1 || numerator == denominator {
                    continue;
                }
                for length in [
                    FilterLength::Standard,
                    FilterLength::Long,
                    FilterLength::ExtraLong,
                ] {
                    let from = numerator * 32_000;
                    let to = denominator * 32_000;
                    let first = select(from, to, length, FilterLengthPolicy::Generic).unwrap();
                    let scaled =
                        select(from * 2, to * 2, length, FilterLengthPolicy::Generic).unwrap();
                    assert_eq!(first.taps, scaled.taps);
                    assert_eq!(first.phases, scaled.phases);
                    assert_bits_eq!(
                        &coefficients(from, to, first),
                        &coefficients(from * 2, to * 2, scaled),
                        "{from}->{to}, {length:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn preset_and_custom_limits_cover_support_time_bank_size_and_extreme_rates() {
        // Independent boundary cases for the minimum, time, bank, and tap caps.
        for (from, to, maximum) in [
            (1, 2, 3),
            (8_000, 11_025, 1_279),
            (48_000, 44_101, 7_679),
            (128_000, 128_001, 8_183),
            (u32::MAX, u32::MAX - 4, 8_183),
            (u32::MAX, 1, 65_535),
        ] {
            for policy in [FilterLengthPolicy::Generic, FilterLengthPolicy::Fixed] {
                let mut previous = 0;
                for length in [
                    FilterLength::Standard,
                    FilterLength::Long,
                    FilterLength::ExtraLong,
                ] {
                    let design = select(from, to, length, policy).unwrap();
                    assert!(design.taps >= 3 && design.taps >= previous && design.taps <= maximum);
                    assert_eq!(design.taps % 2, 1);
                    assert!((design.phases + 1) * design.taps * 8 <= 64 * 1024 * 1024);
                    previous = design.taps;
                }
                assert_eq!(
                    select(from, to, FilterLength::Custom(maximum), policy)
                        .unwrap()
                        .taps,
                    maximum
                );
                assert!(matches!(
                    select(from, to, FilterLength::Custom(maximum + 2), policy),
                    Err(Error::InvalidFilterLength { maximum: actual, .. }) if actual == maximum
                ));
            }
        }
    }
}
