//! Continuous compact-prefix correction, used only by the cold diagnostic.
//!
//! The stored Legendre pieces include the actual prefix window. Their Fourier
//! integrals follow DLMF 10.60.7; no ideal spectral rolloff is substituted.
#[path = "compact_stopband_coefficients.rs"]
mod coefficients;
const COUNT: usize = 65;

fn spherical(x: f64, sin: f64, cos: f64) -> [f64; COUNT] {
    let mut out = [0.; COUNT];
    if x < 0.5 {
        // DLMF 10.53.1 avoids subtractive cancellation and division at zero.
        let mut leading = 1.;
        for (n, value) in out.iter_mut().enumerate() {
            if n != 0 {
                leading *= x / (2 * n + 1) as f64;
            }
            let mut term = leading;
            *value = term;
            for k in 1..=16 {
                term *= -x * x / (2 * k * (2 * n + 2 * k + 1)) as f64;
                *value += term;
                if term.abs() <= value.abs() * f64::EPSILON {
                    break;
                }
            }
        }
        return out;
    }
    let j0 = sin / x;
    let j1 = (j0 - cos) / x;
    if x >= 96. {
        // Ascending recurrence is stable when the argument exceeds the order.
        out[0] = j0;
        out[1] = j1;
        for n in 1..COUNT - 1 {
            out[n + 1] = (2 * n + 1) as f64 / x * out[n] - out[n - 1];
        }
    } else {
        // Miller descent, rescaled to prevent overflow at the small-x boundary.
        // Normalize with the larger of j0/j1, including their individual zeros.
        let mut recurrence = [0.; 194];
        recurrence[192] = 1.;
        for n in (1..=192).rev() {
            recurrence[n - 1] = (2 * n + 1) as f64 / x * recurrence[n] - recurrence[n + 1];
            if recurrence[n - 1].abs() > 1e200 {
                for value in &mut recurrence[n - 1..] {
                    *value *= 1e-200;
                }
            }
        }
        let scale = if j0.abs() >= j1.abs() {
            j0 / recurrence[0]
        } else {
            j1 / recurrence[1]
        };
        for (value, basis) in out.iter_mut().zip(recurrence) {
            *value = basis * scale;
        }
    }
    out
}

pub(crate) fn response(nu: f64) -> f64 {
    if !nu.is_finite() {
        return f64::NAN;
    }
    let frequency = nu.abs();
    let cycle = frequency.rem_euclid(2.);
    let (sin, cos) = (std::f64::consts::PI * cycle).sin_cos();
    let basis = spherical(std::f64::consts::PI * frequency, sin, cos);
    let mut total = 0.;
    let mut residual = 0.;
    for (segment, row) in coefficients::LEGENDRE.iter().enumerate() {
        let center = (2 * segment + 1) as f64;
        let (sin, cos) = (std::f64::consts::PI * (cycle * center).rem_euclid(2.)).sin_cos();
        let phases = [cos, -sin, -cos, sin];
        for (n, (&coefficient, &integral)) in row.iter().zip(&basis).enumerate() {
            let term = 4. * coefficient * integral * phases[n % 4];
            let value = term - residual;
            let next = total + value;
            residual = (next - total) - value;
            total = next;
        }
    }
    total
}

#[cfg(test)]
pub(crate) fn oracle_delta(t: crate::complex::Wide) -> crate::complex::Wide {
    use crate::complex::Wide as W;
    if !(0. ..=160.).contains(&t.value()) {
        return W::from(0.);
    }
    let segment = ((t.value() / 2.).floor() as usize).min(79);
    let x = t - W::from((2 * segment + 1) as f64);
    let coefficients = &crate::compact_stopband_oracle::DELTA_CHEBYSHEV[segment];
    let mut b1 = W::from(0.);
    let mut b2 = b1;
    for &coefficient in coefficients[1..].iter().rev() {
        let value = W::from(coefficient) + W::from(2.) * x * b1 - b2;
        b2 = b1;
        b1 = value;
    }
    W::from(coefficients[0]) + x * b1 - b2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compact_stopband_oracle as oracle;
    use crate::complex::Wide as W;

    #[test]
    fn corrected_prefix_and_original_modal_keep_independent_precision() {
        let modes = crate::prefix::modes();
        for &(t, core_hi, core_lo, delta_hi, delta_lo) in oracle::HELDOUT {
            let t = W::from(t);
            let core = modes.iter().fold(W::from(0.), |sum, (s, c)| {
                sum + (*c * crate::prefix::exp(s.scale(t))).re
            }) * crate::cosine_shape::factor(t);
            let expected_core = W::from(core_hi) + W::from(core_lo);
            let expected_delta = W::from(delta_hi) + W::from(delta_lo);
            assert!((core - expected_core).value().abs() < 2e-28);
            assert!((oracle_delta(t) - expected_delta).value().abs() < 2e-28);
            let actual = crate::prefix::polynomial_core(t);
            assert!((actual - (expected_core + expected_delta)).value().abs() < 2e-28);
        }
    }

    #[test]
    fn compact_response_matches_independent_integrals_and_is_even() {
        for &(nu, expected) in oracle::RESPONSE {
            let actual = response(nu);
            assert!(
                (actual - expected).abs() < 3e-22,
                "nu={nu} actual={actual:e} expected={expected:e}"
            );
            assert_eq!(actual.to_bits(), response(-nu).to_bits());
        }
        assert!(response(f64::NAN).is_nan());
        assert!(response(f64::INFINITY).is_nan());
        assert!(response(f64::MAX).is_finite());
    }

    #[test]
    fn spherical_integrals_cover_series_descent_and_ascent_boundaries() {
        for &(x, values) in oracle::SPHERICAL {
            let (sin, cos) = x.sin_cos();
            let actual = spherical(x, sin, cos);
            for (order, expected) in values.into_iter().enumerate() {
                let error = (actual[order] - expected).abs();
                assert!(
                    error < 3e-14 * expected.abs() + 3e-17,
                    "x={x} order={order} error={error:e}"
                );
            }
        }
    }
}
