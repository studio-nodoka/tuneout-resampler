//! Finite physical edge correction, evaluated only during plan setup.
use crate::complex::{Wide as W, WideComplex as WC};
#[path = "joint_edge_coefficients.rs"]
mod table;
const SUPPORT: f64 = 13230.;
const SPAN: f64 = SUPPORT / 104.;
fn envelope(t: f64, negative: bool) -> WC {
    if t >= SUPPORT {
        return WC::real(0.);
    }
    let i = ((t / SPAN) as usize).min(103);
    let x = 2. * (t / SPAN - i as f64) - 1.;
    let row = if negative {
        &table::NEGATIVE[i]
    } else {
        &table::POSITIVE[i]
    };
    let mut b1 = WC::real(0.);
    let mut b2 = b1;
    for &(re, im) in row[1..].iter().rev() {
        let b = WC {
            re: W::from(re),
            im: W::from(im),
        } + b1.scale(W::from(2. * x))
            - b2;
        b2 = b1;
        b1 = b;
    }
    // Signed envelopes preserve the asymmetric physical impulse.
    WC {
        re: W::from(row[0].0),
        im: W::from(row[0].1),
    } + b1.scale(W::from(x))
        - b2
}
pub(crate) fn apply(fir: &mut [WC], center: usize, end: usize, step: W) {
    for lag in 0..end {
        let t = (step * W::from(lag as f64)).value();
        fir[center + lag] = fir[center + lag] + envelope(t, false).scale(step);
        if lag != 0 {
            fir[center - lag] = fir[center - lag] + envelope(t, true).scale(step);
        }
    }
}

// The independent test operator sums forward Chebyshev polynomials instead
// of using the setup path's backward Clenshaw recurrence.
#[cfg(test)]
pub(crate) fn oracle_envelope(t: f64, negative: bool) -> WC {
    if t >= SUPPORT {
        return WC::real(0.);
    }
    let segment = ((t / SPAN) as usize).min(103);
    let x = W::from(2. * (t / SPAN - segment as f64) - 1.);
    let row = if negative {
        &table::NEGATIVE[segment]
    } else {
        &table::POSITIVE[segment]
    };
    let mut previous = W::from(1.);
    let mut current = x;
    let mut sum = WC::real(0.);
    for (degree, &(re, im)) in row.iter().enumerate() {
        let polynomial = if degree == 0 {
            previous
        } else if degree == 1 {
            current
        } else {
            let next = x * W::from(2.) * current - previous;
            previous = current;
            current = next;
            next
        };
        sum = sum
            + WC {
                re: W::from(re),
                im: W::from(im),
            }
            .scale(polynomial);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_envelopes_match_independent_forward_polynomials() {
        for negative in [false, true] {
            for segment in 0..104 {
                for fraction in [0., 0.123, 0.5, 0.999] {
                    let t = (segment as f64 + fraction) * SPAN;
                    let difference = (envelope(t, negative) - oracle_envelope(t, negative)).value();
                    assert!(difference.abs() < 1e-28, "t={t} negative={negative}");
                }
            }
            assert_eq!(envelope(SUPPORT, negative).value().abs(), 0.);
        }
    }
}
