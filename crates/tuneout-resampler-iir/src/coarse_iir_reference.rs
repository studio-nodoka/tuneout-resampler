//! Independent DD center oracle retained from the pre-FIR modal realization.
use crate::complex::{Complex as C, Wide as W, WideComplex as WC};
use crate::prefix;
struct Center {
    past_first: usize,
    past: Vec<WC>,
    future_first: usize,
    future: Vec<WC>,
}
pub struct CoarseIir {
    center: Center,
}
impl CoarseIir {
    pub fn new(step: W) -> Self {
        let start = (prefix::B / step.value()).ceil() as usize;
        let mut finite = Vec::new();
        for j in 0..start {
            let t = step * W::from(j as f64);
            let window = prefix::turnon(t);
            if window.value() != 0. {
                let coefficient = prefix::envelope(t.value()).scale(window * step);
                finite.push((j, coefficient));
            }
        }
        debug_assert!(finite.windows(2).all(|p| p[0].0 + 1 == p[1].0));
        let center = Center {
            past_first: finite.last().map_or(0, |p| p.0),
            past: finite.iter().rev().map(|p| p.1).collect(),
            future_first: finite.iter().find(|p| p.0 != 0).map_or(0, |p| p.0),
            future: finite
                .iter()
                .filter(|p| p.0 != 0)
                .map(|p| p.1.conj())
                .collect(),
        };
        Self { center }
    }
    fn center_output(&self, q: i128, input: &[C], origin: i128) -> WC {
        let center = &self.center;
        let mut value = WC::real(0.);
        if !center.past.is_empty() {
            let first = usize::try_from(q - center.past_first as i128 - origin)
                .expect("retained turnon past");
            value = crate::wide_dot2::complex_wide_complex(
                &input[first..first + center.past.len()],
                &center.past,
            );
        }
        if !center.future.is_empty() {
            let first = usize::try_from(q + center.future_first as i128 - origin)
                .expect("retained turnon future");
            value = value
                + crate::wide_dot2::complex_wide_complex(
                    &input[first..first + center.future.len()],
                    &center.future,
                );
        }
        value
    }
}

#[cfg(test)]
mod center_tests {
    use super::*;
    fn wc(v: C) -> WC {
        WC {
            re: W::from(v.re),
            im: W::from(v.im),
        }
    }
    #[test]
    fn center_dot2_matches_independent_per_tap_dd_sum() {
        for d in [8., 14.7, 16.] {
            let f = CoarseIir::new(W::from(d));
            let input: Vec<_> = (0..300)
                .map(|i| {
                    C::new(
                        ((i * 31 % 71) as f64 - 35.) / 31.,
                        ((i * 11 % 97) as f64 - 48.) / 97.,
                    )
                })
                .collect();
            let mut maximum: f64 = 0.;
            for q in 100..132 {
                let mut expected = WC::real(0.);
                for lag in 0..(prefix::B / d).ceil() as usize {
                    let t = W::from(d) * W::from(lag as f64);
                    let window = prefix::turnon(t);
                    if window.value() != 0. {
                        let c = prefix::envelope(t.value()).scale(window * W::from(d));
                        expected = expected + c * wc(input[q - lag]);
                        if lag != 0 {
                            expected = expected + c.conj() * wc(input[q + lag]);
                        }
                    }
                }
                let actual = f.center_output(q as i128, &input, 0);
                maximum = maximum.max((actual - expected).value().abs());
            }
            println!(
                "turnon d={d} taps={} maximum error={maximum:e}",
                f.center.past.len()
            );
            assert!(maximum < 1e-27);
        }
    }
}
