//! FixedFejerK4,T12288,blend9/40 time factor. Construction only.
use crate::complex::{Wide as W, WideComplex as WC};
pub(crate) const SHIFT_WEIGHTS: [u32; 7] = [9, 18, 27, 532, 27, 18, 9];
fn pi() -> W {
    W::from(std::f64::consts::PI) + W::from(1.2246467991473532e-16)
}
fn from_cosine(c: W) -> W {
    let c2 = c * c;
    let v = W::from(2.) * c2 - W::from(1.);
    let beta = W::from(0.225) + W::from(-5.551115123125783e-18);
    W::from(1.) + beta * (c2 * v * v - W::from(1.))
}
pub(crate) fn factor(t: W) -> W {
    from_cosine(
        crate::prefix::exp(WC {
            re: W::from(0.),
            im: pi() * t / W::from(24576.),
        })
        .re,
    )
}
pub(crate) fn apply(fir: &mut [WC], infinite: &[WC], center: usize, end: usize, step: W) {
    assert!(infinite.len() >= end);
    assert!(end <= center + 1 && fir.len() >= center + end);
    let pole = crate::prefix::exp(WC {
        re: W::from(0.),
        im: pi() * step / W::from(24576.),
    });
    let mut phase = WC::real(1.);
    for lag in 0..end {
        let delta = (fir[center + lag] + infinite[lag]).scale(from_cosine(phase.re) - W::from(1.));
        fir[center + lag] = fir[center + lag] + delta;
        if lag != 0 {
            fir[center - lag] = fir[center - lag] + delta.conj();
        }
        phase = phase * pole;
    }
}
#[cfg(test)]
#[path = "cosine_shape_oracle.rs"]
mod oracle;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cosine_shape_factor_matches_independent_mp() {
        let mut maximum: f64 = 0.;
        for &(t, hi, lo) in oracle::FACTOR {
            maximum = maximum.max(
                (factor(W::from(t)) - (W::from(hi) + W::from(lo)))
                    .value()
                    .abs(),
            );
        }
        println!("COSINE_FACTOR_MP {maximum:e}");
        assert!(maximum < 2e-29);
    }
    #[test]
    fn cosine_shape_prefix_matches_independent_mp() {
        let mut maximum: f64 = 0.;
        for &(t, hi, lo) in oracle::PREFIX {
            maximum = maximum.max(
                (crate::prefix::polynomial_core(W::from(t))
                    - crate::compact_stopband::oracle_delta(W::from(t))
                    - (W::from(hi) + W::from(lo)))
                .value()
                .abs(),
            );
        }
        println!("COSINE_PREFIX_MP {maximum:e}");
        assert!(maximum < 2e-28);
    }
    #[test]
    fn cosine_shape_recurrence_and_conjugate_count_match_direct_dd() {
        for d in [0.500625, 0.91875, 2.0025, 3.675, 4., 14.7] {
            let step = W::from(d);
            let center = (13230. / d).ceil() as usize;
            let end = center;
            let mut fir = vec![WC::real(0.); 2 * center + 1];
            let mut infinite = vec![WC::real(0.); end];
            for lag in 0..end {
                let a = WC {
                    re: W::from(((lag * 37 % 127) as f64 - 63.) / 127.),
                    im: W::from(((lag * 41 % 131) as f64 - 65.) / 131.),
                };
                fir[center + lag] = a;
                if lag != 0 {
                    fir[center - lag] = a.conj();
                }
                infinite[lag] = a.scale(W::from(0.25));
            }
            let original = fir.clone();
            apply(&mut fir, &infinite, center, end, step);
            let mut maximum: f64 = 0.;
            for lag in 0..end {
                let delta = (original[center + lag] + infinite[lag])
                    .scale(factor(step * W::from(lag as f64)) - W::from(1.));
                maximum = maximum.max(
                    (fir[center + lag] - (original[center + lag] + delta))
                        .value()
                        .abs(),
                );
                if lag != 0 {
                    maximum = maximum.max(
                        (fir[center - lag] - (original[center - lag] + delta.conj()))
                            .value()
                            .abs(),
                    );
                }
            }
            println!("COSINE_SETUP d{d} error{maximum:e}");
            assert!(maximum < 3e-27);
        }
    }
    #[test]
    fn one_sided_setup_preserves_full_hermitian_right_half_bits() {
        fn bits(z: WC) -> [u64; 4] {
            let (rh, rl) = z.re.parts();
            let (ih, il) = z.im.parts();
            [rh.to_bits(), rl.to_bits(), ih.to_bits(), il.to_bits()]
        }
        for count in [1, 4, 8, 16, 31, 32, 36] {
            for (first, end) in [(0, 1), (3, 19), (43, 521)] {
                let coefficients: Vec<_> = (0..count)
                    .map(|j| WC {
                        re: W::from((j * 37 % 127) as f64 / 127. - 0.5),
                        im: W::from((j * 41 % 131) as f64 / 131. - 0.5),
                    })
                    .collect();
                let poles: Vec<_> = (0..count)
                    .map(|j| {
                        crate::prefix::exp(WC {
                            re: W::from(-0.001 * (j + 1) as f64),
                            im: W::from(0.017 * (j + 1) as f64),
                        })
                    })
                    .collect();
                let mut full = vec![WC::real(0.); 2 * end + 1];
                let mut half = vec![WC::real(0.); end];
                crate::setup_modal::add_finite(&mut full, end, first, end, &coefficients, &poles);
                crate::setup_modal::add_causal(&mut half[first..end], &coefficients, &poles);
                for lag in 0..end {
                    assert_eq!(
                        bits(full[end + lag]),
                        bits(half[lag]),
                        "count{count} first{first} end{end} lag{lag}"
                    );
                }
            }
        }
    }
    #[test]
    fn cosine_shape_public_response_is_complete_and_bypass_is_preserved() {
        let shaped = crate::BufferedResampler::new(96000, 44100, 1).unwrap();
        let (nyquist, imag) = shaped.frequency_response(1.);
        println!("COSINE_PUBLIC_NYQUIST {nyquist:.17e} imag{imag:e}");
        assert!((nyquist - 0.4850679049432284).abs() < 2e-11);
        assert!(imag.abs() < 2e-11);
        let bypass = crate::BufferedResampler::new(48000, 48000, 1).unwrap();
        for nu in [0., 0.5, 1., 2.] {
            assert_eq!(bypass.frequency_response(nu), (1., 0.));
        }
    }
}
