//! Constructor-only bounded DD roots with fixed128-step anchors.
//! No rate-specific coefficient bank or process-global cache.
use crate::complex::{Wide, WideComplex};
fn cis(angle: Wide) -> WideComplex {
    let z = WideComplex {
        re: Wide::from(0.),
        im: angle * Wide::from(1. / 16.),
    };
    let mut term = WideComplex::real(1.);
    let mut sum = term;
    for k in 1..=32 {
        term = (term * z).scale(Wide::from(1.) / Wide::from(k as f64));
        sum = sum + term;
    }
    for _ in 0..4 {
        sum = sum * sum;
    }
    sum
}

#[inline(always)]
fn negative(z: Wide) -> Wide {
    let (a, b) = z.parts();
    Wide::from_product_parts(-a, -b)
}
fn fill_roots(roots: &mut [WideComplex], n: usize, scale: Wide, reanchor: usize) {
    if roots.is_empty() {
        return;
    }
    let step = cis(scale);
    let end = (if n.is_multiple_of(4) { n / 8 } else { n / 2 }).min(roots.len() - 1);
    for first in (0..=end).step_by(reanchor) {
        let mut z = cis(scale * Wide::from(first as f64));
        for root in &mut roots[first..=(first + reanchor - 1).min(end)] {
            *root = z;
            z = z * step;
        }
    }
    // Negation and component exchange are exact and create no new phase error.
    roots[0] = WideComplex::real(1.);
    if n.is_multiple_of(4) {
        let q = n / 4;
        if n.is_multiple_of(8) && n / 8 < roots.len() {
            let initial = Wide::from(0.5f64.sqrt());
            let c = initial + (Wide::from(0.5) - initial * initial) / (Wide::from(2.) * initial);
            roots[n / 8] = WideComplex {
                re: c,
                im: negative(c),
            };
        }
        for k in 1..=end {
            let at = q - k;
            if at != k && at < roots.len() {
                let z = roots[k];
                roots[at] = WideComplex {
                    re: negative(z.im),
                    im: negative(z.re),
                };
            }
        }
        for k in 0..q.min(roots.len()) {
            let z = roots[k];
            if k + q < roots.len() {
                roots[k + q] = WideComplex {
                    re: z.im,
                    im: negative(z.re),
                };
            }
            if k + 2 * q < roots.len() {
                roots[k + 2 * q] = WideComplex {
                    re: negative(z.re),
                    im: negative(z.im),
                };
            }
            if k + 3 * q < roots.len() {
                roots[k + 3 * q] = WideComplex {
                    re: negative(z.im),
                    im: z.re,
                };
            }
        }
    } else {
        for k in 1..=((n - 1) / 2).min(roots.len() - 1) {
            if n - k < roots.len() {
                let z = roots[k];
                roots[n - k] = WideComplex {
                    re: z.re,
                    im: negative(z.im),
                };
            }
        }
        if n.is_multiple_of(2) && n / 2 < roots.len() {
            roots[n / 2] = WideComplex::real(-1.);
        }
    }
}

/// First `count` roots exp(-2*pi*i*k/n), with DD arithmetic and exact
/// conjugate/quarter symmetry. This also serves real FFT postprocessing without
/// constructing an otherwise unused full-size Plan. A fixed 128-step reanchor
/// bounds recurrence drift; no process-global cache or FFT-size restriction.
pub(crate) fn unit_circle(n: usize, count: usize) -> Option<Vec<WideComplex>> {
    if n == 0 || n > 65_536 || count > n {
        return None;
    }
    let mut roots = Vec::new();
    roots.try_reserve_exact(count).ok()?;
    roots.resize(count, WideComplex::real(0.));
    let pi = Wide::from(std::f64::consts::PI) + Wide::from(1.224_646_799_147_353_2e-16);
    let scale = (pi * Wide::from(-2.)) / Wide::from(n as f64);
    fill_roots(&mut roots, n, scale, 128);
    Some(roots)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constructor_lattice_roots_match_independent_dd_exponentials() {
        let pi = Wide::from(std::f64::consts::PI) + Wide::from(1.224_646_799_147_353_2e-16);
        for n in [
            1, 2, 3, 4, 5, 7, 8, 17, 64, 80, 147, 160, 294, 320, 640, 1280, 65521, 65536,
        ] {
            let roots = unit_circle(n, n).unwrap();
            let mut maximum: f64 = 0.;
            for j in (0..n).step_by((n / 127).max(1)).chain([0, n / 2, n - 1]) {
                let expected = crate::prefix::exp(WideComplex {
                    re: Wide::from(0.),
                    im: pi * Wide::from(-2.) * Wide::from(j as f64) / Wide::from(n as f64),
                });
                maximum = maximum.max((roots[j] - expected).value().abs());
                if j != 0 {
                    let mirror = roots[n - j];
                    assert!((roots[j] - mirror.conj()).value().abs() < 2e-28);
                }
            }
            assert!(maximum < 3e-28, "n{n},DDerror{maximum:e}");
            println!("SETUP_ROOTS,n={n},max_DD={maximum:e}");
            assert_eq!(roots[0].re.parts(), (1., 0.));
            assert_eq!(roots[0].im.parts(), (0., 0.));
            if n % 4 == 0 {
                assert_eq!(roots[n / 4].re.value(), 0.);
                assert_eq!(roots[n / 4].im.value(), -1.);
            }
        }
        assert!(unit_circle(65537, 1).is_none());
        assert!(unit_circle(0, 0).is_none());
    }
}
