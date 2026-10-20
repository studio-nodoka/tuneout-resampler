//! Constructor-only exact DD modal trajectories, independent modes per lane.
//! Every FIR cell retains the scalar mode-addition order.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    test
))]
use crate::complex::Wide;
use crate::complex::WideComplex;

pub(crate) fn add_finite(
    fir: &mut [WideComplex],
    center: usize,
    first: usize,
    end: usize,
    coefficients: &[WideComplex],
    poles: &[WideComplex],
) {
    assert_eq!(coefficients.len(), poles.len());
    assert!(first <= end && end <= center + 1 && fir.len() > 2 * center);
    dispatch::<true>(fir, center, first, end, coefficients, poles);
}

pub(crate) fn add_causal(
    fir: &mut [WideComplex],
    coefficients: &[WideComplex],
    poles: &[WideComplex],
) {
    assert_eq!(coefficients.len(), poles.len());
    let end = fir.len();
    dispatch::<false>(fir, 0, 0, end, coefficients, poles);
}

fn dispatch<const HERMITIAN: bool>(
    fir: &mut [WideComplex],
    center: usize,
    first: usize,
    end: usize,
    coefficients: &[WideComplex],
    poles: &[WideComplex],
) {
    if first == end || coefficients.is_empty() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            unsafe { x86::avx512::<HERMITIAN>(fir, center, first, end, coefficients, poles) };
            return;
        }
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            unsafe { x86::avx::<HERMITIAN>(fir, center, first, end, coefficients, poles) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { neon::construct::<HERMITIAN>(fir, center, first, end, coefficients, poles) };
        return;
    }
    scalar::<HERMITIAN>(fir, center, first, end, coefficients, poles);
}

fn scalar<const HERMITIAN: bool>(
    fir: &mut [WideComplex],
    center: usize,
    first: usize,
    end: usize,
    coefficients: &[WideComplex],
    poles: &[WideComplex],
) {
    for (&coefficient, &pole) in coefficients.iter().zip(poles) {
        let mut value = coefficient;
        for lag in first..end {
            fir[center + lag] = fir[center + lag] + value;
            if HERMITIAN && lag != 0 {
                fir[center - lag] = fir[center - lag] + value.conj();
            }
            value = value * pole;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! implementation {
    ($name:ident, $feature:literal, $lanes:expr, $v:ty,
     $zero:ident, $load:ident, $store:ident, $add:ident, $sub:ident,
     $mul:ident, $fma:ident, $neg:ident) => {
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $name<const HERMITIAN: bool>(
            fir: &mut [WideComplex],
            center: usize,
            first: usize,
            end: usize,
            coefficients: &[WideComplex],
            poles: &[WideComplex],
        ) {
            unsafe {
                #[derive(Clone, Copy)]
                struct D {
                    hi: $v,
                    lo: $v,
                }
                #[derive(Clone, Copy)]
                struct Z {
                    re: D,
                    im: D,
                }
                macro_rules! norm {
                    ($hi:expr, $lo:expr) => {{
                        let hi = $hi;
                        let lo = $lo;
                        let sum = $add(hi, lo);
                        D {
                            hi: sum,
                            lo: $sub(lo, $sub(sum, hi)),
                        }
                    }};
                }
                macro_rules! plus {
                    ($a:expr, $b:expr) => {{
                        let a = $a;
                        let b = $b;
                        let sum = $add(a.hi, b.hi);
                        let v = $sub(sum, a.hi);
                        let error = $add($sub(a.hi, $sub(sum, v)), $sub(b.hi, v));
                        norm!(sum, $add($add(error, a.lo), b.lo))
                    }};
                }
                macro_rules! minus {
                    ($a:expr, $b:expr) => {{
                        let b = $b;
                        plus!(
                            $a,
                            D {
                                hi: $neg(b.hi),
                                lo: $neg(b.lo)
                            }
                        )
                    }};
                }
                macro_rules! times {
                    ($a:expr, $b:expr) => {{
                        let a = $a;
                        let b = $b;
                        let product = $mul(a.hi, b.hi);
                        let error = $fma(a.hi, b.hi, $neg(product));
                        norm!(
                            product,
                            $add(
                                $add($add(error, $mul(a.hi, b.lo)), $mul(a.lo, b.hi)),
                                $mul(a.lo, b.lo)
                            )
                        )
                    }};
                }
                let mut rh = [0.; $lanes];
                let mut rl = rh;
                let mut ih = rh;
                let mut il = rh;
                let mut pr = rh;
                let mut pi = rh;
                let mut prl = rh;
                let mut pil = rh;
                for batch in (0..coefficients.len()).step_by($lanes) {
                    let count = $lanes.min(coefficients.len() - batch);
                    for lane in 0..$lanes {
                        let index = batch + lane.min(count - 1);
                        (rh[lane], rl[lane]) = coefficients[index].re.parts();
                        (ih[lane], il[lane]) = coefficients[index].im.parts();
                        (pr[lane], prl[lane]) = poles[index].re.parts();
                        (pi[lane], pil[lane]) = poles[index].im.parts();
                    }
                    let mut value = Z {
                        re: D {
                            hi: $load(rh.as_ptr()),
                            lo: $load(rl.as_ptr()),
                        },
                        im: D {
                            hi: $load(ih.as_ptr()),
                            lo: $load(il.as_ptr()),
                        },
                    };
                    let pole = Z {
                        re: D {
                            hi: $load(pr.as_ptr()),
                            lo: $load(prl.as_ptr()),
                        },
                        im: D {
                            hi: $load(pi.as_ptr()),
                            lo: $load(pil.as_ptr()),
                        },
                    };
                    for lag in first..end {
                        $store(rh.as_mut_ptr(), value.re.hi);
                        $store(rl.as_mut_ptr(), value.re.lo);
                        $store(ih.as_mut_ptr(), value.im.hi);
                        $store(il.as_mut_ptr(), value.im.lo);
                        // Preserve the legacy accumulation order at both lags.
                        for lane in 0..count {
                            let c = WideComplex {
                                re: Wide::from_product_parts(rh[lane], rl[lane]),
                                im: Wide::from_product_parts(ih[lane], il[lane]),
                            };
                            fir[center + lag] = fir[center + lag] + c;
                            if HERMITIAN && lag != 0 {
                                fir[center - lag] = fir[center - lag] + c.conj();
                            }
                        }
                        value = Z {
                            re: minus!(times!(value.re, pole.re), times!(value.im, pole.im)),
                            im: plus!(times!(value.re, pole.im), times!(value.im, pole.re)),
                        };
                    }
                }
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn neg4(v: __m256d) -> __m256d {
        unsafe { _mm256_xor_pd(v, _mm256_set1_pd(-0.)) }
    }
    #[inline(always)]
    unsafe fn neg8(v: __m512d) -> __m512d {
        unsafe {
            _mm512_castsi512_pd(_mm512_xor_si512(
                _mm512_castpd_si512(v),
                _mm512_set1_epi64(i64::MIN),
            ))
        }
    }
    implementation!(
        avx512,
        "avx512f,fma",
        8,
        __m512d,
        _mm512_setzero_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        _mm512_sub_pd,
        _mm512_mul_pd,
        _mm512_fmadd_pd,
        neg8
    );
    implementation!(
        avx,
        "avx,fma",
        4,
        __m256d,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        _mm256_sub_pd,
        _mm256_mul_pd,
        _mm256_fmadd_pd,
        neg4
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    #[inline(always)]
    unsafe fn fma(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        unsafe { vfmaq_f64(c, a, b) }
    }
    implementation!(
        construct,
        "neon",
        2,
        float64x2_t,
        zero,
        vld1q_f64,
        vst1q_f64,
        vaddq_f64,
        vsubq_f64,
        vmulq_f64,
        fma,
        vnegq_f64
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn same(a: &[WideComplex], b: &[WideComplex]) {
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            let ap = [
                a.re.parts().0.to_bits(),
                a.re.parts().1.to_bits(),
                a.im.parts().0.to_bits(),
                a.im.parts().1.to_bits(),
            ];
            let bp = [
                b.re.parts().0.to_bits(),
                b.re.parts().1.to_bits(),
                b.im.parts().0.to_bits(),
                b.im.parts().1.to_bits(),
            ];
            assert_eq!(ap, bp, "tap {i}");
        }
    }
    #[test]
    fn finite_modal_trajectories_and_every_backend_keep_all_dd_parts() {
        for modes in [0, 1, 2, 3, 4, 7, 8, 9, 15, 17] {
            for (first, count) in [(0, 0), (0, 1), (0, 129), (7, 513)] {
                for scale in [1e-280, 1e-80, 1., 1e80] {
                    let center = first + count + 1;
                    let initial: Vec<_> = (0..2 * center + 1)
                        .map(|i| WideComplex {
                            re: Wide::from(scale * (i as f64 * 0.07).cos())
                                + Wide::from(scale * 1e-18),
                            im: Wide::from(scale * (i as f64 * 0.11).sin())
                                - Wide::from(scale * 1e-19),
                        })
                        .collect();
                    let coefficients: Vec<_> = (0..modes)
                        .map(|i| WideComplex {
                            re: Wide::from(scale * ((i + 1) as f64 * 0.17).cos())
                                + Wide::from(scale * 3e-19),
                            im: Wide::from(scale * ((i + 1) as f64 * 0.13).sin())
                                - Wide::from(scale * 5e-19),
                        })
                        .collect();
                    let poles: Vec<_> = (0..modes)
                        .map(|i| {
                            let angle = i as f64 * 0.27;
                            WideComplex {
                                re: Wide::from(0.9997 * angle.cos()) + Wide::from(3e-19),
                                im: Wide::from(0.9997 * angle.sin()) - Wide::from(5e-19),
                            }
                        })
                        .collect();
                    let mut expected = initial.clone();
                    scalar::<true>(
                        &mut expected,
                        center,
                        first,
                        first + count,
                        &coefficients,
                        &poles,
                    );
                    let mut actual = initial.clone();
                    add_finite(
                        &mut actual,
                        center,
                        first,
                        first + count,
                        &coefficients,
                        &poles,
                    );
                    same(&actual, &expected);
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if std::is_x86_feature_detected!("avx")
                            && std::is_x86_feature_detected!("fma")
                        {
                            let mut actual = initial.clone();
                            unsafe {
                                x86::avx::<true>(
                                    &mut actual,
                                    center,
                                    first,
                                    first + count,
                                    &coefficients,
                                    &poles,
                                )
                            };
                            same(&actual, &expected);
                        }
                        if std::is_x86_feature_detected!("avx512f")
                            && std::is_x86_feature_detected!("fma")
                        {
                            let mut actual = initial.clone();
                            unsafe {
                                x86::avx512::<true>(
                                    &mut actual,
                                    center,
                                    first,
                                    first + count,
                                    &coefficients,
                                    &poles,
                                )
                            };
                            same(&actual, &expected);
                        }
                    }
                    let mut causal_expected = initial[..count].to_vec();
                    scalar::<false>(&mut causal_expected, 0, 0, count, &coefficients, &poles);
                    let mut causal_actual = initial[..count].to_vec();
                    add_causal(&mut causal_actual, &coefficients, &poles);
                    same(&causal_actual, &causal_expected);
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if std::is_x86_feature_detected!("avx")
                            && std::is_x86_feature_detected!("fma")
                        {
                            let mut actual = initial[..count].to_vec();
                            unsafe {
                                x86::avx::<false>(&mut actual, 0, 0, count, &coefficients, &poles)
                            };
                            same(&actual, &causal_expected);
                        }
                        if std::is_x86_feature_detected!("avx512f")
                            && std::is_x86_feature_detected!("fma")
                        {
                            let mut actual = initial[..count].to_vec();
                            unsafe {
                                x86::avx512::<false>(
                                    &mut actual,
                                    0,
                                    0,
                                    count,
                                    &coefficients,
                                    &poles,
                                )
                            };
                            same(&actual, &causal_expected);
                        }
                    }
                }
            }
        }
    }
}
