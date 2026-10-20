//! Lane-wise compensated real/complex dot products with DD operands.
//!
//! Error-free high products and TwoSum additions retain their residuals; all
//! DD cross products are accumulated into the low lane. Normalization occurs
//! once after the lane reduction. This changes rounding from a per-tap Wide
//! multiply/add. It requires fresh signal qualification before adoption.
//!
//! Install as `#[allow(unsafe_code)] mod wide_dot2;` in the receiving crate.
//! No assumptions about Wide/WideComplex struct layout are made by SIMD code.
use crate::complex::{Complex as C, Wide as W, WideComplex as WC};
use std::collections::VecDeque;

#[inline]
pub(crate) fn real_complex(input: &[f64], coefficients: &[WC]) -> WC {
    debug_assert_eq!(input.len(), coefficients.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        // SAFETY: Features are checked; implementations index bounded slices.
        return unsafe { x86::real_complex(input, coefficients) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::real_complex(input, coefficients) };
    }
    real_complex_scalar(input, coefficients)
}

#[inline]
pub(crate) fn wide_complex_real(input: &[WC], coefficients: &[W]) -> WC {
    debug_assert_eq!(input.len(), coefficients.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        return unsafe { x86::wide_complex_real(input, coefficients) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::wide_complex_real(input, coefficients) };
    }
    wide_complex_real_scalar(input, coefficients)
}

/// Full complex product and sum for binary64 states and DD coefficients.
#[inline]
pub(crate) fn complex_wide_complex(input: &[C], coefficients: &[WC]) -> WC {
    debug_assert_eq!(input.len(), coefficients.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        return unsafe { x86::complex_wide_complex(input, coefficients) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::complex_wide_complex(input, coefficients) };
    }
    complex_wide_complex_scalar(input, coefficients)
}

/// Dot the first `coefficients.len()` queued samples without moving the deque.
/// Its two contiguous pieces are reduced separately, then merged once in DD.
#[inline]
pub(crate) fn wide_complex_real_deque(input: &VecDeque<WC>, coefficients: &[W]) -> WC {
    debug_assert!(input.len() >= coefficients.len());
    let (first, second) = input.as_slices();
    let count = first.len().min(coefficients.len());
    let result = wide_complex_real(&first[..count], &coefficients[..count]);
    if count == coefficients.len() {
        result
    } else {
        result
            + wide_complex_real(
                &second[..coefficients.len() - count],
                &coefficients[count..],
            )
    }
}

#[derive(Default)]
struct Accumulator {
    // Four complex lanes. Components are interleaved explicitly as an array,
    // independently of the source structs' unspecified memory layout.
    high: [f64; 8],
    low: [f64; 8],
}
impl Accumulator {
    #[inline]
    fn add_parts(&mut self, lane: usize, product: f64, residual: f64) {
        let sum = self.high[lane] + product;
        let v = sum - self.high[lane];
        let error = (self.high[lane] - (sum - v)) + (product - v);
        self.low[lane] += error + residual;
        self.high[lane] = sum;
    }

    #[inline]
    fn real_product(&mut self, lane: usize, value: f64, coefficient: W) {
        let (hi, lo) = coefficient.parts();
        let product = value * hi;
        let residual = value.mul_add(hi, -product);
        self.add_parts(lane, product, value.mul_add(lo, residual));
    }

    #[inline]
    fn wide_product(&mut self, lane: usize, value: W, coefficient: W) {
        let (a_hi, a_lo) = value.parts();
        let (b_hi, b_lo) = coefficient.parts();
        let product = a_hi * b_hi;
        let residual = a_hi.mul_add(b_hi, -product);
        // All four terms of (a_hi+a_lo)*(b_hi+b_lo), including low*low.
        let correction = a_lo.mul_add(b_lo, a_lo.mul_add(b_hi, a_hi.mul_add(b_lo, residual)));
        self.add_parts(lane, product, correction);
    }

    fn finish(&self) -> WC {
        fn component(high: &[f64; 8], low: &[f64; 8], part: usize) -> W {
            let mut total = 0.;
            let mut residual = 0.;
            for i in 0..4 {
                let lane = i * 2 + part;
                let sum = total + high[lane];
                let v = sum - total;
                residual += (total - (sum - v)) + (high[lane] - v) + low[lane];
                total = sum;
            }
            // Full TwoSum supports a residual larger than a cancelled total.
            let sum = total + residual;
            let v = sum - total;
            let error = (total - (sum - v)) + (residual - v);
            W::from_product_parts(sum, error)
        }
        WC {
            re: component(&self.high, &self.low, 0),
            im: component(&self.high, &self.low, 1),
        }
    }
}

fn real_complex_scalar(input: &[f64], coefficients: &[WC]) -> WC {
    let mut acc = Accumulator::default();
    for (i, (&x, &c)) in input.iter().zip(coefficients).enumerate() {
        let lane = 2 * (i & 3);
        acc.real_product(lane, x, c.re);
        acc.real_product(lane + 1, x, c.im);
    }
    acc.finish()
}

fn wide_complex_real_scalar(input: &[WC], coefficients: &[W]) -> WC {
    let mut acc = Accumulator::default();
    for (i, (&x, &c)) in input.iter().zip(coefficients).enumerate() {
        let lane = 2 * (i & 3);
        acc.wide_product(lane, x.re, c);
        acc.wide_product(lane + 1, x.im, c);
    }
    acc.finish()
}

fn complex_wide_complex_scalar(input: &[C], coefficients: &[WC]) -> WC {
    let mut acc = Accumulator::default();
    for (i, (&x, &c)) in input.iter().zip(coefficients).enumerate() {
        let lane = 2 * (i & 3);
        acc.real_product(lane, x.re, c.re);
        acc.real_product(lane + 1, x.re, c.im);
        acc.real_product(lane, -x.im, c.im);
        acc.real_product(lane + 1, x.im, c.re);
    }
    acc.finish()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx,fma")]
    unsafe fn add(high: &mut __m256d, low: &mut __m256d, product: __m256d, residual: __m256d) {
        let sum = _mm256_add_pd(*high, product);
        let v = _mm256_sub_pd(sum, *high);
        let error = _mm256_add_pd(
            _mm256_sub_pd(*high, _mm256_sub_pd(sum, v)),
            _mm256_sub_pd(product, v),
        );
        *low = _mm256_add_pd(*low, _mm256_add_pd(error, residual));
        *high = sum;
    }

    #[target_feature(enable = "avx,fma")]
    unsafe fn complex_parts(a: WC, b: WC) -> (__m256d, __m256d) {
        let (ar, alr) = a.re.parts();
        let (ai, ali) = a.im.parts();
        let (br, blr) = b.re.parts();
        let (bi, bli) = b.im.parts();
        (
            _mm256_set_pd(bi, br, ai, ar),
            _mm256_set_pd(bli, blr, ali, alr),
        )
    }

    #[target_feature(enable = "avx,fma")]
    unsafe fn finish(high: [__m256d; 2], low: [__m256d; 2]) -> Accumulator {
        unsafe {
            let mut acc = Accumulator::default();
            for i in 0..2 {
                _mm256_storeu_pd(acc.high.as_mut_ptr().add(4 * i), high[i]);
                _mm256_storeu_pd(acc.low.as_mut_ptr().add(4 * i), low[i]);
            }
            acc
        }
    }

    #[target_feature(enable = "avx,fma")]
    pub(super) unsafe fn real_complex(input: &[f64], coefficients: &[WC]) -> WC {
        unsafe {
            let zero = _mm256_setzero_pd();
            let mut high = [zero; 2];
            let mut low = high;
            let pairs = input.len() / 2;
            for pair in 0..pairs {
                let i = pair * 2;
                let lane = pair & 1;
                let x = _mm256_set_pd(input[i + 1], input[i + 1], input[i], input[i]);
                let (c_hi, c_lo) = complex_parts(coefficients[i], coefficients[i + 1]);
                let product = _mm256_mul_pd(x, c_hi);
                let residual = _mm256_fmsub_pd(x, c_hi, product);
                let residual = _mm256_fmadd_pd(x, c_lo, residual);
                add(&mut high[lane], &mut low[lane], product, residual);
            }
            let mut acc = finish(high, low);
            if input.len() & 1 != 0 {
                let i = input.len() - 1;
                let lane = 2 * (i & 3);
                acc.real_product(lane, input[i], coefficients[i].re);
                acc.real_product(lane + 1, input[i], coefficients[i].im);
            }
            acc.finish()
        }
    }

    #[target_feature(enable = "avx,fma")]
    pub(super) unsafe fn wide_complex_real(input: &[WC], coefficients: &[W]) -> WC {
        unsafe {
            let zero = _mm256_setzero_pd();
            let mut high = [zero; 2];
            let mut low = high;
            let pairs = input.len() / 2;
            for pair in 0..pairs {
                let i = pair * 2;
                let lane = pair & 1;
                let (a_hi, a_lo) = complex_parts(input[i], input[i + 1]);
                let (br, blr) = coefficients[i].parts();
                let (bi, bli) = coefficients[i + 1].parts();
                let b_hi = _mm256_set_pd(bi, bi, br, br);
                let b_lo = _mm256_set_pd(bli, bli, blr, blr);
                let product = _mm256_mul_pd(a_hi, b_hi);
                let residual = _mm256_fmsub_pd(a_hi, b_hi, product);
                let residual = _mm256_fmadd_pd(a_hi, b_lo, residual);
                let residual = _mm256_fmadd_pd(a_lo, b_hi, residual);
                let residual = _mm256_fmadd_pd(a_lo, b_lo, residual);
                add(&mut high[lane], &mut low[lane], product, residual);
            }
            let mut acc = finish(high, low);
            if input.len() & 1 != 0 {
                let i = input.len() - 1;
                let lane = 2 * (i & 3);
                acc.wide_product(lane, input[i].re, coefficients[i]);
                acc.wide_product(lane + 1, input[i].im, coefficients[i]);
            }
            acc.finish()
        }
    }

    #[target_feature(enable = "avx,fma")]
    pub(super) unsafe fn complex_wide_complex(input: &[C], coefficients: &[WC]) -> WC {
        unsafe {
            let zero = _mm256_setzero_pd();
            let mut high = [zero; 2];
            let mut low = high;
            let signs = _mm256_set_pd(0., -0., 0., -0.);
            let pairs = input.len() / 2;
            for pair in 0..pairs {
                let i = pair * 2;
                let lane = pair & 1;
                let (c_hi, c_lo) = complex_parts(coefficients[i], coefficients[i + 1]);
                let xr = _mm256_set_pd(input[i + 1].re, input[i + 1].re, input[i].re, input[i].re);
                let product = _mm256_mul_pd(xr, c_hi);
                let residual = _mm256_fmsub_pd(xr, c_hi, product);
                let residual = _mm256_fmadd_pd(xr, c_lo, residual);
                add(&mut high[lane], &mut low[lane], product, residual);
                let xi = _mm256_set_pd(input[i + 1].im, input[i + 1].im, input[i].im, input[i].im);
                let xi = _mm256_xor_pd(xi, signs);
                let c_hi = _mm256_permute_pd(c_hi, 0b0101);
                let c_lo = _mm256_permute_pd(c_lo, 0b0101);
                let product = _mm256_mul_pd(xi, c_hi);
                let residual = _mm256_fmsub_pd(xi, c_hi, product);
                let residual = _mm256_fmadd_pd(xi, c_lo, residual);
                add(&mut high[lane], &mut low[lane], product, residual);
            }
            let mut acc = finish(high, low);
            if input.len() & 1 != 0 {
                let i = input.len() - 1;
                let lane = 2 * (i & 3);
                let x = input[i];
                let c = coefficients[i];
                acc.real_product(lane, x.re, c.re);
                acc.real_product(lane + 1, x.re, c.im);
                acc.real_product(lane, -x.im, c.im);
                acc.real_product(lane + 1, x.im, c.re);
            }
            acc.finish()
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    unsafe fn add(
        high: &mut float64x2_t,
        low: &mut float64x2_t,
        product: float64x2_t,
        residual: float64x2_t,
    ) {
        let sum = vaddq_f64(*high, product);
        let v = vsubq_f64(sum, *high);
        let error = vaddq_f64(vsubq_f64(*high, vsubq_f64(sum, v)), vsubq_f64(product, v));
        *low = vaddq_f64(*low, vaddq_f64(error, residual));
        *high = sum;
    }

    #[target_feature(enable = "neon")]
    unsafe fn complex_parts(a: WC) -> (float64x2_t, float64x2_t) {
        unsafe {
            let (ar, alr) = a.re.parts();
            let (ai, ali) = a.im.parts();
            (vld1q_f64([ar, ai].as_ptr()), vld1q_f64([alr, ali].as_ptr()))
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn finish(high: [float64x2_t; 4], low: [float64x2_t; 4]) -> WC {
        unsafe {
            let mut acc = Accumulator::default();
            for i in 0..4 {
                vst1q_f64(acc.high.as_mut_ptr().add(2 * i), high[i]);
                vst1q_f64(acc.low.as_mut_ptr().add(2 * i), low[i]);
            }
            acc.finish()
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn real_complex(input: &[f64], coefficients: &[WC]) -> WC {
        unsafe {
            let zero = vdupq_n_f64(0.);
            let mut high = [zero; 4];
            let mut low = high;
            for (i, (&x, &c)) in input.iter().zip(coefficients).enumerate() {
                let lane = i & 3;
                let x = vdupq_n_f64(x);
                let (c_hi, c_lo) = complex_parts(c);
                let product = vmulq_f64(x, c_hi);
                let residual = vfmaq_f64(vnegq_f64(product), x, c_hi);
                let residual = vfmaq_f64(residual, x, c_lo);
                add(&mut high[lane], &mut low[lane], product, residual);
            }
            finish(high, low)
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn wide_complex_real(input: &[WC], coefficients: &[W]) -> WC {
        unsafe {
            let zero = vdupq_n_f64(0.);
            let mut high = [zero; 4];
            let mut low = high;
            for (i, (&x, &c)) in input.iter().zip(coefficients).enumerate() {
                let lane = i & 3;
                let (a_hi, a_lo) = complex_parts(x);
                let (b_hi, b_lo) = c.parts();
                let b_hi = vdupq_n_f64(b_hi);
                let b_lo = vdupq_n_f64(b_lo);
                let product = vmulq_f64(a_hi, b_hi);
                let residual = vfmaq_f64(vnegq_f64(product), a_hi, b_hi);
                let residual = vfmaq_f64(residual, a_hi, b_lo);
                let residual = vfmaq_f64(residual, a_lo, b_hi);
                let residual = vfmaq_f64(residual, a_lo, b_lo);
                add(&mut high[lane], &mut low[lane], product, residual);
            }
            finish(high, low)
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn complex_wide_complex(input: &[C], coefficients: &[WC]) -> WC {
        unsafe {
            let zero = vdupq_n_f64(0.);
            let mut high = [zero; 4];
            let mut low = high;
            for (i, (&x, &c)) in input.iter().zip(coefficients).enumerate() {
                let lane = i & 3;
                let (c_hi, c_lo) = complex_parts(c);
                let xr = vdupq_n_f64(x.re);
                let product = vmulq_f64(xr, c_hi);
                let residual = vfmaq_f64(vnegq_f64(product), xr, c_hi);
                let residual = vfmaq_f64(residual, xr, c_lo);
                add(&mut high[lane], &mut low[lane], product, residual);
                let xi = vld1q_f64([-x.im, x.im].as_ptr());
                let c_hi = vextq_f64(c_hi, c_hi, 1);
                let c_lo = vextq_f64(c_lo, c_lo, 1);
                let product = vmulq_f64(xi, c_hi);
                let residual = vfmaq_f64(vnegq_f64(product), xi, c_hi);
                let residual = vfmaq_f64(residual, xi, c_lo);
                add(&mut high[lane], &mut low[lane], product, residual);
            }
            finish(high, low)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn difference(a: WC, b: WC) -> f64 {
        (a.re - b.re).value().abs().max((a.im - b.im).value().abs())
    }
    #[test]
    fn wide_dot2_audio_scale_prefixes_match_dd_reference() {
        let mut values = Vec::new();
        let mut coefficients = Vec::new();
        let mut wide = Vec::new();
        let mut weights = Vec::new();
        let mut complex = Vec::new();
        for i in 0..259 {
            let x = ((i * 37 % 113) as f64 - 56.) / 59.;
            let a = ((i * 19 % 127) as f64 - 63.) / 131.;
            let b = ((i * 23 % 109) as f64 - 54.) / 113.;
            let c = WC {
                re: W::from(a) + W::from(1e-18),
                im: W::from(b) - W::from(3e-19),
            };
            values.push(x);
            coefficients.push(c);
            complex.push(C::new(x, -x / 3.));
            wide.push(WC {
                re: W::from(x) + W::from(7e-19),
                im: W::from(-x / 3.) - W::from(5e-19),
            });
            weights.push(W::from(a / 19.) + W::from(2e-20));
        }
        for n in 0..=259 {
            let mut real_reference = WC::real(0.);
            let mut wide_reference = WC::real(0.);
            let mut complex_reference = WC::real(0.);
            for i in 0..n {
                real_reference = real_reference + coefficients[i].scale(W::from(values[i]));
                wide_reference = wide_reference + wide[i].scale(weights[i]);
                complex_reference = complex_reference + coefficients[i] * WC::from(complex[i]);
            }
            for actual in [
                real_complex(&values[..n], &coefficients[..n]),
                real_complex_scalar(&values[..n], &coefficients[..n]),
            ] {
                assert!(
                    difference(actual, real_reference) < 1e-27,
                    "real n={n}, error={:e}",
                    difference(actual, real_reference)
                );
            }
            for actual in [
                wide_complex_real(&wide[..n], &weights[..n]),
                wide_complex_real_scalar(&wide[..n], &weights[..n]),
            ] {
                assert!(
                    difference(actual, wide_reference) < 1e-27,
                    "wide n={n}, error={:e}",
                    difference(actual, wide_reference)
                );
            }
            for actual in [
                complex_wide_complex(&complex[..n], &coefficients[..n]),
                complex_wide_complex_scalar(&complex[..n], &coefficients[..n]),
            ] {
                assert!(
                    difference(actual, complex_reference) < 1e-27,
                    "complex n={n}, error={:e}",
                    difference(actual, complex_reference)
                );
            }
        }
        // Force a wrapped queue so that both slice reductions are exercised.
        let mut queue = VecDeque::with_capacity(131);
        queue.extend(wide[..131].iter().copied());
        for _ in 0..91 {
            queue.pop_front();
        }
        queue.extend(wide[131..222].iter().copied());
        let reference = queue
            .iter()
            .zip(&weights[..131])
            .fold(WC::real(0.), |sum, (&x, &w)| sum + x.scale(w));
        assert!(difference(wide_complex_real_deque(&queue, &weights[..131]), reference) < 1e-27);
    }

    #[test]
    fn wide_dot2_retains_low_coefficient_and_product_residuals() {
        let c = [
            WC {
                re: W::from(1e16) + W::from(1.),
                im: W::from(0.),
            },
            WC::real(1e16),
        ];
        assert_eq!(real_complex(&[1., -1.], &c).re.value(), 1.);
        let a = W::from(1. + f64::EPSILON);
        let b = W::from(1. - f64::EPSILON);
        let input = [WC { re: a, im: a }, WC::real(-1.)];
        let output = wide_complex_real(&input, &[b, W::from(1.)]);
        assert_eq!(output.re.value(), -f64::EPSILON * f64::EPSILON);
        assert_eq!((output.im - a * b).value(), 0.);
    }
}
