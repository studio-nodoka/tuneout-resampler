//! Ordinary FIR dot for the new mixed IIR architecture. Like production FIR,
//! use independent accumulators and runtime SIMD selection. This intentionally
//! permits reassociation; changes require numerical and signal-quality checks.
pub(crate) fn dot(x: &[f64], c: &[f64]) -> f64 {
    assert_eq!(x.len(), c.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: Features include OS state, equal slices checked above;
            // all vector implementations use only complete in-bounds vectors.
            return unsafe { x86::avx512(x, c) };
        }
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::avx(x, c) };
        }
        if std::is_x86_feature_detected!("sse2") {
            return unsafe { x86::sse2(x, c) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::dot(x, c) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return unsafe { wasm::dot(x, c) };
    #[allow(unreachable_code)]
    scalar(x, c)
}
fn scalar(x: &[f64], c: &[f64]) -> f64 {
    let bulk = x.len() / 4 * 4;
    let mut s = [0.; 4];
    for i in (0..bulk).step_by(4) {
        for j in 0..4 {
            s[j] += x[i + j] * c[i + j];
        }
    }
    let mut result = (s[0] + s[1]) + (s[2] + s[3]);
    for i in bulk..x.len() {
        result += x[i] * c[i];
    }
    result
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn sse_madd(x: __m128d, c: __m128d, s: __m128d) -> __m128d {
        unsafe { _mm_add_pd(_mm_mul_pd(x, c), s) }
    }
    macro_rules! implementation {
        ($name:ident,$features:literal,$lanes:expr,$zero:ident,$load:ident,$madd:ident,$add:ident,$store:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(x: &[f64], c: &[f64]) -> f64 {
                unsafe {
                    let mut s = [$zero(); 4];
                    let bulk = x.len() / (4 * $lanes) * (4 * $lanes);
                    for i in (0..bulk).step_by(4 * $lanes) {
                        for j in 0..4 {
                            let at = i + j * $lanes;
                            s[j] =
                                $madd($load(x.as_ptr().add(at)), $load(c.as_ptr().add(at)), s[j]);
                        }
                    }
                    let vectors = x.len() / $lanes * $lanes;
                    for i in (bulk..vectors).step_by($lanes) {
                        let j = (i - bulk) / $lanes;
                        s[j] = $madd($load(x.as_ptr().add(i)), $load(c.as_ptr().add(i)), s[j]);
                    }
                    let total = $add($add(s[0], s[1]), $add(s[2], s[3]));
                    let mut lanes = [0.; $lanes];
                    $store(lanes.as_mut_ptr(), total);
                    let mut result = lanes.into_iter().sum::<f64>();
                    for i in vectors..x.len() {
                        result += x[i] * c[i];
                    }
                    result
                }
            }
        };
    }
    implementation!(
        avx512,
        "avx512f,fma",
        8,
        _mm512_setzero_pd,
        _mm512_loadu_pd,
        _mm512_fmadd_pd,
        _mm512_add_pd,
        _mm512_storeu_pd
    );
    implementation!(
        avx,
        "avx,fma",
        4,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_fmadd_pd,
        _mm256_add_pd,
        _mm256_storeu_pd
    );
    implementation!(
        sse2,
        "sse2",
        2,
        _mm_setzero_pd,
        _mm_loadu_pd,
        sse_madd,
        _mm_add_pd,
        _mm_storeu_pd
    );
}
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)] // Fixed SIMD accumulator layout.
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(x: &[f64], c: &[f64]) -> f64 {
        unsafe {
            let mut s = [vdupq_n_f64(0.); 4];
            let bulk = x.len() / 8 * 8;
            for i in (0..bulk).step_by(8) {
                for j in 0..4 {
                    let at = i + 2 * j;
                    s[j] = vfmaq_f64(
                        s[j],
                        vld1q_f64(x.as_ptr().add(at)),
                        vld1q_f64(c.as_ptr().add(at)),
                    );
                }
            }
            let vectors = x.len() / 2 * 2;
            for i in (bulk..vectors).step_by(2) {
                let j = (i - bulk) / 2;
                s[j] = vfmaq_f64(
                    s[j],
                    vld1q_f64(x.as_ptr().add(i)),
                    vld1q_f64(c.as_ptr().add(i)),
                );
            }
            let total = vaddq_f64(vaddq_f64(s[0], s[1]), vaddq_f64(s[2], s[3]));
            let mut lanes = [0.; 2];
            vst1q_f64(lanes.as_mut_ptr(), total);
            let mut result = lanes[0] + lanes[1];
            for i in vectors..x.len() {
                result += x[i] * c[i];
            }
            result
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::needless_range_loop)] // Fixed SIMD accumulator layout.
mod wasm {
    use std::arch::wasm32::*;
    pub(super) unsafe fn dot(x: &[f64], c: &[f64]) -> f64 {
        unsafe {
            let mut s = [f64x2_splat(0.); 4];
            let bulk = x.len() / 8 * 8;
            for i in (0..bulk).step_by(8) {
                for j in 0..4 {
                    let at = i + 2 * j;
                    s[j] = f64x2_add(
                        s[j],
                        f64x2_mul(
                            v128_load(x.as_ptr().add(at).cast()),
                            v128_load(c.as_ptr().add(at).cast()),
                        ),
                    );
                }
            }
            let vectors = x.len() / 2 * 2;
            for i in (bulk..vectors).step_by(2) {
                let j = (i - bulk) / 2;
                s[j] = f64x2_add(
                    s[j],
                    f64x2_mul(
                        v128_load(x.as_ptr().add(i).cast()),
                        v128_load(c.as_ptr().add(i).cast()),
                    ),
                );
            }
            let v = f64x2_add(f64x2_add(s[0], s[1]), f64x2_add(s[2], s[3]));
            let mut result = f64x2_extract_lane::<0>(v) + f64x2_extract_lane::<1>(v);
            for i in vectors..x.len() {
                result += x[i] * c[i];
            }
            result
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Wide;
    fn check(run: unsafe fn(&[f64], &[f64]) -> f64) {
        for n in 0..1032 {
            let x: Vec<_> = (0..n).map(|i| ((i * 31 % 71) as f64 - 35.) / 31.).collect();
            let c: Vec<_> = (0..n).map(|i| ((i * 11 % 97) as f64 - 48.) / 97.).collect();
            let expected = x
                .iter()
                .zip(&c)
                .fold(Wide::from(0.), |s, (&x, &c)| s + Wide::product(x, c));
            let l1 = x.iter().zip(&c).map(|(x, c)| (x * c).abs()).sum::<f64>();
            let value = unsafe { run(&x, &c) };
            assert!(
                (Wide::from(value) - expected).value().abs() < 2e-14 * l1.max(1.),
                "n={n}"
            );
        }
        for at in [0, 1, 15, 16, 31, 32, 1030] {
            let mut x = vec![0.; 1031];
            x[at] = 1.;
            let c: Vec<_> = (0..1031)
                .map(|i| ((i * 11 % 97) as f64 - 48.) / 97.)
                .collect();
            assert_eq!(unsafe { run(&x, &c) }, c[at]);
        }
    }
    #[test]
    fn every_available_backend_matches_independent_dd_reference() {
        check(dot);
        check(scalar);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse2") {
                check(x86::sse2);
            }
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                check(x86::avx);
            }
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
                check(x86::avx512);
            }
        }
    }
}
