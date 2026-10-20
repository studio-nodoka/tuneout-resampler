//! Even real and odd imaginary analysis planes share mirrored input pairs.
pub(crate) fn run(x: &[f64], re: &[f64], im: &[f64]) -> (f64, f64) {
    run_impl::<true>(x, re, im)
}
/// Real carrier analysis needs only the even plane; compile out imaginary loads/products.
pub(crate) fn run_real(x: &[f64], re: &[f64]) -> f64 {
    run_impl::<false>(x, re, &[]).0
}
fn run_impl<const IMAG: bool>(x: &[f64], re: &[f64], im: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), 2 * re.len() + 1);
    if IMAG {
        assert_eq!(re.len(), im.len());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::avx512::<IMAG>(x, re, im) };
        }
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::avx::<IMAG>(x, re, im) };
        }
        if std::is_x86_feature_detected!("sse2") {
            return unsafe { x86::sse2::<IMAG>(x, re, im) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::run::<IMAG>(x, re, im) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return unsafe { wasm::run::<IMAG>(x, re, im) };
    #[allow(unreachable_code)]
    scalar::<IMAG>(x, re, im)
}
fn scalar<const IMAG: bool>(x: &[f64], re: &[f64], im: &[f64]) -> (f64, f64) {
    let mut a = [0.; 4];
    let mut b = a;
    for i in 0..re.len() {
        let l = x[i];
        let r = x[x.len() - 1 - i];
        a[i % 4] += (l + r) * re[i];
        if IMAG {
            b[i % 4] += (l - r) * im[i];
        }
    }
    ((a[0] + a[1]) + (a[2] + a[3]), (b[0] + b[1]) + (b[2] + b[3]))
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn reverse512(v: __m512d) -> __m512d {
        _mm512_permutexvar_pd(_mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7), v)
    }
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn reverse256(v: __m256d) -> __m256d {
        _mm256_permute_pd(_mm256_permute2f128_pd::<1>(v, v), 5)
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn reverse128(v: __m128d) -> __m128d {
        _mm_shuffle_pd::<1>(v, v)
    }
    #[inline(always)]
    unsafe fn sse_madd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
        unsafe { _mm_add_pd(_mm_mul_pd(a, b), c) }
    }
    macro_rules! implementation {
        ($name:ident,$features:literal,$lanes:expr,$zero:ident,$load:ident,$reverse:ident,$add:ident,$sub:ident,$madd:ident,$store:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name<const IMAG: bool>(
                x: &[f64],
                re: &[f64],
                im: &[f64],
            ) -> (f64, f64) {
                unsafe {
                    let mut a = [$zero(); 4];
                    let mut b = a;
                    let bulk = re.len() / (4 * $lanes) * (4 * $lanes);
                    for base in (0..bulk).step_by(4 * $lanes) {
                        for j in 0..4 {
                            let i = base + j * $lanes;
                            let l = $load(x.as_ptr().add(i));
                            let r = $reverse($load(x.as_ptr().add(x.len() - i - $lanes)));
                            a[j] = $madd($add(l, r), $load(re.as_ptr().add(i)), a[j]);
                            if IMAG {
                                b[j] = $madd($sub(l, r), $load(im.as_ptr().add(i)), b[j]);
                            }
                        }
                    }
                    let vectors = re.len() / $lanes * $lanes;
                    for i in (bulk..vectors).step_by($lanes) {
                        let j = (i - bulk) / $lanes;
                        let l = $load(x.as_ptr().add(i));
                        let r = $reverse($load(x.as_ptr().add(x.len() - i - $lanes)));
                        a[j] = $madd($add(l, r), $load(re.as_ptr().add(i)), a[j]);
                        if IMAG {
                            b[j] = $madd($sub(l, r), $load(im.as_ptr().add(i)), b[j]);
                        }
                    }
                    let mut ar = [0.; $lanes];
                    let mut ai = ar;
                    $store(ar.as_mut_ptr(), $add($add(a[0], a[1]), $add(a[2], a[3])));
                    $store(ai.as_mut_ptr(), $add($add(b[0], b[1]), $add(b[2], b[3])));
                    let mut a = ar.into_iter().sum::<f64>();
                    let mut b = ai.into_iter().sum::<f64>();
                    for i in vectors..re.len() {
                        let l = x[i];
                        let r = x[x.len() - 1 - i];
                        a += (l + r) * re[i];
                        if IMAG {
                            b += (l - r) * im[i];
                        }
                    }
                    (a, b)
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
        reverse512,
        _mm512_add_pd,
        _mm512_sub_pd,
        _mm512_fmadd_pd,
        _mm512_storeu_pd
    );
    implementation!(
        avx,
        "avx,fma",
        4,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        reverse256,
        _mm256_add_pd,
        _mm256_sub_pd,
        _mm256_fmadd_pd,
        _mm256_storeu_pd
    );
    implementation!(
        sse2,
        "sse2",
        2,
        _mm_setzero_pd,
        _mm_loadu_pd,
        reverse128,
        _mm_add_pd,
        _mm_sub_pd,
        sse_madd,
        _mm_storeu_pd
    );
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run<const IMAG: bool>(x: &[f64], re: &[f64], im: &[f64]) -> (f64, f64) {
        unsafe {
            let mut a = [vdupq_n_f64(0.); 4];
            let mut b = a;
            let bulk = re.len() / 8 * 8;
            for base in (0..bulk).step_by(8) {
                for j in 0..4 {
                    let i = base + 2 * j;
                    let l = vld1q_f64(x.as_ptr().add(i));
                    let r = vld1q_f64(x.as_ptr().add(x.len() - i - 2));
                    let r = vextq_f64::<1>(r, r);
                    a[j] = vfmaq_f64(a[j], vaddq_f64(l, r), vld1q_f64(re.as_ptr().add(i)));
                    if IMAG {
                        b[j] = vfmaq_f64(b[j], vsubq_f64(l, r), vld1q_f64(im.as_ptr().add(i)));
                    }
                }
            }
            let vectors = re.len() / 2 * 2;
            for i in (bulk..vectors).step_by(2) {
                let j = (i - bulk) / 2;
                let l = vld1q_f64(x.as_ptr().add(i));
                let r = vld1q_f64(x.as_ptr().add(x.len() - i - 2));
                let r = vextq_f64::<1>(r, r);
                a[j] = vfmaq_f64(a[j], vaddq_f64(l, r), vld1q_f64(re.as_ptr().add(i)));
                if IMAG {
                    b[j] = vfmaq_f64(b[j], vsubq_f64(l, r), vld1q_f64(im.as_ptr().add(i)));
                }
            }
            let a = vaddq_f64(vaddq_f64(a[0], a[1]), vaddq_f64(a[2], a[3]));
            let b = vaddq_f64(vaddq_f64(b[0], b[1]), vaddq_f64(b[2], b[3]));
            let mut ar = [0.; 2];
            let mut ai = ar;
            vst1q_f64(ar.as_mut_ptr(), a);
            vst1q_f64(ai.as_mut_ptr(), b);
            let mut a = ar[0] + ar[1];
            let mut b = ai[0] + ai[1];
            for i in vectors..re.len() {
                let l = x[i];
                let r = x[x.len() - 1 - i];
                a += (l + r) * re[i];
                if IMAG {
                    b += (l - r) * im[i];
                }
            }
            (a, b)
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;
    pub(super) unsafe fn run<const IMAG: bool>(x: &[f64], re: &[f64], im: &[f64]) -> (f64, f64) {
        unsafe {
            let mut a = [f64x2_splat(0.); 4];
            let mut b = a;
            let bulk = re.len() / 8 * 8;
            for base in (0..bulk).step_by(8) {
                for j in 0..4 {
                    let i = base + 2 * j;
                    let l = v128_load(x.as_ptr().add(i).cast());
                    let r = v128_load(x.as_ptr().add(x.len() - i - 2).cast());
                    let r = i64x2_shuffle::<1, 0>(r, r);
                    a[j] = f64x2_add(
                        a[j],
                        f64x2_mul(f64x2_add(l, r), v128_load(re.as_ptr().add(i).cast())),
                    );
                    if IMAG {
                        b[j] = f64x2_add(
                            b[j],
                            f64x2_mul(f64x2_sub(l, r), v128_load(im.as_ptr().add(i).cast())),
                        );
                    }
                }
            }
            let vectors = re.len() / 2 * 2;
            for i in (bulk..vectors).step_by(2) {
                let j = (i - bulk) / 2;
                let l = v128_load(x.as_ptr().add(i).cast());
                let r = v128_load(x.as_ptr().add(x.len() - i - 2).cast());
                let r = i64x2_shuffle::<1, 0>(r, r);
                a[j] = f64x2_add(
                    a[j],
                    f64x2_mul(f64x2_add(l, r), v128_load(re.as_ptr().add(i).cast())),
                );
                if IMAG {
                    b[j] = f64x2_add(
                        b[j],
                        f64x2_mul(f64x2_sub(l, r), v128_load(im.as_ptr().add(i).cast())),
                    );
                }
            }
            let a = f64x2_add(f64x2_add(a[0], a[1]), f64x2_add(a[2], a[3]));
            let b = f64x2_add(f64x2_add(b[0], b[1]), f64x2_add(b[2], b[3]));
            let mut a = f64x2_extract_lane::<0>(a) + f64x2_extract_lane::<1>(a);
            let mut b = f64x2_extract_lane::<0>(b) + f64x2_extract_lane::<1>(b);
            for i in vectors..re.len() {
                let l = x[i];
                let r = x[x.len() - 1 - i];
                a += (l + r) * re[i];
                if IMAG {
                    b += (l - r) * im[i];
                }
            }
            (a, b)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mirrored_pairs_match_full_independent_planes() {
        for n in [0, 1, 3, 7, 128, 2048, 4096] {
            let x: Vec<_> = (0..2 * n + 1)
                .map(|i| ((i * 31 % 71) as f64 - 35.) / 31.)
                .collect();
            let a: Vec<_> = (0..n).map(|i| ((i * 11 % 97) as f64 - 48.) / 97.).collect();
            let b: Vec<_> = (0..n)
                .map(|i| ((i * 17 % 103) as f64 - 51.) / 103.)
                .collect();
            let mut aa = a.clone();
            aa.push(0.);
            aa.extend(a.iter().rev());
            let mut bb = b.clone();
            bb.push(0.);
            bb.extend(b.iter().rev().map(|v| -v));
            let actual = run(&x, &a, &b);
            let ar = crate::f64_dot::dot(&x, &aa);
            let ai = crate::f64_dot::dot(&x, &bb);
            assert!((actual.0 - ar).abs() < 2e-13);
            assert!((actual.1 - ai).abs() < 2e-13);
            let expected = scalar::<true>(&x, &a, &b);
            assert!((actual.0 - expected.0).abs() < 2e-13);
            assert!((actual.1 - expected.1).abs() < 2e-13);
        }
    }
    #[test]
    fn cardinal_real_plane_matches_every_available_backend() {
        type Kernel = unsafe fn(&[f64], &[f64], &[f64]) -> (f64, f64);
        fn check(run: Kernel) {
            for n in 0..=257 {
                let x: Vec<_> = (0..2 * n + 1)
                    .map(|i| ((i * 31 % 71) as f64 - 35.) / 31.)
                    .collect();
                let c: Vec<_> = (0..n).map(|i| ((i * 11 % 97) as f64 - 48.) / 97.).collect();
                let exact = x
                    .iter()
                    .take(n)
                    .enumerate()
                    .fold(crate::complex::Wide::from(0.), |s, (i, &l)| {
                        s + crate::complex::Wide::product(l + x[x.len() - 1 - i], c[i])
                    });
                let l1 = (0..n)
                    .map(|i| ((x[i] + x[x.len() - 1 - i]) * c[i]).abs())
                    .sum::<f64>();
                let actual = unsafe { run(&x, &c, &[]) };
                assert!(
                    (crate::complex::Wide::from(actual.0) - exact).value().abs()
                        < 2e-14 * l1.max(1.)
                );
                assert_eq!(actual.1, 0.);
            }
        }
        check(|x, c, _| (run_real(x, c), 0.));
        check(scalar::<false>);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse2") {
                check(x86::sse2::<false>);
            }
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                check(x86::avx::<false>);
            }
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
                check(x86::avx512::<false>);
            }
        }
    }
}
