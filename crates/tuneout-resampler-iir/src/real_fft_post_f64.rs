//! Ordinary proper-real FFT unpacking, preserving scalar operation order.
//! Each positive bin is evaluated independently; its negative bin is an exact
//! conjugate. Fixed SIMD batches never change a bin's summation sequence.
use rustfft::num_complex::Complex as F;
pub(crate) fn forward(input: &[F<f64>], roots: &[F<f64>], output: &mut [F<f64>]) {
    forward_impl::<true>(input, roots, output)
}
/// Only bins0..=m are written. An empty half-input has an empty output.
#[cfg(test)]
pub(crate) fn forward_positive(input: &[F<f64>], roots: &[F<f64>], output: &mut [F<f64>]) {
    forward_impl::<false>(input, roots, output)
}
fn forward_impl<const FULL: bool>(input: &[F<f64>], roots: &[F<f64>], output: &mut [F<f64>]) {
    let m = input.len();
    let expected = if m == 0 {
        0
    } else if FULL {
        m.checked_mul(2).expect("real FFT extent")
    } else {
        m.checked_add(1).expect("positive real FFT extent")
    };
    assert_eq!(output.len(), expected);
    if m == 0 {
        return;
    }
    assert!(roots.len() >= m);
    let z = input[0];
    output[0] = F::new(z.re + z.im, 0.);
    output[m] = F::new(z.re - z.im, 0.);
    #[allow(unused_mut)] // SIMD advances the cursor; scalar targets start at one.
    let mut at = 1;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if std::is_x86_feature_detected!("avx512f") {
            at = x86::avx512::<FULL>(input, roots, output, at);
        } else if std::is_x86_feature_detected!("avx") {
            at = x86::avx::<FULL>(input, roots, output, at);
        } else if std::is_x86_feature_detected!("sse2") {
            at = x86::sse2::<FULL>(input, roots, output, at);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        at = unsafe { neon::run::<FULL>(input, roots, output, at) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        at = unsafe { wasm::run::<FULL>(input, roots, output, at) };
    }
    scalar_from::<FULL>(input, roots, output, at);
}
fn scalar_from<const FULL: bool>(
    input: &[F<f64>],
    roots: &[F<f64>],
    output: &mut [F<f64>],
    at: usize,
) {
    let m = input.len();
    for k in at..m {
        let a = input[k];
        let b = input[m - k];
        let er = (a.re + b.re) * 0.5;
        let ei = (a.im - b.im) * 0.5;
        let or = (a.im + b.im) * 0.5;
        let oi = (b.re - a.re) * 0.5;
        let root = roots[k];
        let re = er + root.re * or - root.im * oi;
        let im = ei + root.re * oi + root.im * or;
        output[k] = F::new(re, im);
        if FULL {
            output[2 * m - k] = F::new(re, -im);
        }
    }
}
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! vector_body {
    ($name:ident,$features:literal,$v:ty,$width:expr,$load:ident,$store:ident,$reverse:ident,$half:expr,$add:ident,$sub:ident,$mul:ident,$neg:ident) => {
        #[target_feature(enable=$features)]
        pub(super) unsafe fn $name<const FULL: bool>(
            input: &[F<f64>],
            roots: &[F<f64>],
            output: &mut [F<f64>],
            mut k: usize,
        ) -> usize {
            unsafe {
                let m = input.len();
                let half: $v = $half;
                // For a full vector k+width<=m, both reversed-input and negative
                // output ranges start above the midpoint/endpoints checked by the
                // public entry point, and all loads/stores stay in their slices.
                while k + $width <= m {
                    let (ar, ai) = $load(input.as_ptr().add(k));
                    let (br, bi) = $load(input.as_ptr().add(m - k - ($width - 1)));
                    let br = $reverse(br);
                    let bi = $reverse(bi);
                    let (wr, wi) = $load(roots.as_ptr().add(k));
                    let er = $mul($add(ar, br), half);
                    let ei = $mul($sub(ai, bi), half);
                    let or = $mul($add(ai, bi), half);
                    let oi = $mul($sub(br, ar), half);
                    let re = $sub($add(er, $mul(wr, or)), $mul(wi, oi));
                    let im = $add($add(ei, $mul(wr, oi)), $mul(wi, or));
                    $store(output.as_mut_ptr().add(k), re, im);
                    if FULL {
                        $store(
                            output.as_mut_ptr().add(2 * m - k - ($width - 1)),
                            $reverse(re),
                            $reverse($neg(im)),
                        );
                    }
                    k += $width;
                }
                k
            }
        }
    };
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::F;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx512f")]
    unsafe fn load512(p: *const F<f64>) -> (__m512d, __m512d) {
        unsafe {
            let a = _mm512_loadu_pd(p.cast());
            let b = _mm512_loadu_pd(p.cast::<f64>().add(8));
            (
                _mm512_permutex2var_pd(a, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), b),
                _mm512_permutex2var_pd(a, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), b),
            )
        }
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn store512(p: *mut F<f64>, re: __m512d, im: __m512d) {
        unsafe {
            _mm512_storeu_pd(
                p.cast(),
                _mm512_permutex2var_pd(re, _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), im),
            );
            _mm512_storeu_pd(
                p.cast::<f64>().add(8),
                _mm512_permutex2var_pd(re, _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15), im),
            );
        }
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn rev512(v: __m512d) -> __m512d {
        _mm512_permutexvar_pd(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), v)
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn neg512(v: __m512d) -> __m512d {
        _mm512_castsi512_pd(_mm512_xor_si512(
            _mm512_castpd_si512(v),
            _mm512_set1_epi64(i64::MIN),
        ))
    }
    vector_body!(
        avx512,
        "avx512f",
        __m512d,
        8,
        load512,
        store512,
        rev512,
        _mm512_set1_pd(0.5),
        _mm512_add_pd,
        _mm512_sub_pd,
        _mm512_mul_pd,
        neg512
    );
    #[target_feature(enable = "avx")]
    unsafe fn load256(p: *const F<f64>) -> (__m256d, __m256d) {
        unsafe {
            let a = _mm256_loadu_pd(p.cast());
            let b = _mm256_loadu_pd(p.cast::<f64>().add(4));
            let lo = _mm256_permute2f128_pd(a, b, 0x20);
            let hi = _mm256_permute2f128_pd(a, b, 0x31);
            (_mm256_shuffle_pd(lo, hi, 0), _mm256_shuffle_pd(lo, hi, 15))
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn store256(p: *mut F<f64>, re: __m256d, im: __m256d) {
        unsafe {
            let lo = _mm256_unpacklo_pd(re, im);
            let hi = _mm256_unpackhi_pd(re, im);
            _mm256_storeu_pd(p.cast(), _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(p.cast::<f64>().add(4), _mm256_permute2f128_pd(lo, hi, 0x31));
        }
    }
    #[target_feature(enable = "avx")]
    unsafe fn rev256(v: __m256d) -> __m256d {
        _mm256_permute_pd(_mm256_permute2f128_pd(v, v, 1), 5)
    }
    #[target_feature(enable = "avx")]
    unsafe fn neg256(v: __m256d) -> __m256d {
        _mm256_xor_pd(v, _mm256_set1_pd(-0.))
    }
    vector_body!(
        avx,
        "avx",
        __m256d,
        4,
        load256,
        store256,
        rev256,
        _mm256_set1_pd(0.5),
        _mm256_add_pd,
        _mm256_sub_pd,
        _mm256_mul_pd,
        neg256
    );
    #[target_feature(enable = "sse2")]
    unsafe fn load128(p: *const F<f64>) -> (__m128d, __m128d) {
        unsafe {
            let a = _mm_loadu_pd(p.cast());
            let b = _mm_loadu_pd(p.cast::<f64>().add(2));
            (_mm_unpacklo_pd(a, b), _mm_unpackhi_pd(a, b))
        }
    }
    #[target_feature(enable = "sse2")]
    unsafe fn store128(p: *mut F<f64>, re: __m128d, im: __m128d) {
        unsafe {
            _mm_storeu_pd(p.cast(), _mm_unpacklo_pd(re, im));
            _mm_storeu_pd(p.cast::<f64>().add(2), _mm_unpackhi_pd(re, im));
        }
    }
    #[target_feature(enable = "sse2")]
    unsafe fn rev128(v: __m128d) -> __m128d {
        _mm_shuffle_pd(v, v, 1)
    }
    #[target_feature(enable = "sse2")]
    unsafe fn neg128(v: __m128d) -> __m128d {
        _mm_xor_pd(v, _mm_set1_pd(-0.))
    }
    vector_body!(
        sse2,
        "sse2",
        __m128d,
        2,
        load128,
        store128,
        rev128,
        _mm_set1_pd(0.5),
        _mm_add_pd,
        _mm_sub_pd,
        _mm_mul_pd,
        neg128
    );
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::F;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    unsafe fn load(p: *const F<f64>) -> (float64x2_t, float64x2_t) {
        unsafe {
            let z = vld2q_f64(p.cast());
            (z.0, z.1)
        }
    }
    #[target_feature(enable = "neon")]
    unsafe fn store(p: *mut F<f64>, re: float64x2_t, im: float64x2_t) {
        unsafe {
            vst2q_f64(p.cast(), float64x2x2_t(re, im));
        }
    }
    #[target_feature(enable = "neon")]
    unsafe fn reverse(v: float64x2_t) -> float64x2_t {
        vextq_f64(v, v, 1)
    }
    vector_body!(
        run,
        "neon",
        float64x2_t,
        2,
        load,
        store,
        reverse,
        vdupq_n_f64(0.5),
        vaddq_f64,
        vsubq_f64,
        vmulq_f64,
        vnegq_f64
    );
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::F;
    use std::arch::wasm32::*;
    #[target_feature(enable = "simd128")]
    unsafe fn load(p: *const F<f64>) -> (v128, v128) {
        unsafe {
            let a = v128_load(p.cast());
            let b = v128_load(p.cast::<f64>().add(2).cast());
            (i64x2_shuffle::<0, 2>(a, b), i64x2_shuffle::<1, 3>(a, b))
        }
    }
    #[target_feature(enable = "simd128")]
    unsafe fn store(p: *mut F<f64>, re: v128, im: v128) {
        unsafe {
            v128_store(p.cast(), i64x2_shuffle::<0, 2>(re, im));
            v128_store(p.cast::<f64>().add(2).cast(), i64x2_shuffle::<1, 3>(re, im));
        }
    }
    #[target_feature(enable = "simd128")]
    unsafe fn reverse(v: v128) -> v128 {
        i64x2_shuffle::<1, 0>(v, v)
    }
    vector_body!(
        run,
        "simd128",
        v128,
        2,
        load,
        store,
        reverse,
        f64x2_splat(0.5),
        f64x2_add,
        f64x2_sub,
        f64x2_mul,
        f64x2_neg
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    fn compare(a: &[F<f64>], b: &[F<f64>], m: usize) {
        for (k, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                [a.re.to_bits(), a.im.to_bits()],
                [b.re.to_bits(), b.im.to_bits()],
                "half={m} bin{k}"
            );
        }
    }
    #[test]
    fn ordinary_real_unpack_matches_scalar_bits_on_tiny_odd_partial_and_scaled_grids() {
        for m in [
            0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257,
        ] {
            for scale in [1e-150, 1., 1e150] {
                let input: Vec<_> = (0..m)
                    .map(|j| {
                        F::new(
                            ((j * 31 % 137) as f64 / 137. - 0.5) * scale,
                            ((j * 37 % 101) as f64 / 101. - 0.5) * scale,
                        )
                    })
                    .collect();
                let roots: Vec<_> = (0..m)
                    .map(|j| F::new((j as f64 * 0.137).cos(), -(j as f64 * 0.137).sin()))
                    .collect();
                let mut expected = vec![F::new(0., 0.); 2 * m];
                if m > 0 {
                    expected[0] = F::new(input[0].re + input[0].im, 0.);
                    expected[m] = F::new(input[0].re - input[0].im, 0.);
                    scalar_from::<true>(&input, &roots, &mut expected, 1);
                }
                let mut actual = vec![F::new(123., 456.); 2 * m];
                forward(&input, &roots, &mut actual);
                compare(&actual, &expected, m);
                let mut positive = vec![F::new(123., 456.); if m == 0 { 0 } else { m + 1 }];
                forward_positive(&input, &roots, &mut positive);
                compare(&positive, &expected[..positive.len()], m);
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if m > 0 {
                    for backend in 0..3 {
                        let available = match backend {
                            0 => std::is_x86_feature_detected!("avx512f"),
                            1 => std::is_x86_feature_detected!("avx"),
                            _ => std::is_x86_feature_detected!("sse2"),
                        };
                        if available {
                            actual.copy_from_slice(&expected);
                            let at = unsafe {
                                match backend {
                                    0 => x86::avx512::<true>(&input, &roots, &mut actual, 1),
                                    1 => x86::avx::<true>(&input, &roots, &mut actual, 1),
                                    _ => x86::sse2::<true>(&input, &roots, &mut actual, 1),
                                }
                            };
                            scalar_from::<true>(&input, &roots, &mut actual, at);
                            compare(&actual, &expected, m);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn ordinary_real_unpack_preserves_basis_dc_nyquist_and_conjugate_signs() {
        for m in [1, 2, 3, 4, 5, 8, 16, 17] {
            for at in 0..m {
                for imaginary in [false, true] {
                    let mut input = vec![F::new(0., 0.); m];
                    if imaginary {
                        input[at].im = 1.
                    } else {
                        input[at].re = 1.;
                    }
                    let roots: Vec<_> = (0..m)
                        .map(|j| {
                            F::new(
                                (std::f64::consts::PI * j as f64 / m as f64).cos(),
                                -(std::f64::consts::PI * j as f64 / m as f64).sin(),
                            )
                        })
                        .collect();
                    let mut expected = vec![F::new(0., 0.); 2 * m];
                    expected[0] = F::new(input[0].re + input[0].im, 0.);
                    expected[m] = F::new(input[0].re - input[0].im, 0.);
                    scalar_from::<true>(&input, &roots, &mut expected, 1);
                    let mut actual = vec![F::new(17., 19.); 2 * m];
                    forward(&input, &roots, &mut actual);
                    compare(&actual, &expected, m);
                    let mut positive = vec![F::new(17., 19.); m + 1];
                    forward_positive(&input, &roots, &mut positive);
                    compare(&positive, &expected[..m + 1], m);
                    assert_eq!(actual[0].im.to_bits(), 0f64.to_bits());
                    assert_eq!(actual[m].im.to_bits(), 0f64.to_bits());
                    for k in 1..m {
                        assert_eq!(actual[2 * m - k].re.to_bits(), actual[k].re.to_bits());
                        assert_eq!(actual[2 * m - k].im.to_bits(), (-actual[k].im).to_bits());
                    }
                }
            }
        }
    }
}
