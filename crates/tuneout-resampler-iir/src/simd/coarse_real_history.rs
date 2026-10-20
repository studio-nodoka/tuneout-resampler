//! Exact Hermitian input-history loads for a full complex spectral product.
//! DC/Nyquist retain their stored signed zeros; only strict negative bins are
//! reflected. Each bin keeps its original backend's multiply/add sequence.
use super::*;
type Mac = fn(&mut [Complex], &[Complex], &[Complex]);
pub(crate) struct RealHistoryMac {
    positive: Mac,
    negative: Mac,
}
impl RealHistoryMac {
    pub(crate) fn apply(&self, sum: &mut [Complex], positive: &[Complex], h: &[Complex]) {
        assert!(positive.len() >= 2);
        let half = positive.len() - 1;
        assert_eq!(sum.len(), half.checked_mul(2).expect("Hermitian extent"));
        assert_eq!(sum.len(), h.len());
        (self.positive)(&mut sum[..half + 1], positive, &h[..half + 1]);
        (self.negative)(&mut sum[half + 1..], &positive[1..half], &h[half + 1..]);
    }
}
impl Kernel {
    pub(crate) fn coarse_real_history_mac(self) -> RealHistoryMac {
        let positive = self.coarse_mac_function();
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return RealHistoryMac {
            positive,
            negative: wasm_checked,
        };
        #[allow(unreachable_code)]
        let negative: Mac = match self.0 {
            Backend::Scalar => negative_scalar::<false>,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => sse_checked,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => {
                if std::is_x86_feature_detected!("fma") {
                    avx_fma_checked
                } else {
                    avx_checked
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => {
                if std::is_x86_feature_detected!("fma") {
                    avx512_fma_checked
                } else {
                    avx512_checked
                }
            }
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => neon_checked,
        };
        RealHistoryMac { positive, negative }
    }
}
fn negative_scalar<const FUSED: bool>(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
    assert_eq!(sum.len(), x.len());
    assert_eq!(sum.len(), h.len());
    for i in 0..sum.len() {
        let z = x[x.len() - 1 - i].conj();
        let w = h[i];
        if FUSED {
            let re = w.re.mul_add(z.re, sum[i].re);
            let im = w.re.mul_add(z.im, sum[i].im);
            sum[i].re = w.im.mul_add(-z.im, re);
            sum[i].im = w.im.mul_add(z.re, im);
        } else {
            sum[i] = sum[i] + z * w;
        }
    }
}
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! checked {
    ($name:ident,$implementation:path) => {
        fn $name(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
            assert_eq!(sum.len(), x.len());
            assert_eq!(sum.len(), h.len());
            // Constructor selection verifies ISA; complete reversed loads remain inside
            // x because each vector starts at len-i-width for i+width<=len.
            unsafe { $implementation(sum, x, h) }
        }
    };
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked!(sse_checked, x86::sse);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked!(avx_checked, x86::avx);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked!(avx_fma_checked, x86::avx_fma);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked!(avx512_checked, x86::avx512);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked!(avx512_fma_checked, x86::avx512_fma);
#[cfg(target_arch = "aarch64")]
checked!(neon_checked, neon::run);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
checked!(wasm_checked, wasm::run);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sse(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        unsafe {
            for i in 0..sum.len() {
                let a = _mm_xor_pd(
                    _mm_loadu_pd(x.as_ptr().add(x.len() - 1 - i).cast()),
                    _mm_set_pd(-0., 0.),
                );
                let b = _mm_loadu_pd(h.as_ptr().add(i).cast());
                let old = _mm_loadu_pd(sum.as_ptr().add(i).cast());
                let re = _mm_mul_pd(_mm_unpacklo_pd(a, a), b);
                let im = _mm_mul_pd(_mm_unpackhi_pd(a, a), _mm_shuffle_pd::<1>(b, b));
                let product = _mm_add_pd(re, _mm_xor_pd(im, _mm_set_pd(0., -0.)));
                _mm_storeu_pd(sum.as_mut_ptr().add(i).cast(), _mm_add_pd(old, product));
            }
        }
    }
    macro_rules! avx_impl {
        ($name:ident,$features:literal,$fused:expr) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
                unsafe {
                    let bulk = sum.len() / 2 * 2;
                    for i in (0..bulk).step_by(2) {
                        let a = _mm256_loadu_pd(x.as_ptr().add(x.len() - i - 2).cast());
                        let a = _mm256_xor_pd(
                            _mm256_permute2f128_pd::<1>(a, a),
                            _mm256_set_pd(-0., 0., -0., 0.),
                        );
                        let b = _mm256_loadu_pd(h.as_ptr().add(i).cast());
                        let old = _mm256_loadu_pd(sum.as_ptr().add(i).cast());
                        let value = if $fused {
                            let signed = _mm256_xor_pd(
                                _mm256_permute_pd::<5>(a),
                                _mm256_set_pd(0., -0., 0., -0.),
                            );
                            let s = _mm256_fmadd_pd(_mm256_movedup_pd(b), a, old);
                            _mm256_fmadd_pd(_mm256_permute_pd::<15>(b), signed, s)
                        } else {
                            let product = _mm256_addsub_pd(
                                _mm256_mul_pd(_mm256_movedup_pd(a), b),
                                _mm256_mul_pd(
                                    _mm256_permute_pd::<15>(a),
                                    _mm256_permute_pd::<5>(b),
                                ),
                            );
                            _mm256_add_pd(old, product)
                        };
                        _mm256_storeu_pd(sum.as_mut_ptr().add(i).cast(), value);
                    }
                    negative_scalar::<$fused>(&mut sum[bulk..], &x[..x.len() - bulk], &h[bulk..]);
                }
            }
        };
    }
    avx_impl!(avx, "avx", false);
    avx_impl!(avx_fma, "avx,fma", true);
    macro_rules! avx512_impl {
        ($name:ident,$features:literal,$fused:expr) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
                unsafe {
                    let bulk = sum.len() / 4 * 4;
                    let reverse = _mm512_setr_epi64(6, 7, 4, 5, 2, 3, 0, 1);
                    let conjugate =
                        _mm512_set_epi64(i64::MIN, 0, i64::MIN, 0, i64::MIN, 0, i64::MIN, 0);
                    let real_sign =
                        _mm512_set_epi64(0, i64::MIN, 0, i64::MIN, 0, i64::MIN, 0, i64::MIN);
                    for i in (0..bulk).step_by(4) {
                        let a = _mm512_loadu_pd(x.as_ptr().add(x.len() - i - 4).cast());
                        let a = _mm512_castsi512_pd(_mm512_xor_si512(
                            _mm512_castpd_si512(_mm512_permutexvar_pd(reverse, a)),
                            conjugate,
                        ));
                        let b = _mm512_loadu_pd(h.as_ptr().add(i).cast());
                        let old = _mm512_loadu_pd(sum.as_ptr().add(i).cast());
                        let value = if $fused {
                            let signed = _mm512_castsi512_pd(_mm512_xor_si512(
                                _mm512_castpd_si512(_mm512_permute_pd::<85>(a)),
                                real_sign,
                            ));
                            let s = _mm512_fmadd_pd(_mm512_movedup_pd(b), a, old);
                            _mm512_fmadd_pd(_mm512_permute_pd::<255>(b), signed, s)
                        } else {
                            let re = _mm512_mul_pd(_mm512_movedup_pd(a), b);
                            let im = _mm512_mul_pd(
                                _mm512_permute_pd::<255>(a),
                                _mm512_permute_pd::<85>(b),
                            );
                            let product = _mm512_mask_sub_pd(_mm512_add_pd(re, im), 0x55, re, im);
                            _mm512_add_pd(old, product)
                        };
                        _mm512_storeu_pd(sum.as_mut_ptr().add(i).cast(), value);
                    }
                    negative_scalar::<$fused>(&mut sum[bulk..], &x[..x.len() - bulk], &h[bulk..]);
                }
            }
        };
    }
    avx512_impl!(avx512, "avx512f", false);
    avx512_impl!(avx512_fma, "avx512f,fma", true);
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        unsafe {
            let conjugate = vsetq_lane_u64::<1>(1u64 << 63, vdupq_n_u64(0));
            let sign = vsetq_lane_u64::<0>(1u64 << 63, vdupq_n_u64(0));
            for i in 0..sum.len() {
                let a = vld1q_f64(x.as_ptr().add(x.len() - 1 - i).cast());
                let a = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), conjugate));
                let b = vld1q_f64(h.as_ptr().add(i).cast());
                let old = vld1q_f64(sum.as_ptr().add(i).cast());
                let swapped = vextq_f64::<1>(a, a);
                let signed = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(swapped), sign));
                let s = vfmaq_f64(old, vdupq_laneq_f64::<0>(b), a);
                let s = vfmaq_f64(s, vdupq_laneq_f64::<1>(b), signed);
                vst1q_f64(sum.as_mut_ptr().add(i).cast(), s);
            }
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::*;
    use std::arch::wasm32::*;
    pub(super) unsafe fn run(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        unsafe {
            for i in 0..sum.len() {
                let a = v128_xor(
                    v128_load(x.as_ptr().add(x.len() - 1 - i).cast()),
                    i64x2(0, i64::MIN),
                );
                let b = v128_load(h.as_ptr().add(i).cast());
                let old = v128_load(sum.as_ptr().add(i).cast());
                let re = f64x2_mul(i64x2_shuffle::<0, 0>(a, a), b);
                let im = f64x2_mul(i64x2_shuffle::<1, 1>(a, a), i64x2_shuffle::<1, 0>(b, b));
                let product = f64x2_add(re, v128_xor(im, i64x2(i64::MIN, 0)));
                v128_store(sum.as_mut_ptr().add(i).cast(), f64x2_add(old, product));
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn compact_history_also_preserves_forced_unfused_avx_models() {
        let mut variants: Vec<(Mac, Mac)> = Vec::new();
        if std::is_x86_feature_detected!("avx") {
            variants.push((mac_avx_checked, avx_checked));
        }
        if std::is_x86_feature_detected!("avx512f") {
            variants.push((mac_avx512_checked, avx512_checked));
        }
        for half in [1, 2, 3, 4, 5, 255, 256, 257, 511, 512, 513] {
            let n = 2 * half;
            let mut positive: Vec<_> = (0..=half)
                .map(|j| {
                    Complex::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            positive[0].im = -0.;
            positive[half].im = 0.;
            let mut full = vec![Complex::default(); n];
            full[..half + 1].copy_from_slice(&positive);
            for k in half + 1..n {
                full[k] = positive[n - k].conj();
            }
            let h: Vec<_> = (0..n)
                .map(|j| {
                    Complex::new(
                        ((j * 17 % 89) as f64 - 44.) / 97.,
                        ((j * 19 % 97) as f64 - 48.) / 101.,
                    )
                })
                .collect();
            for &(normal, negative) in &variants {
                let mut a = vec![Complex::new(-0., 0.); n];
                let mut b = a.clone();
                let compact = RealHistoryMac {
                    positive: normal,
                    negative,
                };
                normal(&mut a, &full, &h);
                compact.apply(&mut b, &positive, &h);
                for (a, b) in a.iter().zip(&b) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits());
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
            }
        }
    }
    #[test]
    fn compact_hermitian_history_mac_preserves_forced_backend_bits_and_endpoints() {
        let kernels = crate::simd::tests::coarse_available();
        for half in [
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 255, 256, 257, 511, 512, 513,
        ] {
            for scale in [1., 1e-150, 1e-300, 1e-319] {
                let n = 2 * half;
                let mut positive: Vec<_> = (0..=half)
                    .map(|j| {
                        Complex::new(
                            if j % 17 == 0 {
                                -0.
                            } else {
                                ((j * 37 % 127) as f64 - 63.) / 127. * scale
                            },
                            if j % 19 == 0 {
                                0.
                            } else {
                                ((j * 41 % 131) as f64 - 65.) / 131. * scale
                            },
                        )
                    })
                    .collect();
                positive[0].im = 0.;
                positive[half].im = -0.;
                let mut full = vec![Complex::default(); n];
                full[..half + 1].copy_from_slice(&positive);
                for k in half + 1..n {
                    full[k] = positive[n - k].conj();
                }
                let h: Vec<_> = (0..n)
                    .map(|j| {
                        Complex::new(
                            ((j * 17 % 89) as f64 - 44.) / 97.,
                            ((j * 19 % 97) as f64 - 48.) / 101.,
                        )
                    })
                    .collect();
                let initial: Vec<_> = (0..n)
                    .map(|j| Complex::new((j as f64 - 17.) * scale / 100., -0.))
                    .collect();
                for &kernel in &kernels {
                    let mut expected = initial.clone();
                    let mut actual = initial.clone();
                    let full_mac = kernel.coarse_mac_function();
                    let compact = kernel.coarse_real_history_mac();
                    for _ in 0..3 {
                        full_mac(&mut expected, &full, &h);
                        compact.apply(&mut actual, &positive, &h);
                    }
                    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.re.to_bits(),
                            b.re.to_bits(),
                            "{kernel:?} half{half} scale{scale} bin{i} real"
                        );
                        assert_eq!(
                            a.im.to_bits(),
                            b.im.to_bits(),
                            "{kernel:?} half{half} scale{scale} bin{i} imag"
                        );
                    }
                }
            }
        }
    }
}
