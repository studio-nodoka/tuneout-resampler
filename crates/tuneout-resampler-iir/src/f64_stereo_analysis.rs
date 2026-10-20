//! Mirrored stereo windows reuse each analysis coefficient vector.
use crate::complex::Complex as C;
pub(crate) fn run<const IMAG: bool>(x: [&[f64]; 2], re: &[f64], im: &[f64]) -> [C; 2] {
    assert!(x.iter().all(|x| x.len() == 2 * re.len() + 1));
    if IMAG {
        assert_eq!(re.len(), im.len());
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::avx512::<IMAG>(x, re, im) };
        }
        // Real-only stereo needs eight accumulators and fits AVX registers.
        if !IMAG && std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::avx::<IMAG>(x, re, im) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::run::<IMAG>(x, re, im) };
    }
    std::array::from_fn(|ch| {
        if IMAG {
            let (a, b) = crate::f64_symmetric_analysis::run(x[ch], re, im);
            C::new(a, b)
        } else {
            C::new(crate::f64_symmetric_analysis::run_real(x[ch], re), 0.)
        }
    })
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx512f")]
    unsafe fn reverse512(v: __m512d) -> __m512d {
        _mm512_permutexvar_pd(_mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7), v)
    }
    #[target_feature(enable = "avx")]
    unsafe fn reverse256(v: __m256d) -> __m256d {
        _mm256_permute_pd(_mm256_permute2f128_pd::<1>(v, v), 5)
    }
    macro_rules! implementation {
        ($name:ident,$features:literal,$lanes:expr,$zero:ident,$load:ident,$reverse:ident,$add:ident,$sub:ident,$madd:ident,$store:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name<const IMAG: bool>(
                x: [&[f64]; 2],
                re: &[f64],
                im: &[f64],
            ) -> [C; 2] {
                unsafe {
                    let mut a = [[$zero(); 4]; 2];
                    let mut b = a;
                    let bulk = re.len() / (4 * $lanes) * (4 * $lanes);
                    for base in (0..bulk).step_by(4 * $lanes) {
                        for j in 0..4 {
                            let i = base + j * $lanes;
                            let cr = $load(re.as_ptr().add(i));
                            let ci = if IMAG {
                                $load(im.as_ptr().add(i))
                            } else {
                                $zero()
                            };
                            for ch in 0..2 {
                                let l = $load(x[ch].as_ptr().add(i));
                                let r =
                                    $reverse($load(x[ch].as_ptr().add(x[ch].len() - i - $lanes)));
                                a[ch][j] = $madd($add(l, r), cr, a[ch][j]);
                                if IMAG {
                                    b[ch][j] = $madd($sub(l, r), ci, b[ch][j]);
                                }
                            }
                        }
                    }
                    let vectors = re.len() / $lanes * $lanes;
                    for i in (bulk..vectors).step_by($lanes) {
                        let j = (i - bulk) / $lanes;
                        let cr = $load(re.as_ptr().add(i));
                        let ci = if IMAG {
                            $load(im.as_ptr().add(i))
                        } else {
                            $zero()
                        };
                        for ch in 0..2 {
                            let l = $load(x[ch].as_ptr().add(i));
                            let r = $reverse($load(x[ch].as_ptr().add(x[ch].len() - i - $lanes)));
                            a[ch][j] = $madd($add(l, r), cr, a[ch][j]);
                            if IMAG {
                                b[ch][j] = $madd($sub(l, r), ci, b[ch][j]);
                            }
                        }
                    }
                    let mut result = [C::default(); 2];
                    for ch in 0..2 {
                        let mut ar = [0.; $lanes];
                        let mut ai = ar;
                        $store(
                            ar.as_mut_ptr(),
                            $add($add(a[ch][0], a[ch][1]), $add(a[ch][2], a[ch][3])),
                        );
                        $store(
                            ai.as_mut_ptr(),
                            $add($add(b[ch][0], b[ch][1]), $add(b[ch][2], b[ch][3])),
                        );
                        result[ch] = C::new(ar.into_iter().sum(), ai.into_iter().sum());
                    }
                    for i in vectors..re.len() {
                        for ch in 0..2 {
                            let l = x[ch][i];
                            let r = x[ch][x[ch].len() - 1 - i];
                            result[ch].re += (l + r) * re[i];
                            if IMAG {
                                result[ch].im += (l - r) * im[i];
                            }
                        }
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
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run<const IMAG: bool>(x: [&[f64]; 2], re: &[f64], im: &[f64]) -> [C; 2] {
        unsafe {
            let mut a = [[vdupq_n_f64(0.); 4]; 2];
            let mut b = a;
            let bulk = re.len() / 8 * 8;
            for base in (0..bulk).step_by(8) {
                for j in 0..4 {
                    let i = base + 2 * j;
                    let cr = vld1q_f64(re.as_ptr().add(i));
                    let ci = if IMAG {
                        vld1q_f64(im.as_ptr().add(i))
                    } else {
                        vdupq_n_f64(0.)
                    };
                    for ch in 0..2 {
                        let l = vld1q_f64(x[ch].as_ptr().add(i));
                        let r = vld1q_f64(x[ch].as_ptr().add(x[ch].len() - i - 2));
                        let r = vextq_f64::<1>(r, r);
                        a[ch][j] = vfmaq_f64(a[ch][j], vaddq_f64(l, r), cr);
                        if IMAG {
                            b[ch][j] = vfmaq_f64(b[ch][j], vsubq_f64(l, r), ci);
                        }
                    }
                }
            }
            let vectors = re.len() / 2 * 2;
            for i in (bulk..vectors).step_by(2) {
                let j = (i - bulk) / 2;
                let cr = vld1q_f64(re.as_ptr().add(i));
                let ci = if IMAG {
                    vld1q_f64(im.as_ptr().add(i))
                } else {
                    vdupq_n_f64(0.)
                };
                for ch in 0..2 {
                    let l = vld1q_f64(x[ch].as_ptr().add(i));
                    let r = vld1q_f64(x[ch].as_ptr().add(x[ch].len() - i - 2));
                    let r = vextq_f64::<1>(r, r);
                    a[ch][j] = vfmaq_f64(a[ch][j], vaddq_f64(l, r), cr);
                    if IMAG {
                        b[ch][j] = vfmaq_f64(b[ch][j], vsubq_f64(l, r), ci);
                    }
                }
            }
            let mut result = [C::default(); 2];
            for ch in 0..2 {
                let ar = vaddq_f64(vaddq_f64(a[ch][0], a[ch][1]), vaddq_f64(a[ch][2], a[ch][3]));
                let ai = vaddq_f64(vaddq_f64(b[ch][0], b[ch][1]), vaddq_f64(b[ch][2], b[ch][3]));
                let mut rr = [0.; 2];
                let mut ii = rr;
                vst1q_f64(rr.as_mut_ptr(), ar);
                vst1q_f64(ii.as_mut_ptr(), ai);
                result[ch] = C::new(rr[0] + rr[1], ii[0] + ii[1]);
            }
            for i in vectors..re.len() {
                for ch in 0..2 {
                    let l = x[ch][i];
                    let r = x[ch][x[ch].len() - 1 - i];
                    result[ch].re += (l + r) * re[i];
                    if IMAG {
                        result[ch].im += (l - r) * im[i];
                    }
                }
            }
            result
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stereo_mirrored_analysis_preserves_both_original_channel_bits() {
        for n in 0..1032 {
            let samples: [Vec<f64>; 2] = std::array::from_fn(|ch| {
                (0..2 * n + 1)
                    .map(|i| ((i * (31 + ch * 2) % 127) as f64 - 63.) / 127.)
                    .collect()
            });
            let x = [&samples[0][..], &samples[1][..]];
            let re: Vec<_> = (0..n)
                .map(|i| ((i * 41 % 131) as f64 - 65.) / 131.)
                .collect();
            let im: Vec<_> = (0..n)
                .map(|i| ((i * 43 % 137) as f64 - 68.) / 137.)
                .collect();
            let complex = run::<true>(x, &re, &im);
            let real = run::<false>(x, &re, &[]);
            for ch in 0..2 {
                let (a, b) = crate::f64_symmetric_analysis::run(x[ch], &re, &im);
                assert_eq!(complex[ch].re.to_bits(), a.to_bits());
                assert_eq!(complex[ch].im.to_bits(), b.to_bits());
                assert_eq!(
                    real[ch].re.to_bits(),
                    crate::f64_symmetric_analysis::run_real(x[ch], &re).to_bits()
                );
                assert_eq!(real[ch].im, 0.);
            }
        }
    }
}
