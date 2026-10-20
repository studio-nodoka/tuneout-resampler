//! Residue-weighted complex states for a provably real analytic input.
//! This is the same modal transfer: y=c*s and y'=p*y+c*x. Six multiplies
//! replace eight per mode. States are weighted once after unweighted seeding.
//! Every supported CPU has an unfused implementation and portable fallback.
use super::*;
impl Kernel {
    pub(crate) fn coarse_weighted_real<const ADVANCE: bool>(
        self,
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: f64,
    ) -> Complex {
        assert_eq!(row.len(), states.len());
        assert_eq!(poles.len(), states.len());
        let incoming = Complex::new(incoming, 0.);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { wasm::weight::<false, ADVANCE>(row, poles, states, incoming) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => weight_scalar::<false, ADVANCE>(row, poles, states, incoming),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe {
                x86::sse_weight::<false, ADVANCE>(row, poles, states, incoming)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe {
                x86::avx_weight::<false, ADVANCE>(
                    row,
                    poles,
                    states,
                    incoming,
                    0,
                    [Complex::default(); 4],
                )
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe {
                x86::avx512_weight::<false, ADVANCE>(row, poles, states, incoming)
            },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe {
                neon::weight::<false, ADVANCE>(row, poles, states, incoming)
            },
        }
    }
}
#[inline(always)]
fn finish<const REAL: bool, const ADVANCE: bool>(
    mut sums: [Complex; 4],
    row: &Planes,
    poles: &Planes,
    states: &mut Planes,
    incoming: Complex,
    start: usize,
) -> Complex {
    for i in start..row.len() {
        let lane = i - start;
        let s = Complex::new(states.re[i], states.im[i]);
        let c = Complex::new(row.re[i], row.im[i]);
        sums[lane].re += s.re;
        if !REAL {
            sums[lane].im += s.im;
        }
        if ADVANCE {
            let z = Complex::new(poles.re[i], poles.im[i]) * s
                + Complex::new(c.re * incoming.re, c.im * incoming.re);
            states.re[i] = z.re;
            states.im[i] = z.im;
        }
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

fn weight_scalar<const REAL: bool, const ADVANCE: bool>(
    row: &Planes,
    poles: &Planes,
    states: &mut Planes,
    incoming: Complex,
) -> Complex {
    let mut sums = [Complex::default(); 4];
    let bulk = row.len() / 4 * 4;
    for i in (0..bulk).step_by(4) {
        for (lane, sum) in sums.iter_mut().enumerate() {
            let j = i + lane;
            let s = Complex::new(states.re[j], states.im[j]);
            let c = Complex::new(row.re[j], row.im[j]);
            sum.re += s.re;
            if !REAL {
                sum.im += s.im;
            }
            if ADVANCE {
                let z = Complex::new(poles.re[j], poles.im[j]) * s
                    + Complex::new(c.re * incoming.re, c.im * incoming.re);
                states.re[j] = z.re;
                states.im[j] = z.im;
            }
        }
    }
    finish::<REAL, ADVANCE>(sums, row, poles, states, incoming, bulk)
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sse_weight<const REAL: bool, const ADVANCE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: Complex,
    ) -> Complex {
        unsafe {
            let mut re = [_mm_setzero_pd(); 2];
            let mut im = [_mm_setzero_pd(); 2];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for part in 0..2 {
                    let j = i + part * 2;
                    let sr = _mm_loadu_pd(states.re.as_ptr().add(j));
                    let si = _mm_loadu_pd(states.im.as_ptr().add(j));
                    let cr = _mm_loadu_pd(row.re.as_ptr().add(j));
                    let ci = _mm_loadu_pd(row.im.as_ptr().add(j));
                    re[part] = _mm_add_pd(re[part], sr);
                    if !REAL {
                        im[part] = _mm_add_pd(im[part], si);
                    }
                    if ADVANCE {
                        let pr = _mm_loadu_pd(poles.re.as_ptr().add(j));
                        let pi = _mm_loadu_pd(poles.im.as_ptr().add(j));
                        _mm_storeu_pd(
                            states.re.as_mut_ptr().add(j),
                            _mm_add_pd(
                                _mm_sub_pd(_mm_mul_pd(pr, sr), _mm_mul_pd(pi, si)),
                                _mm_mul_pd(cr, _mm_set1_pd(incoming.re)),
                            ),
                        );
                        _mm_storeu_pd(
                            states.im.as_mut_ptr().add(j),
                            _mm_add_pd(
                                _mm_add_pd(_mm_mul_pd(pr, si), _mm_mul_pd(pi, sr)),
                                _mm_mul_pd(ci, _mm_set1_pd(incoming.re)),
                            ),
                        );
                    }
                }
            }
            let mut real = [0.; 4];
            let mut imag = [0.; 4];
            for part in 0..2 {
                _mm_storeu_pd(real.as_mut_ptr().add(part * 2), re[part]);
                if !REAL {
                    _mm_storeu_pd(imag.as_mut_ptr().add(part * 2), im[part]);
                }
            }
            finish::<REAL, ADVANCE>(
                std::array::from_fn(|j| Complex::new(real[j], imag[j])),
                row,
                poles,
                states,
                incoming,
                bulk,
            )
        }
    }
    #[target_feature(enable = "avx")]
    pub(super) unsafe fn avx_weight<const REAL: bool, const ADVANCE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: Complex,
        start: usize,
        sums: [Complex; 4],
    ) -> Complex {
        unsafe {
            let initial_re = sums.map(|z| z.re);
            let initial_im = sums.map(|z| z.im);
            let mut re = _mm256_loadu_pd(initial_re.as_ptr());
            let mut im = _mm256_loadu_pd(initial_im.as_ptr());
            let bulk = row.len() / 4 * 4;
            for j in (start..bulk).step_by(4) {
                let sr = _mm256_loadu_pd(states.re.as_ptr().add(j));
                let si = _mm256_loadu_pd(states.im.as_ptr().add(j));
                let cr = _mm256_loadu_pd(row.re.as_ptr().add(j));
                let ci = _mm256_loadu_pd(row.im.as_ptr().add(j));
                re = _mm256_add_pd(re, sr);
                if !REAL {
                    im = _mm256_add_pd(im, si);
                }
                if ADVANCE {
                    let pr = _mm256_loadu_pd(poles.re.as_ptr().add(j));
                    let pi = _mm256_loadu_pd(poles.im.as_ptr().add(j));
                    _mm256_storeu_pd(
                        states.re.as_mut_ptr().add(j),
                        _mm256_add_pd(
                            _mm256_sub_pd(_mm256_mul_pd(pr, sr), _mm256_mul_pd(pi, si)),
                            _mm256_mul_pd(cr, _mm256_set1_pd(incoming.re)),
                        ),
                    );
                    _mm256_storeu_pd(
                        states.im.as_mut_ptr().add(j),
                        _mm256_add_pd(
                            _mm256_add_pd(_mm256_mul_pd(pr, si), _mm256_mul_pd(pi, sr)),
                            _mm256_mul_pd(ci, _mm256_set1_pd(incoming.re)),
                        ),
                    );
                }
            }
            let mut real = [0.; 4];
            let mut imag = [0.; 4];
            _mm256_storeu_pd(real.as_mut_ptr(), re);
            if !REAL {
                _mm256_storeu_pd(imag.as_mut_ptr(), im);
            }
            finish::<REAL, ADVANCE>(
                std::array::from_fn(|j| Complex::new(real[j], imag[j])),
                row,
                poles,
                states,
                incoming,
                bulk,
            )
        }
    }
    #[target_feature(enable = "avx512f,avx")]
    pub(super) unsafe fn avx512_weight<const REAL: bool, const ADVANCE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: Complex,
    ) -> Complex {
        unsafe {
            let mut re = _mm256_setzero_pd();
            let mut im = _mm256_setzero_pd();
            let bulk = row.len() / 8 * 8;
            for j in (0..bulk).step_by(8) {
                let sr = _mm512_loadu_pd(states.re.as_ptr().add(j));
                let si = _mm512_loadu_pd(states.im.as_ptr().add(j));
                let cr = _mm512_loadu_pd(row.re.as_ptr().add(j));
                let ci = _mm512_loadu_pd(row.im.as_ptr().add(j));
                let v = sr;
                re = _mm256_add_pd(re, _mm512_castpd512_pd256(v));
                re = _mm256_add_pd(re, _mm512_extractf64x4_pd::<1>(v));
                if !REAL {
                    let v = si;
                    im = _mm256_add_pd(im, _mm512_castpd512_pd256(v));
                    im = _mm256_add_pd(im, _mm512_extractf64x4_pd::<1>(v));
                }
                if ADVANCE {
                    let pr = _mm512_loadu_pd(poles.re.as_ptr().add(j));
                    let pi = _mm512_loadu_pd(poles.im.as_ptr().add(j));
                    _mm512_storeu_pd(
                        states.re.as_mut_ptr().add(j),
                        _mm512_add_pd(
                            _mm512_sub_pd(_mm512_mul_pd(pr, sr), _mm512_mul_pd(pi, si)),
                            _mm512_mul_pd(cr, _mm512_set1_pd(incoming.re)),
                        ),
                    );
                    _mm512_storeu_pd(
                        states.im.as_mut_ptr().add(j),
                        _mm512_add_pd(
                            _mm512_add_pd(_mm512_mul_pd(pr, si), _mm512_mul_pd(pi, sr)),
                            _mm512_mul_pd(ci, _mm512_set1_pd(incoming.re)),
                        ),
                    );
                }
            }
            let mut next = bulk;
            if row.len() - bulk >= 4 {
                let j = bulk;
                let sr = _mm256_loadu_pd(states.re.as_ptr().add(j));
                let si = _mm256_loadu_pd(states.im.as_ptr().add(j));
                let cr = _mm256_loadu_pd(row.re.as_ptr().add(j));
                let ci = _mm256_loadu_pd(row.im.as_ptr().add(j));
                re = _mm256_add_pd(re, sr);
                if !REAL {
                    im = _mm256_add_pd(im, si);
                }
                if ADVANCE {
                    let pr = _mm256_loadu_pd(poles.re.as_ptr().add(j));
                    let pi = _mm256_loadu_pd(poles.im.as_ptr().add(j));
                    _mm256_storeu_pd(
                        states.re.as_mut_ptr().add(j),
                        _mm256_add_pd(
                            _mm256_sub_pd(_mm256_mul_pd(pr, sr), _mm256_mul_pd(pi, si)),
                            _mm256_mul_pd(cr, _mm256_set1_pd(incoming.re)),
                        ),
                    );
                    _mm256_storeu_pd(
                        states.im.as_mut_ptr().add(j),
                        _mm256_add_pd(
                            _mm256_add_pd(_mm256_mul_pd(pr, si), _mm256_mul_pd(pi, sr)),
                            _mm256_mul_pd(ci, _mm256_set1_pd(incoming.re)),
                        ),
                    );
                }
                next += 4;
            }
            let mut real = [0.; 4];
            let mut imag = [0.; 4];
            _mm256_storeu_pd(real.as_mut_ptr(), re);
            if !REAL {
                _mm256_storeu_pd(imag.as_mut_ptr(), im);
            }
            finish::<REAL, ADVANCE>(
                std::array::from_fn(|j| Complex::new(real[j], imag[j])),
                row,
                poles,
                states,
                incoming,
                next,
            )
        }
    }
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn weight<const REAL: bool, const ADVANCE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: Complex,
    ) -> Complex {
        unsafe {
            let mut re = [vdupq_n_f64(0.); 2];
            let mut im = [vdupq_n_f64(0.); 2];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for part in 0..2 {
                    let j = i + part * 2;
                    let sr = vld1q_f64(states.re.as_ptr().add(j));
                    let si = vld1q_f64(states.im.as_ptr().add(j));
                    let cr = vld1q_f64(row.re.as_ptr().add(j));
                    let ci = vld1q_f64(row.im.as_ptr().add(j));
                    re[part] = vaddq_f64(re[part], sr);
                    if !REAL {
                        im[part] = vaddq_f64(im[part], si);
                    }
                    if ADVANCE {
                        let pr = vld1q_f64(poles.re.as_ptr().add(j));
                        let pi = vld1q_f64(poles.im.as_ptr().add(j));
                        vst1q_f64(
                            states.re.as_mut_ptr().add(j),
                            vaddq_f64(
                                vsubq_f64(vmulq_f64(pr, sr), vmulq_f64(pi, si)),
                                vmulq_f64(cr, vdupq_n_f64(incoming.re)),
                            ),
                        );
                        vst1q_f64(
                            states.im.as_mut_ptr().add(j),
                            vaddq_f64(
                                vaddq_f64(vmulq_f64(pr, si), vmulq_f64(pi, sr)),
                                vmulq_f64(ci, vdupq_n_f64(incoming.re)),
                            ),
                        );
                    }
                }
            }
            let mut real = [0.; 4];
            let mut imag = [0.; 4];
            for part in 0..2 {
                vst1q_f64(real.as_mut_ptr().add(part * 2), re[part]);
                if !REAL {
                    vst1q_f64(imag.as_mut_ptr().add(part * 2), im[part]);
                }
            }
            finish::<REAL, ADVANCE>(
                std::array::from_fn(|j| Complex::new(real[j], imag[j])),
                row,
                poles,
                states,
                incoming,
                bulk,
            )
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::*;
    use std::arch::wasm32::*;
    pub(super) unsafe fn weight<const REAL: bool, const ADVANCE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: Complex,
    ) -> Complex {
        unsafe {
            let mut re = [f64x2_splat(0.); 2];
            let mut im = [f64x2_splat(0.); 2];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for part in 0..2 {
                    let j = i + part * 2;
                    let sr = v128_load(states.re.as_ptr().add(j).cast());
                    let si = v128_load(states.im.as_ptr().add(j).cast());
                    let cr = v128_load(row.re.as_ptr().add(j).cast());
                    let ci = v128_load(row.im.as_ptr().add(j).cast());
                    re[part] = f64x2_add(re[part], sr);
                    if !REAL {
                        im[part] = f64x2_add(im[part], si);
                    }
                    if ADVANCE {
                        let pr = v128_load(poles.re.as_ptr().add(j).cast());
                        let pi = v128_load(poles.im.as_ptr().add(j).cast());
                        v128_store(
                            states.re.as_mut_ptr().add(j).cast(),
                            f64x2_add(
                                f64x2_sub(f64x2_mul(pr, sr), f64x2_mul(pi, si)),
                                f64x2_mul(cr, f64x2_splat(incoming.re)),
                            ),
                        );
                        v128_store(
                            states.im.as_mut_ptr().add(j).cast(),
                            f64x2_add(
                                f64x2_add(f64x2_mul(pr, si), f64x2_mul(pi, sr)),
                                f64x2_mul(ci, f64x2_splat(incoming.re)),
                            ),
                        );
                    }
                }
            }
            let mut real = [0.; 4];
            let mut imag = [0.; 4];
            for part in 0..2 {
                v128_store(real.as_mut_ptr().add(part * 2).cast(), re[part]);
                if !REAL {
                    v128_store(imag.as_mut_ptr().add(part * 2).cast(), im[part]);
                }
            }
            finish::<REAL, ADVANCE>(
                std::array::from_fn(|j| Complex::new(real[j], imag[j])),
                row,
                poles,
                states,
                incoming,
                bulk,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::tests::coarse_available as kernels;
    fn oracle<const ADVANCE: bool>(
        row: &Planes,
        poles: &Planes,
        state: &mut Planes,
        input: f64,
    ) -> Complex {
        let mut sums = [Complex::default(); 4];
        for i in 0..row.len() {
            let lane = i % 4;
            let re = state.re[i];
            let im = state.im[i];
            sums[lane].re += re;
            sums[lane].im += im;
            if ADVANCE {
                state.re[i] = (poles.re[i] * re - poles.im[i] * im) + row.re[i] * input;
                state.im[i] = (poles.re[i] * im + poles.im[i] * re) + row.im[i] * input;
            }
        }
        (sums[0] + sums[1]) + (sums[2] + sums[3])
    }
    fn check<const ADVANCE: bool>(kernel: Kernel, len: usize) {
        let row: Vec<_> = (0..len)
            .map(|j| {
                Complex::new(
                    ((j * 37 % 127) as f64 - 63.) / 127.,
                    ((j * 41 % 131) as f64 - 65.) / 131.,
                )
            })
            .collect();
        let poles: Vec<_> = row.iter().map(|z| *z * Complex::new(0.7, 0.1)).collect();
        let initial: Vec<_> = row.iter().rev().copied().collect();
        let row = Planes::from_complex(&row);
        let poles = Planes::from_complex(&poles);
        let mut actual = Planes::from_complex(&initial);
        let mut expected = Planes::from_complex(&initial);
        for input in [0., 0.37, -0.29] {
            let a = kernel.coarse_weighted_real::<ADVANCE>(&row, &poles, &mut actual, input);
            let e = oracle::<ADVANCE>(&row, &poles, &mut expected, input);
            assert_eq!(a.re.to_bits(), e.re.to_bits(), "{kernel:?} len{len}");
            assert_eq!(a.im.to_bits(), e.im.to_bits());
            for i in 0..len {
                assert_eq!(actual.re[i].to_bits(), expected.re[i].to_bits());
                assert_eq!(actual.im[i].to_bits(), expected.im[i].to_bits());
            }
        }
    }
    #[test]
    fn weighted_real_state_matches_independent_unfused_oracle_on_each_backend() {
        for len in 0..1032 {
            for kernel in kernels() {
                check::<false>(kernel, len);
                check::<true>(kernel, len);
            }
        }
    }
}
