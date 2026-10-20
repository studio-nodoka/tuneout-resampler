//! Split real/imaginary modal arrays. SIMD lanes remain independent modes.
//! The four real/imaginary accumulator lanes retain the qualified reduction.
use super::*;

#[derive(Default)]
pub(crate) struct Planes {
    pub(crate) re: Vec<f64>,
    pub(crate) im: Vec<f64>,
}
impl Planes {
    pub(crate) fn from_complex(v: &[Complex]) -> Self {
        Self {
            re: v.iter().map(|x| x.re).collect(),
            im: v.iter().map(|x| x.im).collect(),
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.re.len()
    }
    pub(crate) fn resize(&mut self, n: usize, value: Complex) {
        self.re.resize(n, value.re);
        self.im.resize(n, value.im);
    }
    pub(crate) fn clear(&mut self) {
        self.re.clear();
        self.im.clear();
    }
    pub(crate) fn bytes(&self) -> usize {
        (self.re.capacity() + self.im.capacity()) * 8
    }
}
impl Kernel {
    pub(crate) fn coarse_weight_planes<const REAL: bool, const ADVANCE: bool>(
        self,
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: Complex,
    ) -> Complex {
        assert_eq!(row.len(), states.len());
        assert_eq!(poles.len(), states.len());
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { wasm::weight::<REAL, ADVANCE>(row, poles, states, incoming) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => weight_scalar::<REAL, ADVANCE>(row, poles, states, incoming),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe {
                x86::sse_weight::<REAL, ADVANCE>(row, poles, states, incoming)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe {
                x86::avx_weight::<REAL, ADVANCE>(
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
                x86::avx512_weight::<REAL, ADVANCE>(row, poles, states, incoming)
            },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { neon::weight::<REAL, ADVANCE>(row, poles, states, incoming) },
        }
    }
    pub(crate) fn coarse_seed_planes<const REVERSE: bool>(
        self,
        poles: &Planes,
        states: &mut Planes,
        samples: &[Complex],
    ) {
        assert_eq!(poles.len(), states.len());
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { wasm::seed::<REVERSE>(poles, states, samples) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => seed_scalar::<REVERSE>(poles, states, samples, 0),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::sse_seed::<REVERSE>(poles, states, samples, 0) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe { x86::avx_seed::<REVERSE>(poles, states, samples, 0) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe { x86::avx512_seed::<REVERSE>(poles, states, samples) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { neon::seed::<REVERSE>(poles, states, samples) },
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
        sums[lane].re += c.re * s.re - c.im * s.im;
        if !REAL {
            sums[lane].im += c.re * s.im + c.im * s.re;
        }
        if ADVANCE {
            let z = Complex::new(poles.re[i], poles.im[i]) * s + incoming;
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
            sum.re += c.re * s.re - c.im * s.im;
            if !REAL {
                sum.im += c.re * s.im + c.im * s.re;
            }
            if ADVANCE {
                let z = Complex::new(poles.re[j], poles.im[j]) * s + incoming;
                states.re[j] = z.re;
                states.im[j] = z.im;
            }
        }
    }
    finish::<REAL, ADVANCE>(sums, row, poles, states, incoming, bulk)
}
fn seed_scalar<const REVERSE: bool>(
    poles: &Planes,
    states: &mut Planes,
    samples: &[Complex],
    start: usize,
) {
    for i in start..poles.len() {
        let pole = Complex::new(poles.re[i], poles.im[i]);
        let mut value = Complex::default();
        if REVERSE {
            for &x in samples.iter().rev() {
                value = pole * value + x;
            }
        } else {
            for &x in samples {
                value = pole * value + x;
            }
        }
        states.re[i] = value.re;
        states.im[i] = value.im;
    }
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
                    re[part] =
                        _mm_add_pd(re[part], _mm_sub_pd(_mm_mul_pd(cr, sr), _mm_mul_pd(ci, si)));
                    if !REAL {
                        im[part] = _mm_add_pd(
                            im[part],
                            _mm_add_pd(_mm_mul_pd(cr, si), _mm_mul_pd(ci, sr)),
                        );
                    }
                    if ADVANCE {
                        let pr = _mm_loadu_pd(poles.re.as_ptr().add(j));
                        let pi = _mm_loadu_pd(poles.im.as_ptr().add(j));
                        _mm_storeu_pd(
                            states.re.as_mut_ptr().add(j),
                            _mm_add_pd(
                                _mm_sub_pd(_mm_mul_pd(pr, sr), _mm_mul_pd(pi, si)),
                                _mm_set1_pd(incoming.re),
                            ),
                        );
                        _mm_storeu_pd(
                            states.im.as_mut_ptr().add(j),
                            _mm_add_pd(
                                _mm_add_pd(_mm_mul_pd(pr, si), _mm_mul_pd(pi, sr)),
                                _mm_set1_pd(incoming.im),
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
                re = _mm256_add_pd(
                    re,
                    _mm256_sub_pd(_mm256_mul_pd(cr, sr), _mm256_mul_pd(ci, si)),
                );
                if !REAL {
                    im = _mm256_add_pd(
                        im,
                        _mm256_add_pd(_mm256_mul_pd(cr, si), _mm256_mul_pd(ci, sr)),
                    );
                }
                if ADVANCE {
                    let pr = _mm256_loadu_pd(poles.re.as_ptr().add(j));
                    let pi = _mm256_loadu_pd(poles.im.as_ptr().add(j));
                    _mm256_storeu_pd(
                        states.re.as_mut_ptr().add(j),
                        _mm256_add_pd(
                            _mm256_sub_pd(_mm256_mul_pd(pr, sr), _mm256_mul_pd(pi, si)),
                            _mm256_set1_pd(incoming.re),
                        ),
                    );
                    _mm256_storeu_pd(
                        states.im.as_mut_ptr().add(j),
                        _mm256_add_pd(
                            _mm256_add_pd(_mm256_mul_pd(pr, si), _mm256_mul_pd(pi, sr)),
                            _mm256_set1_pd(incoming.im),
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
                let v = _mm512_sub_pd(_mm512_mul_pd(cr, sr), _mm512_mul_pd(ci, si));
                re = _mm256_add_pd(re, _mm512_castpd512_pd256(v));
                re = _mm256_add_pd(re, _mm512_extractf64x4_pd::<1>(v));
                if !REAL {
                    let v = _mm512_add_pd(_mm512_mul_pd(cr, si), _mm512_mul_pd(ci, sr));
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
                            _mm512_set1_pd(incoming.re),
                        ),
                    );
                    _mm512_storeu_pd(
                        states.im.as_mut_ptr().add(j),
                        _mm512_add_pd(
                            _mm512_add_pd(_mm512_mul_pd(pr, si), _mm512_mul_pd(pi, sr)),
                            _mm512_set1_pd(incoming.im),
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
                re = _mm256_add_pd(
                    re,
                    _mm256_sub_pd(_mm256_mul_pd(cr, sr), _mm256_mul_pd(ci, si)),
                );
                if !REAL {
                    im = _mm256_add_pd(
                        im,
                        _mm256_add_pd(_mm256_mul_pd(cr, si), _mm256_mul_pd(ci, sr)),
                    );
                }
                if ADVANCE {
                    let pr = _mm256_loadu_pd(poles.re.as_ptr().add(j));
                    let pi = _mm256_loadu_pd(poles.im.as_ptr().add(j));
                    _mm256_storeu_pd(
                        states.re.as_mut_ptr().add(j),
                        _mm256_add_pd(
                            _mm256_sub_pd(_mm256_mul_pd(pr, sr), _mm256_mul_pd(pi, si)),
                            _mm256_set1_pd(incoming.re),
                        ),
                    );
                    _mm256_storeu_pd(
                        states.im.as_mut_ptr().add(j),
                        _mm256_add_pd(
                            _mm256_add_pd(_mm256_mul_pd(pr, si), _mm256_mul_pd(pi, sr)),
                            _mm256_set1_pd(incoming.im),
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
    macro_rules! seed_impl {
        ($name:ident,$features:literal,$width:expr,$zero:ident,$load:ident,$store:ident,$add:ident,$sub:ident,$mul:ident,$splat:ident,$tail:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name<const REVERSE: bool>(
                poles: &Planes,
                states: &mut Planes,
                samples: &[Complex],
                start: usize,
            ) {
                unsafe {
                    let bulk = poles.len() / $width * $width;
                    for i in (start..bulk).step_by($width) {
                        let pr = $load(poles.re.as_ptr().add(i));
                        let pi = $load(poles.im.as_ptr().add(i));
                        let mut re = $zero();
                        let mut im = $zero();
                        for j in 0..samples.len() {
                            let x = samples[if REVERSE { samples.len() - 1 - j } else { j }];
                            let next_re = $add($sub($mul(pr, re), $mul(pi, im)), $splat(x.re));
                            im = $add($add($mul(pr, im), $mul(pi, re)), $splat(x.im));
                            re = next_re;
                        }
                        $store(states.re.as_mut_ptr().add(i), re);
                        $store(states.im.as_mut_ptr().add(i), im);
                    }
                    $tail::<REVERSE>(poles, states, samples, bulk);
                }
            }
        };
    }
    seed_impl!(
        sse_seed,
        "sse2",
        2,
        _mm_setzero_pd,
        _mm_loadu_pd,
        _mm_storeu_pd,
        _mm_add_pd,
        _mm_sub_pd,
        _mm_mul_pd,
        _mm_set1_pd,
        seed_scalar
    );
    seed_impl!(
        avx_seed,
        "avx",
        4,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        _mm256_sub_pd,
        _mm256_mul_pd,
        _mm256_set1_pd,
        seed_scalar
    );
    #[target_feature(enable = "avx512f,avx")]
    pub(super) unsafe fn avx512_seed<const REVERSE: bool>(
        poles: &Planes,
        states: &mut Planes,
        samples: &[Complex],
    ) {
        unsafe {
            let bulk = poles.len() / 8 * 8;
            for i in (0..bulk).step_by(8) {
                let pr = _mm512_loadu_pd(poles.re.as_ptr().add(i));
                let pi = _mm512_loadu_pd(poles.im.as_ptr().add(i));
                let mut re = _mm512_setzero_pd();
                let mut im = _mm512_setzero_pd();
                for j in 0..samples.len() {
                    let x = samples[if REVERSE { samples.len() - 1 - j } else { j }];
                    let next_re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr, re), _mm512_mul_pd(pi, im)),
                        _mm512_set1_pd(x.re),
                    );
                    im = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr, im), _mm512_mul_pd(pi, re)),
                        _mm512_set1_pd(x.im),
                    );
                    re = next_re;
                }
                _mm512_storeu_pd(states.re.as_mut_ptr().add(i), re);
                _mm512_storeu_pd(states.im.as_mut_ptr().add(i), im);
            }
            avx_seed::<REVERSE>(poles, states, samples, bulk);
        }
    }
}

// Architecture-specific plane kernels below are deliberately unfused. Each
// individual modal recurrence has the same f64 product/add ordering everywhere.
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
                    re[part] = vaddq_f64(re[part], vsubq_f64(vmulq_f64(cr, sr), vmulq_f64(ci, si)));
                    if !REAL {
                        im[part] =
                            vaddq_f64(im[part], vaddq_f64(vmulq_f64(cr, si), vmulq_f64(ci, sr)));
                    }
                    if ADVANCE {
                        let pr = vld1q_f64(poles.re.as_ptr().add(j));
                        let pi = vld1q_f64(poles.im.as_ptr().add(j));
                        vst1q_f64(
                            states.re.as_mut_ptr().add(j),
                            vaddq_f64(
                                vsubq_f64(vmulq_f64(pr, sr), vmulq_f64(pi, si)),
                                vdupq_n_f64(incoming.re),
                            ),
                        );
                        vst1q_f64(
                            states.im.as_mut_ptr().add(j),
                            vaddq_f64(
                                vaddq_f64(vmulq_f64(pr, si), vmulq_f64(pi, sr)),
                                vdupq_n_f64(incoming.im),
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
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn seed<const REVERSE: bool>(
        poles: &Planes,
        states: &mut Planes,
        samples: &[Complex],
    ) {
        unsafe {
            let bulk = poles.len() / 2 * 2;
            for i in (0..bulk).step_by(2) {
                let pr = vld1q_f64(poles.re.as_ptr().add(i));
                let pi = vld1q_f64(poles.im.as_ptr().add(i));
                let mut re = vdupq_n_f64(0.);
                let mut im = vdupq_n_f64(0.);
                for j in 0..samples.len() {
                    let x = samples[if REVERSE { samples.len() - 1 - j } else { j }];
                    let next_re = vaddq_f64(
                        vsubq_f64(vmulq_f64(pr, re), vmulq_f64(pi, im)),
                        vdupq_n_f64(x.re),
                    );
                    im = vaddq_f64(
                        vaddq_f64(vmulq_f64(pr, im), vmulq_f64(pi, re)),
                        vdupq_n_f64(x.im),
                    );
                    re = next_re;
                }
                vst1q_f64(states.re.as_mut_ptr().add(i), re);
                vst1q_f64(states.im.as_mut_ptr().add(i), im);
            }
            seed_scalar::<REVERSE>(poles, states, samples, bulk);
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
                    re[part] = f64x2_add(re[part], f64x2_sub(f64x2_mul(cr, sr), f64x2_mul(ci, si)));
                    if !REAL {
                        im[part] =
                            f64x2_add(im[part], f64x2_add(f64x2_mul(cr, si), f64x2_mul(ci, sr)));
                    }
                    if ADVANCE {
                        let pr = v128_load(poles.re.as_ptr().add(j).cast());
                        let pi = v128_load(poles.im.as_ptr().add(j).cast());
                        v128_store(
                            states.re.as_mut_ptr().add(j).cast(),
                            f64x2_add(
                                f64x2_sub(f64x2_mul(pr, sr), f64x2_mul(pi, si)),
                                f64x2_splat(incoming.re),
                            ),
                        );
                        v128_store(
                            states.im.as_mut_ptr().add(j).cast(),
                            f64x2_add(
                                f64x2_add(f64x2_mul(pr, si), f64x2_mul(pi, sr)),
                                f64x2_splat(incoming.im),
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
    pub(super) unsafe fn seed<const REVERSE: bool>(
        poles: &Planes,
        states: &mut Planes,
        samples: &[Complex],
    ) {
        unsafe {
            let bulk = poles.len() / 2 * 2;
            for i in (0..bulk).step_by(2) {
                let pr = v128_load(poles.re.as_ptr().add(i).cast());
                let pi = v128_load(poles.im.as_ptr().add(i).cast());
                let mut re = f64x2_splat(0.);
                let mut im = f64x2_splat(0.);
                for j in 0..samples.len() {
                    let x = samples[if REVERSE { samples.len() - 1 - j } else { j }];
                    let next_re = f64x2_add(
                        f64x2_sub(f64x2_mul(pr, re), f64x2_mul(pi, im)),
                        f64x2_splat(x.re),
                    );
                    im = f64x2_add(
                        f64x2_add(f64x2_mul(pr, im), f64x2_mul(pi, re)),
                        f64x2_splat(x.im),
                    );
                    re = next_re;
                }
                v128_store(states.re.as_mut_ptr().add(i).cast(), re);
                v128_store(states.im.as_mut_ptr().add(i).cast(), im);
            }
            seed_scalar::<REVERSE>(poles, states, samples, bulk);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::tests::coarse_available as kernels;
    fn check_weight<const REAL: bool, const ADVANCE: bool>(
        kernel: Kernel,
        row: &[Complex],
        poles: &[Complex],
        initial: &[Complex],
        incoming: Complex,
    ) {
        let row_planes = Planes::from_complex(row);
        let poles_planes = Planes::from_complex(poles);
        let mut states = Planes::from_complex(initial);
        let expected = kernel.coarse_complex_dot(row, initial);
        let actual = kernel.coarse_weight_planes::<REAL, ADVANCE>(
            &row_planes,
            &poles_planes,
            &mut states,
            incoming,
        );
        assert_eq!(
            actual.re.to_bits(),
            expected.re.to_bits(),
            "{kernel:?} len{}",
            row.len()
        );
        if !REAL {
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        } else {
            assert_eq!(actual.im.to_bits(), 0f64.to_bits());
        }
        for (i, old) in initial.iter().enumerate() {
            let expected = if ADVANCE {
                poles[i] * *old + incoming
            } else {
                *old
            };
            assert_eq!(states.re[i].to_bits(), expected.re.to_bits());
            assert_eq!(states.im[i].to_bits(), expected.im.to_bits());
        }
    }
    #[test]
    fn plane_weight_and_update_keep_every_native_backend_exact() {
        for len in 0..1032 {
            let row: Vec<_> = (0..len)
                .map(|j| {
                    Complex::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let poles: Vec<_> = row.iter().map(|z| *z * Complex::new(0.8, 0.2)).collect();
            let initial: Vec<_> = row.iter().rev().copied().collect();
            let incoming = Complex::new(0.37, -0.29);
            for kernel in kernels() {
                check_weight::<false, false>(kernel, &row, &poles, &initial, incoming);
                check_weight::<false, true>(kernel, &row, &poles, &initial, incoming);
                check_weight::<true, false>(kernel, &row, &poles, &initial, incoming);
                check_weight::<true, true>(kernel, &row, &poles, &initial, incoming);
            }
        }
    }
    #[test]
    fn plane_seeds_keep_every_native_backend_exact_in_both_directions() {
        let poles: Vec<_> = (0..199)
            .map(|i| Complex::new(0.5 + (i % 7) as f64 / 17., ((i * 7 % 19) as f64 - 9.) / 29.))
            .collect();
        let samples: Vec<_> = (0..73)
            .map(|i| {
                Complex::new(
                    ((i * 23 % 53) as f64 - 26.) / 59.,
                    ((i * 13 % 71) as f64 - 35.) / 79.,
                )
            })
            .collect();
        for len in 0..199 {
            let p = Planes::from_complex(&poles[..len]);
            for kernel in kernels() {
                for reverse in [false, true] {
                    let mut expected = vec![Complex::default(); len];
                    let mut actual = Planes::default();
                    actual.resize(len, Complex::default());
                    if reverse {
                        kernel.coarse_seed::<true>(&poles[..len], &mut expected, &samples);
                        kernel.coarse_seed_planes::<true>(&p, &mut actual, &samples);
                    } else {
                        kernel.coarse_seed::<false>(&poles[..len], &mut expected, &samples);
                        kernel.coarse_seed_planes::<false>(&p, &mut actual, &samples);
                    }
                    for (i, z) in expected.iter().enumerate() {
                        assert_eq!(actual.re[i].to_bits(), z.re.to_bits());
                        assert_eq!(actual.im[i].to_bits(), z.im.to_bits());
                    }
                }
            }
        }
    }
}
