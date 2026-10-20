//! Retain residue-weighted states across the complete real-input block.
//! This is the same material modal IIR with six multiplies per mode.
//! Modes remain independent lanes. Every product, addition and reduction keeps
//! the original unfused four-accumulator order; unsupported CPUs use the old path.
use super::*;
impl Kernel {
    pub(crate) fn coarse_resident_weighted_real_available(self, modes: usize) -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            return self.0 == Backend::Avx512 && modes > 0 && modes <= 32;
        }
        #[allow(unreachable_code)]
        {
            let _ = (self, modes);
            false
        }
    }
    pub(crate) fn coarse_resident_weighted_real<const REAL: bool, const REVERSE: bool>(
        self,
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: &[Complex],
        output: &mut [Complex],
    ) {
        assert!(self.coarse_resident_weighted_real_available(row.len()));
        assert_eq!(row.len(), poles.len());
        assert_eq!(row.len(), states.len());
        assert_eq!(incoming.len(), output.len() - usize::from(REVERSE));
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            match row.len().div_ceil(8) {
                1 => x86::run1::<REAL, REVERSE>(row, poles, states, incoming, output),
                2 => x86::run2::<REAL, REVERSE>(row, poles, states, incoming, output),
                3 => x86::run3::<REAL, REVERSE>(row, poles, states, incoming, output),
                4 => {
                    if row.len() == 32 {
                        x86::run4_full::<REAL, REVERSE>(row, poles, states, incoming, output)
                    } else {
                        x86::run4::<REAL, REVERSE>(row, poles, states, incoming, output)
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx512f,avx")]
    pub(super) unsafe fn run1<const REAL: bool, const REVERSE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: &[Complex],
        output: &mut [Complex],
    ) {
        unsafe {
            let remaining = row.len();
            let mask = ((1u16 << remaining) - 1) as u8;
            let low = remaining.min(4);
            let high = remaining.saturating_sub(4);
            let low_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if low > 3 { -1 } else { 0 },
                if low > 2 { -1 } else { 0 },
                if low > 1 { -1 } else { 0 },
                if low > 0 { -1 } else { 0 },
            ));
            let high_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if high > 3 { -1 } else { 0 },
                if high > 2 { -1 } else { 0 },
                if high > 1 { -1 } else { 0 },
                if high > 0 { -1 } else { 0 },
            ));
            let wr0 = _mm512_maskz_loadu_pd(mask, row.re.as_ptr().add(0));
            let wi0 = _mm512_maskz_loadu_pd(mask, row.im.as_ptr().add(0));
            let pr0 = _mm512_maskz_loadu_pd(mask, poles.re.as_ptr().add(0));
            let pi0 = _mm512_maskz_loadu_pd(mask, poles.im.as_ptr().add(0));
            let mut sr0 = _mm512_maskz_loadu_pd(mask, states.re.as_ptr().add(0));
            let mut si0 = _mm512_maskz_loadu_pd(mask, states.im.as_ptr().add(0));
            for step in 0..output.len() {
                let n = if REVERSE {
                    output.len() - 1 - step
                } else {
                    step
                };
                let mut real = _mm256_setzero_pd();
                let mut imag = _mm256_setzero_pd();
                {
                    let value = sr0;
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_castpd512_pd256(value)),
                        low_mask,
                    );
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value)),
                        high_mask,
                    );
                    if !REAL {
                        let value = si0;
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_castpd512_pd256(value)),
                            low_mask,
                        );
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value)),
                            high_mask,
                        );
                    }
                }
                let mut re = [0.; 4];
                let mut im = [0.; 4];
                _mm256_storeu_pd(re.as_mut_ptr(), real);
                if !REAL {
                    _mm256_storeu_pd(im.as_mut_ptr(), imag);
                }
                output[n] = Complex::new(
                    (re[0] + re[1]) + (re[2] + re[3]),
                    if REAL {
                        0.
                    } else {
                        (im[0] + im[1]) + (im[2] + im[3])
                    },
                );
                if !REVERSE || n != 0 {
                    let value = incoming[if REVERSE { n - 1 } else { n }];
                    let input_re = _mm512_set1_pd(value.re);
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr0, sr0), _mm512_mul_pd(pi0, si0)),
                        _mm512_mul_pd(wr0, input_re),
                    );
                    si0 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr0, si0), _mm512_mul_pd(pi0, sr0)),
                        _mm512_mul_pd(wi0, input_re),
                    );
                    sr0 = re;
                }
            }
            _mm512_mask_storeu_pd(states.re.as_mut_ptr().add(0), mask, sr0);
            _mm512_mask_storeu_pd(states.im.as_mut_ptr().add(0), mask, si0);
        }
    }
    #[target_feature(enable = "avx512f,avx")]
    pub(super) unsafe fn run2<const REAL: bool, const REVERSE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: &[Complex],
        output: &mut [Complex],
    ) {
        unsafe {
            let remaining = row.len() - 8;
            let mask = ((1u16 << remaining) - 1) as u8;
            let low = remaining.min(4);
            let high = remaining.saturating_sub(4);
            let low_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if low > 3 { -1 } else { 0 },
                if low > 2 { -1 } else { 0 },
                if low > 1 { -1 } else { 0 },
                if low > 0 { -1 } else { 0 },
            ));
            let high_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if high > 3 { -1 } else { 0 },
                if high > 2 { -1 } else { 0 },
                if high > 1 { -1 } else { 0 },
                if high > 0 { -1 } else { 0 },
            ));
            let wr0 = _mm512_loadu_pd(row.re.as_ptr().add(0));
            let wi0 = _mm512_loadu_pd(row.im.as_ptr().add(0));
            let pr0 = _mm512_loadu_pd(poles.re.as_ptr().add(0));
            let pi0 = _mm512_loadu_pd(poles.im.as_ptr().add(0));
            let mut sr0 = _mm512_loadu_pd(states.re.as_ptr().add(0));
            let mut si0 = _mm512_loadu_pd(states.im.as_ptr().add(0));
            let wr1 = _mm512_maskz_loadu_pd(mask, row.re.as_ptr().add(8));
            let wi1 = _mm512_maskz_loadu_pd(mask, row.im.as_ptr().add(8));
            let pr1 = _mm512_maskz_loadu_pd(mask, poles.re.as_ptr().add(8));
            let pi1 = _mm512_maskz_loadu_pd(mask, poles.im.as_ptr().add(8));
            let mut sr1 = _mm512_maskz_loadu_pd(mask, states.re.as_ptr().add(8));
            let mut si1 = _mm512_maskz_loadu_pd(mask, states.im.as_ptr().add(8));
            for step in 0..output.len() {
                let n = if REVERSE {
                    output.len() - 1 - step
                } else {
                    step
                };
                let mut real = _mm256_setzero_pd();
                let mut imag = _mm256_setzero_pd();
                {
                    let value = sr0;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si0;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr1;
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_castpd512_pd256(value)),
                        low_mask,
                    );
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value)),
                        high_mask,
                    );
                    if !REAL {
                        let value = si1;
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_castpd512_pd256(value)),
                            low_mask,
                        );
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value)),
                            high_mask,
                        );
                    }
                }
                let mut re = [0.; 4];
                let mut im = [0.; 4];
                _mm256_storeu_pd(re.as_mut_ptr(), real);
                if !REAL {
                    _mm256_storeu_pd(im.as_mut_ptr(), imag);
                }
                output[n] = Complex::new(
                    (re[0] + re[1]) + (re[2] + re[3]),
                    if REAL {
                        0.
                    } else {
                        (im[0] + im[1]) + (im[2] + im[3])
                    },
                );
                if !REVERSE || n != 0 {
                    let value = incoming[if REVERSE { n - 1 } else { n }];
                    let input_re = _mm512_set1_pd(value.re);
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr0, sr0), _mm512_mul_pd(pi0, si0)),
                        _mm512_mul_pd(wr0, input_re),
                    );
                    si0 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr0, si0), _mm512_mul_pd(pi0, sr0)),
                        _mm512_mul_pd(wi0, input_re),
                    );
                    sr0 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr1, sr1), _mm512_mul_pd(pi1, si1)),
                        _mm512_mul_pd(wr1, input_re),
                    );
                    si1 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr1, si1), _mm512_mul_pd(pi1, sr1)),
                        _mm512_mul_pd(wi1, input_re),
                    );
                    sr1 = re;
                }
            }
            _mm512_storeu_pd(states.re.as_mut_ptr().add(0), sr0);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(0), si0);
            _mm512_mask_storeu_pd(states.re.as_mut_ptr().add(8), mask, sr1);
            _mm512_mask_storeu_pd(states.im.as_mut_ptr().add(8), mask, si1);
        }
    }
    #[target_feature(enable = "avx512f,avx")]
    pub(super) unsafe fn run3<const REAL: bool, const REVERSE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: &[Complex],
        output: &mut [Complex],
    ) {
        unsafe {
            let remaining = row.len() - 16;
            let mask = ((1u16 << remaining) - 1) as u8;
            let low = remaining.min(4);
            let high = remaining.saturating_sub(4);
            let low_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if low > 3 { -1 } else { 0 },
                if low > 2 { -1 } else { 0 },
                if low > 1 { -1 } else { 0 },
                if low > 0 { -1 } else { 0 },
            ));
            let high_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if high > 3 { -1 } else { 0 },
                if high > 2 { -1 } else { 0 },
                if high > 1 { -1 } else { 0 },
                if high > 0 { -1 } else { 0 },
            ));
            let wr0 = _mm512_loadu_pd(row.re.as_ptr().add(0));
            let wi0 = _mm512_loadu_pd(row.im.as_ptr().add(0));
            let pr0 = _mm512_loadu_pd(poles.re.as_ptr().add(0));
            let pi0 = _mm512_loadu_pd(poles.im.as_ptr().add(0));
            let mut sr0 = _mm512_loadu_pd(states.re.as_ptr().add(0));
            let mut si0 = _mm512_loadu_pd(states.im.as_ptr().add(0));
            let wr1 = _mm512_loadu_pd(row.re.as_ptr().add(8));
            let wi1 = _mm512_loadu_pd(row.im.as_ptr().add(8));
            let pr1 = _mm512_loadu_pd(poles.re.as_ptr().add(8));
            let pi1 = _mm512_loadu_pd(poles.im.as_ptr().add(8));
            let mut sr1 = _mm512_loadu_pd(states.re.as_ptr().add(8));
            let mut si1 = _mm512_loadu_pd(states.im.as_ptr().add(8));
            let wr2 = _mm512_maskz_loadu_pd(mask, row.re.as_ptr().add(16));
            let wi2 = _mm512_maskz_loadu_pd(mask, row.im.as_ptr().add(16));
            let pr2 = _mm512_maskz_loadu_pd(mask, poles.re.as_ptr().add(16));
            let pi2 = _mm512_maskz_loadu_pd(mask, poles.im.as_ptr().add(16));
            let mut sr2 = _mm512_maskz_loadu_pd(mask, states.re.as_ptr().add(16));
            let mut si2 = _mm512_maskz_loadu_pd(mask, states.im.as_ptr().add(16));
            for step in 0..output.len() {
                let n = if REVERSE {
                    output.len() - 1 - step
                } else {
                    step
                };
                let mut real = _mm256_setzero_pd();
                let mut imag = _mm256_setzero_pd();
                {
                    let value = sr0;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si0;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr1;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si1;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr2;
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_castpd512_pd256(value)),
                        low_mask,
                    );
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value)),
                        high_mask,
                    );
                    if !REAL {
                        let value = si2;
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_castpd512_pd256(value)),
                            low_mask,
                        );
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value)),
                            high_mask,
                        );
                    }
                }
                let mut re = [0.; 4];
                let mut im = [0.; 4];
                _mm256_storeu_pd(re.as_mut_ptr(), real);
                if !REAL {
                    _mm256_storeu_pd(im.as_mut_ptr(), imag);
                }
                output[n] = Complex::new(
                    (re[0] + re[1]) + (re[2] + re[3]),
                    if REAL {
                        0.
                    } else {
                        (im[0] + im[1]) + (im[2] + im[3])
                    },
                );
                if !REVERSE || n != 0 {
                    let value = incoming[if REVERSE { n - 1 } else { n }];
                    let input_re = _mm512_set1_pd(value.re);
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr0, sr0), _mm512_mul_pd(pi0, si0)),
                        _mm512_mul_pd(wr0, input_re),
                    );
                    si0 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr0, si0), _mm512_mul_pd(pi0, sr0)),
                        _mm512_mul_pd(wi0, input_re),
                    );
                    sr0 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr1, sr1), _mm512_mul_pd(pi1, si1)),
                        _mm512_mul_pd(wr1, input_re),
                    );
                    si1 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr1, si1), _mm512_mul_pd(pi1, sr1)),
                        _mm512_mul_pd(wi1, input_re),
                    );
                    sr1 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr2, sr2), _mm512_mul_pd(pi2, si2)),
                        _mm512_mul_pd(wr2, input_re),
                    );
                    si2 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr2, si2), _mm512_mul_pd(pi2, sr2)),
                        _mm512_mul_pd(wi2, input_re),
                    );
                    sr2 = re;
                }
            }
            _mm512_storeu_pd(states.re.as_mut_ptr().add(0), sr0);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(0), si0);
            _mm512_storeu_pd(states.re.as_mut_ptr().add(8), sr1);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(8), si1);
            _mm512_mask_storeu_pd(states.re.as_mut_ptr().add(16), mask, sr2);
            _mm512_mask_storeu_pd(states.im.as_mut_ptr().add(16), mask, si2);
        }
    }
    #[target_feature(enable = "avx512f,avx")]
    pub(super) unsafe fn run4<const REAL: bool, const REVERSE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: &[Complex],
        output: &mut [Complex],
    ) {
        unsafe {
            let remaining = row.len() - 24;
            let mask = ((1u16 << remaining) - 1) as u8;
            let low = remaining.min(4);
            let high = remaining.saturating_sub(4);
            let low_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if low > 3 { -1 } else { 0 },
                if low > 2 { -1 } else { 0 },
                if low > 1 { -1 } else { 0 },
                if low > 0 { -1 } else { 0 },
            ));
            let high_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if high > 3 { -1 } else { 0 },
                if high > 2 { -1 } else { 0 },
                if high > 1 { -1 } else { 0 },
                if high > 0 { -1 } else { 0 },
            ));
            let wr0 = _mm512_loadu_pd(row.re.as_ptr().add(0));
            let wi0 = _mm512_loadu_pd(row.im.as_ptr().add(0));
            let pr0 = _mm512_loadu_pd(poles.re.as_ptr().add(0));
            let pi0 = _mm512_loadu_pd(poles.im.as_ptr().add(0));
            let mut sr0 = _mm512_loadu_pd(states.re.as_ptr().add(0));
            let mut si0 = _mm512_loadu_pd(states.im.as_ptr().add(0));
            let wr1 = _mm512_loadu_pd(row.re.as_ptr().add(8));
            let wi1 = _mm512_loadu_pd(row.im.as_ptr().add(8));
            let pr1 = _mm512_loadu_pd(poles.re.as_ptr().add(8));
            let pi1 = _mm512_loadu_pd(poles.im.as_ptr().add(8));
            let mut sr1 = _mm512_loadu_pd(states.re.as_ptr().add(8));
            let mut si1 = _mm512_loadu_pd(states.im.as_ptr().add(8));
            let wr2 = _mm512_loadu_pd(row.re.as_ptr().add(16));
            let wi2 = _mm512_loadu_pd(row.im.as_ptr().add(16));
            let pr2 = _mm512_loadu_pd(poles.re.as_ptr().add(16));
            let pi2 = _mm512_loadu_pd(poles.im.as_ptr().add(16));
            let mut sr2 = _mm512_loadu_pd(states.re.as_ptr().add(16));
            let mut si2 = _mm512_loadu_pd(states.im.as_ptr().add(16));
            let wr3 = _mm512_maskz_loadu_pd(mask, row.re.as_ptr().add(24));
            let wi3 = _mm512_maskz_loadu_pd(mask, row.im.as_ptr().add(24));
            let pr3 = _mm512_maskz_loadu_pd(mask, poles.re.as_ptr().add(24));
            let pi3 = _mm512_maskz_loadu_pd(mask, poles.im.as_ptr().add(24));
            let mut sr3 = _mm512_maskz_loadu_pd(mask, states.re.as_ptr().add(24));
            let mut si3 = _mm512_maskz_loadu_pd(mask, states.im.as_ptr().add(24));
            for step in 0..output.len() {
                let n = if REVERSE {
                    output.len() - 1 - step
                } else {
                    step
                };
                let mut real = _mm256_setzero_pd();
                let mut imag = _mm256_setzero_pd();
                {
                    let value = sr0;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si0;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr1;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si1;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr2;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si2;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr3;
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_castpd512_pd256(value)),
                        low_mask,
                    );
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value)),
                        high_mask,
                    );
                    if !REAL {
                        let value = si3;
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_castpd512_pd256(value)),
                            low_mask,
                        );
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value)),
                            high_mask,
                        );
                    }
                }
                let mut re = [0.; 4];
                let mut im = [0.; 4];
                _mm256_storeu_pd(re.as_mut_ptr(), real);
                if !REAL {
                    _mm256_storeu_pd(im.as_mut_ptr(), imag);
                }
                output[n] = Complex::new(
                    (re[0] + re[1]) + (re[2] + re[3]),
                    if REAL {
                        0.
                    } else {
                        (im[0] + im[1]) + (im[2] + im[3])
                    },
                );
                if !REVERSE || n != 0 {
                    let value = incoming[if REVERSE { n - 1 } else { n }];
                    let input_re = _mm512_set1_pd(value.re);
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr0, sr0), _mm512_mul_pd(pi0, si0)),
                        _mm512_mul_pd(wr0, input_re),
                    );
                    si0 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr0, si0), _mm512_mul_pd(pi0, sr0)),
                        _mm512_mul_pd(wi0, input_re),
                    );
                    sr0 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr1, sr1), _mm512_mul_pd(pi1, si1)),
                        _mm512_mul_pd(wr1, input_re),
                    );
                    si1 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr1, si1), _mm512_mul_pd(pi1, sr1)),
                        _mm512_mul_pd(wi1, input_re),
                    );
                    sr1 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr2, sr2), _mm512_mul_pd(pi2, si2)),
                        _mm512_mul_pd(wr2, input_re),
                    );
                    si2 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr2, si2), _mm512_mul_pd(pi2, sr2)),
                        _mm512_mul_pd(wi2, input_re),
                    );
                    sr2 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr3, sr3), _mm512_mul_pd(pi3, si3)),
                        _mm512_mul_pd(wr3, input_re),
                    );
                    si3 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr3, si3), _mm512_mul_pd(pi3, sr3)),
                        _mm512_mul_pd(wi3, input_re),
                    );
                    sr3 = re;
                }
            }
            _mm512_storeu_pd(states.re.as_mut_ptr().add(0), sr0);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(0), si0);
            _mm512_storeu_pd(states.re.as_mut_ptr().add(8), sr1);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(8), si1);
            _mm512_storeu_pd(states.re.as_mut_ptr().add(16), sr2);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(16), si2);
            _mm512_mask_storeu_pd(states.re.as_mut_ptr().add(24), mask, sr3);
            _mm512_mask_storeu_pd(states.im.as_mut_ptr().add(24), mask, si3);
        }
    }
    #[target_feature(enable = "avx512f,avx")]
    pub(super) unsafe fn run4_full<const REAL: bool, const REVERSE: bool>(
        row: &Planes,
        poles: &Planes,
        states: &mut Planes,
        incoming: &[Complex],
        output: &mut [Complex],
    ) {
        unsafe {
            let remaining = 8usize;
            let mask = ((1u16 << remaining) - 1) as u8;
            let low = remaining.min(4);
            let high = remaining.saturating_sub(4);
            let low_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if low > 3 { -1 } else { 0 },
                if low > 2 { -1 } else { 0 },
                if low > 1 { -1 } else { 0 },
                if low > 0 { -1 } else { 0 },
            ));
            let high_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                if high > 3 { -1 } else { 0 },
                if high > 2 { -1 } else { 0 },
                if high > 1 { -1 } else { 0 },
                if high > 0 { -1 } else { 0 },
            ));
            let wr0 = _mm512_loadu_pd(row.re.as_ptr().add(0));
            let wi0 = _mm512_loadu_pd(row.im.as_ptr().add(0));
            let pr0 = _mm512_loadu_pd(poles.re.as_ptr().add(0));
            let pi0 = _mm512_loadu_pd(poles.im.as_ptr().add(0));
            let mut sr0 = _mm512_loadu_pd(states.re.as_ptr().add(0));
            let mut si0 = _mm512_loadu_pd(states.im.as_ptr().add(0));
            let wr1 = _mm512_loadu_pd(row.re.as_ptr().add(8));
            let wi1 = _mm512_loadu_pd(row.im.as_ptr().add(8));
            let pr1 = _mm512_loadu_pd(poles.re.as_ptr().add(8));
            let pi1 = _mm512_loadu_pd(poles.im.as_ptr().add(8));
            let mut sr1 = _mm512_loadu_pd(states.re.as_ptr().add(8));
            let mut si1 = _mm512_loadu_pd(states.im.as_ptr().add(8));
            let wr2 = _mm512_loadu_pd(row.re.as_ptr().add(16));
            let wi2 = _mm512_loadu_pd(row.im.as_ptr().add(16));
            let pr2 = _mm512_loadu_pd(poles.re.as_ptr().add(16));
            let pi2 = _mm512_loadu_pd(poles.im.as_ptr().add(16));
            let mut sr2 = _mm512_loadu_pd(states.re.as_ptr().add(16));
            let mut si2 = _mm512_loadu_pd(states.im.as_ptr().add(16));
            let wr3 = _mm512_maskz_loadu_pd(mask, row.re.as_ptr().add(24));
            let wi3 = _mm512_maskz_loadu_pd(mask, row.im.as_ptr().add(24));
            let pr3 = _mm512_maskz_loadu_pd(mask, poles.re.as_ptr().add(24));
            let pi3 = _mm512_maskz_loadu_pd(mask, poles.im.as_ptr().add(24));
            let mut sr3 = _mm512_maskz_loadu_pd(mask, states.re.as_ptr().add(24));
            let mut si3 = _mm512_maskz_loadu_pd(mask, states.im.as_ptr().add(24));
            for step in 0..output.len() {
                let n = if REVERSE {
                    output.len() - 1 - step
                } else {
                    step
                };
                let mut real = _mm256_setzero_pd();
                let mut imag = _mm256_setzero_pd();
                {
                    let value = sr0;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si0;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr1;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si1;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr2;
                    real = _mm256_add_pd(real, _mm512_castpd512_pd256(value));
                    real = _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value));
                    if !REAL {
                        let value = si2;
                        imag = _mm256_add_pd(imag, _mm512_castpd512_pd256(value));
                        imag = _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value));
                    }
                }
                {
                    let value = sr3;
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_castpd512_pd256(value)),
                        low_mask,
                    );
                    real = _mm256_blendv_pd(
                        real,
                        _mm256_add_pd(real, _mm512_extractf64x4_pd::<1>(value)),
                        high_mask,
                    );
                    if !REAL {
                        let value = si3;
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_castpd512_pd256(value)),
                            low_mask,
                        );
                        imag = _mm256_blendv_pd(
                            imag,
                            _mm256_add_pd(imag, _mm512_extractf64x4_pd::<1>(value)),
                            high_mask,
                        );
                    }
                }
                let mut re = [0.; 4];
                let mut im = [0.; 4];
                _mm256_storeu_pd(re.as_mut_ptr(), real);
                if !REAL {
                    _mm256_storeu_pd(im.as_mut_ptr(), imag);
                }
                output[n] = Complex::new(
                    (re[0] + re[1]) + (re[2] + re[3]),
                    if REAL {
                        0.
                    } else {
                        (im[0] + im[1]) + (im[2] + im[3])
                    },
                );
                if !REVERSE || n != 0 {
                    let value = incoming[if REVERSE { n - 1 } else { n }];
                    let input_re = _mm512_set1_pd(value.re);
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr0, sr0), _mm512_mul_pd(pi0, si0)),
                        _mm512_mul_pd(wr0, input_re),
                    );
                    si0 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr0, si0), _mm512_mul_pd(pi0, sr0)),
                        _mm512_mul_pd(wi0, input_re),
                    );
                    sr0 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr1, sr1), _mm512_mul_pd(pi1, si1)),
                        _mm512_mul_pd(wr1, input_re),
                    );
                    si1 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr1, si1), _mm512_mul_pd(pi1, sr1)),
                        _mm512_mul_pd(wi1, input_re),
                    );
                    sr1 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr2, sr2), _mm512_mul_pd(pi2, si2)),
                        _mm512_mul_pd(wr2, input_re),
                    );
                    si2 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr2, si2), _mm512_mul_pd(pi2, sr2)),
                        _mm512_mul_pd(wi2, input_re),
                    );
                    sr2 = re;
                    let re = _mm512_add_pd(
                        _mm512_sub_pd(_mm512_mul_pd(pr3, sr3), _mm512_mul_pd(pi3, si3)),
                        _mm512_mul_pd(wr3, input_re),
                    );
                    si3 = _mm512_add_pd(
                        _mm512_add_pd(_mm512_mul_pd(pr3, si3), _mm512_mul_pd(pi3, sr3)),
                        _mm512_mul_pd(wi3, input_re),
                    );
                    sr3 = re;
                }
            }
            _mm512_storeu_pd(states.re.as_mut_ptr().add(0), sr0);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(0), si0);
            _mm512_storeu_pd(states.re.as_mut_ptr().add(8), sr1);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(8), si1);
            _mm512_storeu_pd(states.re.as_mut_ptr().add(16), sr2);
            _mm512_storeu_pd(states.im.as_mut_ptr().add(16), si2);
            _mm512_mask_storeu_pd(states.re.as_mut_ptr().add(24), mask, sr3);
            _mm512_mask_storeu_pd(states.im.as_mut_ptr().add(24), mask, si3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<const REAL: bool, const REVERSE: bool>(len: usize, count: usize) {
        let kernel = Kernel::coarse_detect();
        if !kernel.coarse_resident_weighted_real_available(len) {
            return;
        }
        let row: Vec<_> = (0..len)
            .map(|j| {
                Complex::new(
                    ((j * 37 % 127) as f64 - 63.) / 127.,
                    ((j * 41 % 131) as f64 - 65.) / 131.,
                )
            })
            .collect();
        let poles: Vec<_> = (0..len)
            .map(|j| {
                Complex::new(
                    0.6 + (j % 7) as f64 / 37.,
                    ((j * 11 % 71) as f64 - 35.) / 137.,
                )
            })
            .collect();
        let initial: Vec<_> = row.iter().rev().copied().collect();
        let input: Vec<_> = (0..count - usize::from(REVERSE))
            .map(|j| {
                Complex::new(
                    ((j * 17 % 127) as f64 - 63.) / 127.,
                    ((j * 19 % 131) as f64 - 65.) / 131.,
                )
            })
            .collect();
        let row = Planes::from_complex(&row);
        let poles = Planes::from_complex(&poles);
        let mut actual = Planes::from_complex(&initial);
        let mut expected = Planes::from_complex(&initial);
        let mut output = vec![Complex::default(); count];
        kernel.coarse_resident_weighted_real::<REAL, REVERSE>(
            &row,
            &poles,
            &mut actual,
            &input,
            &mut output,
        );
        for step in 0..count {
            let n = if REVERSE { count - 1 - step } else { step };
            let value = if REVERSE && n == 0 {
                kernel.coarse_weighted_real::<false>(&row, &poles, &mut expected, 0.)
            } else {
                kernel.coarse_weighted_real::<true>(
                    &row,
                    &poles,
                    &mut expected,
                    input[if REVERSE { n - 1 } else { n }].re,
                )
            };
            let value = Complex::new(value.re, if REAL { 0. } else { value.im });
            assert_eq!(
                output[n].re.to_bits(),
                value.re.to_bits(),
                "len{len} count{count} real{REAL} reverse{REVERSE} n{n}"
            );
            assert_eq!(output[n].im.to_bits(), value.im.to_bits());
        }
        for i in 0..len {
            assert_eq!(actual.re[i].to_bits(), expected.re[i].to_bits());
            assert_eq!(actual.im[i].to_bits(), expected.im[i].to_bits());
        }
    }
    #[test]
    fn resident_weighted_blocks_preserve_every_modal_output_and_final_state() {
        for len in 1..=32 {
            for count in [1, 2, 17, 512, 1024] {
                check::<false, false>(len, count);
                check::<false, true>(len, count);
                check::<true, false>(len, count);
                check::<true, true>(len, count);
            }
        }
    }
    #[test]
    fn unsupported_weighted_backends_and_mode_counts_keep_existing_path() {
        let kernel = Kernel(Backend::Scalar);
        for len in [0, 1, 31, 32, 33, 65] {
            assert!(!kernel.coarse_resident_weighted_real_available(len));
        }
        assert!(!Kernel::coarse_detect().coarse_resident_weighted_real_available(0));
        assert!(!Kernel::coarse_detect().coarse_resident_weighted_real_available(33));
    }
}
