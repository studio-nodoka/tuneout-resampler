//! Generic bounded phase tiles for real output projection of coarse samples.
//! Signal support and clocks are unchanged; scalar fallback is always valid.
use crate::complex::{Wide as W, WideComplex as WC};
pub(super) const WIDTH: usize = 8;
const CAP: usize = 8_388_608;
#[derive(Clone, Copy, Default)]
struct Entry {
    row: usize,
    lane: usize,
}
type Kernel = fn(&[f64], [&[f64]; 4], usize, &mut [f64; 16]);
pub(super) struct Plan {
    period: u64,
    complex: bool,
    taps: usize,
    rows: Vec<f64>,
    entries: Vec<Entry>,
    kernel: Kernel,
}
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r
    }
    a
}
impl Plan {
    // Keep the checked clock geometry explicit at this private setup boundary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        input: u32,
        output: u32,
        r: usize,
        weights: &[f64],
        period: u64,
        half: usize,
        carrier_period: u64,
        mut carrier: impl FnMut(u64) -> WC,
    ) -> Option<Self> {
        if weights.is_empty() || period == 1 || !(2 * half).is_multiple_of(4) {
            return None;
        }
        let complex = input < output;
        let taps = 2 * half;
        let joint = period
            .checked_div(gcd(period, carrier_period))?
            .checked_mul(carrier_period)?;
        if joint > 65_536 {
            return None;
        }
        let budget = CAP.checked_sub(weights.len().checked_mul(8)?)?;
        let metadata = usize::try_from(joint)
            .ok()?
            .checked_mul(std::mem::size_of::<Entry>())?;
        if metadata >= budget {
            return None;
        }
        let denominator = u128::from(output) * (r as u128);
        let step = u128::from(input);
        let planes = if complex { 2 } else { 1 };
        let mut first = 0u64;
        let mut length = 0usize;
        while first < joint {
            let center = u128::from(first) * step / denominator;
            let stop = ((center + 1) * denominator)
                .div_ceil(step)
                .min(u128::from(joint)) as u64;
            let tiles = usize::try_from(stop - first).ok()?.div_ceil(WIDTH);
            length = length.checked_add(tiles.checked_mul(taps * planes * WIDTH)?)?;
            first = stop;
        }
        if length.checked_mul(8)?.checked_add(metadata)? > budget {
            return None;
        }
        let mut entries = vec![Entry::default(); joint as usize];
        let mut rows = Vec::with_capacity(length);
        first = 0;
        while first < joint {
            let center = u128::from(first) * step / denominator;
            let stop = ((center + 1) * denominator)
                .div_ceil(step)
                .min(u128::from(joint)) as u64;
            for tile in (first..stop).step_by(WIDTH) {
                let count = (stop - tile).min(WIDTH as u64) as usize;
                let row = rows.len();
                let next = row.checked_add(taps.checked_mul(planes * WIDTH)?)?;
                if next.checked_mul(8)?.checked_add(metadata)? > budget {
                    return None;
                }
                rows.resize(next, 0.);
                for lane in 0..count {
                    let out = tile + lane as u64;
                    let phase = (out % period) as usize;
                    let rotation = carrier(out);
                    let re = rotation.re.value();
                    let im = rotation.im.value();
                    for j in 0..taps {
                        let weight = weights[phase * taps + j];
                        rows[row + j * planes * WIDTH + lane] = weight * re;
                        if complex {
                            rows[row + (j * planes + 1) * WIDTH + lane] = -weight * im;
                        }
                    }
                    entries[out as usize] = Entry { row, lane };
                }
            }
            first = stop;
        }
        Some(Self {
            period: joint,
            complex,
            taps,
            rows,
            entries,
            kernel: select(complex),
        })
    }
    pub(super) fn bytes(&self) -> usize {
        self.rows.capacity() * 8 + self.entries.capacity() * std::mem::size_of::<Entry>()
    }
    pub(super) fn write_stereo(&self, first: u64, input: [&[f64]; 4], values: &mut [W]) {
        assert_eq!(values.len() % 2, 0);
        assert_eq!(input[0].len(), self.taps);
        assert_eq!(input[2].len(), self.taps);
        if self.complex {
            assert_eq!(input[1].len(), self.taps);
            assert_eq!(input[3].len(), self.taps);
        }
        let frames = values.len() / 2;
        let taps = input[0].len();
        let planes = if self.complex { 2 } else { 1 };
        let mut done = 0;
        while done < frames {
            let entry = self.entries[((first + done as u64) % self.period) as usize];
            let count = (frames - done).min(WIDTH - entry.lane);
            let mut samples = [0.; 16];
            (self.kernel)(
                &self.rows[entry.row..entry.row + taps * planes * WIDTH],
                input,
                taps,
                &mut samples,
            );
            for lane in 0..count {
                values[2 * (done + lane)] = W::from(samples[entry.lane + lane]);
                values[2 * (done + lane) + 1] = W::from(samples[8 + entry.lane + lane]);
            }
            done += count;
        }
    }
    pub(super) fn write_mono(&self, first: u64, real: &[f64], imaginary: &[f64], values: &mut [W]) {
        assert_eq!(real.len(), self.taps);
        if self.complex {
            assert_eq!(imaginary.len(), self.taps);
        }
        let taps = real.len();
        let planes = if self.complex { 2 } else { 1 };
        let input = [real, imaginary, real, imaginary];
        let mut done = 0;
        while done < values.len() {
            let entry = self.entries[((first + done as u64) % self.period) as usize];
            let count = (values.len() - done).min(WIDTH - entry.lane);
            let mut samples = [0.; 16];
            (self.kernel)(
                &self.rows[entry.row..entry.row + taps * planes * WIDTH],
                input,
                taps,
                &mut samples,
            );
            for lane in 0..count {
                values[done + lane] = W::from(samples[entry.lane + lane]);
            }
            done += count;
        }
    }
}
fn scalar<const COMPLEX: bool>(
    weights: &[f64],
    input: [&[f64]; 4],
    taps: usize,
    output: &mut [f64; 16],
) {
    let planes = if COMPLEX { 2 } else { 1 };
    let mut a = [[0.; WIDTH]; 4];
    let mut b = a;
    for j in 0..taps {
        let slot = j % 4;
        for lane in 0..WIDTH {
            let w = weights[j * planes * WIDTH + lane];
            a[slot][lane] += w * input[0][j];
            b[slot][lane] += w * input[2][j];
            if COMPLEX {
                let w = weights[(j * planes + 1) * WIDTH + lane];
                a[slot][lane] += w * input[1][j];
                b[slot][lane] += w * input[3][j];
            }
        }
    }
    for lane in 0..WIDTH {
        output[lane] = (a[0][lane] + a[1][lane]) + (a[2][lane] + a[3][lane]);
        output[8 + lane] = (b[0][lane] + b[1][lane]) + (b[2][lane] + b[3][lane]);
    }
}
fn select(complex: bool) -> Kernel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx") {
            if WIDTH == 8 && std::is_x86_feature_detected!("avx512f") {
                return if complex {
                    checked512::<true>
                } else {
                    checked512::<false>
                };
            }
            return if complex {
                checked256::<true>
            } else {
                checked256::<false>
            };
        }
        if std::is_x86_feature_detected!("sse2") {
            return if complex {
                checked128::<true>
            } else {
                checked128::<false>
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return if complex {
            checked_neon::<true>
        } else {
            checked_neon::<false>
        };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return if complex {
        checked_wasm::<true>
    } else {
        checked_wasm::<false>
    };
    #[allow(unreachable_code)]
    if complex {
        scalar::<true>
    } else {
        scalar::<false>
    }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn checked128<const COMPLEX: bool>(
    weights: &[f64],
    input: [&[f64]; 4],
    taps: usize,
    output: &mut [f64; 16],
) {
    unsafe { x86::sse2::<COMPLEX>(weights, input, taps, output) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn checked256<const COMPLEX: bool>(
    weights: &[f64],
    input: [&[f64]; 4],
    taps: usize,
    output: &mut [f64; 16],
) {
    unsafe { x86::avx::<COMPLEX>(weights, input, taps, output) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn checked512<const COMPLEX: bool>(
    weights: &[f64],
    input: [&[f64]; 4],
    taps: usize,
    output: &mut [f64; 16],
) {
    unsafe { x86::avx512::<COMPLEX>(weights, input, taps, output) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn sse_madd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
        unsafe { _mm_add_pd(_mm_mul_pd(a, b), c) }
    }
    macro_rules! implementation {
        ($name:ident,$features:literal,$lanes:expr,$zero:ident,$load:ident,$store:ident,$splat:ident,$fma:ident,$add:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name<const COMPLEX: bool>(
                weights: &[f64],
                input: [&[f64]; 4],
                taps: usize,
                output: &mut [f64; 16],
            ) {
                unsafe {
                    let planes = if COMPLEX { 2 } else { 1 };
                    for column in (0..WIDTH).step_by($lanes) {
                        debug_assert!(taps.is_multiple_of(4));
                        let (mut a0, mut a1, mut a2, mut a3) = ($zero(), $zero(), $zero(), $zero());
                        let (mut b0, mut b1, mut b2, mut b3) = ($zero(), $zero(), $zero(), $zero());
                        for base in (0..taps).step_by(4) {
                            {
                                let j = base + 0;
                                let w = $load(weights.as_ptr().add(j * planes * WIDTH + column));
                                a0 = $fma(w, $splat(input[0][j]), a0);
                                b0 = $fma(w, $splat(input[2][j]), b0);
                                if COMPLEX {
                                    let w = $load(
                                        weights.as_ptr().add((j * planes + 1) * WIDTH + column),
                                    );
                                    a0 = $fma(w, $splat(input[1][j]), a0);
                                    b0 = $fma(w, $splat(input[3][j]), b0);
                                }
                            }
                            {
                                let j = base + 1;
                                let w = $load(weights.as_ptr().add(j * planes * WIDTH + column));
                                a1 = $fma(w, $splat(input[0][j]), a1);
                                b1 = $fma(w, $splat(input[2][j]), b1);
                                if COMPLEX {
                                    let w = $load(
                                        weights.as_ptr().add((j * planes + 1) * WIDTH + column),
                                    );
                                    a1 = $fma(w, $splat(input[1][j]), a1);
                                    b1 = $fma(w, $splat(input[3][j]), b1);
                                }
                            }
                            {
                                let j = base + 2;
                                let w = $load(weights.as_ptr().add(j * planes * WIDTH + column));
                                a2 = $fma(w, $splat(input[0][j]), a2);
                                b2 = $fma(w, $splat(input[2][j]), b2);
                                if COMPLEX {
                                    let w = $load(
                                        weights.as_ptr().add((j * planes + 1) * WIDTH + column),
                                    );
                                    a2 = $fma(w, $splat(input[1][j]), a2);
                                    b2 = $fma(w, $splat(input[3][j]), b2);
                                }
                            }
                            {
                                let j = base + 3;
                                let w = $load(weights.as_ptr().add(j * planes * WIDTH + column));
                                a3 = $fma(w, $splat(input[0][j]), a3);
                                b3 = $fma(w, $splat(input[2][j]), b3);
                                if COMPLEX {
                                    let w = $load(
                                        weights.as_ptr().add((j * planes + 1) * WIDTH + column),
                                    );
                                    a3 = $fma(w, $splat(input[1][j]), a3);
                                    b3 = $fma(w, $splat(input[3][j]), b3);
                                }
                            }
                        }
                        $store(
                            output.as_mut_ptr().add(column),
                            $add($add(a0, a1), $add(a2, a3)),
                        );
                        $store(
                            output.as_mut_ptr().add(8 + column),
                            $add($add(b0, b1), $add(b2, b3)),
                        );
                    }
                }
            }
        };
    }
    implementation!(
        sse2,
        "sse2",
        2,
        _mm_setzero_pd,
        _mm_loadu_pd,
        _mm_storeu_pd,
        _mm_set1_pd,
        sse_madd,
        _mm_add_pd
    );
    implementation!(
        avx,
        "avx,fma",
        4,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_set1_pd,
        _mm256_fmadd_pd,
        _mm256_add_pd
    );
    implementation!(
        avx512,
        "avx512f,fma",
        8,
        _mm512_setzero_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_set1_pd,
        _mm512_fmadd_pd,
        _mm512_add_pd
    );
}

#[cfg(target_arch = "aarch64")]
fn checked_neon<const COMPLEX: bool>(
    weights: &[f64],
    input: [&[f64]; 4],
    taps: usize,
    output: &mut [f64; 16],
) {
    // Selected only after runtime NEON detection or compile-time wasm SIMD.
    unsafe { neon::run::<COMPLEX>(weights, input, taps, output) }
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run<const COMPLEX: bool>(
        weights: &[f64],
        input: [&[f64]; 4],
        taps: usize,
        output: &mut [f64; 16],
    ) {
        unsafe {
            debug_assert!(taps.is_multiple_of(4));
            let planes = if COMPLEX { 2 } else { 1 };
            for column in (0..WIDTH).step_by(2) {
                let (mut a0, mut a1, mut a2, mut a3) = (
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                );
                let (mut b0, mut b1, mut b2, mut b3) = (
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                    vdupq_n_f64(0.),
                );
                for base in (0..taps).step_by(4) {
                    {
                        let j = base;
                        let w = vld1q_f64(weights.as_ptr().add(j * planes * WIDTH + column));
                        a0 = vfmaq_f64(a0, w, vdupq_n_f64(input[0][j]));
                        b0 = vfmaq_f64(b0, w, vdupq_n_f64(input[2][j]));
                        if COMPLEX {
                            let w =
                                vld1q_f64(weights.as_ptr().add((j * planes + 1) * WIDTH + column));
                            a0 = vfmaq_f64(a0, w, vdupq_n_f64(input[1][j]));
                            b0 = vfmaq_f64(b0, w, vdupq_n_f64(input[3][j]));
                        }
                    }
                    {
                        let j = base + 1;
                        let w = vld1q_f64(weights.as_ptr().add(j * planes * WIDTH + column));
                        a1 = vfmaq_f64(a1, w, vdupq_n_f64(input[0][j]));
                        b1 = vfmaq_f64(b1, w, vdupq_n_f64(input[2][j]));
                        if COMPLEX {
                            let w =
                                vld1q_f64(weights.as_ptr().add((j * planes + 1) * WIDTH + column));
                            a1 = vfmaq_f64(a1, w, vdupq_n_f64(input[1][j]));
                            b1 = vfmaq_f64(b1, w, vdupq_n_f64(input[3][j]));
                        }
                    }
                    {
                        let j = base + 2;
                        let w = vld1q_f64(weights.as_ptr().add(j * planes * WIDTH + column));
                        a2 = vfmaq_f64(a2, w, vdupq_n_f64(input[0][j]));
                        b2 = vfmaq_f64(b2, w, vdupq_n_f64(input[2][j]));
                        if COMPLEX {
                            let w =
                                vld1q_f64(weights.as_ptr().add((j * planes + 1) * WIDTH + column));
                            a2 = vfmaq_f64(a2, w, vdupq_n_f64(input[1][j]));
                            b2 = vfmaq_f64(b2, w, vdupq_n_f64(input[3][j]));
                        }
                    }
                    {
                        let j = base + 3;
                        let w = vld1q_f64(weights.as_ptr().add(j * planes * WIDTH + column));
                        a3 = vfmaq_f64(a3, w, vdupq_n_f64(input[0][j]));
                        b3 = vfmaq_f64(b3, w, vdupq_n_f64(input[2][j]));
                        if COMPLEX {
                            let w =
                                vld1q_f64(weights.as_ptr().add((j * planes + 1) * WIDTH + column));
                            a3 = vfmaq_f64(a3, w, vdupq_n_f64(input[1][j]));
                            b3 = vfmaq_f64(b3, w, vdupq_n_f64(input[3][j]));
                        }
                    }
                }
                vst1q_f64(
                    output.as_mut_ptr().add(column),
                    vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3)),
                );
                vst1q_f64(
                    output.as_mut_ptr().add(8 + column),
                    vaddq_f64(vaddq_f64(b0, b1), vaddq_f64(b2, b3)),
                );
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn checked_wasm<const COMPLEX: bool>(
    weights: &[f64],
    input: [&[f64]; 4],
    taps: usize,
    output: &mut [f64; 16],
) {
    // Selected only after runtime NEON detection or compile-time wasm SIMD.
    unsafe { wasm::run::<COMPLEX>(weights, input, taps, output) }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::*;
    use std::arch::wasm32::*;
    pub(super) unsafe fn run<const COMPLEX: bool>(
        weights: &[f64],
        input: [&[f64]; 4],
        taps: usize,
        output: &mut [f64; 16],
    ) {
        unsafe {
            debug_assert!(taps.is_multiple_of(4));
            let planes = if COMPLEX { 2 } else { 1 };
            for column in (0..WIDTH).step_by(2) {
                let (mut a0, mut a1, mut a2, mut a3) = (
                    f64x2_splat(0.),
                    f64x2_splat(0.),
                    f64x2_splat(0.),
                    f64x2_splat(0.),
                );
                let (mut b0, mut b1, mut b2, mut b3) = (
                    f64x2_splat(0.),
                    f64x2_splat(0.),
                    f64x2_splat(0.),
                    f64x2_splat(0.),
                );
                for base in (0..taps).step_by(4) {
                    {
                        let j = base;
                        let w = v128_load(weights.as_ptr().add(j * planes * WIDTH + column).cast());
                        a0 = f64x2_add(a0, f64x2_mul(w, f64x2_splat(input[0][j])));
                        b0 = f64x2_add(b0, f64x2_mul(w, f64x2_splat(input[2][j])));
                        if COMPLEX {
                            let w = v128_load(
                                weights
                                    .as_ptr()
                                    .add((j * planes + 1) * WIDTH + column)
                                    .cast(),
                            );
                            a0 = f64x2_add(a0, f64x2_mul(w, f64x2_splat(input[1][j])));
                            b0 = f64x2_add(b0, f64x2_mul(w, f64x2_splat(input[3][j])));
                        }
                    }
                    {
                        let j = base + 1;
                        let w = v128_load(weights.as_ptr().add(j * planes * WIDTH + column).cast());
                        a1 = f64x2_add(a1, f64x2_mul(w, f64x2_splat(input[0][j])));
                        b1 = f64x2_add(b1, f64x2_mul(w, f64x2_splat(input[2][j])));
                        if COMPLEX {
                            let w = v128_load(
                                weights
                                    .as_ptr()
                                    .add((j * planes + 1) * WIDTH + column)
                                    .cast(),
                            );
                            a1 = f64x2_add(a1, f64x2_mul(w, f64x2_splat(input[1][j])));
                            b1 = f64x2_add(b1, f64x2_mul(w, f64x2_splat(input[3][j])));
                        }
                    }
                    {
                        let j = base + 2;
                        let w = v128_load(weights.as_ptr().add(j * planes * WIDTH + column).cast());
                        a2 = f64x2_add(a2, f64x2_mul(w, f64x2_splat(input[0][j])));
                        b2 = f64x2_add(b2, f64x2_mul(w, f64x2_splat(input[2][j])));
                        if COMPLEX {
                            let w = v128_load(
                                weights
                                    .as_ptr()
                                    .add((j * planes + 1) * WIDTH + column)
                                    .cast(),
                            );
                            a2 = f64x2_add(a2, f64x2_mul(w, f64x2_splat(input[1][j])));
                            b2 = f64x2_add(b2, f64x2_mul(w, f64x2_splat(input[3][j])));
                        }
                    }
                    {
                        let j = base + 3;
                        let w = v128_load(weights.as_ptr().add(j * planes * WIDTH + column).cast());
                        a3 = f64x2_add(a3, f64x2_mul(w, f64x2_splat(input[0][j])));
                        b3 = f64x2_add(b3, f64x2_mul(w, f64x2_splat(input[2][j])));
                        if COMPLEX {
                            let w = v128_load(
                                weights
                                    .as_ptr()
                                    .add((j * planes + 1) * WIDTH + column)
                                    .cast(),
                            );
                            a3 = f64x2_add(a3, f64x2_mul(w, f64x2_splat(input[1][j])));
                            b3 = f64x2_add(b3, f64x2_mul(w, f64x2_splat(input[3][j])));
                        }
                    }
                }
                v128_store(
                    output.as_mut_ptr().add(column).cast(),
                    f64x2_add(f64x2_add(a0, a1), f64x2_add(a2, a3)),
                );
                v128_store(
                    output.as_mut_ptr().add(8 + column).cast(),
                    f64x2_add(f64x2_add(b0, b1), f64x2_add(b2, b3)),
                );
            }
        }
    }
}

#[cfg(test)]
mod portable_kernel_tests {
    use super::*;
    fn oracle<const COMPLEX: bool, const FUSED: bool>(
        weights: &[f64],
        input: [&[f64]; 4],
        taps: usize,
    ) -> [f64; 16] {
        let mut result = [0.; 16];
        let planes = if COMPLEX { 2 } else { 1 };
        for channel in 0..2 {
            for lane in 0..WIDTH {
                let mut sums = [0.; 4];
                for j in 0..taps {
                    let slot = j % 4;
                    let w = weights[j * planes * WIDTH + lane];
                    sums[slot] = if FUSED {
                        w.mul_add(input[2 * channel][j], sums[slot])
                    } else {
                        sums[slot] + w * input[2 * channel][j]
                    };
                    if COMPLEX {
                        let w = weights[(j * planes + 1) * WIDTH + lane];
                        sums[slot] = if FUSED {
                            w.mul_add(input[2 * channel + 1][j], sums[slot])
                        } else {
                            sums[slot] + w * input[2 * channel + 1][j]
                        };
                    }
                }
                result[8 * channel + lane] = (sums[0] + sums[1]) + (sums[2] + sums[3]);
            }
        }
        result
    }
    fn check<const COMPLEX: bool>(kernel: Kernel, fused: bool, taps: usize) {
        let planes = if COMPLEX { 2 } else { 1 };
        let weights: Vec<_> = (0..taps * planes * WIDTH)
            .map(|i| ((i * 37 % 131) as f64 - 65.) / 137.)
            .collect();
        let inputs: [Vec<f64>; 4] = std::array::from_fn(|channel| {
            (0..taps)
                .map(|i| ((i * 31 + channel * 7) % 127) as f64 / 59. - 1.)
                .collect()
        });
        let input = inputs.each_ref().map(|v| v.as_slice());
        let mut actual = [0.; 16];
        kernel(&weights, input, taps, &mut actual);
        let expected = if fused {
            oracle::<COMPLEX, true>(&weights, input, taps)
        } else {
            oracle::<COMPLEX, false>(&weights, input, taps)
        };
        for i in 0..16 {
            assert_eq!(
                actual[i].to_bits(),
                expected[i].to_bits(),
                "complex{COMPLEX} fused{fused} taps{taps} lane{i}"
            );
        }
    }
    #[test]
    fn phase_kernels_match_independent_reduction_and_rounding() {
        for taps in [4, 8, 12, 48, 80] {
            check::<false>(scalar::<false>, false, taps);
            check::<true>(scalar::<true>, false, taps);
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("sse2") {
                    check::<false>(checked128::<false>, false, taps);
                    check::<true>(checked128::<true>, false, taps);
                }
                if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                    check::<false>(checked256::<false>, true, taps);
                    check::<true>(checked256::<true>, true, taps);
                    if std::is_x86_feature_detected!("avx512f") {
                        check::<false>(checked512::<false>, true, taps);
                        check::<true>(checked512::<true>, true, taps);
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                check::<false>(checked_neon::<false>, true, taps);
                check::<true>(checked_neon::<true>, true, taps);
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                check::<false>(checked_wasm::<false>, false, taps);
                check::<true>(checked_wasm::<true>, false, taps);
            }
        }
    }
}
