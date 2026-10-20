//! Reuse exact rational-phase coefficients across independent output periods.
//! The packing and coefficient-broadcast layout follows production FIR. No
//! extra audio is requested: only complete periods in the caller's block run.
pub(crate) fn width() -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            return 8;
        }
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            return 4;
        }
        if std::is_x86_feature_detected!("sse2") {
            return 2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return 2;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return 2;
    #[allow(unreachable_code)]
    1
}
/// Every packed row contains `lanes` periods. Each of four outputs starts at
/// its row offset and shares a coefficient across those independent periods.
pub(crate) fn run(
    lanes: usize,
    packed: &[f64],
    offsets: [usize; 4],
    coeff: [&[f64]; 4],
) -> [[f64; 8]; 4] {
    let taps = coeff[0].len();
    assert!(lanes > 1 && lanes <= 8);
    for j in 0..4 {
        assert_eq!(coeff[j].len(), taps);
        assert!(offsets[j]
            .checked_add(taps)
            .and_then(|n| n.checked_mul(lanes))
            .is_some_and(|n| n <= packed.len()));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if lanes == 8
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("fma")
        {
            return unsafe { x86::avx512(packed, offsets, coeff) };
        }
        if lanes == 4
            && std::is_x86_feature_detected!("avx")
            && std::is_x86_feature_detected!("fma")
        {
            return unsafe { x86::avx(packed, offsets, coeff) };
        }
        if lanes == 2 && std::is_x86_feature_detected!("sse2") {
            return unsafe { x86::sse2(packed, offsets, coeff) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if lanes == 2 && std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::run(packed, offsets, coeff) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if lanes == 2 {
        return unsafe { wasm::run(packed, offsets, coeff) };
    }
    scalar(lanes, packed, offsets, coeff)
}
fn scalar(lanes: usize, x: &[f64], offsets: [usize; 4], c: [&[f64]; 4]) -> [[f64; 8]; 4] {
    let mut result = [[0.; 8]; 4];
    for j in 0..4 {
        for (tap, &w) in c[j].iter().enumerate() {
            for lane in 0..lanes {
                result[j][lane] += x[(offsets[j] + tap) * lanes + lane] * w;
            }
        }
    }
    result
}

/// Larger phase batches use future PCM already admitted by the IIR guard.
/// A separate cache lets a block finish without waiting for a whole batch.
pub(crate) fn cached_periods() -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        return 16;
    }
    width()
}
pub(crate) fn run_cached(
    lanes: usize,
    x: &[f64],
    offsets: [usize; 4],
    c: [&[f64]; 4],
) -> [[f64; 16]; 4] {
    if lanes <= 8 {
        let result = run(lanes, x, offsets, c);
        let mut output = [[0.; 16]; 4];
        for j in 0..4 {
            output[j][..lanes].copy_from_slice(&result[j][..lanes]);
        }
        return output;
    }
    assert_eq!(lanes, 16);
    let taps = c[0].len();
    for j in 0..4 {
        assert_eq!(c[j].len(), taps);
        assert!(offsets[j]
            .checked_add(taps)
            .and_then(|n| n.checked_mul(lanes))
            .is_some_and(|n| n <= x.len()));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        return unsafe { x86::cached16(x, offsets, c) };
    }
    let mut output = [[0.; 16]; 4];
    for j in 0..4 {
        for (tap, &w) in c[j].iter().enumerate() {
            for lane in 0..16 {
                output[j][lane] += x[(offsets[j] + tap) * 16 + lane] * w;
            }
        }
    }
    output
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
        ($name:ident,$features:literal,$n:expr,$zero:ident,$load:ident,$broadcast:ident,$madd:ident,$add:ident,$store:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(
                x: &[f64],
                offsets: [usize; 4],
                c: [&[f64]; 4],
            ) -> [[f64; 8]; 4] {
                unsafe {
                    let mut a = [$zero(); 4];
                    let mut b = [$zero(); 4];
                    let taps = c[0].len();
                    let bulk = taps / 2 * 2;
                    for tap in (0..bulk).step_by(2) {
                        for j in 0..4 {
                            a[j] = $madd(
                                $load(x.as_ptr().add((offsets[j] + tap) * $n)),
                                $broadcast(c[j][tap]),
                                a[j],
                            );
                            b[j] = $madd(
                                $load(x.as_ptr().add((offsets[j] + tap + 1) * $n)),
                                $broadcast(c[j][tap + 1]),
                                b[j],
                            );
                        }
                    }
                    if bulk < taps {
                        for j in 0..4 {
                            a[j] = $madd(
                                $load(x.as_ptr().add((offsets[j] + bulk) * $n)),
                                $broadcast(c[j][bulk]),
                                a[j],
                            );
                        }
                    }
                    let mut result = [[0.; 8]; 4];
                    for j in 0..4 {
                        $store(result[j].as_mut_ptr(), $add(a[j], b[j]));
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
        _mm512_set1_pd,
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
        _mm256_set1_pd,
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
        _mm_set1_pd,
        sse_madd,
        _mm_add_pd,
        _mm_storeu_pd
    );

    #[target_feature(enable = "avx512f,fma")]
    pub(super) unsafe fn cached16(
        x: &[f64],
        offsets: [usize; 4],
        c: [&[f64]; 4],
    ) -> [[f64; 16]; 4] {
        unsafe {
            let x0 = x.as_ptr().add(offsets[0] * 16);
            let x1 = x.as_ptr().add(offsets[1] * 16);
            let x2 = x.as_ptr().add(offsets[2] * 16);
            let x3 = x.as_ptr().add(offsets[3] * 16);
            let c0 = c[0].as_ptr();
            let c1 = c[1].as_ptr();
            let c2 = c[2].as_ptr();
            let c3 = c[3].as_ptr();
            let mut a0 = _mm512_setzero_pd();
            let mut a1 = a0;
            let mut a2 = a0;
            let mut a3 = a0;
            let mut b0 = a0;
            let mut b1 = a0;
            let mut b2 = a0;
            let mut b3 = a0;
            macro_rules! phase {
                ($a:ident,$b:ident,$x:ident,$c:ident,$tap:expr) => {{
                    let weight = _mm512_set1_pd(*$c.add($tap));
                    $a = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 16)), weight, $a);
                    $b = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 16 + 8)), weight, $b);
                }};
            }
            for tap in 0..c[0].len() {
                phase!(a0, b0, x0, c0, tap);
                phase!(a1, b1, x1, c1, tap);
                phase!(a2, b2, x2, c2, tap);
                phase!(a3, b3, x3, c3, tap);
            }
            let mut output = [[0.; 16]; 4];
            _mm512_storeu_pd(output[0].as_mut_ptr(), a0);
            _mm512_storeu_pd(output[0].as_mut_ptr().add(8), b0);
            _mm512_storeu_pd(output[1].as_mut_ptr(), a1);
            _mm512_storeu_pd(output[1].as_mut_ptr().add(8), b1);
            _mm512_storeu_pd(output[2].as_mut_ptr(), a2);
            _mm512_storeu_pd(output[2].as_mut_ptr().add(8), b2);
            _mm512_storeu_pd(output[3].as_mut_ptr(), a3);
            _mm512_storeu_pd(output[3].as_mut_ptr().add(8), b3);
            output
        }
    }
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run(x: &[f64], offsets: [usize; 4], c: [&[f64]; 4]) -> [[f64; 8]; 4] {
        unsafe {
            let mut a = [vdupq_n_f64(0.); 4];
            let mut b = a;
            let taps = c[0].len();
            let bulk = taps / 2 * 2;
            for tap in (0..bulk).step_by(2) {
                for j in 0..4 {
                    a[j] = vfmaq_n_f64(
                        a[j],
                        vld1q_f64(x.as_ptr().add((offsets[j] + tap) * 2)),
                        c[j][tap],
                    );
                    b[j] = vfmaq_n_f64(
                        b[j],
                        vld1q_f64(x.as_ptr().add((offsets[j] + tap + 1) * 2)),
                        c[j][tap + 1],
                    );
                }
            }
            if bulk < taps {
                for j in 0..4 {
                    a[j] = vfmaq_n_f64(
                        a[j],
                        vld1q_f64(x.as_ptr().add((offsets[j] + bulk) * 2)),
                        c[j][bulk],
                    );
                }
            }
            let mut result = [[0.; 8]; 4];
            for j in 0..4 {
                vst1q_f64(result[j].as_mut_ptr(), vaddq_f64(a[j], b[j]));
            }
            result
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;
    pub(super) unsafe fn run(x: &[f64], offsets: [usize; 4], c: [&[f64]; 4]) -> [[f64; 8]; 4] {
        unsafe {
            let mut a = [f64x2_splat(0.); 4];
            let mut b = a;
            let taps = c[0].len();
            let bulk = taps / 2 * 2;
            for tap in (0..bulk).step_by(2) {
                for j in 0..4 {
                    a[j] = f64x2_add(
                        a[j],
                        f64x2_mul(
                            v128_load(x.as_ptr().add((offsets[j] + tap) * 2).cast()),
                            f64x2_splat(c[j][tap]),
                        ),
                    );
                    b[j] = f64x2_add(
                        b[j],
                        f64x2_mul(
                            v128_load(x.as_ptr().add((offsets[j] + tap + 1) * 2).cast()),
                            f64x2_splat(c[j][tap + 1]),
                        ),
                    );
                }
            }
            if bulk < taps {
                for j in 0..4 {
                    a[j] = f64x2_add(
                        a[j],
                        f64x2_mul(
                            v128_load(x.as_ptr().add((offsets[j] + bulk) * 2).cast()),
                            f64x2_splat(c[j][bulk]),
                        ),
                    );
                }
            }
            let mut result = [[0.; 8]; 4];
            for j in 0..4 {
                v128_store(result[j].as_mut_ptr().cast(), f64x2_add(a[j], b[j]));
            }
            result
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Wide;
    #[test]
    fn packed_phase_dots_match_independent_reference_and_sparse_impulses() {
        for lanes in [2, 4, 8] {
            for taps in 0..260 {
                let offsets = [0, 1, 5, 9];
                let rows = taps + 9;
                let x: Vec<_> = (0..rows * lanes)
                    .map(|i| ((i * 31 % 71) as f64 - 35.) / 31.)
                    .collect();
                let cs: [Vec<f64>; 4] = std::array::from_fn(|j| {
                    (0..taps)
                        .map(|i| ((i * 11 + j * 3) % 97) as f64 / 97. - 0.5)
                        .collect()
                });
                let c = cs.each_ref().map(|v| &v[..]);
                let actual = run(lanes, &x, offsets, c);
                for j in 0..4 {
                    for lane in 0..lanes {
                        let expected = (0..taps).fold(Wide::from(0.), |s, i| {
                            s + Wide::product(x[(offsets[j] + i) * lanes + lane], c[j][i])
                        });
                        assert!(
                            (Wide::from(actual[j][lane]) - expected).value().abs() < 1e-12,
                            "lanes={lanes} taps={taps} j={j} lane={lane}"
                        );
                    }
                }
            }
        }
        for lanes in [2, 4, 8] {
            let taps = 129;
            for at in [0, 1, 63, 64, 128] {
                let mut x = vec![0.; taps * lanes];
                for lane in 0..lanes {
                    x[at * lanes + lane] = 1.;
                }
                let c: Vec<_> = (0..taps).map(|i| i as f64 / 131.).collect();
                let actual = run(lanes, &x, [0; 4], [&c; 4]);
                for values in actual {
                    for value in &values[..lanes] {
                        assert_eq!(*value, c[at]);
                    }
                }
            }
        }
    }

    #[test]
    fn cached_phase_dots_match_dd_reference_including_sparse_impulses() {
        for lanes in [2, 4, 8, 16] {
            for taps in 0..260 {
                let offsets = [0, 1, 5, 9];
                let rows = taps + 9;
                let x: Vec<_> = (0..rows * lanes)
                    .map(|i| ((i * 31 % 71) as f64 - 35.) / 31.)
                    .collect();
                let cs: [Vec<f64>; 4] = std::array::from_fn(|j| {
                    (0..taps)
                        .map(|i| ((i * 11 + j * 3) % 97) as f64 / 97. - 0.5)
                        .collect()
                });
                let c = cs.each_ref().map(|v| &v[..]);
                let actual = run_cached(lanes, &x, offsets, c);
                for j in 0..4 {
                    for lane in 0..lanes {
                        let expected = (0..taps).fold(Wide::from(0.), |s, i| {
                            s + Wide::product(x[(offsets[j] + i) * lanes + lane], c[j][i])
                        });
                        assert!(
                            (Wide::from(actual[j][lane]) - expected).value().abs() < 1e-12,
                            "lanes={lanes} taps={taps} row={j} lane={lane}"
                        );
                    }
                }
            }
        }
        for at in [0, 1, 63, 64, 128] {
            let mut x = vec![0.; 129 * 16];
            for lane in 0..16 {
                x[at * 16 + lane] = 1.;
            }
            let c: Vec<_> = (0..129).map(|i| i as f64 / 131.).collect();
            let actual = run_cached(16, &x, [0; 4], [&c; 4]);
            for values in actual {
                for value in values {
                    assert_eq!(value, c[at]);
                }
            }
        }
    }
}
