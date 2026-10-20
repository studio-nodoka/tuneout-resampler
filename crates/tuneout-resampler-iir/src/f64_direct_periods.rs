//! Repeated periods are contiguous when the reduced input step M is one.
//! Load each channel plane directly, avoiding the transposed sample matrix.
pub(crate) fn run(
    periods: usize,
    phases: usize,
    x: &[f64],
    y: &[f64],
    c: [&[f64]; 4],
) -> [[[f64; 16]; 2]; 4] {
    assert!(periods > 0 && periods <= 16);
    assert!((1..=3).contains(&phases));
    let taps = c[0].len();
    assert!(taps
        .checked_add(periods - 1)
        .is_some_and(|n| n <= x.len() && n <= y.len()));
    for row in c.iter().take(phases) {
        assert_eq!(row.len(), taps);
    }
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    macro_rules! select {
        ($module:ident::$run:ident) => {
            match phases {
                1 => unsafe { $module::$run::<1>(x, y, c) },
                2 => unsafe { $module::$run::<2>(x, y, c) },
                _ => unsafe { $module::$run::<3>(x, y, c) },
            }
        };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if periods == 16
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("fma")
        {
            return select!(x86::avx512);
        }
        if periods == 4
            && std::is_x86_feature_detected!("avx")
            && std::is_x86_feature_detected!("fma")
        {
            return select!(x86::avx);
        }
        if periods == 2 && std::is_x86_feature_detected!("sse2") {
            return select!(x86::sse2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if periods == 2 && std::arch::is_aarch64_feature_detected!("neon") {
        return select!(neon::run);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if periods == 2 {
        return select!(wasm::run);
    }
    scalar(periods, phases, x, y, c)
}
fn scalar(
    periods: usize,
    phases: usize,
    x: &[f64],
    y: &[f64],
    c: [&[f64]; 4],
) -> [[[f64; 16]; 2]; 4] {
    let mut out = [[[0.; 16]; 2]; 4];
    for j in 0..phases {
        for (tap, &w) in c[j].iter().enumerate() {
            for period in 0..periods {
                out[j][0][period] += x[tap + period] * w;
                out[j][1][period] += y[tap + period] * w;
            }
        }
    }
    out
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
        ($name:ident,$features:literal,$n:expr,$vectors:expr,$zero:ident,$load:ident,$dup:ident,$madd:ident,$add:ident,$store:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name<const PHASES: usize>(
                x: &[f64],
                y: &[f64],
                c: [&[f64]; 4],
            ) -> [[[f64; 16]; 2]; 4] {
                unsafe {
                    let mut a = [[[$zero(); $vectors]; 2]; PHASES];
                    let mut b = a;
                    let taps = c[0].len();
                    let bulk = taps / 2 * 2;
                    for tap in (0..bulk).step_by(2) {
                        for j in 0..PHASES {
                            let wa = $dup(*c[j].as_ptr().add(tap));
                            let wb = $dup(*c[j].as_ptr().add(tap + 1));
                            for half in 0..$vectors {
                                let at = tap + half * $n;
                                a[j][0][half] = $madd($load(x.as_ptr().add(at)), wa, a[j][0][half]);
                                a[j][1][half] = $madd($load(y.as_ptr().add(at)), wa, a[j][1][half]);
                                b[j][0][half] =
                                    $madd($load(x.as_ptr().add(at + 1)), wb, b[j][0][half]);
                                b[j][1][half] =
                                    $madd($load(y.as_ptr().add(at + 1)), wb, b[j][1][half]);
                            }
                        }
                    }
                    if bulk < taps {
                        for j in 0..PHASES {
                            let w = $dup(*c[j].as_ptr().add(bulk));
                            for half in 0..$vectors {
                                let at = bulk + half * $n;
                                a[j][0][half] = $madd($load(x.as_ptr().add(at)), w, a[j][0][half]);
                                a[j][1][half] = $madd($load(y.as_ptr().add(at)), w, a[j][1][half]);
                            }
                        }
                    }
                    let mut out = [[[0.; 16]; 2]; 4];
                    for j in 0..PHASES {
                        for ch in 0..2 {
                            for half in 0..$vectors {
                                $store(
                                    out[j][ch].as_mut_ptr().add(half * $n),
                                    $add(a[j][ch][half], b[j][ch][half]),
                                );
                            }
                        }
                    }
                    out
                }
            }
        };
    }
    implementation!(
        avx512,
        "avx512f,fma",
        8,
        2,
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
        1,
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
        1,
        _mm_setzero_pd,
        _mm_loadu_pd,
        _mm_set1_pd,
        sse_madd,
        _mm_add_pd,
        _mm_storeu_pd
    );
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run<const PHASES: usize>(
        x: &[f64],
        y: &[f64],
        c: [&[f64]; 4],
    ) -> [[[f64; 16]; 2]; 4] {
        unsafe {
            let mut a = [[vdupq_n_f64(0.); 2]; PHASES];
            let mut b = a;
            let taps = c[0].len();
            let bulk = taps / 2 * 2;
            for tap in (0..bulk).step_by(2) {
                for j in 0..PHASES {
                    a[j][0] = vfmaq_n_f64(a[j][0], vld1q_f64(x.as_ptr().add(tap)), c[j][tap]);
                    a[j][1] = vfmaq_n_f64(a[j][1], vld1q_f64(y.as_ptr().add(tap)), c[j][tap]);
                    b[j][0] =
                        vfmaq_n_f64(b[j][0], vld1q_f64(x.as_ptr().add(tap + 1)), c[j][tap + 1]);
                    b[j][1] =
                        vfmaq_n_f64(b[j][1], vld1q_f64(y.as_ptr().add(tap + 1)), c[j][tap + 1]);
                }
            }
            if bulk < taps {
                for j in 0..PHASES {
                    a[j][0] = vfmaq_n_f64(a[j][0], vld1q_f64(x.as_ptr().add(bulk)), c[j][bulk]);
                    a[j][1] = vfmaq_n_f64(a[j][1], vld1q_f64(y.as_ptr().add(bulk)), c[j][bulk]);
                }
            }
            let mut out = [[[0.; 16]; 2]; 4];
            for j in 0..PHASES {
                for ch in 0..2 {
                    vst1q_f64(out[j][ch].as_mut_ptr(), vaddq_f64(a[j][ch], b[j][ch]));
                }
            }
            out
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;
    pub(super) unsafe fn run<const PHASES: usize>(
        x: &[f64],
        y: &[f64],
        c: [&[f64]; 4],
    ) -> [[[f64; 16]; 2]; 4] {
        unsafe {
            let mut a = [[f64x2_splat(0.); 2]; PHASES];
            let mut b = a;
            let taps = c[0].len();
            let bulk = taps / 2 * 2;
            for tap in (0..bulk).step_by(2) {
                for j in 0..PHASES {
                    let wa = f64x2_splat(c[j][tap]);
                    let wb = f64x2_splat(c[j][tap + 1]);
                    a[j][0] = f64x2_add(
                        a[j][0],
                        f64x2_mul(v128_load(x.as_ptr().add(tap).cast()), wa),
                    );
                    a[j][1] = f64x2_add(
                        a[j][1],
                        f64x2_mul(v128_load(y.as_ptr().add(tap).cast()), wa),
                    );
                    b[j][0] = f64x2_add(
                        b[j][0],
                        f64x2_mul(v128_load(x.as_ptr().add(tap + 1).cast()), wb),
                    );
                    b[j][1] = f64x2_add(
                        b[j][1],
                        f64x2_mul(v128_load(y.as_ptr().add(tap + 1).cast()), wb),
                    );
                }
            }
            if bulk < taps {
                for j in 0..PHASES {
                    let w = f64x2_splat(c[j][bulk]);
                    a[j][0] = f64x2_add(
                        a[j][0],
                        f64x2_mul(v128_load(x.as_ptr().add(bulk).cast()), w),
                    );
                    a[j][1] = f64x2_add(
                        a[j][1],
                        f64x2_mul(v128_load(y.as_ptr().add(bulk).cast()), w),
                    );
                }
            }
            let mut out = [[[0.; 16]; 2]; 4];
            for j in 0..PHASES {
                for ch in 0..2 {
                    v128_store(
                        out[j][ch].as_mut_ptr().cast(),
                        f64x2_add(a[j][ch], b[j][ch]),
                    );
                }
            }
            out
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Wide;
    #[test]
    fn direct_periods_match_independent_dd_and_isolated_channels() {
        for periods in [1, 2, 4, 8, 16] {
            for phases in 1..=3 {
                for taps in 0..260 {
                    let x: Vec<_> = (0..taps + periods)
                        .map(|i| ((i * 31 % 71) as f64 - 35.) / 31.)
                        .collect();
                    let y: Vec<_> = x.iter().map(|x| -x * 0.7).collect();
                    let cs: [Vec<f64>; 4] = std::array::from_fn(|j| {
                        (0..taps)
                            .map(|i| ((i * 11 + j * 3) % 97) as f64 / 97. - 0.5)
                            .collect()
                    });
                    let c = cs.each_ref().map(|v| &v[..]);
                    let out = run(periods, phases, &x, &y, c);
                    for (j, phase) in out.iter().enumerate().take(phases) {
                        for (ch, output) in phase.iter().enumerate() {
                            let samples = if ch == 0 { &x } else { &y };
                            for (period, &value) in output.iter().enumerate().take(periods) {
                                let expected = (0..taps).fold(Wide::from(0.), |s, i| {
                                    s + Wide::product(samples[i + period], c[j][i])
                                });
                                assert!((Wide::from(value) - expected).value().abs() < 1e-12);
                            }
                        }
                    }
                }
            }
        }
        for periods in [1, 2, 4, 8, 16] {
            for phases in 1..=3 {
                for channel in 0..2 {
                    let mut x = vec![0.; 129 + periods];
                    let mut y = x.clone();
                    if channel == 0 {
                        x[64] = 1.
                    } else {
                        y[64] = 1.
                    }
                    let c: Vec<_> = (0..129).map(|i| i as f64 / 131.).collect();
                    let out = run(periods, phases, &x, &y, [&c; 4]);
                    for phase in out.iter().take(phases) {
                        for (ch, output) in phase.iter().enumerate() {
                            for (period, &value) in output.iter().enumerate().take(periods) {
                                assert_eq!(value, if ch == channel { c[64 - period] } else { 0. });
                            }
                        }
                    }
                }
            }
        }
    }
}
