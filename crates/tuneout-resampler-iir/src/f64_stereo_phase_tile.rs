//! Exact rational phase tiles with adjacent channels in each packed row.
//! Every row is [left period0, right period0, left period1, right period1,...].
//! This shares coefficient broadcasts between channels without adding lookahead.
pub(crate) fn periods() -> usize {
    crate::f64_phase_tile::cached_periods()
}
pub(crate) fn run(
    periods: usize,
    x: &[f64],
    offsets: [usize; 4],
    c: [&[f64]; 4],
) -> [[f64; 32]; 4] {
    assert!(periods > 0 && periods <= 16);
    let lanes = periods * 2;
    let taps = c[0].len();
    for j in 0..4 {
        assert_eq!(c[j].len(), taps);
        assert!(offsets[j]
            .checked_add(taps)
            .and_then(|n| n.checked_mul(lanes))
            .is_some_and(|n| n <= x.len()));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if periods == 16
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("fma")
        {
            return unsafe { x86::avx512(x, offsets, c) };
        }
        if periods == 4
            && std::is_x86_feature_detected!("avx")
            && std::is_x86_feature_detected!("fma")
        {
            return unsafe { x86::avx(x, offsets, c) };
        }
        if periods == 2 && std::is_x86_feature_detected!("sse2") {
            return unsafe { x86::sse2(x, offsets, c) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if periods == 2 && std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::run(x, offsets, c) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if periods == 2 {
        return unsafe { wasm::run(x, offsets, c) };
    }
    scalar(periods, x, offsets, c)
}
fn scalar(periods: usize, x: &[f64], offsets: [usize; 4], c: [&[f64]; 4]) -> [[f64; 32]; 4] {
    let lanes = 2 * periods;
    let mut out = [[0.; 32]; 4];
    for j in 0..4 {
        for (tap, &w) in c[j].iter().enumerate() {
            for lane in 0..lanes {
                out[j][lane] += x[(offsets[j] + tap) * lanes + lane] * w;
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
    macro_rules! pair {
        ($name:ident,$features:literal,$n:expr,$zero:ident,$load:ident,$dup:ident,$madd:ident,$store:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(
                x: &[f64],
                offsets: [usize; 4],
                c: [&[f64]; 4],
            ) -> [[f64; 32]; 4] {
                unsafe {
                    let mut a = [$zero(); 4];
                    let mut b = a;
                    for tap in 0..c[0].len() {
                        for j in 0..4 {
                            let w = $dup(c[j][tap]);
                            let at = (offsets[j] + tap) * ($n * 2);
                            a[j] = $madd($load(x.as_ptr().add(at)), w, a[j]);
                            b[j] = $madd($load(x.as_ptr().add(at + $n)), w, b[j]);
                        }
                    }
                    let mut out = [[0.; 32]; 4];
                    for j in 0..4 {
                        $store(out[j].as_mut_ptr(), a[j]);
                        $store(out[j].as_mut_ptr().add($n), b[j]);
                    }
                    out
                }
            }
        };
    }
    pair!(
        avx,
        "avx,fma",
        4,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_set1_pd,
        _mm256_fmadd_pd,
        _mm256_storeu_pd
    );
    pair!(
        sse2,
        "sse2",
        2,
        _mm_setzero_pd,
        _mm_loadu_pd,
        _mm_set1_pd,
        sse_madd,
        _mm_storeu_pd
    );
    #[target_feature(enable = "avx512f,fma")]
    pub(super) unsafe fn avx512(x: &[f64], offsets: [usize; 4], c: [&[f64]; 4]) -> [[f64; 32]; 4] {
        unsafe {
            let x0 = x.as_ptr().add(offsets[0] * 32);
            let x1 = x.as_ptr().add(offsets[1] * 32);
            let x2 = x.as_ptr().add(offsets[2] * 32);
            let x3 = x.as_ptr().add(offsets[3] * 32);
            let c0 = c[0].as_ptr();
            let c1 = c[1].as_ptr();
            let c2 = c[2].as_ptr();
            let c3 = c[3].as_ptr();
            let zero = _mm512_setzero_pd();
            let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
            let (mut b0, mut b1, mut b2, mut b3) = (zero, zero, zero, zero);
            let (mut d0, mut d1, mut d2, mut d3) = (zero, zero, zero, zero);
            let (mut e0, mut e1, mut e2, mut e3) = (zero, zero, zero, zero);
            macro_rules! phase {
                ($a:ident,$b:ident,$d:ident,$e:ident,$x:ident,$c:ident,$tap:expr) => {{
                    let w = _mm512_set1_pd(*$c.add($tap));
                    $a = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 32)), w, $a);
                    $b = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 32 + 8)), w, $b);
                    $d = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 32 + 16)), w, $d);
                    $e = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 32 + 24)), w, $e);
                }};
            }
            for tap in 0..c[0].len() {
                phase!(a0, b0, d0, e0, x0, c0, tap);
                phase!(a1, b1, d1, e1, x1, c1, tap);
                phase!(a2, b2, d2, e2, x2, c2, tap);
                phase!(a3, b3, d3, e3, x3, c3, tap);
            }
            let mut out = [[0.; 32]; 4];
            macro_rules! store {
                ($j:expr,$a:ident,$b:ident,$d:ident,$e:ident) => {{
                    let p = out[$j].as_mut_ptr();
                    _mm512_storeu_pd(p, $a);
                    _mm512_storeu_pd(p.add(8), $b);
                    _mm512_storeu_pd(p.add(16), $d);
                    _mm512_storeu_pd(p.add(24), $e);
                }};
            }
            store!(0, a0, b0, d0, e0);
            store!(1, a1, b1, d1, e1);
            store!(2, a2, b2, d2, e2);
            store!(3, a3, b3, d3, e3);
            out
        }
    }
    #[target_feature(enable = "avx512f,fma")]
    pub(super) unsafe fn avx512_into(
        x: &[f64],
        offsets: [usize; 4],
        c: [&[f64]; 4],
        outputs: [&mut [f64]; 2],
        stride: usize,
        first: usize,
    ) {
        unsafe {
            let x0 = x.as_ptr().add(offsets[0] * 32);
            let x1 = x.as_ptr().add(offsets[1] * 32);
            let x2 = x.as_ptr().add(offsets[2] * 32);
            let x3 = x.as_ptr().add(offsets[3] * 32);
            let c0 = c[0].as_ptr();
            let c1 = c[1].as_ptr();
            let c2 = c[2].as_ptr();
            let c3 = c[3].as_ptr();
            let zero = _mm512_setzero_pd();
            let (mut a0, mut a1, mut a2, mut a3) = (zero, zero, zero, zero);
            let (mut b0, mut b1, mut b2, mut b3) = (zero, zero, zero, zero);
            let (mut d0, mut d1, mut d2, mut d3) = (zero, zero, zero, zero);
            let (mut e0, mut e1, mut e2, mut e3) = (zero, zero, zero, zero);
            macro_rules! phase {
                ($a:ident,$b:ident,$d:ident,$e:ident,$x:ident,$c:ident,$tap:expr) => {{
                    let w = _mm512_set1_pd(*$c.add($tap));
                    $a = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 32)), w, $a);
                    $b = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 32 + 8)), w, $b);
                    $d = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 32 + 16)), w, $d);
                    $e = _mm512_fmadd_pd(_mm512_loadu_pd($x.add($tap * 32 + 24)), w, $e);
                }};
            }
            for tap in 0..c[0].len() {
                phase!(a0, b0, d0, e0, x0, c0, tap);
                phase!(a1, b1, d1, e1, x1, c1, tap);
                phase!(a2, b2, d2, e2, x2, c2, tap);
                phase!(a3, b3, d3, e3, x3, c3, tap);
            }
            // Each four-phase by two-period block is transposed in256-bit
            // registers. Every store is four adjacent phases in one channel.
            let left = outputs[0].as_mut_ptr();
            let right = outputs[1].as_mut_ptr();
            macro_rules! pair_store {
                ($a:expr,$b:expr,$c:expr,$d:expr,$period:expr) => {{
                    let a = $a;
                    let b = $b;
                    let c = $c;
                    let d = $d;
                    let ab_l = _mm256_unpacklo_pd(a, b);
                    let cd_l = _mm256_unpacklo_pd(c, d);
                    let ab_r = _mm256_unpackhi_pd(a, b);
                    let cd_r = _mm256_unpackhi_pd(c, d);
                    let p = $period;
                    let at0 = p * stride + first;
                    let at1 = at0 + stride;
                    _mm256_storeu_pd(left.add(at0), _mm256_permute2f128_pd::<0x20>(ab_l, cd_l));
                    _mm256_storeu_pd(left.add(at1), _mm256_permute2f128_pd::<0x31>(ab_l, cd_l));
                    _mm256_storeu_pd(right.add(at0), _mm256_permute2f128_pd::<0x20>(ab_r, cd_r));
                    _mm256_storeu_pd(right.add(at1), _mm256_permute2f128_pd::<0x31>(ab_r, cd_r));
                }};
            }
            macro_rules! group_store {
                ($a:ident,$b:ident,$c:ident,$d:ident,$period:expr) => {{
                    pair_store!(
                        _mm512_castpd512_pd256($a),
                        _mm512_castpd512_pd256($b),
                        _mm512_castpd512_pd256($c),
                        _mm512_castpd512_pd256($d),
                        $period
                    );
                    pair_store!(
                        _mm512_extractf64x4_pd::<1>($a),
                        _mm512_extractf64x4_pd::<1>($b),
                        _mm512_extractf64x4_pd::<1>($c),
                        _mm512_extractf64x4_pd::<1>($d),
                        $period + 2
                    );
                }};
            }
            group_store!(a0, a1, a2, a3, 0);
            group_store!(b0, b1, b2, b3, 4);
            group_store!(d0, d1, d2, d3, 8);
            group_store!(e0, e1, e2, e3, 12);
        }
    }
}
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)] // Phase/tap indices name fixed SIMD lanes.
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run(x: &[f64], offsets: [usize; 4], c: [&[f64]; 4]) -> [[f64; 32]; 4] {
        unsafe {
            let mut a = [vdupq_n_f64(0.); 4];
            let mut b = a;
            for tap in 0..c[0].len() {
                for j in 0..4 {
                    let at = (offsets[j] + tap) * 4;
                    a[j] = vfmaq_n_f64(a[j], vld1q_f64(x.as_ptr().add(at)), c[j][tap]);
                    b[j] = vfmaq_n_f64(b[j], vld1q_f64(x.as_ptr().add(at + 2)), c[j][tap]);
                }
            }
            let mut out = [[0.; 32]; 4];
            for j in 0..4 {
                vst1q_f64(out[j].as_mut_ptr(), a[j]);
                vst1q_f64(out[j].as_mut_ptr().add(2), b[j]);
            }
            out
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::needless_range_loop)] // Phase/tap indices name fixed SIMD lanes.
mod wasm {
    use std::arch::wasm32::*;
    pub(super) unsafe fn run(x: &[f64], offsets: [usize; 4], c: [&[f64]; 4]) -> [[f64; 32]; 4] {
        unsafe {
            let mut a = [f64x2_splat(0.); 4];
            let mut b = a;
            for tap in 0..c[0].len() {
                for j in 0..4 {
                    let at = (offsets[j] + tap) * 4;
                    let w = f64x2_splat(c[j][tap]);
                    a[j] = f64x2_add(a[j], f64x2_mul(v128_load(x.as_ptr().add(at).cast()), w));
                    b[j] = f64x2_add(b[j], f64x2_mul(v128_load(x.as_ptr().add(at + 2).cast()), w));
                }
            }
            let mut out = [[0.; 32]; 4];
            for j in 0..4 {
                v128_store(out[j].as_mut_ptr().cast(), a[j]);
                v128_store(out[j].as_mut_ptr().add(2).cast(), b[j]);
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
    fn stereo_tiles_match_dd_reference_and_channel_impulses() {
        for periods in [1, 2, 4, 8, 16] {
            for taps in 0..260 {
                let lanes = 2 * periods;
                let offsets = [0, 1, 5, 9];
                let x: Vec<_> = (0..(taps + 9) * lanes)
                    .map(|i| ((i * 31 % 71) as f64 - 35.) / 31.)
                    .collect();
                let cs: [Vec<f64>; 4] = std::array::from_fn(|j| {
                    (0..taps)
                        .map(|i| ((i * 11 + j * 3) % 97) as f64 / 97. - 0.5)
                        .collect()
                });
                let c = cs.each_ref().map(|v| &v[..]);
                let out = run(periods, &x, offsets, c);
                for j in 0..4 {
                    for lane in 0..lanes {
                        let expected = (0..taps).fold(Wide::from(0.), |s, i| {
                            s + Wide::product(x[(offsets[j] + i) * lanes + lane], c[j][i])
                        });
                        assert!((Wide::from(out[j][lane]) - expected).value().abs() < 1e-12);
                    }
                }
            }
        }
        for periods in [1, 2, 4, 8, 16] {
            for channel in 0..2 {
                for at in [0, 1, 63, 64, 128] {
                    let lanes = 2 * periods;
                    let mut x = vec![0.; 129 * lanes];
                    for period in 0..periods {
                        x[at * lanes + 2 * period + channel] = 1.;
                    }
                    let c: Vec<_> = (0..129).map(|i| i as f64 / 131.).collect();
                    let out = run(periods, &x, [0; 4], [&c; 4]);
                    for row in out {
                        for (lane, &value) in row.iter().enumerate().take(lanes) {
                            assert_eq!(value, if lane % 2 == channel { c[at] } else { 0. });
                        }
                    }
                }
            }
        }
    }
}

/// Cache-line blocks consume eight adjacent input frames before opening the
/// next four-period stripe. Every packed row segment is one64-byte line.
pub(crate) fn pack(
    windows: [&[f64]; 2],
    start: usize,
    used: usize,
    periods: usize,
    stride: usize,
    packed: &mut [f64],
) {
    assert!(periods > 0);
    let lanes = periods.checked_mul(2).expect("admitted stereo lanes");
    let extent = used
        .checked_add((periods - 1).checked_mul(stride).expect("period extent"))
        .expect("packed input extent");
    let end = start.checked_add(extent).expect("packed input right edge");
    assert!(end <= windows[0].len() && end <= windows[1].len());
    assert!(used.checked_mul(lanes).is_some_and(|n| n <= packed.len()));
    // SAFETY: row<used,period<periods. The greatest input index is below
    // start+extent and the greatest destination is below used*lanes; those
    // complete extents were checked above before creating raw pointers.
    unsafe {
        let x = windows[0].as_ptr().add(start);
        let y = windows[1].as_ptr().add(start);
        let out = packed.as_mut_ptr();
        for r0 in (0..used).step_by(8) {
            for p0 in (0..periods).step_by(4) {
                for row in r0..(r0 + 8).min(used) {
                    for period in p0..(p0 + 4).min(periods) {
                        let at = row + period * stride;
                        let to = row * lanes + period * 2;
                        *out.add(to) = *x.add(at);
                        *out.add(to + 1) = *y.add(at);
                    }
                }
            }
        }
    }
}
#[cfg(test)]
mod packing_tests {
    use super::*;
    #[test]
    fn blocked_pack_matches_simple_matrix_and_preserves_unused_scratch() {
        for periods in [1, 2, 3, 4, 8, 16, 32] {
            for used in 0..35 {
                for stride in [0, 1, 5, 17] {
                    let n = used + (periods - 1) * stride + 3;
                    let x: Vec<_> = (0..n).map(|i| i as f64 + 0.5).collect();
                    let y: Vec<_> = x.iter().map(|x| -x).collect();
                    let mut out = vec![123.; used * periods * 2 + 7];
                    pack([&x, &y], 3, used, periods, stride, &mut out);
                    for row in 0..used {
                        for period in 0..periods {
                            let at = row * periods * 2 + period * 2;
                            let expected = 3 + row + period * stride;
                            assert_eq!(out[at], x[expected]);
                            assert_eq!(out[at + 1], y[expected]);
                        }
                    }
                    assert!(out[used * periods * 2..].iter().all(|&v| v == 123.));
                }
            }
        }
    }
}

/// Write the four independent phase rows directly into the two caches.
/// The checked extents are sufficient for every private raw SIMD access.
#[allow(clippy::too_many_arguments)] // Explicit source and destination geometry.
pub(crate) fn run_into(
    periods: usize,
    x: &[f64],
    offsets: [usize; 4],
    c: [&[f64]; 4],
    outputs: [&mut [f64]; 2],
    stride: usize,
    first: usize,
    count: usize,
) {
    assert!(count <= 4 && first.checked_add(count).is_some_and(|end| end <= stride));
    assert!(periods
        .checked_mul(stride)
        .is_some_and(|n| outputs.iter().all(|out| n <= out.len())));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if periods == 16
        && count == 4
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("fma")
    {
        let taps = c[0].len();
        for j in 0..4 {
            assert_eq!(c[j].len(), taps);
            assert!(offsets[j]
                .checked_add(taps)
                .and_then(|n| n.checked_mul(32))
                .is_some_and(|n| n <= x.len()));
        }
        // SAFETY: features are checked; all coefficient/input extents above
        // cover exactly16 periods, and every four-phase output fits stride.
        unsafe { x86::avx512_into(x, offsets, c, outputs, stride, first) };
        return;
    }
    let result = run(periods, x, offsets, c);
    for j in 0..count {
        for period in 0..periods {
            for ch in 0..2 {
                outputs[ch][period * stride + first + j] = result[j][period * 2 + ch];
            }
        }
    }
}
#[cfg(test)]
mod direct_store_tests {
    use super::*;
    #[test]
    fn direct_phase_cache_stores_preserve_every_native_fma_bit_and_sentinels() {
        for periods in [1, 2, 4, 8, 16] {
            for taps in [0, 1, 2, 3, 7, 8, 9, 31, 32, 33, 127, 128, 129, 321, 1397] {
                let lanes = periods * 2;
                let offsets = [0, 1, 5, 9];
                let x: Vec<_> = (0..(taps + 9) * lanes)
                    .map(|j| ((j * 29 % 73) as f64 - 36.) / 79.)
                    .collect();
                let coefficients: [Vec<f64>; 4] = std::array::from_fn(|j| {
                    (0..taps)
                        .map(|i| ((i * 31 + j * 7) % 97) as f64 / 101. - 0.5)
                        .collect()
                });
                let c = coefficients.each_ref().map(|r| &r[..]);
                let expected = run(periods, &x, offsets, c);
                let stride = 23;
                for first in [0, 5, 19] {
                    for count in [1, 2, 3, 4] {
                        let mut left = vec![123.; periods * stride + 7];
                        let mut right = left.clone();
                        run_into(
                            periods,
                            &x,
                            offsets,
                            c,
                            [&mut left, &mut right],
                            stride,
                            first,
                            count,
                        );
                        for (ch, out) in [left, right].iter().enumerate() {
                            for (i, &actual) in out.iter().enumerate() {
                                let p = i / stride;
                                let j = i % stride;
                                let value = if p < periods && j >= first && j < first + count {
                                    expected[j - first][p * 2 + ch]
                                } else {
                                    123.
                                };
                                assert_eq!(actual.to_bits(),value.to_bits(),"periods{periods},taps{taps},first{first},count{count},ch{ch},i{i}");
                            }
                        }
                    }
                }
            }
        }
    }
}
