//! Four synthesis planes share coefficients without changing their reductions.
pub(crate) fn run(x: [&[f64]; 4], weights: &[f64]) -> [f64; 4] {
    assert!(x.iter().all(|v| v.len() == weights.len()));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
        return unsafe { x86::run(x, weights) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::run(x, weights) };
    }
    // Narrower register files retain the qualified paired implementation.
    let (a, b) = crate::f64_pairdot::shared_coefficients(x[0], x[1], weights);
    let (c, d) = crate::f64_pairdot::shared_coefficients(x[2], x[3], weights);
    [a, b, c, d]
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "avx512f,fma")]
    // Fixed transposed indices preserve the qualified accumulator order.
    #[allow(clippy::needless_range_loop)]
    pub(super) unsafe fn run(x: [&[f64]; 4], weights: &[f64]) -> [f64; 4] {
        unsafe {
            let mut accum = [[_mm512_setzero_pd(); 4]; 4];
            let bulk = weights.len() / 32 * 32;
            for base in (0..bulk).step_by(32) {
                for j in 0..4 {
                    let at = base + 8 * j;
                    let c = _mm512_loadu_pd(weights.as_ptr().add(at));
                    for plane in 0..4 {
                        accum[plane][j] = _mm512_fmadd_pd(
                            _mm512_loadu_pd(x[plane].as_ptr().add(at)),
                            c,
                            accum[plane][j],
                        );
                    }
                }
            }
            let vectors = weights.len() / 8 * 8;
            for at in (bulk..vectors).step_by(8) {
                let j = (at - bulk) / 8;
                let c = _mm512_loadu_pd(weights.as_ptr().add(at));
                for plane in 0..4 {
                    accum[plane][j] = _mm512_fmadd_pd(
                        _mm512_loadu_pd(x[plane].as_ptr().add(at)),
                        c,
                        accum[plane][j],
                    );
                }
            }
            let mut output = [0.; 4];
            for plane in 0..4 {
                let a = accum[plane];
                let total = _mm512_add_pd(_mm512_add_pd(a[0], a[1]), _mm512_add_pd(a[2], a[3]));
                let mut lanes = [0.; 8];
                _mm512_storeu_pd(lanes.as_mut_ptr(), total);
                output[plane] = lanes.into_iter().sum();
            }
            for i in vectors..weights.len() {
                for plane in 0..4 {
                    output[plane] += x[plane][i] * weights[i];
                }
            }
            output
        }
    }
}
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)] // Same transposed accumulator order as x86.
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run(x: [&[f64]; 4], weights: &[f64]) -> [f64; 4] {
        unsafe {
            let mut accum = [[vdupq_n_f64(0.); 4]; 4];
            let bulk = weights.len() / 8 * 8;
            for base in (0..bulk).step_by(8) {
                for j in 0..4 {
                    let at = base + 2 * j;
                    let c = vld1q_f64(weights.as_ptr().add(at));
                    for plane in 0..4 {
                        accum[plane][j] =
                            vfmaq_f64(accum[plane][j], vld1q_f64(x[plane].as_ptr().add(at)), c);
                    }
                }
            }
            let vectors = weights.len() / 2 * 2;
            for at in (bulk..vectors).step_by(2) {
                let j = (at - bulk) / 2;
                let c = vld1q_f64(weights.as_ptr().add(at));
                for plane in 0..4 {
                    accum[plane][j] =
                        vfmaq_f64(accum[plane][j], vld1q_f64(x[plane].as_ptr().add(at)), c);
                }
            }
            let mut output = [0.; 4];
            for plane in 0..4 {
                let a = accum[plane];
                let total = vaddq_f64(vaddq_f64(a[0], a[1]), vaddq_f64(a[2], a[3]));
                let mut lanes = [0.; 2];
                vst1q_f64(lanes.as_mut_ptr(), total);
                output[plane] = lanes[0] + lanes[1];
            }
            for i in vectors..weights.len() {
                for plane in 0..4 {
                    output[plane] += x[plane][i] * weights[i];
                }
            }
            output
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stereo_synthesis_matches_independent_paired_planes_for_all_lengths() {
        for n in 0..1032 {
            let planes: [Vec<f64>; 4] = std::array::from_fn(|p| {
                (0..n)
                    .map(|i| ((i * (31 + 2 * p) % 127) as f64 - 63.) / 127.)
                    .collect()
            });
            let x = std::array::from_fn(|p| &planes[p][..]);
            let weights: Vec<_> = (0..n)
                .map(|i| ((i * 41 % 131) as f64 - 65.) / 131.)
                .collect();
            let (a, b) = crate::f64_pairdot::shared_coefficients(x[0], x[1], &weights);
            let (c, d) = crate::f64_pairdot::shared_coefficients(x[2], x[3], &weights);
            let expected = [a, b, c, d];
            let actual = run(x, &weights);
            for (a, b) in actual.iter().zip(expected) {
                assert_eq!(a.to_bits(), b.to_bits(), "n{n}");
            }
        }
    }
}
