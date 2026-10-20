//! Two ordinary dots sharing one input plane or one coefficient row.
//! The accumulation/reduction order matches f64_dot on each backend.
#[cfg(test)]
pub(crate) fn shared_input(x: &[f64], a: &[f64], b: &[f64]) -> (f64, f64) {
    run::<true>(x, a, b)
}
pub(crate) fn shared_coefficients(x: &[f64], y: &[f64], c: &[f64]) -> (f64, f64) {
    run::<false>(x, y, c)
}
fn run<const INPUT: bool>(x: &[f64], y: &[f64], z: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::avx512::<INPUT>(x, y, z) };
        }
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            return unsafe { x86::avx::<INPUT>(x, y, z) };
        }
        if std::is_x86_feature_detected!("sse2") {
            return unsafe { x86::sse2::<INPUT>(x, y, z) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { neon::run::<INPUT>(x, y, z) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    return unsafe { wasm::run::<INPUT>(x, y, z) };
    #[allow(unreachable_code)]
    scalar::<INPUT>(x, y, z)
}
fn scalar<const INPUT: bool>(x: &[f64], y: &[f64], z: &[f64]) -> (f64, f64) {
    let bulk = x.len() / 4 * 4;
    let mut a = [0.; 4];
    let mut b = a;
    for i in (0..bulk).step_by(4) {
        for j in 0..4 {
            let at = i + j;
            if INPUT {
                a[j] += x[at] * y[at];
                b[j] += x[at] * z[at];
            } else {
                a[j] += x[at] * z[at];
                b[j] += y[at] * z[at];
            }
        }
    }
    let mut a = (a[0] + a[1]) + (a[2] + a[3]);
    let mut b = (b[0] + b[1]) + (b[2] + b[3]);
    for i in bulk..x.len() {
        if INPUT {
            a += x[i] * y[i];
            b += x[i] * z[i]
        } else {
            a += x[i] * z[i];
            b += y[i] * z[i]
        }
    }
    (a, b)
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
        ($name:ident,$features:literal,$lanes:expr,$zero:ident,$load:ident,$madd:ident,$add:ident,$store:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name<const INPUT: bool>(
                x: &[f64],
                y: &[f64],
                z: &[f64],
            ) -> (f64, f64) {
                unsafe {
                    let mut a = [$zero(); 4];
                    let mut b = a;
                    let bulk = x.len() / (4 * $lanes) * (4 * $lanes);
                    for i in (0..bulk).step_by(4 * $lanes) {
                        for j in 0..4 {
                            let at = i + j * $lanes;
                            let vx = $load(x.as_ptr().add(at));
                            let vy = $load(y.as_ptr().add(at));
                            let vz = $load(z.as_ptr().add(at));
                            if INPUT {
                                a[j] = $madd(vx, vy, a[j]);
                                b[j] = $madd(vx, vz, b[j]);
                            } else {
                                a[j] = $madd(vx, vz, a[j]);
                                b[j] = $madd(vy, vz, b[j]);
                            }
                        }
                    }
                    let vectors = x.len() / $lanes * $lanes;
                    for i in (bulk..vectors).step_by($lanes) {
                        let j = (i - bulk) / $lanes;
                        let vx = $load(x.as_ptr().add(i));
                        let vy = $load(y.as_ptr().add(i));
                        let vz = $load(z.as_ptr().add(i));
                        if INPUT {
                            a[j] = $madd(vx, vy, a[j]);
                            b[j] = $madd(vx, vz, b[j]);
                        } else {
                            a[j] = $madd(vx, vz, a[j]);
                            b[j] = $madd(vy, vz, b[j]);
                        }
                    }
                    let va = $add($add(a[0], a[1]), $add(a[2], a[3]));
                    let vb = $add($add(b[0], b[1]), $add(b[2], b[3]));
                    let mut la = [0.; $lanes];
                    let mut lb = la;
                    $store(la.as_mut_ptr(), va);
                    $store(lb.as_mut_ptr(), vb);
                    let mut a = la.into_iter().sum::<f64>();
                    let mut b = lb.into_iter().sum::<f64>();
                    for i in vectors..x.len() {
                        if INPUT {
                            a += x[i] * y[i];
                            b += x[i] * z[i]
                        } else {
                            a += x[i] * z[i];
                            b += y[i] * z[i]
                        }
                    }
                    (a, b)
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
        sse_madd,
        _mm_add_pd,
        _mm_storeu_pd
    );
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run<const INPUT: bool>(x: &[f64], y: &[f64], z: &[f64]) -> (f64, f64) {
        unsafe {
            let mut a = [vdupq_n_f64(0.); 4];
            let mut b = a;
            let bulk = x.len() / 8 * 8;
            for i in (0..bulk).step_by(8) {
                for j in 0..4 {
                    let at = i + 2 * j;
                    let vx = vld1q_f64(x.as_ptr().add(at));
                    let vy = vld1q_f64(y.as_ptr().add(at));
                    let vz = vld1q_f64(z.as_ptr().add(at));
                    if INPUT {
                        a[j] = vfmaq_f64(a[j], vx, vy);
                        b[j] = vfmaq_f64(b[j], vx, vz)
                    } else {
                        a[j] = vfmaq_f64(a[j], vx, vz);
                        b[j] = vfmaq_f64(b[j], vy, vz)
                    }
                }
            }
            let vectors = x.len() / 2 * 2;
            for i in (bulk..vectors).step_by(2) {
                let j = (i - bulk) / 2;
                let vx = vld1q_f64(x.as_ptr().add(i));
                let vy = vld1q_f64(y.as_ptr().add(i));
                let vz = vld1q_f64(z.as_ptr().add(i));
                if INPUT {
                    a[j] = vfmaq_f64(a[j], vx, vy);
                    b[j] = vfmaq_f64(b[j], vx, vz)
                } else {
                    a[j] = vfmaq_f64(a[j], vx, vz);
                    b[j] = vfmaq_f64(b[j], vy, vz)
                }
            }
            let va = vaddq_f64(vaddq_f64(a[0], a[1]), vaddq_f64(a[2], a[3]));
            let vb = vaddq_f64(vaddq_f64(b[0], b[1]), vaddq_f64(b[2], b[3]));
            let mut la = [0.; 2];
            let mut lb = la;
            vst1q_f64(la.as_mut_ptr(), va);
            vst1q_f64(lb.as_mut_ptr(), vb);
            let mut a = la[0] + la[1];
            let mut b = lb[0] + lb[1];
            for i in vectors..x.len() {
                if INPUT {
                    a += x[i] * y[i];
                    b += x[i] * z[i]
                } else {
                    a += x[i] * z[i];
                    b += y[i] * z[i]
                }
            }
            (a, b)
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;
    pub(super) unsafe fn run<const INPUT: bool>(x: &[f64], y: &[f64], z: &[f64]) -> (f64, f64) {
        unsafe {
            let mut a = [f64x2_splat(0.); 4];
            let mut b = a;
            let bulk = x.len() / 8 * 8;
            for i in (0..bulk).step_by(8) {
                for j in 0..4 {
                    let at = i + 2 * j;
                    let vx = v128_load(x.as_ptr().add(at).cast());
                    let vy = v128_load(y.as_ptr().add(at).cast());
                    let vz = v128_load(z.as_ptr().add(at).cast());
                    if INPUT {
                        a[j] = f64x2_add(a[j], f64x2_mul(vx, vy));
                        b[j] = f64x2_add(b[j], f64x2_mul(vx, vz))
                    } else {
                        a[j] = f64x2_add(a[j], f64x2_mul(vx, vz));
                        b[j] = f64x2_add(b[j], f64x2_mul(vy, vz))
                    }
                }
            }
            let vectors = x.len() / 2 * 2;
            for i in (bulk..vectors).step_by(2) {
                let j = (i - bulk) / 2;
                let vx = v128_load(x.as_ptr().add(i).cast());
                let vy = v128_load(y.as_ptr().add(i).cast());
                let vz = v128_load(z.as_ptr().add(i).cast());
                if INPUT {
                    a[j] = f64x2_add(a[j], f64x2_mul(vx, vy));
                    b[j] = f64x2_add(b[j], f64x2_mul(vx, vz))
                } else {
                    a[j] = f64x2_add(a[j], f64x2_mul(vx, vz));
                    b[j] = f64x2_add(b[j], f64x2_mul(vy, vz))
                }
            }
            let va = f64x2_add(f64x2_add(a[0], a[1]), f64x2_add(a[2], a[3]));
            let vb = f64x2_add(f64x2_add(b[0], b[1]), f64x2_add(b[2], b[3]));
            let mut a = f64x2_extract_lane::<0>(va) + f64x2_extract_lane::<1>(va);
            let mut b = f64x2_extract_lane::<0>(vb) + f64x2_extract_lane::<1>(vb);
            for i in vectors..x.len() {
                if INPUT {
                    a += x[i] * y[i];
                    b += x[i] * z[i]
                } else {
                    a += x[i] * z[i];
                    b += y[i] * z[i]
                }
            }
            (a, b)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pairs_are_identical_to_the_independent_backend_dots() {
        for n in 0..1032 {
            let x: Vec<_> = (0..n).map(|i| ((i * 31 % 71) as f64 - 35.) / 31.).collect();
            let y: Vec<_> = (0..n).map(|i| ((i * 11 % 97) as f64 - 48.) / 97.).collect();
            let z: Vec<_> = (0..n)
                .map(|i| ((i * 17 % 103) as f64 - 51.) / 103.)
                .collect();
            let xy = crate::f64_dot::dot(&x, &y);
            let xz = crate::f64_dot::dot(&x, &z);
            let yz = crate::f64_dot::dot(&y, &z);
            let a = shared_input(&x, &y, &z);
            let b = shared_coefficients(&x, &y, &z);
            assert_eq!(a.0.to_bits(), xy.to_bits(), "shared input n={n}");
            assert_eq!(a.1.to_bits(), xz.to_bits());
            assert_eq!(b.0.to_bits(), xz.to_bits(), "shared coefficients n={n}");
            assert_eq!(b.1.to_bits(), yz.to_bits());
        }
    }
}
