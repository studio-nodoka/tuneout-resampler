//! SIMD across independent rational phases during coefficient construction.
//!
//! This reproduces the scalar double-double recurrence, operation for
//! operation. FMA is used only for the existing error-free product residual.
//! Phase exponentials and coefficient rounding retain their original paths.
use crate::complex::Wide as W;

pub(crate) fn rows(alpha: W, radius: usize, l: u32) -> Option<Vec<Vec<f64>>> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            // SAFETY: Runtime dispatch checks OS state and all required ISA;
            // the private implementation loads/stores fixed-size local arrays.
            return Some(unsafe { x86::avx512(alpha, radius, l) });
        }
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            return Some(unsafe { x86::avx(alpha, radius, l) });
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return Some(unsafe { neon::construct(alpha, radius, l) });
    }
    let _ = (alpha, radius, l);
    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! implementation {
    ($name:ident,$feature:literal,$n:expr,$v:ty,$zero:ident,$set:ident,$load:ident,$store:ident,$add:ident,$sub:ident,$mul:ident,$fma:ident,$neg:ident) => {
        #[target_feature(enable=$feature)]
        pub(super) unsafe fn $name(alpha: W, radius: usize, l: u32) -> Vec<Vec<f64>> {
            unsafe {
                #[derive(Clone, Copy)]
                struct D {
                    hi: $v,
                    lo: $v,
                }
                macro_rules! norm {
                    ($hi:expr,$lo:expr) => {{
                        let hi = $hi;
                        let lo = $lo;
                        let sum = $add(hi, lo);
                        D {
                            hi: sum,
                            lo: $sub(lo, $sub(sum, hi)),
                        }
                    }};
                }
                macro_rules! plus {
                    ($a:expr,$b:expr) => {{
                        let a = $a;
                        let b = $b;
                        let sum = $add(a.hi, b.hi);
                        let v = $sub(sum, a.hi);
                        let e = $add($sub(a.hi, $sub(sum, v)), $sub(b.hi, v));
                        norm!(sum, $add($add(e, a.lo), b.lo))
                    }};
                }
                macro_rules! minus {
                    ($a:expr,$b:expr) => {{
                        let b = $b;
                        plus!(
                            $a,
                            D {
                                hi: $neg(b.hi),
                                lo: $neg(b.lo)
                            }
                        )
                    }};
                }
                macro_rules! times {
                    ($a:expr,$b:expr) => {{
                        let a = $a;
                        let b = $b;
                        let product = $mul(a.hi, b.hi);
                        let error = $fma(a.hi, b.hi, $neg(product));
                        norm!(
                            product,
                            $add(
                                $add($add(error, $mul(a.hi, b.lo)), $mul(a.lo, b.hi)),
                                $mul(a.lo, b.lo)
                            )
                        )
                    }};
                }
                macro_rules! splat {
                    ($w:expr) => {{
                        let (hi, lo) = $w.parts();
                        D {
                            hi: $set(hi),
                            lo: $set(lo),
                        }
                    }};
                }
                macro_rules! zero {
                    () => {
                        D {
                            hi: $zero(),
                            lo: $zero(),
                        }
                    };
                }
                macro_rules! one {
                    () => {
                        D {
                            hi: $set(1.),
                            lo: $zero(),
                        }
                    };
                }
                let alpha_v = splat!(alpha);
                let two = splat!(W::from(2.));
                let span = splat!(W::from(1. / (crate::prefix::B - crate::prefix::A)));
                let lower = splat!(W::from(crate::prefix::A));
                let taps = 2 * radius + 1;
                let mut result = vec![vec![0.; taps]; l as usize];
                // The opposite nonzero phase reverses indices1..taps;
                // index0 lies beyond the compact support for both phases.
                let representatives = l as usize / 2 + 1;
                for phase in (0..representatives).step_by($n) {
                    let active = $n.min(representatives - phase);
                    let u: Vec<W> = (0..$n)
                        .map(|lane| {
                            W::from((phase + lane.min(active - 1)) as f64) / W::from(f64::from(l))
                        })
                        .collect();
                    for future in [false, true] {
                        let offset: Vec<W> = u
                            .iter()
                            .map(|&u| if future { W::from(1.) - u } else { u })
                            .collect();
                        let mut hi = [0.; $n];
                        let mut lo = hi;
                        for lane in 0..$n {
                            (hi[lane], lo[lane]) = offset[lane].parts();
                        }
                        let offset_v = D {
                            hi: $load(hi.as_ptr()),
                            lo: $load(lo.as_ptr()),
                        };
                        let end = if future { radius } else { radius + 1 };
                        for j in 0..end {
                            let t = times!(plus!(splat!(W::from(j as f64)), offset_v), alpha_v);
                            let mut t_values = [0.; $n];
                            $store(t_values.as_mut_ptr(), $add(t.hi, t.lo));
                            let in_support =
                                t_values.iter().take(active).any(|&t| t < crate::prefix::B);
                            if in_support {
                                let segments =
                                    t_values.map(|t| ((t / 2.).floor() as usize).min(79));
                                let centers = segments.map(|j| (2 * j + 1) as f64);
                                let x = minus!(
                                    t,
                                    D {
                                        hi: $load(centers.as_ptr()),
                                        lo: $zero()
                                    }
                                );
                                let mut h = zero!();
                                for k in (0..=crate::prefix::core_polynomial::DEGREE).rev() {
                                    for lane in 0..$n {
                                        (hi[lane], lo[lane]) =
                                            crate::prefix::core_polynomial::COEFFICIENTS
                                                [segments[lane]][k];
                                    }
                                    h = plus!(
                                        times!(h, x),
                                        D {
                                            hi: $load(hi.as_ptr()),
                                            lo: $load(lo.as_ptr())
                                        }
                                    );
                                }
                                let window = if t_values
                                    .iter()
                                    .take(active)
                                    .all(|&t| t <= crate::prefix::A)
                                {
                                    one!()
                                } else {
                                    // The original division by512 is an exact power-of-two scale.
                                    let x =
                                        minus!(times!(two, times!(minus!(t, lower), span)), one!());
                                    let mut b1 = zero!();
                                    let mut b2 = zero!();
                                    for &(a, b) in
                                        crate::prefix::partition::COEFFICIENTS[1..].iter().rev()
                                    {
                                        let c = plus!(splat!(W::from(a)), splat!(W::from(b)));
                                        let b = minus!(plus!(c, times!(times!(two, x), b1)), b2);
                                        b2 = b1;
                                        b1 = b;
                                    }
                                    let (a, b) = crate::prefix::partition::COEFFICIENTS[0];
                                    let c = plus!(splat!(W::from(a)), splat!(W::from(b)));
                                    minus!(one!(), minus!(plus!(c, times!(x, b1)), b2))
                                };
                                // turnon clamps each endpoint independently, including
                                // a SIMD group straddling the lower boundary.
                                let mut wh = [0.; $n];
                                let mut wl = wh;
                                $store(wh.as_mut_ptr(), window.hi);
                                $store(wl.as_mut_ptr(), window.lo);
                                for lane in 0..$n {
                                    if t_values[lane] <= crate::prefix::A {
                                        wh[lane] = 1.;
                                        wl[lane] = 0.;
                                    }
                                }
                                let window = D {
                                    hi: $load(wh.as_ptr()),
                                    lo: $load(wl.as_ptr()),
                                };
                                let value = times!(times!(h, alpha_v), window);
                                let mut out = [0.; $n];
                                $store(out.as_mut_ptr(), $add(value.hi, value.lo));
                                let index = if future { radius + j + 1 } else { radius - j };
                                for lane in 0..active {
                                    if t_values[lane] < crate::prefix::B {
                                        result[phase + lane][index] = out[lane];
                                    }
                                }
                            }
                        }
                    }
                }
                for phase in 1..(l as usize).div_ceil(2) {
                    let (source, destination) = result.split_at_mut(l as usize - phase);
                    for j in 1..taps {
                        destination[0][j] = source[phase][taps - j];
                    }
                }
                result
            }
        }
    };
}

pub(crate) fn weights(x: &[W], half: usize, beta: f64, den: W) -> Option<Vec<f64>> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
            return Some(unsafe { x86::weights_avx512(x, half, beta, den) });
        }
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            return Some(unsafe { x86::weights_avx(x, half, beta, den) });
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return Some(unsafe { neon::weights_neon(x, half, beta, den) });
    }
    let _ = (x, half, beta, den);
    None
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! weight_implementation {
    ($name:ident,$feature:literal,$n:expr,$v:ty,$zero:ident,$set:ident,$load:ident,$store:ident,$add:ident,$sub:ident,$mul:ident,$div:ident,$fma:ident,$neg:ident)=>{
        #[target_feature(enable=$feature)]
        pub(super) unsafe fn $name(input:&[W],half:usize,beta:f64,den:W)->Vec<f64> {
            unsafe {
                #[derive(Clone,Copy)]struct D {hi:$v,lo:$v}
                #[derive(Clone,Copy)]struct Z {re:D,im:D}
                macro_rules! norm {($hi:expr,$lo:expr)=>{{let hi=$hi;let lo=$lo;let sum=$add(hi,lo);D{hi:sum,lo:$sub(lo,$sub(sum,hi))}}}}
                macro_rules! plus {($a:expr,$b:expr)=>{{let a=$a;let b=$b;let sum=$add(a.hi,b.hi);let v=$sub(sum,a.hi);let e=$add($sub(a.hi,$sub(sum,v)),$sub(b.hi,v));norm!(sum,$add($add(e,a.lo),b.lo))}}}
                macro_rules! minus {($a:expr,$b:expr)=>{{let b=$b;plus!($a,D{hi:$neg(b.hi),lo:$neg(b.lo)})}}}
                macro_rules! times {($a:expr,$b:expr)=>{{let a=$a;let b=$b;let product=$mul(a.hi,b.hi);let error=$fma(a.hi,b.hi,$neg(product));norm!(product,$add($add($add(error,$mul(a.hi,b.lo)),$mul(a.lo,b.hi)),$mul(a.lo,b.lo)))}}}
                macro_rules! splat {($w:expr)=>{{let (hi,lo)=$w.parts();D{hi:$set(hi),lo:$set(lo)}}}}
                macro_rules! zero {()=>{D{hi:$zero(),lo:$zero()}}}
                macro_rules! one {()=>{D{hi:$set(1.),lo:$zero()}}}

                macro_rules! quotient {($a:expr,$b:expr)=>{{let a=$a;let b=$b;let first=D{hi:$div(a.hi,b.hi),lo:$zero()};let r=minus!(a,times!(b,first));plus!(first,D{hi:$div($add(r.hi,r.lo),b.hi),lo:$zero()})}}}
                macro_rules! ztimes {($a:expr,$b:expr)=>{{let a=$a;let b=$b;Z{re:minus!(times!(a.re,b.re),times!(a.im,b.im)),im:plus!(times!(a.re,b.im),times!(a.im,b.re))}}}}
                macro_rules! zdivide {($a:expr,$k:expr)=>{{let a=$a;let k=splat!(W::from($k));let n=plus!(times!(k,k),times!(zero!(),zero!()));Z{re:quotient!(plus!(times!(a.re,k),times!(a.im,zero!())),n),im:quotient!(minus!(times!(a.im,k),times!(a.re,zero!())),n)}}}}
                macro_rules! values {($a:expr)=>{{let a=$a;let mut out=[0.;$n];$store(out.as_mut_ptr(),$add(a.hi,a.lo));out}}}
                macro_rules! from_lanes {($a:expr)=>{{let a=$a;let mut hi=[0.;$n];let mut lo=hi;for lane in 0..$n{(hi[lane],lo[lane])=a[lane].parts();}D{hi:$load(hi.as_ptr()),lo:$load(lo.as_ptr())}}}}
                macro_rules! choose {($current:expr,$old:expr,$active:expr)=>{{let current=$current;let old=$old;let active=$active;let mut hi=[0.;$n];let mut lo=hi;let mut old_hi=hi;let mut old_lo=hi;$store(hi.as_mut_ptr(),current.hi);$store(lo.as_mut_ptr(),current.lo);$store(old_hi.as_mut_ptr(),old.hi);$store(old_lo.as_mut_ptr(),old.lo);for lane in 0..$n{if !active[lane]{hi[lane]=old_hi[lane];lo[lane]=old_lo[lane];}}D{hi:$load(hi.as_ptr()),lo:$load(lo.as_ptr())}}}}
                let pi=splat!(W::from(std::f64::consts::PI)+W::from(1.2246467991473532e-16));
                let mut result=Vec::with_capacity(input.len());
                for first in (0..input.len()).step_by($n){
                    let count=$n.min(input.len()-first);let mut xs=[W::from(0.);$n];
                    let mut valid=[false;$n];let mut nonzero=valid;
                    for lane in 0..$n{let x=input[first+lane.min(count-1)];valid[lane]=x.value().abs()<=half as f64;nonzero[lane]=x.value()!=0.;xs[lane]=if valid[lane]{x}else{W::from(0.)};}
                    let x=from_lanes!(xs);let angle=times!(pi,x);let angle_values=values!(angle);
                    let mut extra=[0usize;$n];let mut scales=[W::from(0.);$n];
                    for lane in 0..$n{let size=angle_values[lane].abs();extra[lane]=if size>4.{(size/4.).log2().ceil()as usize}else{0};scales[lane]=W::from(1./(16usize<<extra[lane])as f64);}
                    let z=Z{re:times!(zero!(),from_lanes!(scales)),im:times!(angle,from_lanes!(scales))};
                    let mut sum=Z{re:one!(),im:zero!()};let mut term=sum;
                    for k in 1..=28{term=zdivide!(ztimes!(term,z),k as f64);sum=Z{re:plus!(sum.re,term.re),im:plus!(sum.im,term.im)};}
                    for iteration in 0..4+*extra.iter().max().unwrap(){let active=extra.map(|e|iteration<4+e);let squared=ztimes!(sum,sum);sum=Z{re:choose!(squared.re,sum.re,active),im:choose!(squared.im,sum.im,active)};}
                    let safe_angle=choose!(angle,one!(),nonzero);let sinc=choose!(quotient!(sum.im,safe_angle),one!(),nonzero);
                    let q=minus!(one!(),quotient!(times!(x,x),splat!(W::from((half*half)as f64))));let qvalues=values!(q);
                    let roots=qvalues.map(|v|W::from(if v<=0.{0.}else{v.sqrt()}));let roots=from_lanes!(roots);
                    let positive=qvalues.map(|q|q>0.);let safe_root=choose!(roots,one!(),positive);
                    let root=choose!(plus!(roots,quotient!(minus!(q,times!(roots,roots)),times!(splat!(W::from(2.)),safe_root))),zero!(),positive);
                    let b=times!(root,splat!(W::from(beta)));let q=quotient!(times!(b,b),splat!(W::from(4.)));
                    let mut s=one!();let mut t=s;let mut active=[true;$n];
                    for k in 1..=160{let next_t=quotient!(times!(t,q),splat!(W::from((k*k)as f64)));let next_s=plus!(s,next_t);t=choose!(next_t,t,active);s=choose!(next_s,s,active);let tv=values!(t);let sv=values!(s);for lane in 0..$n{if tv[lane].abs()<sv[lane].abs()*1e-34{active[lane]=false;}}if !active.iter().any(|&a|a){break}}
                    let output=values!(quotient!(times!(sinc,s),splat!(den)));
                    for lane in 0..count{result.push(if valid[lane]{output[lane]}else{0.});}
                }result
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn neg4(v: __m256d) -> __m256d {
        unsafe { _mm256_xor_pd(v, _mm256_set1_pd(-0.)) }
    }
    #[inline(always)]
    unsafe fn neg8(v: __m512d) -> __m512d {
        unsafe {
            _mm512_castsi512_pd(_mm512_xor_si512(
                _mm512_castpd_si512(v),
                _mm512_set1_epi64(i64::MIN),
            ))
        }
    }
    implementation!(
        avx512,
        "avx512f,fma",
        8,
        __m512d,
        _mm512_setzero_pd,
        _mm512_set1_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        _mm512_sub_pd,
        _mm512_mul_pd,
        _mm512_fmadd_pd,
        neg8
    );
    implementation!(
        avx,
        "avx,fma",
        4,
        __m256d,
        _mm256_setzero_pd,
        _mm256_set1_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        _mm256_sub_pd,
        _mm256_mul_pd,
        _mm256_fmadd_pd,
        neg4
    );
    weight_implementation!(
        weights_avx512,
        "avx512f,fma",
        8,
        __m512d,
        _mm512_setzero_pd,
        _mm512_set1_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        _mm512_sub_pd,
        _mm512_mul_pd,
        _mm512_div_pd,
        _mm512_fmadd_pd,
        neg8
    );
    weight_implementation!(
        weights_avx,
        "avx,fma",
        4,
        __m256d,
        _mm256_setzero_pd,
        _mm256_set1_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        _mm256_sub_pd,
        _mm256_mul_pd,
        _mm256_div_pd,
        _mm256_fmadd_pd,
        neg4
    );
}
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    #[inline(always)]
    unsafe fn zero() -> float64x2_t {
        unsafe { vdupq_n_f64(0.) }
    }
    #[inline(always)]
    unsafe fn fma(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        unsafe { vfmaq_f64(c, a, b) }
    }
    implementation!(
        construct,
        "neon",
        2,
        float64x2_t,
        zero,
        vdupq_n_f64,
        vld1q_f64,
        vst1q_f64,
        vaddq_f64,
        vsubq_f64,
        vmulq_f64,
        fma,
        vnegq_f64
    );
    weight_implementation!(
        weights_neon,
        "neon",
        2,
        float64x2_t,
        zero,
        vdupq_n_f64,
        vld1q_f64,
        vst1q_f64,
        vaddq_f64,
        vsubq_f64,
        vmulq_f64,
        vdivq_f64,
        fma,
        vnegq_f64
    );
}

#[cfg(all(
    test,
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
mod tests {
    use super::*;
    #[test]
    fn all_available_setup_backends_match_dispatched_coefficients() {
        let alpha = W::from(147.) / W::from(320.);
        let radius = (crate::prefix::B / alpha.value()).ceil() as usize;
        if let Some(expected) = rows(alpha, radius, 7) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                let actual = unsafe { x86::avx(alpha, radius, 7) };
                for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                let actual = unsafe { neon::construct(alpha, radius, 7) };
                for (a, b) in actual.iter().flatten().zip(expected.iter().flatten()) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
        let xs: Vec<_> = (0..1027)
            .map(|i| W::from((i * 103 % 8191) as f64) / W::from(64.) - W::from(64.))
            .collect();
        let den = W::from(1.7658818543684496e19);
        if let Some(expected) = weights(&xs, 64, 48., den) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                let actual = unsafe { x86::weights_avx(&xs, 64, 48., den) };
                for (a, b) in actual.iter().zip(expected) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                let actual = unsafe { neon::weights_neon(&xs, 64, 48., den) };
                for (a, b) in actual.iter().zip(expected) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
    }
}
