//! Constructor-only analytic Bessel series. No rate or sampled-kernel bank.
use crate::complex::{Wide as W, WideComplex as WC};
pub(crate) struct Kaiser {
    coefficients: Vec<W>,
    #[cfg(test)]
    pub(crate) tail_bound: f64,
}
fn pi() -> W {
    W::from(std::f64::consts::PI) + W::from(1.2246467991473532e-16)
}
impl Kaiser {
    pub(crate) fn new(beta: f64, den: W) -> Option<Self> {
        if !beta.is_finite() || beta < 0. || !den.value().is_finite() || den.value() <= 0. {
            return None;
        }
        let gamma = W::from(beta) * W::from(beta) / W::from(4.);
        let mut term = W::from(1.) / den;
        let mut coefficients = vec![term];
        if beta == 0. {
            return Some(Self {
                coefficients,
                #[cfg(test)]
                tail_bound: 0.,
            });
        }
        for k in 1..=160 {
            term = term * gamma / W::from((k * k) as f64);
            if !term.value().is_finite() {
                return None;
            }
            coefficients.push(term);
            let ratio = gamma.value() / ((k + 1) * (k + 1)) as f64;
            if ratio < 1. {
                let tail = term.value().abs() * ratio / (1. - ratio) * (1. + 1e-12);
                if tail < 1e-36 {
                    return Some(Self {
                        coefficients,
                        #[cfg(test)]
                        tail_bound: tail,
                    });
                }
            }
        }
        None
    }
    pub(crate) fn window(&self, q: W) -> W {
        let q = if q.value() <= 0. { W::from(0.) } else { q };
        let mut value = W::from(0.);
        for &coefficient in self.coefficients.iter().rev() {
            value = value * q + coefficient;
        }
        value
    }
    #[cfg(test)]
    pub(crate) fn weight(&self, x: W, half: usize) -> W {
        self.weight_with_sine(x, half, None)
    }
    pub(crate) fn weight_with_sine(&self, x: W, half: usize, numerator: Option<W>) -> W {
        if x.value().abs() > half as f64 {
            return W::from(0.);
        }
        let angle = pi() * x;
        let sinc = if x.value() == 0. {
            W::from(1.)
        } else {
            numerator.unwrap_or_else(|| {
                crate::prefix::exp(WC {
                    re: W::from(0.),
                    im: angle,
                })
                .im
            }) / angle
        };
        sinc * self.window(W::from(1.) - x * x / W::from((half * half) as f64))
    }
    #[cfg(test)]
    pub(crate) fn weights(&self, input: &[W], half: usize) -> Vec<f64> {
        self.weights_impl(input, half, None)
    }
    pub(crate) fn weights_phased(&self, input: &[W], half: usize, sines: &[W]) -> Vec<f64> {
        assert!(half > 0);
        assert_eq!(input.len(), sines.len() * 2 * half);
        self.weights_impl(input, half, Some(sines))
    }
    fn weights_impl(&self, input: &[W], half: usize, sines: Option<&[W]>) -> Vec<f64> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
                return unsafe { x86::weights_avx512(input, half, &self.coefficients, sines) };
            }
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                return unsafe { x86::weights_avx(input, half, &self.coefficients, sines) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon::weights_neon(input, half, &self.coefficients, sines) };
        }
        input
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let numerator = sines.map(|sines| {
                    let value = sines[i / (2 * half)];
                    if (half - 1 + i) & 1 == 0 {
                        value
                    } else {
                        W::from(0.) - value
                    }
                });
                self.weight_with_sine(x, half, numerator).value()
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_bessel_polynomial_matches_independent_sqrt_series() {
        for beta in [0., 1., 8., 16., 32., 48., 64.] {
            let den = crate::core_pipeline::i0(W::from(beta));
            let p = Kaiser::new(beta, den).unwrap();
            assert!(p.tail_bound < 1e-35);
            let mut maximum: f64 = 0.;
            for j in 0..=4096 {
                let q = W::from(j as f64) / W::from(4096.);
                let root = if j == 0 {
                    W::from(0.)
                } else {
                    let a = W::from(q.value().sqrt());
                    a + (q - a * a) / (W::from(2.) * a)
                };
                let expected = crate::core_pipeline::i0(root * W::from(beta)) / den;
                let e = (p.window(q) - expected).value().abs();
                maximum = maximum.max(e);
            }
            assert!(maximum < 1e-27, "beta{beta},error{maximum:e}");
            println!(
                "KAISER_POLY,beta={beta},degree={},tail_bound={:e},max_DD={maximum:e}",
                p.coefficients.len() - 1,
                p.tail_bound
            );
        }
    }
    #[test]
    fn polynomial_windowed_sinc_dense_phases_and_backends_match_reference() {
        for (half, beta) in [(20, 32.), (24, 32.), (40, 32.), (64, 48.)] {
            let den = crate::core_pipeline::i0(W::from(beta));
            let p = Kaiser::new(beta, den).unwrap();
            let mut arguments = Vec::new();
            for phase in 0..127 {
                let u = W::from(phase as f64) / W::from(127.);
                for j in -(half as i64) + 1..=half as i64 {
                    arguments.push(u - W::from(j as f64));
                }
            }
            arguments.extend([
                W::from(0.),
                W::from(half as f64),
                W::from(-(half as f64)),
                W::from(half as f64 + 0.1),
            ]);
            let vector = p.weights(&arguments, half);
            let mut maximum: f64 = 0.;
            let mut changed = 0;
            for (&x, &v) in arguments.iter().zip(&vector) {
                let expected = crate::core_pipeline::weight(x, half, beta, den);
                let actual = p.weight(x, half);
                maximum = maximum.max((actual - expected).value().abs());
                if v.to_bits() != expected.value().to_bits() {
                    changed += 1;
                }
                assert!((v - actual.value()).abs() < 2e-16);
            }
            assert!(maximum < 1e-27, "half{half},beta{beta},error{maximum:e}");
            println!("KAISER_SINC,half={half},beta={beta},max_DD={maximum:e},rounded_changes={changed}/{}",arguments.len());
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! kaiser_implementation {
    ($name:ident,$feature:literal,$n:expr,$v:ty,$zero:ident,$set:ident,$load:ident,$store:ident,$add:ident,$sub:ident,$mul:ident,$div:ident,$fma:ident,$neg:ident)=>{
        #[target_feature(enable=$feature)]
        pub(super) unsafe fn $name(input:&[W],half:usize,coefficients:&[W],sines:Option<&[W]>)->Vec<f64> {
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
                    let x=from_lanes!(xs);let angle=times!(pi,x);let sinc=if let Some(sines)=sines{
                        let numerators:[W;$n]=std::array::from_fn(|lane|{let index=first+lane.min(count-1);let value=sines[index/(2*half)];if(half-1+index)&1==0{value}else{W::from(0.)-value}});
                        let safe_angle=choose!(angle,one!(),nonzero);choose!(quotient!(from_lanes!(numerators),safe_angle),one!(),nonzero)
                    }else{let angle_values=values!(angle);
                    let mut extra=[0usize;$n];let mut scales=[W::from(0.);$n];
                    for lane in 0..$n{let size=angle_values[lane].abs();extra[lane]=if size>4.{(size/4.).log2().ceil()as usize}else{0};scales[lane]=W::from(1./(16usize<<extra[lane])as f64);}
                    let z=Z{re:times!(zero!(),from_lanes!(scales)),im:times!(angle,from_lanes!(scales))};
                    let mut sum=Z{re:one!(),im:zero!()};let mut term=sum;
                    for k in 1..=28{term=zdivide!(ztimes!(term,z),k as f64);sum=Z{re:plus!(sum.re,term.re),im:plus!(sum.im,term.im)};}
                    for iteration in 0..4+*extra.iter().max().unwrap(){let active=extra.map(|e|iteration<4+e);let squared=ztimes!(sum,sum);sum=Z{re:choose!(squared.re,sum.re,active),im:choose!(squared.im,sum.im,active)};}
                    let safe_angle=choose!(angle,one!(),nonzero);choose!(quotient!(sum.im,safe_angle),one!(),nonzero)
                    };
                    let q=minus!(one!(),quotient!(times!(x,x),splat!(W::from((half*half)as f64))));let positive=values!(q).map(|q|q>0.);let q=choose!(q,zero!(),positive);
                    let mut window=zero!();for &coefficient in coefficients.iter().rev(){window=plus!(times!(window,q),splat!(coefficient));}
                    let output=values!(times!(sinc,window));
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
    kaiser_implementation!(
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
    kaiser_implementation!(
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
    kaiser_implementation!(
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

#[cfg(test)]
#[path = "setup_kaiser_phase_tests.rs"]
mod phase_tests;

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod forced_backends {
    use super::*;
    #[test]
    fn every_available_kaiser_backend_retains_the_scalar_coefficient_bits() {
        for half in [20, 24, 40] {
            let den = crate::core_pipeline::i0(W::from(32.));
            let p = Kaiser::new(32., den).unwrap();
            let mut args = Vec::new();
            let mut sines = Vec::new();
            for phase in 1..17 {
                let u = W::from(phase as f64) / W::from(17.);
                sines.push(
                    crate::prefix::exp(WC {
                        re: W::from(0.),
                        im: pi() * u,
                    })
                    .im,
                );
                for j in -(half as i64) + 1..=half as i64 {
                    args.push(u - W::from(j as f64));
                }
            }
            let expected: Vec<_> = args
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let z = sines[i / (2 * half)];
                    let z = if (half - 1 + i) & 1 == 0 {
                        z
                    } else {
                        W::from(0.) - z
                    };
                    p.weight_with_sine(x, half, Some(z)).value().to_bits()
                })
                .collect();
            if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
                let actual =
                    unsafe { x86::weights_avx(&args, half, &p.coefficients, Some(&sines)) };
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected
                );
            }
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("fma") {
                let actual =
                    unsafe { x86::weights_avx512(&args, half, &p.coefficients, Some(&sines)) };
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }
}
