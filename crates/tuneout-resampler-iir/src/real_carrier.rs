//! Same DD real carrier component, with a checked native FMA context.
use crate::complex::{Wide as W, WideComplex as WC};
pub(super) type Finish = fn(f64, f64, WC) -> W;
#[inline(always)]
fn scalar(re: f64, im: f64, rotation: WC) -> W {
    W::from(re) * rotation.re - W::from(im) * rotation.im
}
pub(super) fn select() -> Finish {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("fma") {
        return checked_fma;
    }
    scalar
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn checked_fma(re: f64, im: f64, rotation: WC) -> W {
    // This private function pointer is returned only after FMA detection.
    unsafe { hardware_fma(re, im, rotation) }
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn hardware_fma(re: f64, im: f64, rotation: WC) -> W {
    scalar(re, im, rotation)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_carrier_component_matches_the_original_dd_expression() {
        for index in 0..4096 {
            let x = (index * 31 % 257) as f64 / 257. - 0.5;
            let y = (index * 43 % 263) as f64 / 263. - 0.5;
            let rotation = crate::prefix::exp(WC {
                re: W::from(0.),
                im: W::from(index as f64 / 173.),
            });
            let original = (WC::from(crate::complex::Complex::new(x, y)) * rotation).re;
            assert_eq!(
                select()(x, y, rotation).parts(),
                original.parts(),
                "phase {index}"
            );
        }
    }
}
