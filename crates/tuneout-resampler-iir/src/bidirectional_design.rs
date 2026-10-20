//! Impulse-invariant polyphase sampling of one normalized continuous IIR.
//!
//! h(t) = Re sum c_j exp(s_j |t|), with t in lower-rate sample units.
//! L = output/gcd, M = input/gcd, K = max(L,M), a = L/K.
//! The input-rate poles are exp(s_j a). At output phase r, the past and
//! future coefficients are a*c_j*exp(s_j*r/K) and
//! a*c_j*exp(s_j*(L-r)/K). The current-input coefficient is h(r/K)*a.
//! These follow directly from the sampled convolution, without a virtual
//! high-rate recurrence, bilinear frequency warping, or special rate tables.

use crate::complex::{Complex, Wide, WideComplex};
use crate::{design, Error};
use std::f64::consts::PI;

#[path = "normalized_model.rs"]
pub(crate) mod model;
pub(crate) const PASSBAND: f64 = model::PASSBAND;
pub(crate) const STOPBAND: f64 = model::STOPBAND;
pub(crate) const ATTENUATION_DB: f64 = 115.0; // historical seed metadata only
pub(crate) const MODE_COUNT: usize = model::MODE_COUNT;
pub(crate) const MAX_TABLE_PHASES: u32 = 512;
// The stronger normalized model needs just over one million guard frames at
// 64:1. This is a search bound only; constructor byte limits independently
// constrain every actual buffer and channel count.
pub(crate) const MAX_GUARD_FRAMES: usize = 2_097_152;
// Uniformly tighter than the public 1e-14 omitted-future contract.
const TAIL_ERROR: f64 = 1e-17;

#[derive(Clone, Copy)]
struct Mode {
    s: WideComplex,
    c: WideComplex,
}

pub(crate) struct TwoSidedDesign {
    pub interpolation: u32,
    pub decimation: u32,
    pub passband: f64,
    pub flatness_moments: usize,
    pub max_flatness_coefficient_change: f64,
    pub causal_poles: Vec<Complex>,
    pub causal_phases: Vec<Complex>,
    pub future_poles: Vec<Complex>,
    pub future_phases: Vec<Complex>,
    pub future_guards: Vec<usize>,
    pub direct: Vec<f64>,
    pub guard_frames: usize,
    pub guard_bound: f64,
    pub max_dc_correction: f64,
    pub rational_order: usize,
    modes: Vec<Mode>,
    phase_residues: Vec<WideComplex>,
    #[cfg(test)]
    alpha: f64,
    weights: Vec<(f64, f64)>,
}

fn modes() -> Vec<Mode> {
    model::MODES
        .iter()
        .map(|m| {
            let w = |i| Wide::from(m[i]) + Wide::from(m[i + 1]);
            Mode {
                s: WideComplex { re: w(0), im: w(2) },
                c: WideComplex { re: w(4), im: w(6) },
            }
        })
        .collect()
}

/// Double-double exponential for coefficient setup; |z| <= 4 here.
/// Scaling to |z/16| <= 1/4 and 24 Taylor terms gives a truncation below
/// 4e-42 before four squarings. Runtime states remain binary64.
fn exp_wide(z: WideComplex) -> WideComplex {
    let z = z * WideComplex::real(1.0 / 16.0);
    let mut sum = WideComplex::real(1.0);
    let mut term = sum;
    for k in 1..=24 {
        term = term * z / WideComplex::real(k as f64);
        sum = sum + term;
    }
    for _ in 0..4 {
        sum = sum * sum;
    }
    sum
}

#[cfg(test)]
fn exponential(s: Complex, t: f64) -> Complex {
    let (sin, cos) = (s.im * t).sin_cos();
    let radius = (s.re * t).exp();
    Complex::new(radius * cos, radius * sin)
}

pub(crate) fn make(input: u32, output: u32) -> Result<TwoSidedDesign, Error> {
    make_impl(input, output, true)
}
/// The mixed wrapper retains the original admission/continuous diagnostics,
/// but never executes this full-rate phase bank.
pub(crate) fn make_metadata(input: u32, output: u32) -> Result<TwoSidedDesign, Error> {
    make_impl(input, output, false)
}
fn make_impl(input: u32, output: u32, cache_phases: bool) -> Result<TwoSidedDesign, Error> {
    if input == 0 || output == 0 {
        return Err(Error::ZeroSampleRate);
    }
    let gcd = design::gcd(input, output);
    let interpolation = output / gcd;
    let decimation = input / gcd;
    let maximum = interpolation.max(decimation);
    let alpha_w = Wide::from(f64::from(interpolation)) / Wide::from(f64::from(maximum));
    let alpha = alpha_w.value();
    let modes = modes();
    let poles: Vec<_> = modes
        .iter()
        .map(|m| {
            exp_wide(
                m.s * WideComplex {
                    re: alpha_w,
                    im: Wide::from(0.0),
                },
            )
            .value()
        })
        .collect();
    if poles.iter().any(|p| !p.abs().is_finite() || p.abs() >= 1.0) {
        return Err(Error::InvalidDesign);
    }
    // Every phase has |b_j| <= a |c_j|. This continuous-phase envelope also
    // covers uncached phases. Inflate both coefficient and radius bounds
    // beyond f64 rounding; no sampled-phase maximum is assumed.
    let weights: Vec<_> = modes
        .iter()
        .zip(&poles)
        .map(|(m, p)| {
            let radius = (p.abs() + 4.0 * f64::EPSILON).min(1.0);
            (
                radius,
                alpha * m.c.value().abs() * (1.0 + 1e-12) / (1.0 - radius),
            )
        })
        .collect();
    if weights.iter().any(|&(r, w)| r >= 1.0 || !w.is_finite()) {
        return Err(Error::InvalidDesign);
    }
    let guard_frames = minimum_guard(&weights, TAIL_ERROR)?;
    let future_guards = allocate_guards(&weights, guard_frames, TAIL_ERROR);
    let guard_bound = allocated_tail_bound(&weights, &future_guards);
    let phase_residues = modes.iter().map(|mode| mode.c.scale(alpha_w)).collect();
    let mut d = TwoSidedDesign {
        interpolation,
        decimation,
        passband: PASSBAND,
        flatness_moments: 9,
        max_flatness_coefficient_change: 0.0,
        causal_poles: poles.clone(),
        future_poles: poles,
        causal_phases: Vec::new(),
        future_phases: Vec::new(),
        direct: Vec::new(),
        future_guards,
        guard_frames,
        guard_bound,
        max_dc_correction: 0.0,
        rational_order: 2 * MODE_COUNT,
        modes,
        phase_residues,
        #[cfg(test)]
        alpha,
        weights,
    };
    if cache_phases && interpolation <= MAX_TABLE_PHASES {
        d.causal_phases
            .reserve_exact(interpolation as usize * MODE_COUNT);
        d.future_phases
            .reserve_exact(interpolation as usize * MODE_COUNT);
        d.direct.reserve_exact(interpolation as usize);
        let mut causal = [Complex::default(); MODE_COUNT];
        let mut future = causal;
        let mut causal_low = causal;
        let mut future_low = causal;
        for phase in 0..interpolation {
            let direct = d.past_phase(phase, &mut causal, &mut causal_low);
            d.future_phase(phase, &mut future, &mut future_low);
            d.causal_phases.extend_from_slice(&causal);
            d.future_phases.extend_from_slice(&future);
            d.direct.push(direct.value());
        }
    }
    Ok(d)
}

impl TwoSidedDesign {
    pub fn tabled(&self) -> bool {
        !self.direct.is_empty()
    }

    /// Setup-only control: round the DD result at the original storage boundary.
    #[cfg(test)]
    pub fn phase(&self, phase: u32, causal: &mut [Complex], future: &mut [Complex]) -> f64 {
        debug_assert!(phase < self.interpolation);
        if self.tabled() {
            let start = phase as usize * MODE_COUNT;
            causal.copy_from_slice(&self.causal_phases[start..start + MODE_COUNT]);
            future.copy_from_slice(&self.future_phases[start..start + MODE_COUNT]);
            return self.direct[phase as usize];
        }
        let mut low = [Complex::default(); MODE_COUNT];
        let direct = self.past_phase(phase, causal, &mut low);
        self.future_phase(phase, future, &mut low);
        direct.value()
    }

    /// Exact rational phase in DD; same setup for cached and on-demand rows.
    /// The two directions are separate so an uncached output does not compute
    /// an unused row. Residue times alpha is precomputed once in DD.
    fn half_phase<const DIRECT: bool>(
        &self,
        numerator: u32,
        high: &mut [Complex],
        low: &mut [Complex],
    ) -> Wide {
        let maximum = Wide::from(f64::from(self.interpolation.max(self.decimation)));
        let time = Wide::from(f64::from(numerator)) / maximum;
        let mut direct = Wide::from(0.0);
        for (((mode, residue), a), b) in self
            .modes
            .iter()
            .zip(&self.phase_residues)
            .zip(high)
            .zip(low)
        {
            let coefficient = *residue * exp_wide(mode.s.scale(time));
            (*a, *b) = coefficient.split();
            if DIRECT {
                direct = direct + coefficient.re;
            }
        }
        direct
    }

    pub fn past_phase(&self, phase: u32, high: &mut [Complex], low: &mut [Complex]) -> Wide {
        debug_assert!(phase < self.interpolation);
        self.half_phase::<true>(phase, high, low)
    }

    pub fn future_phase(&self, phase: u32, high: &mut [Complex], low: &mut [Complex]) {
        debug_assert!(phase < self.interpolation);
        self.half_phase::<false>(self.interpolation - phase, high, low);
    }

    /// Continuous normalized model response. Sampling introduces periodic
    /// copies; use actual phase-bank transfer tests to measure those images.
    pub fn response(&self, relative: f64) -> Complex {
        let w = Wide::from(PI) * Wide::from(relative);
        let w2 = WideComplex {
            re: w * w,
            im: Wide::from(0.0),
        };
        let mut h = Wide::from(0.0);
        for m in &self.modes {
            h = h + (WideComplex::real(-2.0) * m.c * m.s / (m.s * m.s + w2)).re;
        }
        Complex::new(h.value(), 0.0)
    }

    pub fn model_bytes(&self) -> usize {
        self.modes.len() * std::mem::size_of::<Mode>()
            + self.phase_residues.len() * std::mem::size_of::<WideComplex>()
            + self.weights.len() * std::mem::size_of::<(f64, f64)>()
    }
}

include!("tail.rs");

#[cfg(test)]
#[path = "modal_tests.rs"]
pub(crate) mod tests;

#[cfg(test)]
#[path = "phase_precision_tests.rs"]
mod phase_precision_tests;
