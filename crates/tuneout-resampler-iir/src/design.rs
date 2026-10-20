//! Two parallel all-pass branches, warped to the required low-pass band.
//!
//! `prototype` adapts the elliptic design equations in Laurent de Soras's
//! HIIR PolyphaseIir2Designer (2005), released under WTFPL v2. See NOTICE.md.
//! The rational polyphase decomposition and precision handling are local.

use std::f64::consts::PI;

use crate::complex::{Complex, Wide, WideComplex};
use crate::Error;

pub(crate) const PASSBAND: f64 = 0.96;
pub(crate) const DESIGN_ATTENUATION_DB: f64 = 210.0;
pub(crate) const MAX_FACTOR: u32 = 4096;

pub(crate) struct Prototype {
    pub interpolation: u32,
    pub decimation: u32,
    pub coefficients: Vec<f64>,
    pub warp: f64,
}

pub(crate) struct Design {
    pub interpolation: u32,
    pub decimation: u32,
    pub coefficients: Vec<f64>,
    pub warp: f64,
    pub branches: [Vec<Complex>; 2],
    pub poles: Vec<Complex>,
    pub phases: Vec<Complex>,
    pub direct: Vec<f64>,
    pub max_dc_correction: f64,
}

pub(crate) fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

pub(crate) fn make(input: u32, output: u32) -> Result<Design, Error> {
    make_with_attenuation(input, output, DESIGN_ATTENUATION_DB)
}

pub(crate) fn make_with_attenuation(
    input: u32,
    output: u32,
    attenuation: f64,
) -> Result<Design, Error> {
    let Prototype {
        interpolation,
        decimation,
        coefficients,
        warp,
    } = make_prototype(input, output, attenuation, PASSBAND)?;
    let mut branches = [Vec::new(), Vec::new()];
    let alpha = WideComplex::real(warp);
    let one = WideComplex::real(1.0);
    for (i, &a) in coefficients.iter().enumerate() {
        let original = WideComplex::from(Complex::new(0.0, a.sqrt()));
        let pole = ((original + alpha) / (one + alpha * original)).value();
        branches[i % 2].extend([pole, pole.conj()]);
    }
    // The z^-1 in the second prototype branch becomes another all-pass pole.
    branches[1].push(Complex::new(warp, 0.0));

    let mut modes = Vec::new();
    let mut h0 = Wide::from(0.0);
    for branch in &branches {
        let mut feedthrough = one;
        for &p in branch {
            if !p.re.is_finite() || !p.im.is_finite() || p.abs() >= 1.0 {
                return Err(Error::InvalidDesign);
            }
            feedthrough = feedthrough * (WideComplex::real(0.0) - WideComplex::from(p).conj());
        }
        h0 = h0 + feedthrough.re * Wide::from(0.5);
        for (index, &p) in branch.iter().enumerate() {
            if p.im < 0.0 {
                continue;
            }
            let pole = WideComplex::from(p);
            // Residue of 1/(1-p*q); form ratios before multiplying, avoiding
            // tiny high-order numerator/denominator polynomials entirely.
            let mut residue = (one - pole.conj() * pole) / pole;
            for (other_index, &other) in branch.iter().enumerate() {
                if other_index != index {
                    let other = WideComplex::from(other);
                    residue = residue * ((one - other.conj() * pole) / (pole - other));
                }
            }
            // Re(conjugate pair) supplies factor two, cancelling branch /2.
            if p.im == 0.0 {
                residue = residue * WideComplex::real(0.5);
            }
            modes.push((pole, residue * WideComplex::real(f64::from(interpolation))));
        }
    }

    let mut phases = Vec::with_capacity(interpolation as usize * modes.len());
    let mut direct = Vec::with_capacity(interpolation as usize);
    let mut powers = vec![one; modes.len()];
    for phase in 0..interpolation {
        let mut impulse = Wide::from(0.0);
        for (i, &(pole, residue)) in modes.iter().enumerate() {
            let coefficient = residue * powers[i];
            impulse = impulse + coefficient.re;
            phases.push(coefficient.value());
            powers[i] = powers[i] * pole;
        }
        direct.push(if phase == 0 {
            (h0 * Wide::from(f64::from(interpolation))).value()
        } else {
            impulse.value()
        });
    }
    // Every recursive state runs at the INPUT rate: lambda = p^L. The phase
    // rows give h[L*n+d]. No inserted zero or discarded output is processed.
    let poles: Vec<_> = powers.into_iter().map(WideComplex::value).collect();
    // A finite-rejection interpolator need not have exactly unity gain in
    // every phase (the residual images also contribute to DC). Calibrate the
    // actual rounded recursive branches, just as a FIR phase can be normalized
    // by its tap sum. This is a fixed coefficient correction, not an adaptive
    // gain or an FIR stage. It changes the ideal all-pass prototype slightly;
    // independent streaming/alias tests cover the corrected realization.
    let mut max_dc_correction: f64 = 0.0;
    for (phase, row) in phases.chunks_exact_mut(modes.len()).enumerate() {
        let mut gain = Wide::from(direct[phase]);
        for (&coefficient, &pole) in row.iter().zip(&poles) {
            let pole = WideComplex::from(pole);
            gain = gain + (WideComplex::from(coefficient) * pole / (one - pole)).re;
        }
        let correction = Wide::from(1.0) / gain;
        max_dc_correction = max_dc_correction.max((correction - Wide::from(1.0)).value().abs());
        for c in row {
            c.re = (Wide::from(c.re) * correction).value();
            c.im = (Wide::from(c.im) * correction).value();
        }
        direct[phase] = (Wide::from(direct[phase]) * correction).value();
    }
    if poles
        .iter()
        .chain(phases.iter())
        .any(|p| !p.re.is_finite() || !p.im.is_finite())
        || direct.iter().any(|x| !x.is_finite())
    {
        return Err(Error::InvalidDesign);
    }
    Ok(Design {
        interpolation,
        decimation,
        coefficients,
        warp,
        branches,
        poles,
        phases,
        direct,
        max_dc_correction,
    })
}

/// Build only ratio and elliptic prototype parameters; no recursive phase bank.
pub(crate) fn make_prototype(
    input: u32,
    output: u32,
    attenuation: f64,
    passband: f64,
) -> Result<Prototype, Error> {
    let divisor = gcd(input, output);
    let interpolation = output / divisor;
    let decimation = input / divisor;
    let maximum = interpolation.max(decimation);
    if maximum > MAX_FACTOR {
        return Err(Error::RatioTooLarge {
            interpolation,
            decimation,
        });
    }

    // Stopband begins at the LOWER Nyquist, not above it. Map both band edges
    // to a symmetric half-band prototype using a first-order all-pass warp:
    // q -> (q - alpha) / (1 - alpha*q), q = z^-1.
    let wp = PI * passband / f64::from(maximum);
    let ws = PI / f64::from(maximum);
    let scale = ((wp * 0.5).tan() * (ws * 0.5).tan()).sqrt();
    let warp = (1.0 - scale) / (1.0 + scale);
    let transition = 0.5 - 2.0 * ((wp * 0.5).tan() / scale).atan() / PI;
    let coefficients = prototype(attenuation, transition);
    Ok(Prototype {
        interpolation,
        decimation,
        coefficients,
        warp,
    })
}

fn prototype(attenuation: f64, transition: f64) -> Vec<f64> {
    let k = ((1.0 - 2.0 * transition) * PI / 4.0).tan().powi(2);
    // The truncated nome expansion in the original designer limits rejection
    // at our narrow transition, even when the requested order is increased.
    // q = exp(-pi*K(k')/K(k)), K(k) = pi/(2*AGM(1,k')). Computing the nome
    // through the arithmetic-geometric mean removes that design error. This
    // is construction-only work, allowing fewer modes at higher accuracy.
    let q = (-PI * agm(1.0, (1.0 - k * k).sqrt()) / agm(1.0, k)).exp();
    let power = 10.0_f64.powf(-attenuation / 10.0);
    let a = power / (1.0 - power);
    let order = ((a * a / 16.0).ln() / q.ln()).ceil() as usize | 1;
    let order = order.max(3);
    (1..=(order - 1) / 2)
        .map(|c| {
            let theta = c as f64 * PI / order as f64;
            let mut numerator = 0.0;
            let mut denominator = 0.5;
            // For this bounded design domain, 32 terms exceed f64 precision.
            for i in 0_i32..32 {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                numerator += sign * q.powi(i * (i + 1)) * (f64::from(2 * i + 1) * theta).sin();
                if i > 0 {
                    denominator += sign * q.powi(i * i) * (f64::from(2 * i) * theta).cos();
                }
            }
            let w2 = (numerator * q.sqrt().sqrt() / denominator).powi(2);
            let x = ((1.0 - w2 * k) * (1.0 - w2 / k)).max(0.0).sqrt() / (1.0 + w2);
            (1.0 - x) / (1.0 + x)
        })
        .collect()
}

fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..16 {
        let arithmetic = (a + b) * 0.5;
        let geometric = (a * b).sqrt();
        if arithmetic == a && geometric == b {
            break;
        }
        (a, b) = (arithmetic, geometric);
    }
    (a + b) * 0.5
}

impl Design {
    /// Analytic prototype group delay at DC, expressed in output frames.
    pub fn dc_delay(&self) -> f64 {
        (1.0 + self.warp) / (1.0 - self.warp)
            * (0.5
                + self
                    .coefficients
                    .iter()
                    .map(|c| (1.0 - c) / (1.0 + c))
                    .sum::<f64>())
            / f64::from(self.decimation)
    }

    pub fn response(&self, relative_to_lower_nyquist: f64) -> Complex {
        let frequency =
            PI * relative_to_lower_nyquist / f64::from(self.interpolation.max(self.decimation));
        let z = Complex::new(frequency.cos(), frequency.sin());
        let one = Complex::new(1.0, 0.0);
        let mut sum = Complex::default();
        for branch in &self.branches {
            let mut response = one;
            for &pole in branch {
                response = response * ((one - pole.conj() * z) / (z - pole));
            }
            sum = sum + response;
        }
        Complex::new(sum.re * 0.5, sum.im * 0.5)
    }
}
