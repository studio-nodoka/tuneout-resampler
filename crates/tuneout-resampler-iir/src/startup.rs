//! A bounded leading segment estimates the missing history at a file boundary.
//!
//! Burg's forward/backward recursion estimates a stable autoregressive model.
//! Its backward continuation is summed analytically at each recursive pole;
//! no FIR stage, synthesized padding buffer or per-sample predictor is used.
//! See the crate's research notes for the derivation and limitations.

use crate::complex::{Complex, Wide, WideComplex};

pub(crate) const FRAMES: usize = 256;
const ORDER: usize = 16;

pub(crate) fn initialize(
    input: &[f64],
    channels: usize,
    poles: &[Complex],
    states: &mut [Complex],
) {
    initialize_inner(input, channels, poles, states, false);
}

/// As `initialize`, but sums x[-k] * pole^(k-1). This also handles pole=0.
pub(crate) fn initialize_lagged(
    input: &[f64],
    channels: usize,
    poles: &[Complex],
    states: &mut [Complex],
) {
    initialize_inner(input, channels, poles, states, true);
}

fn initialize_inner(
    input: &[f64],
    channels: usize,
    poles: &[Complex],
    states: &mut [Complex],
    lagged: bool,
) {
    for channel in 0..channels {
        let mut normalized: Vec<_> = input
            .chunks_exact(channels)
            .map(|frame| frame[channel])
            .collect();
        let peak = normalized.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if peak == 0.0 || !peak.is_finite() {
            continue;
        }
        for value in &mut normalized {
            *value /= peak;
        }
        let a = predictable_model(&normalized).unwrap_or_else(|| burg(&normalized));
        let order = a.len() - 1;
        // For backward samples y(k)=x(-k), sum a[j]*y(k-j)=0. The history
        // y(-j)=x(j) gives S(z)=sum(k>=1) y(k)*z^k=B(z)/A(z), where
        // B[r]=-sum(j=r..P) a[j]*x(j-r), 1<=r<=P. Runtime state before x(0)
        // is precisely S(lambda), so no long warmup needs to be rendered.
        let b: Vec<_> = (1..=order)
            .map(|r| {
                (r..=order).fold(Wide::from(0.0), |sum, j| {
                    sum - Wide::from(a[j]) * Wide::from(normalized[j - r])
                })
            })
            .collect();
        let state = &mut states[channel * poles.len()..(channel + 1) * poles.len()];
        let mut valid = true;
        for (s, &pole) in state.iter_mut().zip(poles) {
            let z = WideComplex::from(pole);
            let mut denominator = WideComplex::real(a[order]);
            let mut numerator = WideComplex::real(0.0);
            for j in (0..order).rev() {
                denominator = denominator * z + WideComplex::real(a[j]);
                numerator = numerator
                    + WideComplex {
                        re: b[j],
                        im: Wide::from(0.0),
                    };
                if j != 0 || !lagged {
                    numerator = numerator * z;
                }
            }
            let value = (numerator / denominator).value();
            // Keep uncertain/ill-conditioned predictions bounded. This only
            // selects initial state; the stable DSP poles never change.
            if !value.re.is_finite()
                || !value.im.is_finite()
                || value.abs() > 8.0 / (1.0 - pole.abs())
            {
                valid = false;
                break;
            }
            *s = Complex::new(value.re * peak, value.im * peak);
            valid &= s.re.is_finite() && s.im.is_finite();
        }
        if !valid {
            state.fill(Complex::default());
        }
    }
}

/// Continue only the finite endpoint needed by timestamp compensation. The
/// normal streaming loop never runs this predictor. Work and amplitude remain
/// bounded even for a poor model or an extreme supported sample-rate ratio.
pub(crate) fn extend(input: &[f64], channels: usize, frames: usize) -> Vec<f64> {
    let mut output = vec![0.0; frames * channels];
    for channel in 0..channels {
        let mut normalized: Vec<_> = input.chunks_exact(channels).map(|f| f[channel]).collect();
        let peak = normalized.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if peak == 0.0 || !peak.is_finite() {
            continue;
        }
        for value in &mut normalized {
            *value /= peak;
        }
        let (a, stride) =
            predictable_strided_model(&normalized).unwrap_or_else(|| (burg(&normalized), 1));
        let order = (a.len() - 1) * stride;
        let mut history: Vec<_> = normalized[normalized.len() - order..].to_vec();
        let fallback = *normalized.last().unwrap();
        let mut valid = true;
        for frame in output.chunks_exact_mut(channels) {
            let next = -a[1..]
                .iter()
                .zip(history.iter().rev().skip(stride - 1).step_by(stride))
                .map(|(a, x)| a * x)
                .sum::<f64>();
            // This guards an extrapolation, not the user's original PCM.
            // Reject the complete channel forecast if any part is unreliable.
            valid = next.is_finite() && next.abs() <= 8.0 && (next * peak).is_finite();
            if !valid {
                break;
            }
            frame[channel] = next * peak;
            history.rotate_left(1);
            history[order - 1] = next;
        }
        if !valid {
            for frame in output.chunks_exact_mut(channels) {
                frame[channel] = fallback * peak;
            }
        }
    }
    output
}

fn burg(x: &[f64]) -> Vec<f64> {
    if x.len() < 4 {
        // A bounded constant continuation is preferable to fitting a model
        // without enough observations. A(z)=1-z still converges at |lambda|<1.
        return vec![1.0, -1.0];
    }
    let mut forward = x[1..].to_vec();
    let mut backward = x[..x.len() - 1].to_vec();
    let energy: f64 = x.iter().map(|v| v * v).sum();
    let mut a = vec![1.0];
    for _ in 0..ORDER.min(x.len() / 4) {
        let denominator: f64 = forward
            .iter()
            .zip(&backward)
            .map(|(f, b)| f * f + b * b)
            .sum();
        if denominator < energy * 1e-24 {
            break;
        }
        let numerator: f64 = forward.iter().zip(&backward).map(|(f, b)| f * b).sum();
        let reflection = (-2.0 * numerator / denominator).clamp(-1.0 + 1e-12, 1.0 - 1e-12);
        let mut next = vec![0.0; a.len() + 1];
        for j in 0..a.len() {
            next[j] += a[j];
            next[j + 1] += reflection * a[a.len() - 1 - j];
        }
        a = next;
        for (f, b) in forward.iter_mut().zip(&mut backward) {
            (*f, *b) = (*f + reflection * *b, *b + reflection * *f);
        }
        forward.remove(0);
        backward.pop();
    }
    a
}

// A near-deterministic record (DC, sinusoid, or a small mixture) should not
// acquire extra near-unit-circle modes merely to fit roundoff. Incremental,
// twice-orthogonalized QR finds the smallest accurate AR model. Use it only
// when the relative one-step residual is below 1e-20 and a Schur test confirms
// stability; ordinary audio/noise retains the Burg estimate. This is local
// order selection, not recognition of any test frequency or sample-rate pair.
fn predictable_model(x: &[f64]) -> Option<Vec<f64>> {
    predictable_model_at_stride(x, 1)
}

fn predictable_strided_model(x: &[f64]) -> Option<(Vec<f64>, usize)> {
    // Different lag spacings improve conditioning when the same short record
    // contains several nearby modes. Every fit uses all observed phases of
    // that spacing, the original residual threshold, and the Schur check.
    // The smallest coefficient norm limits rounding amplification during the
    // finite forecast. Ordinary audio retains the original Burg fallback.
    [1, 2, 4, 8]
        .into_iter()
        .flat_map(|stride| {
            [
                predictable_model_at_stride(x, stride),
                reciprocal_model(x, stride),
            ]
            .into_iter()
            .flatten()
            .map(move |a| {
                let norm = a.iter().map(|v| v.abs()).sum::<f64>();
                (a, stride, norm)
            })
        })
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(a, stride, _)| (a, stride))
}

// Both model fits use the same two-pass QR column reduction and addition order.
fn orthogonalize(
    v: &mut [f64],
    basis: &[Vec<f64>],
    r: &mut [[f64; ORDER]; ORDER],
    column: usize,
) -> f64 {
    for _ in 0..2 {
        for (j, q) in basis.iter().enumerate() {
            let dot: f64 = v.iter().zip(q).map(|(a, b)| a * b).sum();
            r[j][column] += dot;
            for (value, q) in v.iter_mut().zip(q) {
                *value -= dot * q;
            }
        }
    }
    v.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn reciprocal_model(x: &[f64], stride: usize) -> Option<Vec<f64>> {
    // A stationary real modal signal has a reciprocal annihilating
    // polynomial: a[j] = a[order-j]. Fit that constraint directly instead of
    // allowing roundoff to split marginal conjugate pole pairs across the
    // unit circle. The same small radial contraction and stability check as
    // the unconstrained fit remain mandatory; no frequency is prescribed.
    for pairs in 1..=ORDER / 2 {
        let order = 2 * pairs;
        let offset = order * stride;
        if x.len() < offset + 32 {
            break;
        }
        let count = x.len() - offset;
        let mut residual: Vec<_> = (offset..x.len()).map(|i| -x[i] - x[i - offset]).collect();
        let energy: f64 = x[offset..].iter().map(|v| v * v).sum();
        if energy == 0. {
            continue;
        }
        let minimum_norm = energy.sqrt() * 1e-12;
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut r = [[0.; ORDER]; ORDER];
        let mut projections = [0.; ORDER];
        let mut valid = true;
        for column in 0..pairs {
            let lag = (column + 1) * stride;
            let mut v: Vec<_> = (offset..x.len())
                .map(|i| {
                    if column + 1 == pairs {
                        x[i - lag]
                    } else {
                        x[i - lag] + x[i - offset + lag]
                    }
                })
                .collect();
            let norm = orthogonalize(&mut v, &basis, &mut r, column);
            if norm < minimum_norm {
                valid = false;
                break;
            }
            r[column][column] = norm;
            for value in &mut v {
                *value /= norm;
            }
            projections[column] = residual.iter().zip(&v).map(|(a, b)| a * b).sum();
            for i in 0..count {
                residual[i] -= projections[column] * v[i];
            }
            basis.push(v);
        }
        if !valid || residual.iter().map(|v| v * v).sum::<f64>() >= energy * 1e-20 {
            continue;
        }
        let mut a = vec![0.; order + 1];
        a[0] = 1.;
        a[order] = 1.;
        for i in (0..pairs).rev() {
            let correction: f64 = (i + 1..pairs).map(|j| r[i][j] * a[j + 1]).sum();
            a[i + 1] = (projections[i] - correction) / r[i][i];
            a[order - i - 1] = a[i + 1];
        }
        for (j, v) in a.iter_mut().enumerate().skip(1) {
            *v *= (1. - 1e-12_f64).powi(j as i32);
        }
        if schur_stable(&a) {
            return Some(a);
        }
    }
    None
}

fn predictable_model_at_stride(x: &[f64], stride: usize) -> Option<Vec<f64>> {
    let offset = ORDER * stride;
    if x.len() < 2 * offset {
        return None;
    }
    let count = x.len() - offset;
    let mut residual: Vec<_> = x[offset..].iter().map(|x| -x).collect();
    let energy: f64 = residual.iter().map(|v| v * v).sum();
    if energy == 0.0 {
        return None;
    }
    let minimum_norm = energy.sqrt() * 1e-12;
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut r = [[0.0; ORDER]; ORDER];
    let mut projections = [0.0; ORDER];
    for column in 0..ORDER {
        let lag = (column + 1) * stride;
        let mut v = x[offset - lag..x.len() - lag].to_vec();
        let norm = orthogonalize(&mut v, &basis, &mut r, column);
        if norm < minimum_norm {
            return None;
        }
        r[column][column] = norm;
        for value in &mut v {
            *value /= norm;
        }
        let projection: f64 = residual.iter().zip(&v).map(|(a, b)| a * b).sum();
        projections[column] = projection;
        for i in 0..count {
            residual[i] -= projection * v[i];
        }
        basis.push(v);
        if residual.iter().map(|v| v * v).sum::<f64>() < energy * 1e-20 {
            let order = column + 1;
            let mut a = vec![0.0; order + 1];
            a[0] = 1.0;
            for i in (0..order).rev() {
                let correction: f64 = (i + 1..order).map(|j| r[i][j] * a[j + 1]).sum();
                a[i + 1] = (projections[i] - correction) / r[i][i];
            }
            // Move marginal pure-tone/DC poles just inside the unit circle.
            for (j, value) in a.iter_mut().enumerate().skip(1) {
                *value *= (1.0 - 1e-12_f64).powi(j as i32);
            }
            if schur_stable(&a) {
                return Some(a);
            }
            return None;
        }
    }
    None
}

fn schur_stable(a: &[f64]) -> bool {
    let mut a: Vec<_> = a.iter().copied().map(Wide::from).collect();
    let mut next = [Wide::from(0.0); ORDER];
    for order in (1..a.len()).rev() {
        let reflection = a[order];
        if !reflection.value().is_finite() || reflection.value().abs() >= 1.0 {
            return false;
        }
        let denominator = Wide::from(1.0) - reflection * reflection;
        for j in 1..order {
            next[j - 1] = (a[j] - reflection * a[order - j]) / denominator;
        }
        a[1..order].copy_from_slice(&next[..order - 1]);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_continuation_matches_unseen_tones_and_bounds_poor_models() {
        let tone = |n: usize| {
            0.31 * (n as f64 * 0.073 + 0.4).cos() + 0.19 * (n as f64 * 0.219 - 0.8).sin()
        };
        let input: Vec<_> = (0..FRAMES).flat_map(|n| [tone(n), 0.0]).collect();
        let future = extend(&input, 2, 28);
        for (n, f) in future.chunks_exact(2).enumerate() {
            assert!((f[0] - tone(FRAMES + n)).abs() < 1e-8);
            assert_eq!(f[1], 0.0);
        }
        // A stable model can overshoot far outside its observations. Exercise
        // the longest supported downsampling horizon and the fallback guard.
        let ramp: Vec<_> = (0..FRAMES)
            .map(|n| n as f64 / (FRAMES - 1) as f64)
            .collect();
        let forecast = extend(&ramp, 1, 6 * 4096);
        assert!(forecast.iter().all(|v| v.is_finite() && v.abs() <= 8.0));
        assert!(forecast.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn analytic_state_matches_independent_backward_recursion() {
        let input: Vec<_> = (0..FRAMES)
            .map(|i| 0.5 * (i as f64 * 1.51).sin() + 0.1 * (i as f64 * 0.07).cos())
            .collect();
        let poles = [Complex::new(0.1, 0.2), Complex::new(0.6, 0.5)];
        let mut states = vec![Complex::default(); poles.len()];
        initialize(&input, 1, &poles, &mut states);
        // Use the same normalized fit, but explicitly generate the backward
        // samples and accumulate the infinite sum to its negligible tail.
        let peak = input.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let normalized: Vec<_> = input.iter().map(|x| x / peak).collect();
        let a = predictable_model(&normalized).unwrap_or_else(|| burg(&normalized));
        let mut history: Vec<_> = input[..a.len() - 1].iter().rev().copied().collect();
        let mut sums = vec![Complex::default(); poles.len()];
        let mut powers = poles.to_vec();
        for _ in 0..2048 {
            let next = -a[1..]
                .iter()
                .zip(history.iter().rev())
                .map(|(a, x)| a * x)
                .sum::<f64>();
            history.push(next);
            for j in 0..poles.len() {
                sums[j] = sums[j] + powers[j] * Complex::new(next, 0.0);
                powers[j] = powers[j] * poles[j];
            }
        }
        for (actual, expected) in states.iter().zip(sums) {
            assert!((*actual - expected).abs() < 1e-11);
        }
    }
}
