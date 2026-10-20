//! Actual coarse-grid causal/anticausal IIR, plus a finite coarse complex FIR.
//! Significant untapered core modes remain infinite recurrences; eight selected
//! slow modes use finite recursive intervals from the accepted analytic model.
use crate::complex::{Complex as C, Wide as W, WideComplex as WC};
use crate::late_tail_coefficients as tail;
use crate::prefix;
#[derive(Clone)]
struct Mode {
    start: usize,
    end: Option<usize>,
    guard: usize,
    p: C,
    leave: C,
    coefficient: WC,
}
struct Correction {
    first: usize,
    re: Vec<f64>,
    im: Vec<f64>,
    reverse_re: Vec<f64>,
    reverse_im: Vec<f64>,
}
struct Center {
    past_first: usize,
    past: Vec<WC>,
    future_first: usize,
    future: Vec<WC>,
}
pub struct CoarseIir {
    correction: Correction,
    parent_correction: Correction,
    modes: Vec<Mode>,
    past_coefficients: Vec<WC>,
    future_coefficients: Vec<WC>,
    center: Center,
    pub max_lag: usize,
}
#[derive(Default)]
pub struct State {
    past: Vec<C>,
    next: Option<i128>,
}
fn wc(v: C) -> WC {
    WC {
        re: W::from(v.re),
        im: W::from(v.im),
    }
}
fn add(a: C, b: C) -> C {
    a + b
}
fn times(a: C, b: C) -> C {
    a * b
}
fn power(mut p: C, mut n: usize) -> C {
    let mut z = C::new(1., 0.);
    while n != 0 {
        if n & 1 != 0 {
            z = z * p
        }
        p = p * p;
        n >>= 1;
    }
    z
}
impl CoarseIir {
    pub fn new(step: W) -> Self {
        let d = step.value();
        let start = (prefix::B / d).ceil() as usize;
        let pi = W::from(std::f64::consts::PI) + W::from(1.2246467991473532e-16);
        let selected = [34, 35, 36, 37, 38, 39, 41, 46];
        let mut modes = Vec::new();
        let mut _omitted_core_l1 = 0.;
        for (j, (s, c)) in prefix::modes().into_iter().enumerate() {
            let s = WC {
                re: s.re,
                im: s.im - pi,
            };
            let p = prefix::exp(s.scale(step)).value();
            let coefficient = c * prefix::exp(s.scale(step * W::from(start as f64))).scale(step);
            let end = if selected.contains(&j) {
                Some((13230. / d).ceil() as usize)
            } else {
                None
            };
            let radius = p.abs();
            let weight = coefficient.value().abs() / (1. - radius);
            let guard = end.map_or_else(
                || {
                    if weight < 1e-23 {
                        0
                    } else {
                        ((1e-23 / weight).ln() / radius.ln()).ceil().max(1.) as usize
                    }
                },
                |end| end - start,
            );
            if guard == 0 {
                let (rh, rl) = coefficient.re.parts();
                let (ih, il) = coefficient.im.parts();
                let radius = (p.abs() + 4. * f64::EPSILON).min(1.);
                _omitted_core_l1 +=
                    2. * (rh.abs() + rl.abs() + ih.abs() + il.abs()) * (1. + 1e-12) / (1. - radius);
                continue;
            }
            let leave = end.map_or(C::default(), |end| power(p, end - start));
            modes.push(Mode {
                start,
                end,
                guard,
                p,
                leave,
                coefficient,
            });
        }
        // Finite late correction is sampled from the identical generic modal
        // model in DD, then evaluated as a coarse ordinary complex FIR. The
        // significant retained core modes remain actual IIR recurrences.
        let selected: Vec<_> = tail::MODES
            .iter()
            .filter(|&&(_, b, _, si, _, _)| si > std::f64::consts::PI / 2. && b <= 13230)
            .collect();
        let first = 0; // The fixed-shape delta begins with the old turnon center.
        let last = selected
            .iter()
            .map(|&&(_, b, _, _, _, _)| (b as f64 / d).ceil() as usize)
            .max()
            .unwrap_or(0);
        let mut coefficients = vec![WC::real(0.); last - first];
        for &&(a, b, sr, si, cr, ci) in &selected {
            let begin = (a as f64 / d).ceil() as usize;
            let end = (b as f64 / d).ceil() as usize;
            let s = WC {
                re: W::from(sr),
                im: W::from(si) - pi,
            };
            let pole = prefix::exp(s.scale(step));
            let sign = if a % 2 == 0 { 1. } else { -1. };
            let c = WC {
                re: W::from(cr * sign),
                im: W::from(ci * sign),
            };
            let offset = step * W::from(begin as f64) - W::from(a as f64);
            let mut coefficient = (c * prefix::exp(s.scale(offset))).scale(step);
            for lag in begin..end {
                coefficients[lag - first] = coefficients[lag - first] + coefficient;
                coefficient = coefficient * pole;
            }
        }
        let parent_re: Vec<_> = coefficients.iter().map(|z| z.re.value()).collect();
        let parent_im: Vec<_> = coefficients.iter().map(|z| z.im.value()).collect();
        let parent_correction = Correction {
            first,
            reverse_re: parent_re.iter().rev().copied().collect(),
            reverse_im: parent_im.iter().rev().copied().collect(),
            re: parent_re,
            im: parent_im,
        };
        let mut whole = coefficients.clone();
        for m in &modes {
            let mut value = m.coefficient;
            let pole = WC::from(m.p);
            let end = m.end.unwrap_or(last).min(last);
            for sample in whole.iter_mut().take(end).skip(m.start) {
                *sample = *sample + value;
                value = value * pole;
            }
        }
        for (lag, sample) in whole.iter_mut().enumerate().take(start) {
            let t = step * W::from(lag as f64);
            let window = prefix::turnon(t);
            if window.value() != 0. {
                *sample = *sample + prefix::envelope(t.value()).scale(window * step);
            }
        }
        for lag in 0..last {
            let t = step * W::from(lag as f64);
            coefficients[lag] =
                coefficients[lag] + whole[lag].scale(crate::cosine_shape::factor(t) - W::from(1.));
        }
        // The finite phase correction has independent past/future envelopes.
        // Include it in this direct convolution oracle without using the FFT
        // setup or its Clenshaw polynomial evaluator. Count its center once.
        let past: Vec<_> = coefficients
            .iter()
            .enumerate()
            .map(|(lag, &c)| {
                c + crate::joint_edge::oracle_envelope((step * W::from(lag as f64)).value(), false)
                    .scale(step)
            })
            .collect();
        let future: Vec<_> = coefficients
            .iter()
            .enumerate()
            .map(|(lag, &c)| {
                if lag == 0 {
                    c
                } else {
                    c + crate::joint_edge::oracle_envelope(
                        (step * W::from(lag as f64)).value(),
                        true,
                    )
                    .conj()
                    .scale(step)
                }
            })
            .collect();
        let correction = Correction {
            first,
            reverse_re: past.iter().rev().map(|z| z.re.value()).collect(),
            reverse_im: past.iter().rev().map(|z| z.im.value()).collect(),
            re: future.iter().map(|z| z.re.value()).collect(),
            im: future.iter().map(|z| z.im.value()).collect(),
        };
        let mut finite = Vec::new();
        for j in 0..start {
            let t = step * W::from(j as f64);
            let window = prefix::turnon(t);
            if window.value() != 0. {
                let coefficient = prefix::envelope(t.value()).scale(window * step);
                finite.push((j, coefficient));
            }
        }
        debug_assert!(finite.windows(2).all(|p| p[0].0 + 1 == p[1].0));
        let center = Center {
            past_first: finite.last().map_or(0, |p| p.0),
            past: finite.iter().rev().map(|p| p.1).collect(),
            future_first: finite.iter().find(|p| p.0 != 0).map_or(0, |p| p.0),
            future: finite
                .iter()
                .filter(|p| p.0 != 0)
                .map(|p| p.1.conj())
                .collect(),
        };
        let max_lag = modes
            .iter()
            .map(|m| m.start + m.guard)
            .max()
            .unwrap_or(start)
            .max(start);
        let past_coefficients = modes.iter().map(|m| m.coefficient).collect();
        let future_coefficients = modes.iter().map(|m| m.coefficient.conj()).collect();
        Self {
            correction,
            parent_correction,
            modes,
            past_coefficients,
            future_coefficients,
            center,
            max_lag: max_lag.max(last),
        }
    }
    pub fn mode_count(&self) -> usize {
        self.modes.len()
    }
    pub fn correction_taps(&self) -> usize {
        self.correction.re.len()
    }
    fn center_output(&self, q: i128, input: &[C], origin: i128) -> WC {
        let center = &self.center;
        let mut value = WC::real(0.);
        if !center.past.is_empty() {
            let first = usize::try_from(q - center.past_first as i128 - origin)
                .expect("retained turnon past");
            value = crate::wide_dot2::complex_wide_complex(
                &input[first..first + center.past.len()],
                &center.past,
            );
        }
        if !center.future.is_empty() {
            let first = usize::try_from(q + center.future_first as i128 - origin)
                .expect("retained turnon future");
            value = value
                + crate::wide_dot2::complex_wide_complex(
                    &input[first..first + center.future.len()],
                    &center.future,
                );
        }
        value
    }
    /// Fixed-coarse-block interface. `input` contains its predictor/zero padded
    /// neighborhood. Use a caller-independent block size for exact chunk invariance.
    pub fn process(
        &self,
        state: &mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> Vec<WC> {
        assert!(count > 0);
        let last = start + count as i128 - 1;
        let at = |index: i128| {
            input[usize::try_from(index - origin)
                .expect("coarse input starts before requested guard")]
        };
        assert!(start - self.max_lag as i128 >= origin);
        assert!(last + (self.max_lag as i128) < origin + input.len() as i128);
        if state.next.is_none() {
            state.past.resize(self.modes.len(), C::default());
            for (j, m) in self.modes.iter().enumerate() {
                let mut s = C::default();
                for lag in (m.start..m.start + m.guard).rev() {
                    s = add(times(m.p, s), at(start - lag as i128));
                }
                state.past[j] = s;
            }
        } else {
            assert_eq!(state.next, Some(start), "contiguous coarse processing")
        }
        let mut future = vec![C::default(); self.modes.len()];
        for (j, m) in self.modes.iter().enumerate() {
            let mut s = C::default();
            for lag in (m.start..m.start + m.guard).rev() {
                s = add(times(m.p.conj(), s), at(last + lag as i128));
            }
            future[j] = s;
        }
        let input_re: Vec<_> = input.iter().map(|z| z.re).collect();
        let input_im: Vec<_> = input.iter().map(|z| z.im).collect();
        let mut output = vec![WC::real(0.); count];
        for (n, row) in output.iter_mut().enumerate() {
            let q = start + n as i128;
            *row = crate::wide_dot2::complex_wide_complex(&state.past, &self.past_coefficients);
            for (m, s) in self.modes.iter().zip(state.past.iter_mut()) {
                *s = add(times(m.p, *s), at(q + 1 - m.start as i128));
                if let Some(end) = m.end {
                    *s = *s - times(m.leave, at(q + 1 - end as i128));
                }
            }
            let correction = &self.correction;
            let len = correction.re.len();
            if len != 0 {
                let past = usize::try_from(q - (correction.first + len - 1) as i128 - origin)
                    .expect("retained correction past");
                let future = usize::try_from(q + correction.first as i128 - origin)
                    .expect("retained correction future");
                let dot = crate::f64_dot::dot;
                let pr = dot(&input_re[past..past + len], &correction.reverse_re)
                    - dot(&input_im[past..past + len], &correction.reverse_im);
                let pi = dot(&input_im[past..past + len], &correction.reverse_re)
                    + dot(&input_re[past..past + len], &correction.reverse_im);
                let fr = dot(&input_re[future..future + len], &correction.re)
                    + dot(&input_im[future..future + len], &correction.im);
                let fi = dot(&input_im[future..future + len], &correction.re)
                    - dot(&input_re[future..future + len], &correction.im);
                *row = *row + wc(C::new(pr + fr, pi + fi));
            }
            *row = *row + self.center_output(q, input, origin);
        }
        for n in (0..count).rev() {
            let q = start + n as i128;
            output[n] = output[n]
                + crate::wide_dot2::complex_wide_complex(&future, &self.future_coefficients);
            for (m, s) in self.modes.iter().zip(future.iter_mut()) {
                if n != 0 {
                    *s = add(times(m.p.conj(), *s), at(q - 1 + m.start as i128));
                    if let Some(end) = m.end {
                        *s = *s - times(m.leave.conj(), at(q - 1 + end as i128));
                    }
                }
            }
        }
        state.next = Some(last + 1);
        output
    }
}

#[cfg(test)]
mod correction_tests {
    use super::*;
    #[test]
    fn constructed_fir_correction_preserves_generic_modal_model() {
        for d in [8., 14.7, 16.] {
            let f = CoarseIir::new(W::from(d));
            let pi = W::from(std::f64::consts::PI) + W::from(1.2246467991473532e-16);
            let mut maximum: f64 = 0.;
            let mut l1: f64 = 0.;
            for j in (0..f.parent_correction.re.len()).step_by(13) {
                let lag = f.parent_correction.first + j;
                let t = W::from(lag as f64) * W::from(d);
                let mut expected = WC::real(0.);
                for &(a, b, sr, si, cr, ci) in tail::MODES {
                    if si <= std::f64::consts::PI / 2.
                        || b > 13230
                        || t.value() < a as f64
                        || t.value() >= b as f64
                    {
                        continue;
                    }
                    let sign = if a % 2 == 0 { 1. } else { -1. };
                    let c = WC {
                        re: W::from(cr * sign),
                        im: W::from(ci * sign),
                    };
                    expected = expected
                        + (c * prefix::exp(
                            WC {
                                re: W::from(sr),
                                im: W::from(si) - pi,
                            }
                            .scale(t - W::from(a as f64)),
                        ))
                        .scale(W::from(d));
                }
                let actual = WC::from(C::new(f.parent_correction.re[j], f.parent_correction.im[j]));
                maximum = maximum.max((actual - expected).value().abs());
                l1 += actual.value().abs();
            }
            println!("d={d} core modes={} correction taps={} selected L1={l1:e} max coefficient error={maximum:e}",f.mode_count(),f.correction_taps());
            assert!(maximum < 1e-25);
        }
    }
}

#[cfg(test)]
mod center_tests {
    use super::*;
    #[test]
    fn center_dot2_matches_independent_per_tap_dd_sum() {
        for d in [8., 14.7, 16.] {
            let f = CoarseIir::new(W::from(d));
            let input: Vec<_> = (0..300)
                .map(|i| {
                    C::new(
                        ((i * 31 % 71) as f64 - 35.) / 31.,
                        ((i * 11 % 97) as f64 - 48.) / 97.,
                    )
                })
                .collect();
            let mut maximum: f64 = 0.;
            for q in 100..132 {
                let mut expected = WC::real(0.);
                for lag in 0..(prefix::B / d).ceil() as usize {
                    let t = W::from(d) * W::from(lag as f64);
                    let window = prefix::turnon(t);
                    if window.value() != 0. {
                        let c = prefix::envelope(t.value()).scale(window * W::from(d));
                        expected = expected + c * wc(input[q - lag]);
                        if lag != 0 {
                            expected = expected + c.conj() * wc(input[q + lag]);
                        }
                    }
                }
                let actual = f.center_output(q as i128, &input, 0);
                maximum = maximum.max((actual - expected).value().abs());
            }
            println!(
                "turnon d={d} taps={} maximum error={maximum:e}",
                f.center.past.len()
            );
            assert!(maximum < 1e-27);
        }
    }
}
