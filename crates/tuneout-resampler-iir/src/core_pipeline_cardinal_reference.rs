//! Reference DD analysis/synthesis around the actual coarse IIR primitive.
//! This direct implementation is a correctness prototype. Production work can
//! share analysis/synthesis transforms with the finite-prefix DD FFT.
use crate::complex::{Complex as C, Wide as W, WideComplex as WC};
use crate::{coarse_iir, prefix};
use std::collections::VecDeque;
const BATCH: usize = 1024;
const ANALYSIS_HALF: usize = 20;
const ANALYSIS_BETA: f64 = 32.;
fn pi() -> W {
    W::from(std::f64::consts::PI) + W::from(1.2246467991473532e-16)
}
fn i0(x: W) -> W {
    let q = x * x / W::from(4.);
    let mut s = W::from(1.);
    let mut t = s;
    for k in 1..=160 {
        t = t * q / W::from((k * k) as f64);
        s = s + t;
        if t.value().abs() < s.value().abs() * 1e-34 {
            break;
        }
    }
    s
}
fn weight(x: W, half: usize, beta: f64, den: W) -> W {
    if x.value().abs() > half as f64 {
        return W::from(0.);
    }
    let sinc = if x.value() == 0. {
        W::from(1.)
    } else {
        prefix::exp(WC {
            re: W::from(0.),
            im: pi() * x,
        })
        .im / (pi() * x)
    };
    // sqrt is Newton-refined in DD from the binary64 starting point.
    let q = W::from(1.) - x * x / W::from((half * half) as f64);
    let root = if q.value() <= 0. {
        W::from(0.)
    } else {
        let a = W::from(q.value().sqrt());
        a + (q - a * a) / (W::from(2.) * a)
    };
    sinc * i0(root * W::from(beta)) / den
}
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r
    }
    a
}
#[inline]
fn parity(value: f64, index: i128) -> f64 {
    if index & 1 == 0 {
        value
    } else {
        -value
    }
}
fn rotation(index: i128, l: u32, k: u32) -> WC {
    let modulus = 2 * i128::from(k);
    let rem = (index * i128::from(l)).rem_euclid(modulus);
    if rem == 0 {
        return WC::real(1.);
    }
    if rem == i128::from(k) {
        return WC::real(-1.);
    }
    prefix::exp(WC {
        re: W::from(0.),
        im: pi() * W::from(rem as f64) / W::from(f64::from(k)),
    })
}
pub struct CorePipeline {
    input: u32,
    output: u32,
    l: u32,
    k: u32,
    r: usize,
    analysis_re: Vec<f64>,
    analysis_im: Vec<f64>,
    synthesis: Vec<f64>,
    period: u64,
    den: W,
    input_rotation: Vec<WC>,
    output_rotation: Vec<WC>,
    pub coarse: coarse_iir::CoarseIir,
}
#[derive(Default)]
pub struct PipelineState {
    output_clock: Option<(u64, i128, u64)>,
    low: VecDeque<C>,
    low_first: i128,
    q_re: Vec<f64>,
    q_im: Vec<f64>,
    q_head: usize,
    q_first: i128,
    coarse: coarse_iir::State,
}
impl CorePipeline {
    fn factor(input: u32, output: u32) -> usize {
        let a = input.min(output) as f64 / input as f64;
        let limit = (prefix::GRID / a).floor() as usize;
        1usize << (usize::BITS - 1 - limit.leading_zeros())
    }
    pub fn new(input: u32, output: u32) -> Self {
        let g = gcd(input as u64, output as u64) as u32;
        let l = output / g;
        let k = l.max(input / g);
        let r = Self::factor(input, output);
        let alpha = W::from(f64::from(l)) / W::from(f64::from(k));
        let step = alpha * W::from(r as f64);
        let half = prefix::HALF;
        let den = i0(W::from(prefix::BETA));
        let analysis_den = i0(W::from(ANALYSIS_BETA));
        let mut analysis = Vec::with_capacity(2 * ANALYSIS_HALF * r + 1);
        let mut norm = W::from(0.);
        if l == k && r == 1 {
            analysis.push(WC::real(1.));
            norm = W::from(1.);
        } else {
            for offset in -(ANALYSIS_HALF as i128 * r as i128)..=ANALYSIS_HALF as i128 * r as i128 {
                let w = weight(
                    W::from(offset as f64) / W::from(r as f64),
                    ANALYSIS_HALF,
                    ANALYSIS_BETA,
                    analysis_den,
                ) / W::from(r as f64);
                norm = norm + w;
                analysis.push(if l == k {
                    WC {
                        re: if offset & 1 == 0 { w } else { W::from(0.) - w },
                        im: W::from(0.),
                    }
                } else {
                    rotation(offset, l, k).conj().scale(w)
                });
            }
        }
        for z in &mut analysis {
            *z = z.scale(W::from(1.) / norm)
        }
        // The analytic real input analysis window is Hermitian. Construct its
        // mirror explicitly rather than retaining tiny DD exp(2*pi-i*x) versus
        // exp(-i*x) construction differences.
        for i in 0..analysis.len() / 2 {
            let other = analysis.len() - 1 - i;
            analysis[other] = analysis[i].conj();
        }
        let center = analysis.len() / 2;
        analysis[center].im = W::from(0.);
        let analysis_re: Vec<f64> = analysis.iter().map(|z| z.re.value()).collect();
        let analysis_im: Vec<f64> = if l == k {
            Vec::new()
        } else {
            analysis.iter().map(|z| z.im.value()).collect()
        };
        // Actual normalized analysis-component L1 is valid at every rate.
        // For the 128-tap Kaiser/sinc synthesis, |w| sums to at most
        // 2+2*H64/pi < 5.02 for any fractional phase. Use 8 for numerical
        // margin and unit carrier rotations, giving <=1e-23 global omission.
        let analysis_l1 = analysis_re
            .iter()
            .enumerate()
            .map(|(j, r)| r.abs() + analysis_im.get(j).map_or(0., |i| i.abs()))
            .sum::<f64>()
            * (1. + 1e-12);
        let budget = 1e-23 / (8. * analysis_l1);
        let coarse = coarse_iir::CoarseIir::with_pruning_budget(step, budget);
        let denominator = u64::from(output) * r as u64;
        let period = denominator / gcd(u64::from(input), denominator);
        let mut synthesis = Vec::new();
        if (period as u128) * (2 * half) as u128 * 16 <= 8_388_608 {
            let mut arguments = Vec::with_capacity(period as usize * 2 * half);
            for out in 1..period {
                let rem = (out as u128 * u128::from(input)) % u128::from(denominator);
                let u = W::from(rem as f64) / W::from(denominator as f64);
                for j in -(half as i64) + 1..=half as i64 {
                    arguments.push(u - W::from(j as f64));
                }
            }
            synthesis = vec![0.; 2 * half];
            synthesis[half - 1] = 1.;
            synthesis.extend(
                crate::setup_prefix::weights(&arguments, half, prefix::BETA, den).unwrap_or_else(
                    || {
                        arguments
                            .into_iter()
                            .map(|x| weight(x, half, prefix::BETA, den).value())
                            .collect()
                    },
                ),
            );
        }
        let ip = 2 * u64::from(k) / gcd(2 * u64::from(k), r as u64 * u64::from(l));
        let input_rotation = if l != k && ip <= 65536 {
            (0..ip)
                .map(|m| rotation(i128::from(m) * r as i128, l, k).conj())
                .collect()
        } else {
            Vec::new()
        };
        let op = 2 * u64::from(output) / gcd(2 * u64::from(output), u64::from(input.min(output)));
        let output_rotation = if input < output && op <= 65536 {
            (0..op)
                .map(|out| output_rotation(out, input, output))
                .collect()
        } else {
            Vec::new()
        };
        Self {
            input,
            output,
            l,
            k,
            r,
            analysis_re,
            analysis_im,
            synthesis,
            period,
            den,
            input_rotation,
            output_rotation,
            coarse,
        }
    }
    pub fn omitted_future_bound(&self) -> f64 {
        let gain = self
            .analysis_re
            .iter()
            .enumerate()
            .map(|(j, r)| r.abs() + self.analysis_im.get(j).map_or(0., |i| i.abs()))
            .sum::<f64>()
            * (1. + 1e-12);
        self.coarse.omitted_future_bound() * gain * 8.
    }
    #[inline]
    fn aligned_value(&self, out: u64, center: i128, value: f64) -> W {
        // rem==0 implies out*Fi/Fo=center*r. The carrier is then real even
        // for upsampling; downsampling always has exp(i*pi*out).
        let index = if self.input < self.output {
            center * self.r as i128
        } else {
            i128::from(out)
        };
        W::from(parity(value, index))
    }
    fn low_sample(&self, m: i128, window: &[f64], origin: i128) -> C {
        let center = m * self.r as i128;
        let p = (ANALYSIS_HALF * self.r) as i128;
        if self.l == self.k && self.r == 1 {
            let value =
                window[usize::try_from(center - origin).expect("retained coarse analysis center")];
            return C::new(parity(value, center), 0.);
        }
        let begin =
            usize::try_from(center - p - origin).expect("retained coarse analysis left edge");
        let samples = &window[begin..begin + self.analysis_re.len()];
        let half = self.analysis_re.len() / 2;
        if self.l == self.k {
            let re = crate::f64_symmetric_analysis::run_real(samples, &self.analysis_re[..half])
                + samples[half] * self.analysis_re[half];
            return C::new(parity(re, center), 0.);
        }
        let (re, im) = crate::f64_symmetric_analysis::run(
            samples,
            &self.analysis_re[..half],
            &self.analysis_im[..half],
        );
        let z = WC::from(C::new(re + samples[half] * self.analysis_re[half], im));
        let rot = if self.input_rotation.is_empty() {
            rotation(center, self.l, self.k).conj()
        } else {
            self.input_rotation[m.rem_euclid(self.input_rotation.len() as i128) as usize]
        };
        // This oracle isolates the original per-output clock from the candidate's
        // cached clock. Use its intentional ordinary input-rotation operator;
        // the separate independent WC test bounds the changed arithmetic.
        z.value() * rot.value()
    }
    fn ensure(
        &self,
        state: &mut PipelineState,
        first: i128,
        last: i128,
        window: &[f64],
        origin: i128,
    ) {
        if state.q_re.len() == state.q_head {
            // A period-one synthesis row needs only its center, but retain
            // the original coarse block origin and recurrence seed schedule.
            let anchor = if self.period == 1 {
                first - prefix::HALF as i128 + 1
            } else {
                first
            };
            state.q_first = anchor.div_euclid(BATCH as i128) * BATCH as i128;
        }
        let mut next = state.q_first + (state.q_re.len() - state.q_head) as i128;
        while next <= last {
            let lo = next - self.coarse.max_lag as i128;
            let hi = next + BATCH as i128 - 1 + self.coarse.max_lag as i128;
            if state.low.is_empty() {
                state.low_first = lo
            }
            while state.low_first < lo {
                state.low.pop_front().expect("monotone coarse window");
                state.low_first += 1;
            }
            let mut n = state.low_first + state.low.len() as i128;
            while n <= hi {
                state.low.push_back(self.low_sample(n, window, origin));
                n += 1;
            }
            let low = state.low.make_contiguous();
            let output = if self.input < self.output {
                // The reference retains its original analytic construction;
                // exp(i*pi*integer) has only a DD roundoff imaginary residue.
                // Match the new state coordinates while independently testing
                // this reference's rational synthesis and carrier clocks.
                for z in low.iter_mut() {
                    assert!(z.im.abs() < 1e-28);
                    z.im = 0.;
                }
                self.coarse
                    .process_real_input(&mut state.coarse, next, BATCH, low, state.low_first)
            } else {
                self.coarse
                    .process(&mut state.coarse, next, BATCH, low, state.low_first)
            };
            // Compact only at fixed coarse-block boundaries. The 128-tap
            // synthesis window stays contiguous and its logical dot lanes do
            // not depend on allocation capacity or VecDeque wrap position.
            if state.q_head != 0 {
                let remaining = state.q_re.len() - state.q_head;
                state.q_re.copy_within(state.q_head.., 0);
                state.q_re.truncate(remaining);
                if self.input < self.output {
                    state.q_im.copy_within(state.q_head.., 0);
                    state.q_im.truncate(remaining);
                }
                state.q_head = 0;
            }
            state.q_re.reserve(output.len());
            if self.input < self.output {
                state.q_im.reserve(output.len());
            }
            for z in output {
                let z = z.value();
                state.q_re.push(z.re);
                if self.input < self.output {
                    state.q_im.push(z.im);
                }
            }
            next += BATCH as i128;
        }
        let consumed = usize::try_from(first - state.q_first).expect("monotone synthesis clock");
        state.q_head += consumed;
        state.q_first = first;
        debug_assert!(
            self.input >= self.output && state.q_im.is_empty()
                || state.q_re.len() == state.q_im.len()
        );
        debug_assert!(state.q_head <= state.q_re.len());
        debug_assert!(state.q_re.len() - state.q_head <= 2 * prefix::HALF + BATCH);
    }
    /// Outputs sharing the same coarse center only extend/check the coarse
    /// stream once. Their exact rational clock advances in one integer step.
    /// The ordinary SIMD dots retain their original coefficient and sum order.
    pub fn output_block(
        &self,
        state: &mut PipelineState,
        first: u64,
        values: &mut [W],
        window: &[f64],
        origin: i128,
    ) {
        if self.synthesis.is_empty() {
            for (i, value) in values.iter_mut().enumerate() {
                *value = self.output(state, first + i as u64, window, origin);
            }
            return;
        }
        let finish = crate::real_carrier::select();
        let denominator = u64::from(self.output) * self.r as u64;
        let (mut center, mut rem) = match state.output_clock {
            Some((next, center, rem)) if next == first => (center, rem),
            _ => {
                let ticks = u128::from(first) * u128::from(self.input);
                (
                    (ticks / u128::from(denominator)) as i128,
                    (ticks % u128::from(denominator)) as u64,
                )
            }
        };
        // Clock division belongs to the block, not to each output group.
        let step = u64::from(self.input);
        let whole = step / denominator;
        let fraction = step % denominator;
        let outputs_per_center = denominator / step;
        let remainder_per_center = denominator % step;
        let mut phase = (first % self.period) as usize;
        let mut done = 0;
        while done < values.len() {
            // Following a complete center group rem<step. Only the first group
            // after a chunk boundary can need the general quotient.
            let same_center = if whole != 0 {
                1
            } else if rem < step {
                outputs_per_center + u64::from(rem < remainder_per_center)
            } else {
                (denominator - 1 - rem) / step + 1
            };
            let same_center = usize::try_from(same_center).unwrap_or(usize::MAX);
            let count = (values.len() - done).min(same_center);
            if self.period == 1 {
                self.ensure(state, center, center, window, origin);
            } else {
                self.ensure(
                    state,
                    center - prefix::HALF as i128 + 1,
                    center + prefix::HALF as i128,
                    window,
                    origin,
                );
            }
            for lane in 0..count {
                let at = first + done as u64 + lane as u64;
                let row = phase * 2 * prefix::HALF;
                values[done + lane] = if rem == 0 && lane == 0 {
                    let index = state.q_head
                        + if self.period == 1 {
                            0
                        } else {
                            prefix::HALF - 1
                        };
                    self.aligned_value(at, center, state.q_re[index])
                } else {
                    let range = state.q_head..state.q_head + 2 * prefix::HALF;
                    let weights = &self.synthesis[row..row + 2 * prefix::HALF];
                    if self.input >= self.output {
                        W::from(parity(
                            crate::f64_dot::dot(&state.q_re[range], weights),
                            i128::from(at),
                        ))
                    } else {
                        let (re, im) = crate::f64_pairdot::shared_coefficients(
                            &state.q_re[range.clone()],
                            &state.q_im[range],
                            weights,
                        );
                        let rot = if self.output_rotation.is_empty() {
                            output_rotation(at, self.input, self.output)
                        } else {
                            self.output_rotation[(at % self.output_rotation.len() as u64) as usize]
                        };
                        finish(re, im, rot)
                    }
                };
                phase += 1;
                if phase == self.period as usize {
                    phase = 0;
                }
            }
            // A complete center group crosses at most once. For decimation
            // count==1; for interpolation count is bounded by same_center.
            // Thus rem+count*fraction<2*denominator, including partial chunks.
            let next = rem + count as u64 * fraction;
            center += i128::from(count as u64 * whole);
            if next >= denominator {
                center += 1;
                rem = next - denominator;
            } else {
                rem = next;
            }
            done += count;
            state.output_clock = Some((first + done as u64, center, rem));
        }
    }
    pub fn output(&self, state: &mut PipelineState, out: u64, window: &[f64], origin: i128) -> W {
        let denominator = u64::from(self.output) * self.r as u64;
        let (center, rem) = match state.output_clock {
            Some((next, center, rem)) if next == out => (center, rem),
            _ => {
                let ticks = u128::from(out) * u128::from(self.input);
                (
                    (ticks / u128::from(denominator)) as i128,
                    (ticks % u128::from(denominator)) as u64,
                )
            }
        };
        // denominator<=16*max(input_rate,output_rate), because factor() is
        // the largest power of two not above16/alpha. The addition fits u64.
        let next = rem + u64::from(self.input);
        state.output_clock = out.checked_add(1).map(|out| {
            (
                out,
                center + i128::from(next / denominator),
                next % denominator,
            )
        });
        if self.period == 1 {
            self.ensure(state, center, center, window, origin);
        } else {
            self.ensure(
                state,
                center - prefix::HALF as i128 + 1,
                center + prefix::HALF as i128,
                window,
                origin,
            );
        }
        if rem == 0 {
            let index = state.q_head
                + if self.period == 1 {
                    0
                } else {
                    prefix::HALF - 1
                };
            return self.aligned_value(out, center, state.q_re[index]);
        }
        let row = if self.synthesis.is_empty() {
            None
        } else {
            Some((out % self.period) as usize * 2 * prefix::HALF)
        };
        let temporary;
        let weights = if let Some(row) = row {
            &self.synthesis[row..row + 2 * prefix::HALF]
        } else {
            let u = W::from(rem as f64) / W::from(denominator as f64);
            temporary = (0..2 * prefix::HALF)
                .map(|i| {
                    let j = i as i128 - prefix::HALF as i128 + 1;
                    weight(u - W::from(j as f64), prefix::HALF, prefix::BETA, self.den).value()
                })
                .collect::<Vec<_>>();
            &temporary[..]
        };
        let range = state.q_head..state.q_head + 2 * prefix::HALF;
        if self.input >= self.output {
            return W::from(parity(
                crate::f64_dot::dot(&state.q_re[range], weights),
                i128::from(out),
            ));
        }
        let (re, im) = crate::f64_pairdot::shared_coefficients(
            &state.q_re[range.clone()],
            &state.q_im[range],
            weights,
        );
        let z = WC::from(C::new(re, im));
        let rot = if self.output_rotation.is_empty() {
            output_rotation(out, self.input, self.output)
        } else {
            self.output_rotation[(out % self.output_rotation.len() as u64) as usize]
        };
        let value = z.value();
        crate::real_carrier::select()(value.re, value.im, rot)
    }
}

fn output_rotation(out: u64, input: u32, output: u32) -> WC {
    let lower = u64::from(input.min(output));
    let modulus = 2 * u64::from(output);
    let rem = (u128::from(out) * u128::from(lower)) % u128::from(modulus);
    if rem == 0 {
        return WC::real(1.);
    }
    if rem == u128::from(output) {
        return WC::real(-1.);
    }
    prefix::exp(WC {
        re: W::from(0.),
        im: pi() * W::from(rem as f64) / W::from(f64::from(output)),
    })
}
