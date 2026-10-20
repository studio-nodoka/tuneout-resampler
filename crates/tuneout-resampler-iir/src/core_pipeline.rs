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
pub(crate) fn i0(x: W) -> W {
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
pub(crate) fn weight(x: W, half: usize, beta: f64, den: W) -> W {
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
    if 2 * rem == i128::from(k) {
        return WC {
            re: W::from(0.),
            im: W::from(1.),
        };
    }
    if 2 * rem == 3 * i128::from(k) {
        return WC {
            re: W::from(0.),
            im: W::from(-1.),
        };
    }
    prefix::exp(WC {
        re: W::from(0.),
        im: pi() * W::from(rem as f64) / W::from(f64::from(k)),
    })
}
pub struct CorePipeline {
    projected: Option<crate::projected_synthesis::Plan>,
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
impl PipelineState {
    pub fn reset(&mut self) {
        self.output_clock = None;
        self.low.clear();
        self.q_re.clear();
        self.q_im.clear();
        self.q_head = 0;
        self.low_first = 0;
        self.q_first = 0;
        self.coarse = coarse_iir::State::default();
    }
    pub fn bytes(&self) -> usize {
        self.low.capacity() * 16
            + (self.q_re.capacity() + self.q_im.capacity()) * 8
            + self.coarse.bytes()
    }
    fn append_coarse(&mut self, complex: bool) {
        // Compact only at fixed coarse-block boundaries. Keep the original
        // contiguous synthesis window and coefficient/reduction lane alignment.
        if self.q_head != 0 {
            let remaining = self.q_re.len() - self.q_head;
            self.q_re.copy_within(self.q_head.., 0);
            self.q_re.truncate(remaining);
            if complex {
                self.q_im.copy_within(self.q_head.., 0);
                self.q_im.truncate(remaining);
            }
            self.q_head = 0;
        }
        let output = self.coarse.output();
        self.q_re.reserve(output.len());
        if complex {
            self.q_im.reserve(output.len());
        }
        for &z in output {
            let z = z.value();
            self.q_re.push(z.re);
            if complex {
                self.q_im.push(z.im);
            }
        }
    }
}
impl CorePipeline {
    fn factor(input: u32, output: u32) -> usize {
        let a = input.min(output) as f64 / input as f64;
        let limit = (prefix::GRID / a).floor() as usize;
        1usize << (usize::BITS - 1 - limit.leading_zeros())
    }
    pub fn lookahead_for(input: u32, output: u32) -> usize {
        let a = input.min(output) as f64 / input as f64;
        if (prefix::GRID / a).floor() > usize::MAX as f64 {
            return usize::MAX;
        }
        let r = Self::factor(input, output);
        let d = r as f64 * a;
        let maxlag = (13230. / d).ceil() as usize;
        maxlag
            .checked_add(ANALYSIS_HALF + prefix::HALF + BATCH)
            .and_then(|x| x.checked_mul(r))
            .unwrap_or(usize::MAX)
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
        let analysis_poly = crate::setup_kaiser::Kaiser::new(ANALYSIS_BETA, analysis_den);
        let mut analysis = Vec::with_capacity(2 * ANALYSIS_HALF * r + 1);
        let mut norm = W::from(0.);
        let analysis_sines: Vec<_> = (0..r)
            .map(|phase| {
                prefix::exp(WC {
                    re: W::from(0.),
                    im: pi() * W::from(phase as f64) / W::from(r as f64),
                })
                .im
            })
            .collect();
        let ag = gcd(2 * u64::from(k), u64::from(l));
        let ap = 2 * u64::from(k) / ag;
        let astep = u64::from(l) / ag;
        let analysis_roots = if l != k && ap <= 65536 {
            crate::setup_roots::unit_circle(ap as usize, ap as usize)
        } else {
            None
        };
        if l == k && r == 1 {
            analysis.push(WC::real(1.));
            norm = W::from(1.);
        } else {
            for offset in -(ANALYSIS_HALF as i128 * r as i128)..=ANALYSIS_HALF as i128 * r as i128 {
                let x = W::from(offset as f64) / W::from(r as f64);
                let rem = offset.rem_euclid(r as i128) as usize;
                // Preserve the old tiny integer-sinc residues. Fractional arguments
                // share sin(pi*u), with exact integer translation/sign identities.
                let numerator = if rem == 0 {
                    prefix::exp(WC {
                        re: W::from(0.),
                        im: pi() * x,
                    })
                    .im
                } else {
                    let value = analysis_sines[rem];
                    if offset.div_euclid(r as i128) & 1 == 0 {
                        value
                    } else {
                        W::from(0.) - value
                    }
                };
                let w = analysis_poly.as_ref().map_or_else(
                    || weight(x, ANALYSIS_HALF, ANALYSIS_BETA, analysis_den),
                    |p| p.weight_with_sine(x, ANALYSIS_HALF, Some(numerator)),
                ) / W::from(r as f64);
                norm = norm + w;
                analysis.push(if l == k {
                    WC {
                        re: if offset & 1 == 0 { w } else { W::from(0.) - w },
                        im: W::from(0.),
                    }
                } else {
                    analysis_roots
                        .as_ref()
                        .map_or_else(
                            || rotation(offset, l, k).conj(),
                            |roots| {
                                roots[(offset * i128::from(astep)).rem_euclid(i128::from(ap))
                                    as usize]
                            },
                        )
                        .scale(w)
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
            // Conjugate rational phases u and 1-u have reversed even-kernel
            // rows. Build one representative, retaining the original cache cap.
            let representatives = period / 2;
            let mut arguments = Vec::with_capacity(representatives as usize * 2 * half);
            let mut phase_sines = Vec::with_capacity(representatives as usize);
            for out in 1..=representatives {
                let rem = (out as u128 * u128::from(input)) % u128::from(denominator);
                let u = W::from(rem as f64) / W::from(denominator as f64);
                phase_sines.push(
                    prefix::exp(WC {
                        re: W::from(0.),
                        im: pi() * u,
                    })
                    .im,
                );
                for j in -(half as i64) + 1..=half as i64 {
                    arguments.push(u - W::from(j as f64));
                }
            }
            let rows = if let Some(poly) = crate::setup_kaiser::Kaiser::new(prefix::BETA, den) {
                poly.weights_phased(&arguments, half, &phase_sines)
            } else {
                crate::setup_prefix::weights(&arguments, half, prefix::BETA, den).unwrap_or_else(
                    || {
                        arguments
                            .into_iter()
                            .map(|x| weight(x, half, prefix::BETA, den).value())
                            .collect()
                    },
                )
            };
            let width = 2 * half;
            synthesis = vec![0.; period as usize * width];
            synthesis[half - 1] = 1.;
            for out in 1..=representatives {
                let first = (out as usize - 1) * width;
                let row = &rows[first..first + width];
                synthesis[out as usize * width..(out as usize + 1) * width].copy_from_slice(row);
                let mirror = (period - out) as usize;
                if mirror != out as usize {
                    for j in 0..width {
                        synthesis[mirror * width + j] = row[width - 1 - j];
                    }
                }
            }
        }
        let ip = 2 * u64::from(k) / gcd(2 * u64::from(k), r as u64 * u64::from(l));
        let input_rotation = if l != k && ip <= 65536 {
            let numerator =
                r as u64 * u64::from(l) / gcd(2 * u64::from(k), r as u64 * u64::from(l));
            crate::setup_roots::unit_circle(ip as usize, ip as usize)
                .map(|roots| {
                    (0..ip)
                        .map(|m| roots[((m * numerator) % ip) as usize])
                        .collect()
                })
                .unwrap_or_else(|| {
                    (0..ip)
                        .map(|m| rotation(i128::from(m) * r as i128, l, k).conj())
                        .collect()
                })
        } else {
            Vec::new()
        };
        let op = 2 * u64::from(output) / gcd(2 * u64::from(output), u64::from(input.min(output)));
        let output_rotation = if input < output && op <= 65536 {
            let numerator = u64::from(input) / gcd(2 * u64::from(output), u64::from(input));
            crate::setup_roots::unit_circle(op as usize, op as usize)
                .map(|roots| {
                    (0..op)
                        .map(|out| roots[((op - (out * numerator) % op) % op) as usize])
                        .collect()
                })
                .unwrap_or_else(|| {
                    (0..op)
                        .map(|out| output_rotation(out, input, output))
                        .collect()
                })
        } else {
            Vec::new()
        };
        let projected = crate::projected_synthesis::Plan::new(
            input,
            output,
            r,
            &synthesis,
            period,
            half,
            if input < output { op } else { 2 },
            |out| {
                if input < output && !output_rotation.is_empty() {
                    output_rotation[(out % output_rotation.len() as u64) as usize]
                } else {
                    self::output_rotation(out, input, output)
                }
            },
        );
        Self {
            projected,
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
    pub fn coefficient_bytes(&self) -> usize {
        self.projected.as_ref().map_or(0, |p| p.bytes())
            + (self.analysis_re.capacity() + self.analysis_im.capacity()) * 8
            + (self.input_rotation.capacity() + self.output_rotation.capacity()) * 32
            + self.synthesis.capacity() * 8
            + self.coarse.coefficient_bytes()
    }
    pub fn multiplications_per_input(&self) -> f64 {
        (2. * self.analysis_re.len() as f64
            + 20.
            + self.coarse.multiplications_per_coarse_frame(BATCH))
            / self.r as f64
            + (4. * prefix::HALF as f64 + 10.) * self.output as f64 / self.input as f64
    }
    pub fn future_seed_mode_frames(&self) -> usize {
        self.coarse.future_seed_mode_frames()
    }
    pub fn min_guard_input_frames(&self) -> usize {
        self.coarse.minimum_guard_frames() * self.r
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
    #[inline]
    fn synthesis_value(
        &self,
        out: u64,
        real: &[f64],
        imaginary: &[f64],
        weights: &[f64],
        finish: crate::real_carrier::Finish,
    ) -> W {
        if self.input >= self.output {
            return W::from(parity(crate::f64_dot::dot(real, weights), i128::from(out)));
        }
        let rot = if self.output_rotation.is_empty() {
            output_rotation(out, self.input, self.output)
        } else {
            self.output_rotation[(out % self.output_rotation.len() as u64) as usize]
        };
        // These zeros are assigned only by exact integer clock identities.
        // Quarter-turn carriers project one synthesis plane with a sign.
        if rot.im.parts() == (0., 0.) {
            let re = crate::f64_dot::dot(real, weights);
            return W::from(if rot.re.value() > 0. { re } else { -re });
        }
        if rot.re.parts() == (0., 0.) {
            let im = crate::f64_dot::dot(imaginary, weights);
            return W::from(if rot.im.value() > 0. { -im } else { im });
        }
        let (re, im) = crate::f64_pairdot::shared_coefficients(real, imaginary, weights);
        finish(re, im, rot)
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
        let z = C::new(re + samples[half] * self.analysis_re[half], im);
        let rot = if self.input_rotation.is_empty() {
            rotation(center, self.l, self.k).conj()
        } else {
            self.input_rotation[m.rem_euclid(self.input_rotation.len() as i128) as usize]
        };
        analysis_rotation(z, rot)
    }
    #[inline(always)]
    fn low_sample_stereo(
        &self,
        m: i128,
        windows: [&[f64]; 2],
        origin: i128,
        rotation_phase: usize,
    ) -> [C; 2] {
        let center = m * self.r as i128;
        if self.l == self.k && self.r == 1 {
            let index = usize::try_from(center - origin).expect("retained stereo analysis center");
            return windows.map(|x| C::new(parity(x[index], center), 0.));
        }
        let p = (self.analysis_re.len() / 2) as i128;
        let begin =
            usize::try_from(center - p - origin).expect("retained stereo analysis left edge");
        let x = windows.map(|x| &x[begin..begin + self.analysis_re.len()]);
        let half = self.analysis_re.len() / 2;
        if self.l == self.k {
            let z = crate::f64_stereo_analysis::run::<false>(x, &self.analysis_re[..half], &[]);
            return std::array::from_fn(|ch| {
                C::new(
                    parity(z[ch].re + x[ch][half] * self.analysis_re[half], center),
                    0.,
                )
            });
        }
        let z = crate::f64_stereo_analysis::run::<true>(
            x,
            &self.analysis_re[..half],
            &self.analysis_im[..half],
        );
        let rot = if self.input_rotation.is_empty() {
            rotation(center, self.l, self.k).conj()
        } else {
            self.input_rotation[rotation_phase]
        };
        std::array::from_fn(|ch| {
            analysis_rotation(
                C::new(z[ch].re + x[ch][half] * self.analysis_re[half], z[ch].im),
                rot,
            )
        })
    }
    fn ensure_stereo(
        &self,
        left: &mut PipelineState,
        right: &mut PipelineState,
        first: i128,
        last: i128,
        windows: [&[f64]; 2],
        origin: i128,
    ) {
        debug_assert_eq!(left.q_first, right.q_first);
        debug_assert_eq!(
            left.q_re.len() - left.q_head,
            right.q_re.len() - right.q_head
        );
        if left.q_re.len() == left.q_head {
            let anchor = if self.period == 1 {
                first - prefix::HALF as i128 + 1
            } else {
                first
            };
            let start = anchor.div_euclid(BATCH as i128) * BATCH as i128;
            left.q_first = start;
            right.q_first = start;
        }
        let mut next = left.q_first + (left.q_re.len() - left.q_head) as i128;
        while next <= last {
            let lo = next - self.coarse.max_lag as i128;
            let hi = next + BATCH as i128 - 1 + self.coarse.max_lag as i128;
            for state in [&mut *left, &mut *right] {
                if state.low.is_empty() {
                    state.low_first = lo;
                }
                while state.low_first < lo {
                    state
                        .low
                        .pop_front()
                        .expect("monotone stereo coarse window");
                    state.low_first += 1;
                }
            }
            debug_assert_eq!(left.low_first, right.low_first);
            debug_assert_eq!(left.low.len(), right.low.len());
            let mut n = left.low_first + left.low.len() as i128;
            let rotation_period = self.input_rotation.len();
            let mut rotation_phase = if rotation_period == 0 {
                0
            } else {
                n.rem_euclid(rotation_period as i128) as usize
            };
            while n <= hi {
                let z = self.low_sample_stereo(n, windows, origin, rotation_phase);
                left.low.push_back(z[0]);
                right.low.push_back(z[1]);
                n += 1;
                if rotation_period != 0 {
                    rotation_phase += 1;
                    if rotation_phase == rotation_period {
                        rotation_phase = 0;
                    }
                }
            }
            for state in [&mut *left, &mut *right] {
                let low = state.low.make_contiguous();
                if self.input >= self.output {
                    self.coarse
                        .process_real(&mut state.coarse, next, BATCH, low, state.low_first)
                } else {
                    self.coarse.process_real_input(
                        &mut state.coarse,
                        next,
                        BATCH,
                        low,
                        state.low_first,
                    )
                };
            }
            for state in [&mut *left, &mut *right] {
                state.append_coarse(self.input < self.output);
            }
            next += BATCH as i128;
        }
        let consumed =
            usize::try_from(first - left.q_first).expect("monotone stereo synthesis clock");
        for state in [left, right] {
            state.q_head += consumed;
            state.q_first = first;
            debug_assert!(
                self.input >= self.output && state.q_im.is_empty()
                    || state.q_re.len() == state.q_im.len()
            );
            debug_assert!(state.q_head <= state.q_re.len());
            debug_assert!(state.q_re.len() - state.q_head <= 2 * prefix::HALF + BATCH);
        }
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
            if self.input >= self.output {
                self.coarse
                    .process_real(&mut state.coarse, next, BATCH, low, state.low_first)
            } else {
                self.coarse
                    .process_real_input(&mut state.coarse, next, BATCH, low, state.low_first)
            };
            state.append_coarse(self.input < self.output);
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
            if let Some(projected) = &self.projected {
                let range = state.q_head..state.q_head + 2 * prefix::HALF;
                let imaginary = if self.input < self.output {
                    &state.q_im[range.clone()]
                } else {
                    &[]
                };
                projected.write_mono(
                    first + done as u64,
                    &state.q_re[range],
                    imaginary,
                    &mut values[done..done + count],
                );
                if rem == 0 {
                    values[done] = self.aligned_value(
                        first + done as u64,
                        center,
                        state.q_re[state.q_head + prefix::HALF - 1],
                    );
                }
                phase = (phase + count) % self.period as usize;
            } else {
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
                        let imaginary = if self.input < self.output {
                            &state.q_im[range.clone()]
                        } else {
                            &[]
                        };
                        self.synthesis_value(at, &state.q_re[range], imaginary, weights, finish)
                    };
                    phase += 1;
                    if phase == self.period as usize {
                        phase = 0;
                    }
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
    /// One rational clock, coefficient row and carrier lookup serves stereo.
    pub fn output_block_stereo(
        &self,
        states: &mut [PipelineState],
        first: u64,
        values: &mut [W],
        windows: [&[f64]; 2],
        origin: i128,
    ) {
        assert_eq!(values.len() % 2, 0);
        let [left, right] = states else {
            panic!("two stereo pipeline states")
        };
        let frames = values.len() / 2;
        if frames == 0 {
            return;
        }
        if self.synthesis.is_empty() {
            for i in 0..frames {
                let at = first + i as u64;
                values[2 * i] = self.output(left, at, windows[0], origin);
                values[2 * i + 1] = self.output(right, at, windows[1], origin);
            }
            return;
        }
        debug_assert_eq!(left.output_clock, right.output_clock);
        let finish = crate::real_carrier::select();
        let denominator = u64::from(self.output) * self.r as u64;
        let (mut center, mut rem) = match left.output_clock {
            Some((next, center, rem)) if next == first => (center, rem),
            _ => {
                let ticks = u128::from(first) * u128::from(self.input);
                (
                    (ticks / u128::from(denominator)) as i128,
                    (ticks % u128::from(denominator)) as u64,
                )
            }
        };
        let rotation_period = self.output_rotation.len();
        let mut rotation_phase = if rotation_period == 0 {
            0
        } else {
            (first % rotation_period as u64) as usize
        };
        let step = u64::from(self.input);
        let whole = step / denominator;
        let fraction = step % denominator;
        let outputs_per_center = denominator / step;
        let remainder_per_center = denominator % step;
        let mut phase = (first % self.period) as usize;
        let mut done = 0;
        while done < frames {
            let same_center = if whole != 0 {
                1
            } else if rem < step {
                outputs_per_center + u64::from(rem < remainder_per_center)
            } else {
                (denominator - 1 - rem) / step + 1
            };
            let count = (frames - done).min(usize::try_from(same_center).unwrap_or(usize::MAX));
            let (a, b) = if self.period == 1 {
                (center, center)
            } else {
                (
                    center - prefix::HALF as i128 + 1,
                    center + prefix::HALF as i128,
                )
            };
            self.ensure_stereo(left, right, a, b, windows, origin);
            if let Some(projected) = &self.projected {
                let lrange = left.q_head..left.q_head + 2 * prefix::HALF;
                let rrange = right.q_head..right.q_head + 2 * prefix::HALF;
                let lim = if self.input < self.output {
                    &left.q_im[lrange.clone()]
                } else {
                    &[]
                };
                let rim = if self.input < self.output {
                    &right.q_im[rrange.clone()]
                } else {
                    &[]
                };
                projected.write_stereo(
                    first + done as u64,
                    [&left.q_re[lrange], lim, &right.q_re[rrange], rim],
                    &mut values[2 * done..2 * (done + count)],
                );
                if rem == 0 {
                    let at = first + done as u64;
                    values[2 * done] =
                        self.aligned_value(at, center, left.q_re[left.q_head + prefix::HALF - 1]);
                    values[2 * done + 1] =
                        self.aligned_value(at, center, right.q_re[right.q_head + prefix::HALF - 1]);
                }
                phase = (phase + count) % self.period as usize;
                if rotation_period != 0 {
                    rotation_phase = (rotation_phase + count) % rotation_period;
                }
            } else {
                for lane in 0..count {
                    let at = first + done as u64 + lane as u64;
                    let pair = if rem == 0 && lane == 0 {
                        let offset = if self.period == 1 {
                            0
                        } else {
                            prefix::HALF - 1
                        };
                        [
                            self.aligned_value(at, center, left.q_re[left.q_head + offset]),
                            self.aligned_value(at, center, right.q_re[right.q_head + offset]),
                        ]
                    } else {
                        let row = phase * 2 * prefix::HALF;
                        let weights = &self.synthesis[row..row + 2 * prefix::HALF];
                        let lrange = left.q_head..left.q_head + 2 * prefix::HALF;
                        let rrange = right.q_head..right.q_head + 2 * prefix::HALF;
                        if self.input >= self.output {
                            let (l, r) = crate::f64_pairdot::shared_coefficients(
                                &left.q_re[lrange],
                                &right.q_re[rrange],
                                weights,
                            );
                            [
                                W::from(parity(l, i128::from(at))),
                                W::from(parity(r, i128::from(at))),
                            ]
                        } else {
                            let rot = if self.output_rotation.is_empty() {
                                output_rotation(at, self.input, self.output)
                            } else {
                                self.output_rotation[rotation_phase]
                            };
                            if rot.im.parts() == (0., 0.) {
                                let (l, r) = crate::f64_pairdot::shared_coefficients(
                                    &left.q_re[lrange],
                                    &right.q_re[rrange],
                                    weights,
                                );
                                if rot.re.value() > 0. {
                                    [W::from(l), W::from(r)]
                                } else {
                                    [W::from(-l), W::from(-r)]
                                }
                            } else if rot.re.parts() == (0., 0.) {
                                let (l, r) = crate::f64_pairdot::shared_coefficients(
                                    &left.q_im[lrange],
                                    &right.q_im[rrange],
                                    weights,
                                );
                                if rot.im.value() > 0. {
                                    [W::from(-l), W::from(-r)]
                                } else {
                                    [W::from(l), W::from(r)]
                                }
                            } else {
                                let p = crate::f64_stereo_synthesis::run(
                                    [
                                        &left.q_re[lrange.clone()],
                                        &left.q_im[lrange],
                                        &right.q_re[rrange.clone()],
                                        &right.q_im[rrange],
                                    ],
                                    weights,
                                );
                                [finish(p[0], p[1], rot), finish(p[2], p[3], rot)]
                            }
                        }
                    };
                    values[2 * (done + lane)] = pair[0];
                    values[2 * (done + lane) + 1] = pair[1];
                    phase += 1;
                    if phase == self.period as usize {
                        phase = 0;
                    }
                    if rotation_period != 0 {
                        rotation_phase += 1;
                        if rotation_phase == rotation_period {
                            rotation_phase = 0;
                        }
                    }
                }
            }
            let next = rem + count as u64 * fraction;
            center += i128::from(count as u64 * whole);
            if next >= denominator {
                center += 1;
                rem = next - denominator;
            } else {
                rem = next;
            }
            done += count;
            let clock = Some((first + done as u64, center, rem));
            left.output_clock = clock;
            right.output_clock = clock;
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
        if let Some(projected) = &self.projected {
            let range = state.q_head..state.q_head + 2 * prefix::HALF;
            let imaginary = if self.input < self.output {
                &state.q_im[range.clone()]
            } else {
                &[]
            };
            let mut value = [W::from(0.)];
            projected.write_mono(out, &state.q_re[range], imaginary, &mut value);
            return value[0];
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
        let imaginary = if self.input < self.output {
            &state.q_im[range.clone()]
        } else {
            &[]
        };
        self.synthesis_value(
            out,
            &state.q_re[range],
            imaginary,
            weights,
            crate::real_carrier::select(),
        )
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
    if 2 * rem == u128::from(output) {
        return WC {
            re: W::from(0.),
            im: W::from(1.),
        };
    }
    if 2 * rem == 3 * u128::from(output) {
        return WC {
            re: W::from(0.),
            im: W::from(-1.),
        };
    }
    prefix::exp(WC {
        re: W::from(0.),
        im: pi() * W::from(rem as f64) / W::from(f64::from(output)),
    })
}

#[cfg(test)]
#[path = "core_pipeline_cardinal_reference.rs"]
mod cardinal_reference;
#[cfg(test)]
#[path = "core_pipeline_cardinal_tests.rs"]
mod cardinal_tests;

#[cfg(test)]
mod matrix_tests {
    use super::*;
    #[test]
    fn matrix_synthesis_matches_original_outputs_and_clock() {
        for (input, output) in [
            (44100, 48000),
            (96000, 44100),
            (192000, 44100),
            (48000, 96000),
        ] {
            let p = CorePipeline::new(input, output);
            let count = 1024;
            let end = CorePipeline::lookahead_for(input, output);
            let origin = -(end as i128);
            let window: Vec<_> = (0..2 * end + 8192)
                .map(|i| ((i * 31 % 71) as f64 - 35.) / 71.)
                .collect();
            let mut a = PipelineState::default();
            let mut b = PipelineState::default();
            let mut maximum: f64 = 0.;
            let mut total = 0;
            for size in [13, 1, count, 57, 511] {
                let mut values = vec![W::from(0.); size];
                p.output_block(&mut a, total, &mut values, &window, origin);
                for (i, v) in values.iter().enumerate() {
                    let expected = p.output(&mut b, total + i as u64, &window, origin);
                    maximum = maximum.max((*v - expected).value().abs());
                    assert_eq!(v.parts(), expected.parts());
                }
                total += size as u64;
                assert_eq!(a.output_clock, b.output_clock);
                assert_eq!(a.q_first, b.q_first);
            }
            println!("{input}->{output}: max matrix error {maximum:e}");
            assert!(maximum < 2e-17);
        }
    }
}

#[cfg(test)]
mod symmetric_analysis_tests {
    use super::*;
    #[test]
    fn analysis_symmetry_is_exact_and_pairing_error_is_small() {
        for (input, output) in [
            (44100, 48000),
            (96000, 44100),
            (192000, 44100),
            (48000, 44117),
        ] {
            let p = CorePipeline::new(input, output);
            let n = p.analysis_re.len();
            let h = n / 2;
            let imaginary = if p.analysis_im.is_empty() {
                vec![0.; n]
            } else {
                p.analysis_im.clone()
            };
            for i in 0..h {
                assert_eq!(
                    p.analysis_re[i].to_bits(),
                    p.analysis_re[n - 1 - i].to_bits()
                );
                assert_eq!(imaginary[i], -imaginary[n - 1 - i]);
            }
            assert_eq!(imaginary[h], 0.);
            let x: Vec<_> = (0..n).map(|i| ((i * 31 % 71) as f64 - 35.) / 31.).collect();
            let (re, im) =
                crate::f64_symmetric_analysis::run(&x, &p.analysis_re[..h], &imaginary[..h]);
            let re = re + x[h] * p.analysis_re[h];
            let (ar, ai) = crate::f64_pairdot::shared_input(&x, &p.analysis_re, &imaginary);
            let e = (re - ar).hypot(im - ai);
            println!("{input}->{output} taps{n} paired error{e:e}");
            assert!(e < 1e-14);
        }
    }
}

#[cfg(test)]
mod setup_weight_tests {
    use super::*;
    #[test]
    fn vector_weight_matches_scalar_coefficient_bits() {
        let den = i0(W::from(prefix::BETA));
        let mut arguments = Vec::new();
        for phase in 0..257 {
            let u = W::from(phase as f64) / W::from(257.);
            for j in -(prefix::HALF as i64) + 1..=prefix::HALF as i64 {
                arguments.push(u - W::from(j as f64));
            }
        }
        arguments.extend([
            W::from(0.),
            W::from(-64.),
            W::from(64.),
            W::from(-64.1),
            W::from(64.1),
        ]);
        let Some(actual) =
            crate::setup_prefix::weights(&arguments, prefix::HALF, prefix::BETA, den)
        else {
            return;
        };
        for (i, (a, &x)) in actual.iter().zip(&arguments).enumerate() {
            let b = weight(x, prefix::HALF, prefix::BETA, den).value();
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "i={i} x={} actual={a:e} expected={b:e}",
                x.value()
            );
        }
    }
}

#[cfg(test)]
mod setup_cross_tests {
    use super::*;
    #[test]
    fn setup_cross_numerical_weights() {
        let den = i0(W::from(prefix::BETA));
        let mut arguments = Vec::new();
        for phase in [0, 1, 2, 3, 5, 7, 9, 15, 16] {
            let u = W::from(phase as f64) / W::from(17.);
            for j in -(prefix::HALF as i64) + 1..=prefix::HALF as i64 {
                arguments.push(u - W::from(j as f64));
            }
        }
        arguments.extend([
            W::from(0.),
            W::from(-64.),
            W::from(64.),
            W::from(-64.1),
            W::from(64.1),
        ]);
        let Some(actual) =
            crate::setup_prefix::weights(&arguments, prefix::HALF, prefix::BETA, den)
        else {
            return;
        };
        for (a, &x) in actual.iter().zip(&arguments) {
            let b = weight(x, prefix::HALF, prefix::BETA, den).value();
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[cfg(test)]
mod stereo_pipeline_exact_tests {
    use super::*;
    #[test]
    fn stereo_synthesis_keeps_both_independent_mono_clocks_and_dd_parts() {
        for (input, output) in [
            (44100, 48000),
            (48000, 44100),
            (96000, 44100),
            (192000, 44100),
            (48000, 96000),
            (48000, 44117),
        ] {
            let p = CorePipeline::new(input, output);
            let guard = CorePipeline::lookahead_for(input, output);
            for first in [0u64, 137] {
                let count = 257usize;
                let origin = -(guard as i128);
                let total = 2 * guard
                    + ((first + count as u64) * u64::from(input) / u64::from(output)) as usize
                    + 1024;
                let windows: [Vec<f64>; 2] = std::array::from_fn(|ch| {
                    (0..total)
                        .map(|j| ((j * (31 + 2 * ch) % 127) as f64 - 63.) / 127.)
                        .collect()
                });
                let mut stereo = [PipelineState::default(), PipelineState::default()];
                let mut mono = [PipelineState::default(), PipelineState::default()];
                let mut done = 0;
                for n in [1usize, 3, 17, 2, 64, 5, 113, 52] {
                    let mut actual = vec![W::from(0.); n * 2];
                    p.output_block_stereo(
                        &mut stereo,
                        first + done as u64,
                        &mut actual,
                        [&windows[0], &windows[1]],
                        origin,
                    );
                    for ch in 0..2 {
                        let mut expected = vec![W::from(0.); n];
                        p.output_block(
                            &mut mono[ch],
                            first + done as u64,
                            &mut expected,
                            &windows[ch],
                            origin,
                        );
                        for (i, want) in expected.iter().enumerate() {
                            assert_eq!(
                                actual[2 * i + ch].parts(),
                                want.parts(),
                                "{input}->{output} first{first} offset{} channel{ch}",
                                done + i
                            );
                        }
                        assert_eq!(stereo[ch].output_clock, mono[ch].output_clock);
                        assert_eq!(stereo[ch].q_first, mono[ch].q_first);
                        assert_eq!(stereo[ch].q_head, mono[ch].q_head);
                    }
                    done += n;
                }
                assert_eq!(done, count);
            }
        }
    }
}

#[cfg(test)]
mod projected_model_tests {
    use super::*;
    #[test]
    fn projected_phase_tables_match_independent_dd_operator_with_normwise_bound() {
        for (input, output) in [
            (44100, 48000),
            (48000, 44100),
            (96000, 44100),
            (192000, 44100),
            (48000, 96000),
            (32000, 48000),
            (48000, 44117),
            (48000, 48001),
        ] {
            let p = CorePipeline::new(input, output);
            let Some(projected) = &p.projected else {
                assert!(p.synthesis.is_empty() || p.period == 1);
                continue;
            };
            let n = 2 * prefix::HALF;
            let re: Vec<_> = (0..n)
                .map(|j| ((j * 37 % 127) as f64 - 63.) / 127.)
                .collect();
            let im: Vec<_> = (0..n)
                .map(|j| ((j * 41 % 131) as f64 - 65.) / 131.)
                .collect();
            let carrier_period = if input < output {
                2 * u64::from(output) / gcd(2 * u64::from(output), u64::from(input.min(output)))
            } else {
                2
            };
            let period = p.period / gcd(p.period, carrier_period) * carrier_period;
            let mut maximum: f64 = 0.;
            let mut bound_max: f64 = 0.;
            for out in 0..period {
                let weights =
                    &p.synthesis[(out % p.period) as usize * n..(out % p.period) as usize * n + n];
                let rotation = output_rotation(out, input, output);
                let mut expected = WC::real(0.);
                let mut norm = 0.;
                for j in 0..n {
                    let q = WC::from(C::new(re[j], if input < output { im[j] } else { 0. }));
                    expected = expected + q.scale(W::from(weights[j]));
                    norm += weights[j].abs()
                        * (re[j].abs() * rotation.re.value().abs()
                            + if input < output {
                                im[j].abs() * rotation.im.value().abs()
                            } else {
                                0.
                            });
                }
                let expected = (expected * rotation).re;
                let mut actual = [W::from(0.)];
                projected.write_mono(
                    out,
                    &re,
                    if input < output { &im } else { &[] },
                    &mut actual,
                );
                let error = (actual[0] - expected).value().abs();
                let operations = (8 * n + 64) as f64;
                let gamma = operations * f64::EPSILON / (1. - operations * f64::EPSILON);
                let bound = gamma * norm + f64::MIN_POSITIVE;
                assert!(
                    error <= bound,
                    "{input}->{output} out{out} error{error:e} bound{bound:e}"
                );
                maximum = maximum.max(error);
                bound_max = bound_max.max(bound);
            }
            println!("PROJECTED_MODEL,{input},{output},period{period},width{},maxDDerror{maximum:e},normwise_bound{bound_max:e},cache_bytes{}",crate::projected_synthesis::WIDTH,projected.bytes());
            assert!(p.synthesis.capacity() * 8 + projected.bytes() <= 8_388_608);
        }
    }
}

// The coarse analysis result is already ordinary f64. Retain the full DD
// carrier table, round its components once and rotate in ordinary arithmetic.
// This removes DD product residuals from the per-input hot path.
#[inline(always)]
fn analysis_rotation(value: C, rotation: WC) -> C {
    value * rotation.value()
}
#[cfg(test)]
mod ordinary_rotation_tests {
    use super::*;
    #[test]
    fn ordinary_input_rotation_has_a_separate_dd_error_bound() {
        for phase in 0..=640 {
            let rot = rotation(phase, 147, 640);
            for scale in [1e-100, 1., 1e100] {
                for value in [
                    C::new(0., 0.),
                    C::new(0., -scale),
                    C::new(scale, 0.),
                    C::new(scale, -0.71 * scale),
                ] {
                    let actual = WC::from(analysis_rotation(value, rot));
                    let expected = WC::from(value) * rot;
                    let norm = (value.re.abs() + value.im.abs())
                        * (rot.re.value().abs() + rot.im.value().abs());
                    let error = (actual - expected).value();
                    assert!(error.re.abs() + error.im.abs() <= 16. * f64::EPSILON * norm);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "setup_synthesis_symmetry_tests.rs"]
mod setup_synthesis_symmetry_tests;
