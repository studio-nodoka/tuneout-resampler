//! Actual coarse-grid causal/anticausal IIR, plus a finite coarse complex FIR.
//! Significant untapered core modes remain infinite recurrences; eight selected
//! slow modes use finite recursive intervals from the accepted analytic model.
use crate::complex::{Complex as C, Wide as W, WideComplex as WC};
use crate::late_tail_coefficients as tail;
use crate::prefix;
use crate::simd::Planes;
#[derive(Clone)]
struct Mode {
    start: usize,
    end: Option<usize>,
    guard: usize,
    p: C,
    coefficient: WC,
}
struct SeedGroup {
    guard: usize,
    #[cfg(test)]
    poles: Vec<C>,
    #[cfg(test)]
    reverse_poles: Vec<C>,
    split_poles: Planes,
    split_reverse_poles: Planes,
    indices: Vec<usize>,
}
struct Correction {
    first: usize,
    re: Vec<f64>,
    im: Vec<f64>,
}
struct Center {
    past_first: usize,
    past: Vec<WC>,
    future_first: usize,
    future: Vec<WC>,
}
#[cfg(test)]
struct OrdinaryRow {
    re: Vec<C>,
}
#[cfg(test)]
impl OrdinaryRow {
    fn new(v: &[WC]) -> Self {
        Self {
            re: v.iter().map(|c| c.value()).collect(),
        }
    }
    fn dot(&self, k: crate::simd::Kernel, v: &[C]) -> WC {
        WC::from(k.coarse_complex_dot(&self.re, v))
    }
}
pub struct CoarseIir {
    past_split: Planes,
    future_split: Planes,
    poles_split: Planes,
    reverse_poles_split: Planes,
    kernel: crate::simd::Kernel,
    core_start: usize,
    seed_groups: Vec<SeedGroup>,
    #[cfg(test)]
    center_past_ordinary: OrdinaryRow,
    #[cfg(test)]
    center_future_ordinary: OrdinaryRow,
    omitted_core_l1: f64,
    #[cfg(test)]
    correction: Correction,
    fft_correction: crate::coarse_fft::Plan,
    #[cfg(test)]
    quantization_l1: f64,
    #[cfg(test)]
    quantization_bound: f64,
    modes: Vec<Mode>,
    center_taps: usize,
    #[cfg(test)]
    center: Center,
    pub max_lag: usize,
}
#[derive(Default)]
pub struct State {
    past: Planes,
    future: Planes,
    seed: Planes,
    resident: Vec<C>,
    output: Vec<WC>,
    real_input: bool,
    next: Option<i128>,
    fft_correction: crate::coarse_fft::State,
}
impl State {
    pub fn bytes(&self) -> usize {
        self.past.bytes()
            + self.future.bytes()
            + self.seed.bytes()
            + self.resident.capacity() * 16
            + self.output.capacity() * std::mem::size_of::<WC>()
            + self.fft_correction.bytes()
    }
    pub fn output(&self) -> &[WC] {
        &self.output
    }
}
fn wc(v: C) -> WC {
    WC {
        re: W::from(v.re),
        im: W::from(v.im),
    }
}
impl CoarseIir {
    pub fn new(step: W) -> Self {
        let d = step.value();
        let start = (prefix::B / d).ceil() as usize;
        let pi = W::from(std::f64::consts::PI) + W::from(1.2246467991473532e-16);
        let selected = [34, 35, 36, 37, 38, 39, 41, 46];
        let mut modes = Vec::new();
        let mut omitted_core_l1 = 0.;
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
                omitted_core_l1 +=
                    2. * (rh.abs() + rl.abs() + ih.abs() + il.abs()) * (1. + 1e-12) / (1. - radius);
                continue;
            }
            modes.push(Mode {
                start,
                end,
                guard,
                p,
                coefficient,
            });
        }
        // The eight finite slow core intervals join their cancelling finite
        // correction in one FFT FIR. Every untapered infinite mode remains a
        // material recursive branch; no arbitrary infinite tail is appended.
        let finite_core: Vec<_> = modes.iter().filter(|m| m.end.is_some()).cloned().collect();
        modes.retain(|m| m.end.is_none());
        // A bounded upward guard bucket lets neighboring modes share SIMD seed
        // intervals. The original omitted-state bound only improves; padding
        // fits inside the already admitted finite branch whenever possible.
        let admitted = (13230. / d).ceil() as usize;
        for m in &mut modes {
            let padded = m.guard.div_ceil(128) * 128;
            m.guard = padded.min(admitted.saturating_sub(m.start)).max(m.guard);
        }
        // Finite late correction is sampled from the identical generic modal
        // model in DD, then evaluated as a coarse ordinary complex FIR. The
        // significant retained core modes remain actual IIR recurrences.
        let selected: Vec<_> = tail::MODES
            .iter()
            .filter(|&&(_, b, _, si, _, _)| si > std::f64::consts::PI / 2. && b <= 13230)
            .collect();
        let first = selected
            .iter()
            .map(|&&(a, _, _, _, _, _)| (a as f64 / d).ceil() as usize)
            .min()
            .unwrap_or(0);
        let last = selected
            .iter()
            .map(|&&(_, b, _, _, _, _)| (b as f64 / d).ceil() as usize)
            .max()
            .unwrap_or(0);
        let mut coefficients = vec![WC::real(0.); last - first];
        // Contiguous modes with one interval share an exact DD SIMD
        // trajectory batch. Each tap keeps the original mode-addition order.
        let mut next = 0;
        while next < selected.len() {
            let (a, b, _, _, _, _) = *selected[next];
            let mut stop = next + 1;
            while stop < selected.len() && selected[stop].0 == a && selected[stop].1 == b {
                stop += 1;
            }
            let begin = (a as f64 / d).ceil() as usize;
            let end = (b as f64 / d).ceil() as usize;
            let mut initial = Vec::with_capacity(stop - next);
            let mut poles = Vec::with_capacity(stop - next);
            for &&(a, _, sr, si, cr, ci) in &selected[next..stop] {
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
                initial.push((c * prefix::exp(s.scale(offset))).scale(step));
                poles.push(pole);
            }
            crate::setup_modal::add_causal(
                &mut coefficients[begin - first..end - first],
                &initial,
                &poles,
            );
            next = stop;
        }
        let re: Vec<_> = coefficients.iter().map(|z| z.re.value()).collect();
        let im: Vec<_> = coefficients.iter().map(|z| z.im.value()).collect();
        #[cfg(test)]
        let quantization_bound = 2. * coefficient_rounding_bound(&coefficients);
        #[cfg(test)]
        let quantization_l1 = 2.
            * coefficients
                .iter()
                .zip(re.iter().zip(&im))
                .map(|(c, (&re, &im))| {
                    let e = *c - WC::from(C::new(re, im));
                    let (rh, rl) = e.re.parts();
                    let (ih, il) = e.im.parts();
                    rh.abs() + rl.abs() + ih.abs() + il.abs()
                })
                .sum::<f64>()
            * (1. + 1e-12);
        let correction = Correction { first, re, im };
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
        let past_coefficients: Vec<_> = modes.iter().map(|m| m.coefficient.value()).collect();
        let future_coefficients: Vec<_> =
            modes.iter().map(|m| m.coefficient.conj().value()).collect();
        #[cfg(test)]
        let center_past_ordinary = OrdinaryRow::new(&center.past);
        #[cfg(test)]
        let center_future_ordinary = OrdinaryRow::new(&center.future);
        let max_lag = max_lag.max(last);
        let mut fir_wide = vec![WC::real(0.); 2 * max_lag + 1];
        for j in 0..correction.re.len() {
            let lag = correction.first + j;
            let c = WC::from(C::new(correction.re[j], correction.im[j]));
            fir_wide[max_lag + lag] = fir_wide[max_lag + lag] + c;
            if lag != 0 {
                fir_wide[max_lag - lag] = fir_wide[max_lag - lag] + c.conj();
            }
        }
        if let Some(first_mode) = finite_core.first() {
            if finite_core
                .iter()
                .all(|m| m.start == first_mode.start && m.end == first_mode.end)
            {
                let initial: Vec<_> = finite_core.iter().map(|m| m.coefficient).collect();
                let poles: Vec<_> = finite_core.iter().map(|m| WC::from(m.p)).collect();
                crate::setup_modal::add_finite(
                    &mut fir_wide,
                    max_lag,
                    first_mode.start,
                    first_mode.end.unwrap(),
                    &initial,
                    &poles,
                );
            } else {
                // Future model revisions may use different finite intervals.
                for m in &finite_core {
                    let mut c = m.coefficient;
                    let pole = WC::from(m.p);
                    for lag in m.start..m.end.unwrap() {
                        fir_wide[max_lag + lag] = fir_wide[max_lag + lag] + c;
                        if lag != 0 {
                            fir_wide[max_lag - lag] = fir_wide[max_lag - lag] + c.conj();
                        }
                        c = c * pole;
                    }
                }
            }
        }
        // Fold the exact existing finite turnon center into the same finite
        // convolution as the eight finite modal intervals and late correction.
        // All admitted infinite modes remain material IIR recurrences.
        for (j, &c) in center.past.iter().enumerate() {
            let lag = center.past_first - j;
            fir_wide[max_lag + lag] = fir_wide[max_lag + lag] + c;
        }
        for (j, &c) in center.future.iter().enumerate() {
            let lag = center.future_first + j;
            fir_wide[max_lag - lag] = fir_wide[max_lag - lag] + c;
        }
        // Setup-only redistribution of the complete existing coarse kernel.
        // The retained infinite modes, guards and all processing loops stay unchanged.
        let shape_end = (13230. / d).ceil() as usize;
        // Only nonnegative lags are read by the fixed shape correction.
        // Keep the existing modal addition order while skipping unused mirrors.
        let mut infinite_impulse = vec![WC::real(0.); shape_end];
        if let Some(first_mode) = modes.first() {
            assert!(modes.iter().all(|m| m.start == first_mode.start));
            let initial: Vec<_> = modes.iter().map(|m| m.coefficient).collect();
            let poles: Vec<_> = modes.iter().map(|m| WC::from(m.p)).collect();
            crate::setup_modal::add_causal(
                &mut infinite_impulse[first_mode.start..shape_end],
                &initial,
                &poles,
            );
        }
        crate::cosine_shape::apply(&mut fir_wide, &infinite_impulse, max_lag, shape_end, step);
        crate::joint_edge::apply(&mut fir_wide, max_lag, shape_end, step);
        let fir: Vec<_> = fir_wide.iter().map(|z| z.value()).collect();
        #[cfg(test)]
        let quantization_bound = quantization_bound + coefficient_rounding_bound(&fir_wide);
        #[cfg(test)]
        let quantization_l1 = quantization_l1
            + fir_wide
                .iter()
                .zip(&fir)
                .map(|(c, &z)| {
                    let e = *c - WC::from(z);
                    let (rh, rl) = e.re.parts();
                    let (ih, il) = e.im.parts();
                    rh.abs() + rl.abs() + ih.abs() + il.abs()
                })
                .sum::<f64>()
                * (1. + 1e-12);
        let fft_correction = crate::coarse_fft::Plan::new(&fir, max_lag);
        let poles: Vec<C> = modes.iter().map(|m| m.p).collect();
        let reverse_poles: Vec<C> = modes.iter().map(|m| m.p.conj()).collect();
        let core_start = modes.first().map_or(0, |m| m.start);
        assert!(modes
            .iter()
            .all(|m| m.end.is_none() && m.start == core_start));
        let mut by_guard = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for (j, m) in modes.iter().enumerate() {
            by_guard.entry(m.guard).or_default().push(j);
        }
        let seed_groups = by_guard
            .into_iter()
            .map(|(guard, indices)| {
                let poles: Vec<_> = indices.iter().map(|&j| modes[j].p).collect();
                let reverse_poles: Vec<_> = indices.iter().map(|&j| modes[j].p.conj()).collect();
                SeedGroup {
                    guard,
                    split_poles: Planes::from_complex(&poles),
                    split_reverse_poles: Planes::from_complex(&reverse_poles),
                    #[cfg(test)]
                    poles,
                    #[cfg(test)]
                    reverse_poles,
                    indices,
                }
            })
            .collect();
        Self {
            past_split: Planes::from_complex(&past_coefficients),
            future_split: Planes::from_complex(&future_coefficients),
            poles_split: Planes::from_complex(&poles),
            reverse_poles_split: Planes::from_complex(&reverse_poles),
            kernel: crate::simd::Kernel::coarse_detect(),
            core_start,
            seed_groups,
            #[cfg(test)]
            center_past_ordinary,
            #[cfg(test)]
            center_future_ordinary,
            omitted_core_l1,
            #[cfg(test)]
            correction,
            fft_correction,
            #[cfg(test)]
            quantization_l1,
            #[cfg(test)]
            quantization_bound,
            modes,
            center_taps: center.past.len() + center.future.len(),
            #[cfg(test)]
            center,
            max_lag,
        }
    }
    /// Full finite FIR coefficients are retained; this argument is accepted
    /// for wrapper compatibility and does not prune the operator.
    pub fn with_pruning_budget(step: W, _budget: f64) -> Self {
        Self::new(step)
    }
    pub fn infinite_mode_count(&self) -> usize {
        self.modes.len()
    }
    pub fn finite_core_mode_count(&self) -> usize {
        // Finite core intervals were absorbed into fft_correction at setup.
        0
    }
    pub fn finite_correction_mode_count(&self) -> usize {
        0
    }
    pub fn center_taps(&self) -> usize {
        self.center_taps
    }
    pub fn minimum_guard_frames(&self) -> usize {
        self.modes.iter().map(|m| m.guard).min().unwrap_or(0)
    }
    pub fn omitted_future_bound(&self) -> f64 {
        self.omitted_core_l1
            + self
                .modes
                .iter()
                .map(|m| {
                    let (rh, rl) = m.coefficient.re.parts();
                    let (ih, il) = m.coefficient.im.parts();
                    let radius = (m.p.abs() + 4. * f64::EPSILON).min(1.);
                    (rh.abs() + rl.abs() + ih.abs() + il.abs())
                        * (1. + 1e-12)
                        * radius.powf(m.guard as f64)
                        / (1. - radius)
                })
                .sum::<f64>()
    }
    /// Multiply-equivalent estimate, with each FMA counted once. Includes
    /// ordinary complex FIR dots, core updates and DD weighting, center Dot2,
    /// and amortized future seeds; excludes copies, allocation and construction.
    pub fn multiplications_per_coarse_frame(&self, batch: usize) -> f64 {
        (16 * self.modes.len()
            + 8 * self.finite_core_mode_count()
            + 8 * self.center_taps()
            + (4 * 1024 * 10 + 4 * 1024 * self.fft_correction.active_partition_count()) / 512)
            as f64
            + 4. * self.future_seed_mode_frames() as f64 / batch as f64
    }
    #[cfg(test)]
    pub fn mode_count(&self) -> usize {
        self.modes.len()
    }
    #[cfg(test)]
    pub fn correction_taps(&self) -> usize {
        self.correction.re.len()
    }
    pub fn coefficient_bytes(&self) -> usize {
        let split_bytes = self.past_split.bytes()
            + self.future_split.bytes()
            + self.poles_split.bytes()
            + self.reverse_poles_split.bytes()
            + self
                .seed_groups
                .iter()
                .map(|g| g.split_poles.bytes() + g.split_reverse_poles.bytes())
                .sum::<usize>();
        let reference_bytes = {
            #[cfg(test)]
            {
                (self.center_past_ordinary.re.capacity()
                    + self.center_future_ordinary.re.capacity())
                    * 16
                    + (self.correction.re.capacity() + self.correction.im.capacity()) * 8
                    + (self.center.past.capacity() + self.center.future.capacity()) * 32
            }
            #[cfg(not(test))]
            {
                0
            }
        };
        split_bytes
            + self.fft_correction.coefficient_bytes()
            + self.modes.capacity() * std::mem::size_of::<Mode>()
            + reference_bytes
    }
    #[cfg(test)]
    fn center_output(&self, q: i128, input: &[C], origin: i128) -> WC {
        let center = &self.center;
        let mut value = WC::real(0.);
        if !center.past.is_empty() {
            let first = usize::try_from(q - center.past_first as i128 - origin)
                .expect("retained turnon past");
            value = self
                .center_past_ordinary
                .dot(self.kernel, &input[first..first + center.past.len()]);
        }
        if !center.future.is_empty() {
            let first = usize::try_from(q + center.future_first as i128 - origin)
                .expect("retained turnon future");
            value = value
                + self
                    .center_future_ordinary
                    .dot(self.kernel, &input[first..first + center.future.len()]);
        }
        value
    }
    pub fn future_seed_mode_frames(&self) -> usize {
        self.modes.iter().map(|m| m.guard).sum()
    }
    #[cfg(test)]
    fn seed_states<const FUTURE: bool>(
        &self,
        output: &mut [C],
        scratch: &mut Vec<C>,
        q: i128,
        input: &[C],
        origin: i128,
    ) {
        for g in &self.seed_groups {
            scratch.resize(g.indices.len(), C::default());
            let begin = if FUTURE {
                q + self.core_start as i128
            } else {
                q - self.core_start as i128 - g.guard as i128 + 1
            };
            let first = usize::try_from(begin - origin).expect("admitted grouped seed window");
            let samples = &input[first..first + g.guard];
            if FUTURE {
                self.kernel
                    .coarse_seed::<true>(&g.reverse_poles, scratch, samples);
            } else {
                self.kernel.coarse_seed::<false>(&g.poles, scratch, samples);
            }
            for (j, &index) in g.indices.iter().enumerate() {
                output[index] = scratch[j];
            }
        }
    }
    fn seed_planes<const FUTURE: bool>(
        &self,
        output: &mut Planes,
        scratch: &mut Planes,
        q: i128,
        input: &[C],
        origin: i128,
    ) {
        for g in &self.seed_groups {
            scratch.resize(g.indices.len(), C::default());
            let begin = if FUTURE {
                q + self.core_start as i128
            } else {
                q - self.core_start as i128 - g.guard as i128 + 1
            };
            let first = usize::try_from(begin - origin).expect("admitted split-plane seed window");
            let samples = &input[first..first + g.guard];
            if FUTURE {
                self.kernel
                    .coarse_seed_planes::<true>(&g.split_reverse_poles, scratch, samples);
            } else {
                self.kernel
                    .coarse_seed_planes::<false>(&g.split_poles, scratch, samples);
            }
            for (j, &index) in g.indices.iter().enumerate() {
                output.re[index] = scratch.re[j];
                output.im[index] = scratch.im[j];
            }
        }
    }
    #[cfg(test)]
    pub fn reset(&self, state: &mut State) {
        state.past.clear();
        state.future.clear();
        state.resident.clear();
        state.output.clear();
        state.real_input = false;
        state.next = None;
        state.fft_correction.reset();
    }
    /// Fixed-coarse-block interface. `input` contains its predictor/zero padded
    /// neighborhood. Use a caller-independent block size for exact chunk invariance.
    #[cfg(test)]
    pub fn process<'a>(
        &self,
        state: &'a mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> &'a [WC] {
        self.process_impl::<false, false>(state, start, count, input, origin)
    }
    pub fn process_real<'a>(
        &self,
        state: &'a mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> &'a [WC] {
        self.process_impl::<true, false>(state, start, count, input, origin)
    }
    /// Valid only when analytic input imaginary parts are structurally zero.
    /// The caller selects this from sample-clock geometry, never signal content.
    pub fn process_real_input<'a>(
        &self,
        state: &'a mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> &'a [WC] {
        debug_assert!(input.iter().all(|z| z.im == 0.));
        self.process_impl::<false, true>(state, start, count, input, origin)
    }
    fn weight_seed(&self, coefficients: &Planes, states: &mut Planes) {
        for j in 0..states.len() {
            let z =
                C::new(coefficients.re[j], coefficients.im[j]) * C::new(states.re[j], states.im[j]);
            states.re[j] = z.re;
            states.im[j] = z.im;
        }
    }
    fn process_impl<'a, const REAL: bool, const REAL_INPUT: bool>(
        &self,
        state: &'a mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> &'a [WC] {
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
            self.seed_planes::<false>(&mut state.past, &mut state.seed, start, input, origin);
            state.real_input = REAL_INPUT;
            if REAL_INPUT {
                self.weight_seed(&self.past_split, &mut state.past);
            }
        } else {
            assert_eq!(state.next, Some(start), "contiguous coarse processing");
            assert_eq!(
                state.real_input, REAL_INPUT,
                "fixed modal-state coordinates"
            )
        }
        state.future.resize(self.modes.len(), C::default());
        let future = &mut state.future;
        self.seed_planes::<true>(future, &mut state.seed, last, input, origin);
        if REAL_INPUT {
            self.weight_seed(&self.future_split, future);
        }
        assert!(count.is_multiple_of(crate::coarse_fft::BLOCK));
        // Both forward paths assign every live cell before merging backward
        // values. Retain initialized storage without clearing it between blocks.
        state.output.resize(count, WC::real(0.));
        let output = &mut state.output;
        let resident = self.kernel.coarse_resident_available(self.modes.len());
        // Forward and finite correction are independent state machines. Fill
        // forward values first, then merge each borrowed FFT half-block once.
        // Every sample retains the old (forward + finite) + backward DD order.
        if resident {
            state.resident.resize(count, C::default());
            let first = usize::try_from(start + 1 - self.core_start as i128 - origin)
                .expect("retained resident past");
            if REAL_INPUT {
                self.kernel.coarse_resident_weighted_real::<false, false>(
                    &self.past_split,
                    &self.poles_split,
                    &mut state.past,
                    &input[first..first + count],
                    &mut state.resident,
                );
            } else {
                self.kernel.coarse_resident::<REAL, false>(
                    &self.past_split,
                    &self.poles_split,
                    &mut state.past,
                    &input[first..first + count],
                    &mut state.resident,
                );
            }
        } else {
            for (n, row) in output.iter_mut().enumerate() {
                let q = start + n as i128;
                *row = WC::from(if REAL_INPUT {
                    self.kernel.coarse_weighted_real::<true>(
                        &self.past_split,
                        &self.poles_split,
                        &mut state.past,
                        at(q + 1 - self.core_start as i128).re,
                    )
                } else {
                    self.kernel.coarse_weight_planes::<REAL, true>(
                        &self.past_split,
                        &self.poles_split,
                        &mut state.past,
                        at(q + 1 - self.core_start as i128),
                    )
                });
            }
        }
        for n in (0..count).step_by(crate::coarse_fft::BLOCK) {
            let finite = if REAL {
                self.fft_correction.process_real_borrowed(
                    &mut state.fft_correction,
                    start + n as i128,
                    crate::coarse_fft::BLOCK,
                    input,
                    origin,
                )
            } else if REAL_INPUT {
                self.fft_correction.process_borrowed::<true>(
                    &mut state.fft_correction,
                    start + n as i128,
                    crate::coarse_fft::BLOCK,
                    input,
                    origin,
                )
            } else {
                self.fft_correction.process_borrowed::<false>(
                    &mut state.fft_correction,
                    start + n as i128,
                    crate::coarse_fft::BLOCK,
                    input,
                    origin,
                )
            };
            if resident {
                // Read existing forward Complex scratch directly. No Wide
                // materialization pass or extra output read/write is needed.
                for (i, &value) in finite.iter().enumerate() {
                    output[n + i] = WC::from(state.resident[n + i])
                        + if REAL {
                            WC {
                                re: W::from(value.re),
                                im: W::from(0.),
                            }
                        } else {
                            wc(value)
                        };
                }
            } else {
                for (i, &value) in finite.iter().enumerate() {
                    output[n + i] = output[n + i]
                        + if REAL {
                            WC {
                                re: W::from(value.re),
                                im: W::from(0.),
                            }
                        } else {
                            wc(value)
                        };
                }
            }
        }
        if resident {
            let first = usize::try_from(start + self.core_start as i128 - origin)
                .expect("retained resident future");
            if REAL_INPUT {
                self.kernel.coarse_resident_weighted_real::<false, true>(
                    &self.future_split,
                    &self.reverse_poles_split,
                    future,
                    &input[first..first + count - 1],
                    &mut state.resident,
                );
            } else {
                self.kernel.coarse_resident::<REAL, true>(
                    &self.future_split,
                    &self.reverse_poles_split,
                    future,
                    &input[first..first + count - 1],
                    &mut state.resident,
                );
            }
            for (n, row) in output.iter_mut().enumerate() {
                *row = *row + WC::from(state.resident[n]);
            }
        } else {
            for n in (0..count).rev() {
                let q = start + n as i128;
                let weighted = WC::from(if REAL_INPUT {
                    if n != 0 {
                        self.kernel.coarse_weighted_real::<true>(
                            &self.future_split,
                            &self.reverse_poles_split,
                            future,
                            at(q - 1 + self.core_start as i128).re,
                        )
                    } else {
                        self.kernel.coarse_weighted_real::<false>(
                            &self.future_split,
                            &self.reverse_poles_split,
                            future,
                            0.,
                        )
                    }
                } else if n != 0 {
                    self.kernel.coarse_weight_planes::<REAL, true>(
                        &self.future_split,
                        &self.reverse_poles_split,
                        future,
                        at(q - 1 + self.core_start as i128),
                    )
                } else {
                    self.kernel.coarse_weight_planes::<REAL, false>(
                        &self.future_split,
                        &self.reverse_poles_split,
                        future,
                        C::default(),
                    )
                });
                output[n] = output[n] + weighted;
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
            for j in (0..f.correction.re.len()).step_by(13) {
                let lag = f.correction.first + j;
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
                let actual = WC::from(C::new(f.correction.re[j], f.correction.im[j]));
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
            let coefficient_l1 = f
                .center
                .past
                .iter()
                .chain(&f.center.future)
                .map(|c| c.re.value().abs() + c.im.value().abs())
                .sum::<f64>();
            let input_max = input
                .iter()
                .map(|c| c.re.abs() + c.im.abs())
                .fold(0., f64::max);
            let n = (16 * (f.center.past.len() + f.center.future.len()) + 64) as f64;
            let gamma = n * f64::EPSILON / (1. - n * f64::EPSILON);
            let bound = gamma * coefficient_l1 * input_max;
            println!("turnon d={d} taps={} maximum error={maximum:e} normwise ordinary-dot bound={bound:e}",f.center.past.len());
            assert!(maximum <= bound);
        }
    }
}

#[cfg(test)]
fn coefficient_rounding_bound(coefficients: &[WC]) -> f64 {
    coefficients
        .iter()
        .map(|z| {
            let (rh, rl) = z.re.parts();
            let (ih, il) = z.im.parts();
            rh.abs() + rl.abs() + ih.abs() + il.abs()
        })
        .sum::<f64>()
        * f64::EPSILON
        * (1. + 1e-12)
}

#[cfg(test)]
mod fft_model_tests {
    use super::*;
    #[test]
    fn reused_output_overwrites_poisoned_storage_across_sizes_and_reset() {
        let bits = |w: W| {
            let (hi, lo) = w.parts();
            [hi.to_bits(), lo.to_bits()]
        };
        for d in [2.0025, 4.] {
            let mut f = CoarseIir::new(W::from(d));
            let origin = -(f.max_lag as i128);
            for kernel in crate::simd::test_kernels() {
                f.kernel = kernel;
                for kind in 0..3 {
                    let input: Vec<_> = (0..2 * f.max_lag + 3072)
                        .map(|j| {
                            C::new(
                                ((j * 37 % 127) as f64 - 63.) / 127.,
                                if kind == 2 {
                                    0.
                                } else {
                                    ((j * 41 % 131) as f64 - 65.) / 131.
                                },
                            )
                        })
                        .collect();
                    let mut reused = State {
                        output: vec![WC::real(f64::NAN); 1024],
                        ..State::default()
                    };
                    let pointer = reused.output.as_ptr();
                    let capacity = reused.output.capacity();
                    let mut fresh = State::default();
                    for _ in 0..2 {
                        let mut start = 0;
                        for count in [1024, 512, 1024, 512] {
                            reused.output.fill(WC::real(f64::NAN));
                            // Recreate the old per-block allocation as the storage oracle.
                            fresh.output = Vec::new();
                            for state in [&mut reused, &mut fresh] {
                                match kind {
                                    0 => f.process(state, start, count, &input, origin),
                                    1 => f.process_real(state, start, count, &input, origin),
                                    _ => f.process_real_input(state, start, count, &input, origin),
                                };
                            }
                            assert_eq!(reused.output.as_ptr(), pointer);
                            assert_eq!(reused.output.capacity(), capacity);
                            assert_eq!(reused.output().len(), count);
                            for (a, b) in reused.output().iter().zip(fresh.output()) {
                                assert_eq!(bits(a.re), bits(b.re));
                                assert_eq!(bits(a.im), bits(b.im));
                            }
                            start += count as i128;
                        }
                        f.reset(&mut reused);
                        f.reset(&mut fresh);
                        assert!(reused.output().is_empty());
                        assert_eq!(reused.output.as_ptr(), pointer);
                    }
                }
            }
        }
    }
    #[test]
    fn partitioned_correction_preserves_full_coarse_operator_and_reset() {
        for d in [2.0025, 3.675, 4.] {
            let f = CoarseIir::new(W::from(d));
            let old = crate::coarse_iir_fir_reference::CoarseIir::new(W::from(d));
            let count = 512;
            let start = -256i128;
            let origin = start - f.max_lag as i128;
            let blocks = 2 * f.fft_correction.partition_count() + 3;
            let input: Vec<_> = (0..2 * f.max_lag + blocks * count)
                .map(|j| {
                    C::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let mut a = State::default();
            let mut b = crate::coarse_iir_fir_reference::State::default();
            let mut maximum: f64 = 0.;
            let mut first = Vec::new();
            for block in 0..blocks {
                let at = start + (block * count) as i128;
                let actual = f.process(&mut a, at, count, &input, origin);
                let expected = old.process(&mut b, at, count, &input, origin);
                if block == 0 {
                    first = actual.to_vec();
                }
                for (a, b) in actual.iter().zip(&expected) {
                    maximum = maximum.max((*a - *b).value().abs());
                }
            }
            f.reset(&mut a);
            let again = f.process(&mut a, start, count, &input, origin);
            for (a, b) in first.iter().zip(again) {
                assert_eq!(a.re.parts(), b.re.parts());
                assert_eq!(a.im.parts(), b.im.parts());
            }
            println!("FFT_CORRECTION d={d} coremodes{} partitions{}/{} quantizationL1{:e} FFTnormwise_screen{:e} maximum_fulloperator_difference{maximum:e}",f.mode_count(),f.fft_correction.active_partition_count(),f.fft_correction.partition_count(),f.quantization_l1,f.fft_correction.roundoff_screen(1.));
            assert!(maximum < 3e-16);
            // The added signed correction changes the coefficient norm. Check
            // its actual storage rounding against that norm, independently of
            // the unchanged absolute full-operator error requirement above.
            assert!(f.quantization_l1 <= f.quantization_bound);
        }
    }
}

#[cfg(test)]
mod grouped_seed_tests {
    use super::*;
    #[test]
    fn grouped_seeds_match_independent_same_guard_scalar_states() {
        for d in [0.500625, 0.91875, 1., 2.0025, 3.675, 4.] {
            let f = CoarseIir::new(W::from(d));
            let origin = -(f.max_lag as i128);
            let q = 0i128;
            let input: Vec<_> = (0..2 * f.max_lag + 1)
                .map(|j| {
                    C::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let mut past = vec![C::default(); f.mode_count()];
            let mut future = past.clone();
            let mut scratch = Vec::new();
            f.seed_states::<false>(&mut past, &mut scratch, q, &input, origin);
            f.seed_states::<true>(&mut future, &mut scratch, q, &input, origin);
            for (j, m) in f.modes.iter().enumerate() {
                let mut a = C::default();
                let mut b = C::default();
                for lag in (m.start..m.start + m.guard).rev() {
                    a = m.p * a + input[(q - lag as i128 - origin) as usize];
                    b = m.p.conj() * b + input[(q + lag as i128 - origin) as usize];
                }
                assert_eq!(past[j].re.to_bits(), a.re.to_bits());
                assert_eq!(past[j].im.to_bits(), a.im.to_bits());
                assert_eq!(future[j].re.to_bits(), b.re.to_bits());
                assert_eq!(future[j].im.to_bits(), b.im.to_bits());
            }
            println!(
                "GROUPED_SEED d={d} modes{} groups{} maxlag{} bound{:e}",
                f.mode_count(),
                f.seed_groups.len(),
                f.max_lag,
                f.omitted_future_bound()
            );
        }
    }
}

#[cfg(test)]
mod weighted_real_model_tests {
    use super::*;
    #[test]
    fn weighted_real_coordinates_preserve_generic_operator_and_reset() {
        for d in [0.500625, 0.91875, 1., 2.0025, 3.675, 4.] {
            let f = CoarseIir::new(W::from(d));
            let count = 512;
            let blocks = 5;
            let start = -256;
            let origin = start - f.max_lag as i128;
            let input: Vec<_> = (0..2 * f.max_lag + blocks * count)
                .map(|j| C::new(((j * 37 % 127) as f64 - 63.) / 127., 0.))
                .collect();
            let mut old = State::default();
            let mut new = State::default();
            let mut maximum: f64 = 0.;
            let mut first = Vec::new();
            for block in 0..blocks {
                let at = start + (block * count) as i128;
                let a = f.process(&mut old, at, count, &input, origin);
                let b = f.process_real_input(&mut new, at, count, &input, origin);
                if block == 0 {
                    first = b.to_vec();
                }
                for (a, b) in a.iter().zip(b) {
                    maximum = maximum.max((*a - *b).value().abs());
                }
            }
            f.reset(&mut new);
            let again = f.process_real_input(&mut new, start, count, &input, origin);
            for (a, b) in first.iter().zip(again) {
                assert_eq!(a.re.parts(), b.re.parts());
                assert_eq!(a.im.parts(), b.im.parts());
            }
            let modal_l1: f64 = f
                .modes
                .iter()
                .map(|m| m.coefficient.value().abs() / (1. - m.p.abs()))
                .sum();
            let radius = f.modes.iter().map(|m| m.p.abs()).fold(0., f64::max);
            let bound = 64. * f64::EPSILON * modal_l1 / (1. - radius);
            println!("WEIGHTED_REAL,d{d},modes{},max_error{maximum:e},independent_normwise_bound{bound:e},modal_l1{modal_l1:e}",f.mode_count());
            assert!(maximum <= bound);
            assert!(maximum < 1e-15);
        }
    }
}
