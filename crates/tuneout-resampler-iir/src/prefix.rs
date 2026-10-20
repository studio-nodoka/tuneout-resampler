//! Generic DD construction of the compact prefix for a multirate modal IIR.
use crate::bidirectional_design::model;
use crate::complex::{Wide, WideComplex as WC};
#[path = "core_polynomial_coefficients.rs"]
pub(crate) mod core_polynomial;
pub(crate) const A: f64 = 32.;
pub(crate) const B: f64 = 160.;
pub(crate) const GRID: f64 = 4.;
pub(crate) const HALF: usize = 24;
pub(crate) const BETA: f64 = 32.;
fn pi() -> Wide {
    Wide::from(std::f64::consts::PI) + Wide::from(1.2246467991473532e-16)
}
pub(crate) fn exp(z: WC) -> WC {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("fma") {
        return unsafe { exp_fma(z) };
    }
    exp_scalar(z)
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn exp_fma(z: WC) -> WC {
    exp_scalar(z)
}
// All Wide methods retain their exact original arithmetic. Inlining puts the
// existing error-free product residuals into the checked hardware-FMA context.
#[inline(always)]
fn exp_scalar(z: WC) -> WC {
    let size = z.value().abs();
    let extra = if size > 4. {
        (size / 4.).log2().ceil() as usize
    } else {
        0
    };
    let z = z.scale(Wide::from(1. / (16usize << extra) as f64));
    let mut sum = WC::real(1.);
    let mut term = sum;
    for k in 1..=28 {
        term = term * z / WC::real(k as f64);
        sum = sum + term
    }
    for _ in 0..4 + extra {
        sum = sum * sum
    }
    sum
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    #[test]
    fn repeated_periods_cover_arbitrary_phase_and_short_remainders() {
        for (input, output) in [
            (96000, 44100),
            (192000, 44100),
            (44100, 48000),
            (48000, 96000),
        ] {
            let p = Prefix::new(input, output);
            for first in [0, 137, 10001] {
                let count = 12 * p.l as usize + 37;
                let start = (u128::from(first) * u128::from(p.m) / u128::from(p.l)) as i128;
                let end =
                    (u128::from(first + count as u64) * u128::from(p.m) / u128::from(p.l)) as i128;
                let origin = start - p.radius as i128;
                let n = (end - start) as usize + 2 * p.radius + 1;
                let window: Vec<_> = (0..n).map(|i| ((i * 31 % 71) as f64 - 35.) / 31.).collect();
                let mut actual = vec![0.; count];
                let mut scratch = Vec::new();
                p.output_block(first, &mut actual, &window, origin, &mut scratch);
                for (i, &value) in actual.iter().enumerate() {
                    let expected = p.output(first + i as u64, &window, origin).value();
                    assert!(
                        (value - expected).abs() < 2e-13,
                        "{input}->{output} first={first} i={i}"
                    );
                }
            }
        }
    }
}
pub(crate) fn modes() -> Vec<(WC, WC)> {
    mode_values().collect()
}
fn mode_values() -> impl Iterator<Item = (WC, WC)> {
    model::MODES.iter().map(|m| {
        let w = |i| Wide::from(m[i]) + Wide::from(m[i + 1]);
        (WC { re: w(0), im: w(2) }, WC { re: w(4), im: w(6) })
    })
}
pub(crate) fn envelope(t: f64) -> WC {
    let t = Wide::from(t.abs());
    let mut result = WC::real(0.);
    for (s, c) in mode_values() {
        result = result
            + c * exp(WC {
                re: s.re * t,
                im: (s.im - pi()) * t,
            })
    }
    result
}
#[path = "partition_coefficients.rs"]
pub(crate) mod partition;
/// Kaiser-CDF beta44, evaluated by a DD Chebyshev series from the frozen
/// model-only high-precision exporter. Endpoints clamp to exact zero/one.
pub(crate) fn turnon(t: Wide) -> Wide {
    if t.value() <= A {
        return Wide::from(0.);
    }
    if t.value() >= B {
        return Wide::from(1.);
    }
    let x = Wide::from(2.) * (t - Wide::from(A)) / Wide::from(B - A) - Wide::from(1.);
    let mut b1 = Wide::from(0.);
    let mut b2 = b1;
    for &(hi, lo) in partition::COEFFICIENTS[1..].iter().rev() {
        let b = Wide::from(hi) + Wide::from(lo) + Wide::from(2.) * x * b1 - b2;
        b2 = b1;
        b1 = b;
    }
    let (hi, lo) = partition::COEFFICIENTS[0];
    Wide::from(hi) + Wide::from(lo) + x * b1 - b2
}
#[derive(Default)]
pub struct PrefixState {
    first: Option<u64>,
    values: Vec<f64>,
}
impl PrefixState {
    pub fn reset(&mut self) {
        self.first = None;
        self.values.clear();
    }
    pub fn bytes(&self) -> usize {
        self.values.capacity() * 8
    }
}
pub struct Prefix {
    input: u32,
    l: u32,
    m: u32,
    alpha: Wide,
    radius: usize,
    rows: Vec<Vec<f64>>,
    first: Vec<i64>,
}
impl Prefix {
    pub fn new(input: u32, output: u32) -> Self {
        let g = crate::design::gcd(input, output);
        let l = output / g;
        let m = input / g;
        let alpha = Wide::from(f64::from(l)) / Wide::from(f64::from(l.max(m)));
        let radius = (B / alpha.value()).ceil() as usize;
        let mut p = Self {
            input,
            l,
            m,
            alpha,
            radius,
            rows: Vec::new(),
            first: Vec::new(),
        };
        if l <= 512 && u128::from(l) * (2 * radius + 1) as u128 * 8 <= 33_554_432 {
            if let Some(rows) = crate::setup_prefix::rows(alpha, radius, l) {
                p.rows = rows;
                p.first = vec![-(radius as i64); l as usize];
            } else {
                let representatives = l / 2 + 1;
                for phase in 0..representatives {
                    let (first, row) = p.make_row(phase);
                    p.first.push(first);
                    p.rows.push(row)
                }
                for phase in representatives..l {
                    let source = &p.rows[(l - phase) as usize];
                    let mut row = vec![0.; source.len()];
                    for j in 1..row.len() {
                        row[j] = source[row.len() - j];
                    }
                    p.first.push(-(radius as i64));
                    p.rows.push(row);
                }
            }
        }
        p
    }
    fn make_row(&self, phase: u32) -> (i64, Vec<f64>) {
        let u = Wide::from(f64::from(phase)) / Wide::from(f64::from(self.l));
        let mut result = vec![Wide::from(0.); 2 * self.radius + 1];
        for future in [false, true] {
            let offset = if future { Wide::from(1.) - u } else { u };
            for j in 0..=self.radius {
                let lag = if future { -(j as i64) - 1 } else { j as i64 };
                if lag < -(self.radius as i64) || lag > self.radius as i64 {
                    break;
                }
                let t = (Wide::from(j as f64) + offset) * self.alpha;
                if t.value() < B {
                    result[(lag + self.radius as i64) as usize] =
                        polynomial_core(t) * self.alpha * (Wide::from(1.) - turnon(t));
                }
            }
        }
        result.reverse();
        (
            -(self.radius as i64),
            result.into_iter().map(Wide::value).collect(),
        )
    }
    #[cfg(test)]
    fn make_row_reference(&self, phase: u32) -> (i64, Vec<f64>) {
        let u = Wide::from(f64::from(phase)) / Wide::from(f64::from(self.l));
        let bank = modes();
        let poles: Vec<_> = bank.iter().map(|(s, _)| exp(s.scale(self.alpha))).collect();
        let mut result = vec![Wide::from(0.); 2 * self.radius + 1];
        // Input lags j: normalized distance=(j+u)*alpha. Generate both halves
        // from small exponentials and DD recurrence, avoiding large-angle f64
        // trig or per-tap transcendental calls.
        for future in [false, true] {
            let offset = if future { Wide::from(1.) - u } else { u };
            let mut values: Vec<_> = bank
                .iter()
                .map(|(s, c)| *c * exp(s.scale(self.alpha * offset)))
                .collect();
            for j in 0..=self.radius {
                let lag = if future { -(j as i64) - 1 } else { j as i64 };
                if lag < -(self.radius as i64) || lag > self.radius as i64 {
                    break;
                }
                let t = (Wide::from(j as f64) + offset) * self.alpha;
                if t.value() < B {
                    let h = values.iter().fold(Wide::from(0.), |sum, z| sum + z.re)
                        * crate::cosine_shape::factor(t)
                        + crate::compact_stopband::oracle_delta(t);
                    result[(lag + self.radius as i64) as usize] =
                        h * self.alpha * (Wide::from(1.) - turnon(t));
                }
                for (z, pole) in values.iter_mut().zip(&poles) {
                    *z = *z * *pole
                }
            }
        }
        result.reverse();
        (
            -(self.radius as i64),
            result.into_iter().map(Wide::value).collect(),
        )
    }

    /// Repeated exact rational phases share coefficients across periods. Only
    /// already requested outputs are batched, so this introduces no lookahead.
    /// Scratch has a fixed4MiB admission cap for every ratio and channel count.
    pub fn output_block(
        &self,
        first: u64,
        values: &mut [f64],
        window: &[f64],
        origin: i128,
        scratch: &mut Vec<f64>,
    ) {
        let mut done = 0;
        let l = self.l as usize;
        let m = self.m as usize;
        if !self.rows.is_empty() && l >= 4 {
            let mut lanes = crate::f64_phase_tile::width();
            while lanes > 1 {
                let count = lanes * l;
                let taps = self.rows[0].len();
                let rows = taps + m;
                let fits = rows
                    .checked_mul(lanes)
                    .and_then(|n| n.checked_mul(8))
                    .is_some_and(|n| n <= 4_194_304)
                    && (lanes as u128) * (m as u128) * 1000 <= u128::from(self.input) * 60;
                while fits && values.len() - done >= count {
                    let base = first + done as u64;
                    let ticks = u128::from(base) * u128::from(self.m);
                    let anchor = (ticks / u128::from(self.l)) as i128;
                    let phase = (ticks % u128::from(self.l)) as usize;
                    let first_lag = self.first[0];
                    let start = usize::try_from(anchor + i128::from(first_lag) - origin)
                        .expect("retained phase-tile left edge");
                    let used = taps + (phase + (l - 1) * m) / l;
                    assert!(start + used + (lanes - 1) * m <= window.len());
                    // Every live element is overwritten by the packed-input loop below.
                    let packed = aligned_scratch(scratch, used * lanes);
                    for row in 0..used {
                        for lane in 0..lanes {
                            packed[row * lanes + lane] = window[start + row + lane * m];
                        }
                    }
                    for group in (0..l).step_by(4) {
                        let offsets =
                            std::array::from_fn(|j| (phase + (group + j).min(l - 1) * m) / l);
                        let coeff = std::array::from_fn(|j| {
                            &self.rows[(phase + (group + j).min(l - 1) * m) % l][..]
                        });
                        let output = crate::f64_phase_tile::run(lanes, packed, offsets, coeff);
                        for j in 0..4 {
                            if group + j < l {
                                for lane in 0..lanes {
                                    values[done + lane * l + group + j] = output[j][lane];
                                }
                            }
                        }
                    }
                    done += count;
                }
                lanes /= 2;
            }
        }
        for (i, value) in values.iter_mut().enumerate().skip(done) {
            *value = self.output(first + i as u64, window, origin).value();
        }
    }
    pub fn cached_phases(&self) -> usize {
        self.rows.len()
    }
    pub fn taps_per_output(&self) -> usize {
        2 * self.radius + 1
    }
    /// Cache complete phase periods already covered by the supplied IIR
    /// window. This changes no input requirement or output availability.
    pub fn output_cached(
        &self,
        state: &mut PrefixState,
        first: u64,
        values: &mut [f64],
        window: &[f64],
        origin: i128,
        scratch: &mut Vec<f64>,
    ) {
        let lanes = crate::f64_phase_tile::cached_periods();
        let l = self.l as usize;
        let m = self.m as usize;
        // Empty phase banks are the generic on-demand fallback. Check this
        // before lane/row products, including on32-bit debug builds.
        if lanes <= 1 || l < 4 || self.rows.is_empty() {
            return self.output_block(first, values, window, origin, scratch);
        }
        let tile = lanes * l;
        let taps = self.rows[0].len();
        let Some(rows) = taps.checked_add(m) else {
            return self.output_block(first, values, window, origin, scratch);
        };
        let eligible = rows
            .checked_mul(lanes)
            .and_then(|n| n.checked_mul(8))
            .is_some_and(|n| n <= 4_194_304)
            && (lanes as u128) * (m as u128) * 1000 <= u128::from(self.input) * 60;
        if !eligible {
            return self.output_block(first, values, window, origin, scratch);
        }
        let mut done = 0;
        while done < values.len() {
            let out = first + done as u64;
            let cached = state
                .first
                .is_some_and(|base| out >= base && out - base < state.values.len() as u64);
            if !cached {
                let base = out / tile as u64 * tile as u64;
                if base.checked_add(tile as u64).is_none() {
                    self.output_block(out, &mut values[done..], window, origin, scratch);
                    return;
                }
                let anchor = (u128::from(base) * u128::from(self.m) / u128::from(self.l)) as i128;
                // base is a whole number of exact rational phase periods.
                let left = anchor + i128::from(self.first[0]) - origin;
                let used = taps + (l - 1) * m / l;
                let extent = used + (lanes - 1) * m;
                let available = left >= 0
                    && usize::try_from(left)
                        .ok()
                        .and_then(|left| left.checked_add(extent))
                        .is_some_and(|end| end <= window.len());
                if !available {
                    self.output_block(out, &mut values[done..], window, origin, scratch);
                    return;
                }
                let start = left as usize;
                // Every live element is overwritten by the packed-input loop below.
                let packed = aligned_scratch(scratch, used * lanes);
                for row in 0..used {
                    for lane in 0..lanes {
                        packed[row * lanes + lane] = window[start + row + lane * m];
                    }
                }
                state.values.resize(tile, 0.);
                for group in (0..l).step_by(4) {
                    let offsets = std::array::from_fn(|j| (group + j).min(l - 1) * m / l);
                    let coeff =
                        std::array::from_fn(|j| &self.rows[((group + j).min(l - 1) * m) % l][..]);
                    let output = crate::f64_phase_tile::run_cached(lanes, packed, offsets, coeff);
                    for (j, row) in output.iter().enumerate() {
                        if group + j < l {
                            for (lane, &value) in row.iter().enumerate().take(lanes) {
                                state.values[lane * l + group + j] = value;
                            }
                        }
                    }
                }
                state.first = Some(base);
            }
            let offset = (out - state.first.unwrap()) as usize;
            let count = (values.len() - done).min(state.values.len() - offset);
            values[done..done + count].copy_from_slice(&state.values[offset..offset + count]);
            done += count;
        }
    }
    pub fn radius(&self) -> usize {
        self.radius
    }
    pub fn bytes(&self) -> usize {
        self.rows.iter().map(|r| r.capacity() * 8).sum::<usize>()
            + self.rows.capacity() * std::mem::size_of::<Vec<f64>>()
            + self.first.capacity() * 8
    }
    pub fn output(&self, out: u64, window: &[f64], origin: i128) -> Wide {
        let clock = u128::from(out) * u128::from(self.m);
        let anchor = (clock / u128::from(self.l)) as i128;
        let phase = (clock % u128::from(self.l)) as u32;
        let temporary;
        let (first, row) = if self.rows.is_empty() {
            temporary = self.make_row(phase);
            (temporary.0, &temporary.1[..])
        } else {
            (self.first[phase as usize], &self.rows[phase as usize][..])
        };
        let begin = usize::try_from(anchor + i128::from(first) - origin)
            .expect("retained direct prefix edge");
        Wide::from(crate::f64_dot::dot(&window[begin..begin + row.len()], row))
    }
}

#[cfg(test)]
mod cache_geometry_tests {
    use super::*;
    #[test]
    fn huge_reduced_phase_clock_uses_bounded_on_demand_fallback() {
        let prefix = Prefix::new(u32::MAX, u32::MAX - 1);
        assert!(prefix.rows.is_empty());
        let input = vec![0.; 2 * prefix.radius() + 1];
        let origin = -(prefix.radius() as i128);
        let mut state = PrefixState::default();
        let mut scratch = Vec::new();
        let mut values = [17.];
        prefix.output_cached(&mut state, 0, &mut values, &input, origin, &mut scratch);
        assert_eq!(values, [0.]);
        assert_eq!(state.bytes(), 0);
        prefix.output_cached(&mut state, 0, &mut [], &[], 0, &mut scratch);
    }
}

#[cfg(test)]
mod setup_equivalence_tests {
    use super::*;
    #[test]
    fn vector_construction_is_exactly_scalar_coefficient_construction() {
        for (input, output) in [
            (44100, 48000),
            (48000, 44100),
            (96000, 44100),
            (192000, 44100),
            (48000, 96000),
            (48000, 30000),
            (44100, 50400),
            (44117, 48013),
        ] {
            let a = Prefix::new(input, output);
            for (phase, row) in a.rows.iter().enumerate() {
                let (first, b) = a.make_row_reference(phase as u32);
                assert_eq!(a.first[phase], first);
                for (tap, (a, b)) in row.iter().zip(&b).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{input}->{output} phase={phase} tap={tap} actual={a:e} expected={b:e}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod setup_cross_tests {
    use super::*;
    #[test]
    fn setup_cross_numerical_prefix() {
        for (input, output) in [(48000, 30000), (44100, 50400), (96000, 60000)] {
            let p = Prefix::new(input, output);
            assert!(!p.rows.is_empty());
            for (phase, row) in p.rows.iter().enumerate() {
                let (first, expected) = p.make_row_reference(phase as u32);
                assert_eq!(p.first[phase], first);
                for (a, b) in row.iter().zip(expected) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
            let scaled = Prefix::new(input * 2, output * 2);
            for (a, b) in p.rows.iter().flatten().zip(scaled.rows.iter().flatten()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
}

// Both channel caches share the same absolute rational-period tile. Packing
// channels together lets every coefficient broadcast serve left and right.
impl Prefix {
    pub(crate) fn stereo_cache_eligible(&self) -> bool {
        let periods = crate::f64_stereo_phase_tile::periods();
        if periods <= 1 || self.rows.is_empty() {
            return false;
        }
        if self.l < 4 {
            return self.m == 1;
        }
        self.rows[0]
            .len()
            .checked_add(self.m as usize)
            .and_then(|n| n.checked_mul(periods * 2))
            .and_then(|n| n.checked_mul(8))
            .is_some_and(|n| n <= 4_194_304)
            && (periods as u128) * u128::from(self.m) * 1000 <= u128::from(self.input) * 60
    }
    pub(crate) fn output_cached_stereo(
        &self,
        states: &mut [PrefixState],
        first: u64,
        values: &mut [f64],
        windows: [&[f64]; 2],
        origin: i128,
        scratch: &mut Vec<f64>,
    ) {
        assert_eq!(states.len(), 2);
        assert_eq!(values.len() % 2, 0);
        assert!(self.stereo_cache_eligible());
        let periods = crate::f64_stereo_phase_tile::periods();
        let lanes = periods * 2;
        let l = self.l as usize;
        let m = self.m as usize;
        let tile = periods * l;
        let taps = self.rows[0].len();
        let mut done = 0;
        while done < values.len() / 2 {
            let out = first + done as u64;
            let cached = states.iter().all(|state| {
                state
                    .first
                    .is_some_and(|base| out >= base && out - base < state.values.len() as u64)
            });
            if !cached {
                let base = out / tile as u64 * tile as u64;
                let anchor = (u128::from(base) * u128::from(self.m) / u128::from(self.l)) as i128;
                let left = anchor + i128::from(self.first[0]) - origin;
                let used = taps + (l - 1) * m / l;
                let extent = used + (periods - 1) * m;
                let available = base.checked_add(tile as u64).is_some()
                    && left >= 0
                    && usize::try_from(left)
                        .ok()
                        .and_then(|left| left.checked_add(extent))
                        .is_some_and(|end| windows.iter().all(|w| end <= w.len()));
                if !available {
                    for index in done..values.len() / 2 {
                        for ch in 0..2 {
                            values[2 * index + ch] = self
                                .output(first + index as u64, windows[ch], origin)
                                .value();
                        }
                    }
                    return;
                }
                let start = left as usize;
                for state in states.iter_mut() {
                    state.values.resize(tile, 0.);
                }
                if m == 1 && l < 4 {
                    let coeff = std::array::from_fn(|j| &self.rows[j.min(l - 1)][..]);
                    let output = crate::f64_direct_periods::run(
                        periods,
                        l,
                        &windows[0][start..start + extent],
                        &windows[1][start..start + extent],
                        coeff,
                    );
                    for (j, phase) in output.iter().enumerate().take(l) {
                        for (period, (&left, &right)) in
                            phase[0].iter().zip(&phase[1]).enumerate().take(periods)
                        {
                            states[0].values[period * l + j] = left;
                            states[1].values[period * l + j] = right;
                        }
                    }
                } else {
                    // Every live element is overwritten by the packed-input loop below.
                    let packed = aligned_scratch(scratch, used * lanes);
                    crate::f64_stereo_phase_tile::pack(windows, start, used, periods, m, packed);
                    for group in (0..l).step_by(4) {
                        let offsets = std::array::from_fn(|j| (group + j).min(l - 1) * m / l);
                        let coeff =
                            std::array::from_fn(
                                |j| &self.rows[((group + j).min(l - 1) * m) % l][..],
                            );
                        let (left, right) = states.split_at_mut(1);
                        crate::f64_stereo_phase_tile::run_into(
                            periods,
                            packed,
                            offsets,
                            coeff,
                            [&mut left[0].values, &mut right[0].values],
                            l,
                            group,
                            (l - group).min(4),
                        );
                    }
                }
                for state in states.iter_mut() {
                    state.first = Some(base);
                }
            }
            let offsets = [
                (out - states[0].first.unwrap()) as usize,
                (out - states[1].first.unwrap()) as usize,
            ];
            let count = (values.len() / 2 - done)
                .min(states[0].values.len() - offsets[0])
                .min(states[1].values.len() - offsets[1]);
            for i in 0..count {
                values[2 * (done + i)] = states[0].values[offsets[0] + i];
                values[2 * (done + i) + 1] = states[1].values[offsets[1] + i];
            }
            done += count;
        }
    }
}

#[cfg(test)]
mod exp_hardware_tests {
    use super::*;
    #[test]
    fn guarded_exponentials_preserve_every_original_dd_part() {
        for t in [
            0., 0.001, 0.5, 1., 3.675, 4., 31.999, 32., 64.125, 128., 160., 1000.125, 6615., 13230.,
        ] {
            for (s, _) in modes() {
                for z in [
                    s.scale(Wide::from(t)),
                    WC {
                        re: s.re,
                        im: s.im - pi(),
                    }
                    .scale(Wide::from(t)),
                ] {
                    let a = exp(z);
                    let b = exp_scalar(z);
                    assert_eq!(a.re.parts(), b.re.parts());
                    assert_eq!(a.im.parts(), b.im.parts());
                }
            }
        }
    }
}

pub(crate) fn polynomial_core(t: Wide) -> Wide {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("fma") {
        return unsafe { polynomial_core_fma(t) };
    }
    polynomial_core_scalar(t)
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn polynomial_core_fma(t: Wide) -> Wide {
    polynomial_core_scalar(t)
}
#[inline(always)]
fn polynomial_core_scalar(t: Wide) -> Wide {
    if !(0. ..=160.).contains(&t.value()) {
        return mode_values().fold(Wide::from(0.), |sum, (s, c)| sum + (c * exp(s.scale(t))).re)
            * crate::cosine_shape::factor(t);
    }
    let segment = ((t.value() / 2.).floor() as usize).min(79);
    let x = t - Wide::from((2 * segment + 1) as f64);
    let mut h = Wide::from(0.);
    for &(hi, lo) in core_polynomial::COEFFICIENTS[segment].iter().rev() {
        h = h * x + (Wide::from(hi) + Wide::from(lo));
    }
    h
}
#[cfg(test)]
mod polynomial_tests {
    use super::*;
    #[test]
    fn model_only_polynomial_matches_independent_modal_dd_at_heldout_points() {
        let bank = modes();
        let mut maximum: f64 = 0.;
        let mut at = 0.;
        for segment in 0..80 {
            for local in [
                0., 0.00390625, 0.125, 0.375, 0.625, 1., 1.375, 1.875, 1.99609375, 2.,
            ] {
                let t = Wide::from((2 * segment) as f64) + Wide::from(local);
                let expected = bank.iter().fold(Wide::from(0.), |sum, (s, c)| {
                    sum + (*c * exp(s.scale(t))).re
                }) * crate::cosine_shape::factor(t)
                    + crate::compact_stopband::oracle_delta(t);
                let actual = polynomial_core(t);
                let e = (actual - expected).value().abs();
                if e > maximum {
                    maximum = e;
                    at = t.value();
                }
            }
        }
        println!("CORE_POLYNOMIAL max_heldout_DD_model_error{maximum:e} at{at}");
        assert!(maximum < 2e-28);
    }
}

/// Reuse initialized storage and choose a64-byte-aligned live input tile.
/// The row stride remains unchanged. Seven trailing f64 slots are enough
/// because Vec<f64> already guarantees8-byte alignment.
fn aligned_scratch(scratch: &mut Vec<f64>, len: usize) -> &mut [f64] {
    scratch.resize(
        len.checked_add(7).expect("admitted packed input extent"),
        0.,
    );
    let skip = scratch.as_ptr().align_offset(64);
    assert!(skip <= 7);
    &mut scratch[skip..skip + len]
}
