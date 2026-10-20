//! Uniformly partitioned overlap-save for the finite coarse correction.
//! The material IIR core remains separate. Input blocks start at output_start
//! +zero_lag; zero_lag need not be a multiple of the partition length.
use crate::complex::Complex as C;
use rustfft::{num_complex::Complex as F, Fft, FftPlanner};
use std::sync::Arc;
pub(crate) const BLOCK: usize = 512;
const FFT: usize = 2 * BLOCK;
pub(crate) struct Plan {
    real_mac: crate::simd::RealHistoryMac,
    mac: fn(&mut [C], &[C], &[C]),
    spectra: Vec<Vec<C>>,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    real_forward: Arc<dyn Fft<f64>>,
    real_roots: Vec<F<f64>>,
    scratch_len: usize,
    zero_lag: usize,
    #[cfg(test)]
    l1: f64,
}
#[derive(Default)]
pub(crate) struct State {
    history: Vec<Vec<C>>,
    head: usize,
    next: Option<i128>,
    real_input: bool,
    scratch: Vec<C>,
    work: Vec<C>,
    real_work: Vec<C>,
}
impl State {
    pub(crate) fn bytes(&self) -> usize {
        self.history.capacity() * std::mem::size_of::<Vec<C>>()
            + self
                .history
                .iter()
                .map(|v| v.capacity() * 16)
                .sum::<usize>()
            + (self.scratch.capacity() + self.work.capacity() + self.real_work.capacity()) * 16
    }
    #[cfg(test)]
    pub(crate) fn reset(&mut self) {
        self.history.clear();
        self.scratch.clear();
        self.real_work.clear();
        self.real_input = false;
        self.head = 0;
        self.next = None;
    }
}
impl Plan {
    pub(crate) fn new(coefficients: &[C], zero_lag: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(FFT);
        let inverse = planner.plan_fft_inverse(FFT);
        let real_forward = planner.plan_fft_forward(BLOCK);
        let real_roots = (0..BLOCK)
            .map(|k| {
                if k == 0 {
                    return F::new(1., 0.);
                }
                if 2 * k == BLOCK {
                    return F::new(0., -1.);
                }
                let angle = -2. * std::f64::consts::PI * k as f64 / FFT as f64;
                F::new(angle.cos(), angle.sin())
            })
            .collect();
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len())
            .max(real_forward.get_inplace_scratch_len());
        let mut work = vec![C::default(); scratch_len];
        let mut spectra = Vec::new();
        for chunk in coefficients.chunks(BLOCK) {
            if chunk.iter().all(|c| c.re == 0. && c.im == 0.) {
                spectra.push(Vec::new());
                continue;
            }
            let mut row = vec![C::default(); FFT];
            row[..chunk.len()].copy_from_slice(chunk);
            planned_transform(&mut row, forward.as_ref(), &mut work, false);
            spectra.push(row);
        }
        Self {
            real_mac: crate::simd::Kernel::coarse_detect().coarse_real_history_mac(),
            mac: crate::simd::Kernel::coarse_detect().coarse_mac_function(),
            spectra,
            forward,
            inverse,
            real_forward,
            real_roots,
            scratch_len,
            zero_lag,
            #[cfg(test)]
            l1: coefficients.iter().map(|c| c.re.abs() + c.im.abs()).sum(),
        }
    }
    /// Owned spectra plus Arc handles; excludes opaque RustFFT plan storage.
    pub(crate) fn coefficient_bytes(&self) -> usize {
        self.spectra.capacity() * std::mem::size_of::<Vec<C>>()
            + self
                .spectra
                .iter()
                .map(|v| v.capacity() * 16)
                .sum::<usize>()
            + 3 * std::mem::size_of::<Arc<dyn Fft<f64>>>()
            + self.real_roots.capacity() * 16
    }
    #[cfg(test)]
    pub(crate) fn partition_count(&self) -> usize {
        self.spectra.len()
    }
    pub(crate) fn active_partition_count(&self) -> usize {
        self.spectra.iter().filter(|p| !p.is_empty()).count()
    }
    /// Standard normwise roundoff screen, scaled by the caller's complex input
    /// component maximum. It includes FFT stages and partition accumulation;
    /// it is not a claimed tight forward-error proof or a quality score.
    #[cfg(test)]
    pub(crate) fn roundoff_screen(&self, maximum: f64) -> f64 {
        self.l1
            * maximum
            * f64::EPSILON
            * (64. * FFT.ilog2() as f64 + 16. * self.spectra.len() as f64)
    }
    fn input_spectrum<const REAL_INPUT: bool>(
        &self,
        state: &mut State,
        start: i128,
        input: &[C],
        origin: i128,
    ) {
        state.scratch.resize(FFT, C::default());
        if REAL_INPUT {
            // Preserve exactly the old admitted read range, including an odd
            // first retained sample and the padded initial history partitions.
            state.real_work.resize(BLOCK, C::default());
            for (i, z) in state.real_work.iter_mut().enumerate() {
                let at = start - BLOCK as i128 + 2 * i as i128;
                if at >= origin {
                    let j = usize::try_from(at - origin).expect("real coarse FFT even index");
                    assert!(j < input.len());
                    z.re = input[j].re;
                } else {
                    z.re = 0.;
                }
                if at + 1 >= origin {
                    let j = usize::try_from(at + 1 - origin).expect("real coarse FFT odd index");
                    assert!(j < input.len());
                    z.im = input[j].re;
                } else {
                    z.im = 0.;
                }
            }
            state.work.resize(self.scratch_len, C::default());
            self.real_forward
                .process_with_scratch(fft_slice(&mut state.real_work), fft_slice(&mut state.work));
            // Restore every bin of the full conjugate spectrum. Both live
            // buffers are overwritten; retaining their storage avoids a zero-fill.
            crate::real_fft_post_f64::forward(
                fft_slice(&mut state.real_work),
                &self.real_roots,
                fft_slice(&mut state.scratch),
            );
            return;
        }
        // Only the prefix before admitted input needs zero padding. Copy the
        // contiguous remainder once, without clearing samples we overwrite.
        let first = start - BLOCK as i128;
        let padding = (origin - first).clamp(0, FFT as i128) as usize;
        state.scratch[..padding].fill(C::default());
        if padding < FFT {
            let index =
                usize::try_from(first + padding as i128 - origin).expect("coarse FFT input index");
            state.scratch[padding..].copy_from_slice(&input[index..index + FFT - padding]);
        }
        state.work.resize(self.scratch_len, C::default());
        planned_transform(
            &mut state.scratch,
            self.forward.as_ref(),
            &mut state.work,
            false,
        );
    }
    #[cfg(test)]
    pub(crate) fn process(
        &self,
        state: &mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> Vec<C> {
        self.process_borrowed::<false>(state, start, count, input, origin)
            .to_vec()
    }
    /// The caller proves imaginary input is identically zero from clock geometry.
    #[cfg(test)]
    pub(crate) fn process_real_input(
        &self,
        state: &mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> Vec<C> {
        self.process_borrowed::<true>(state, start, count, input, origin)
            .to_vec()
    }
    /// Valid half-block borrowed until the next call, without temporary Vec.
    pub(crate) fn process_borrowed<'a, const REAL_INPUT: bool>(
        &self,
        state: &'a mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> &'a [C] {
        self.process_impl::<REAL_INPUT>(state, start, count, input, origin)
    }
    pub(crate) fn process_real_borrowed<'a>(
        &self,
        state: &'a mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> &'a [C] {
        let values = self.process_impl::<false>(state, start, count, input, origin);
        for z in values.iter_mut() {
            z.im = 0.;
        }
        values
    }
    fn process_impl<'a, const REAL_INPUT: bool>(
        &self,
        state: &'a mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> &'a mut [C] {
        assert_eq!(
            count, BLOCK,
            "partitioned correction follows the fixed coarse block"
        );
        let parts = self.spectra.len();
        if parts == 0 {
            state.scratch.resize(count, C::default());
            state.scratch.fill(C::default());
            return &mut state.scratch;
        }
        let input_start = start + self.zero_lag as i128;
        let history_len = if REAL_INPUT { BLOCK + 1 } else { FFT };
        if state.next.is_none() {
            state.real_input = REAL_INPUT;
            state.history = (0..parts)
                .map(|_| vec![C::default(); history_len])
                .collect();
            state.head = 0;
            for part in 0..parts {
                self.input_spectrum::<REAL_INPUT>(
                    state,
                    input_start - (part * BLOCK) as i128,
                    input,
                    origin,
                );
                let index = (parts - part) % parts;
                if REAL_INPUT {
                    state.history[index].copy_from_slice(&state.scratch[..history_len]);
                } else {
                    std::mem::swap(&mut state.history[index], &mut state.scratch);
                }
            }
        } else {
            assert_eq!(state.next, Some(start), "contiguous FFT correction clock");
            assert_eq!(
                state.real_input, REAL_INPUT,
                "fixed real/complex FFT history"
            );
            state.head = (state.head + 1) % parts;
            self.input_spectrum::<REAL_INPUT>(state, input_start, input, origin);
            if REAL_INPUT {
                state.history[state.head].copy_from_slice(&state.scratch[..history_len]);
            } else {
                std::mem::swap(&mut state.history[state.head], &mut state.scratch);
            }
        }
        state.scratch.fill(C::default());
        for (part, h) in self.spectra.iter().enumerate() {
            if h.is_empty() {
                continue;
            }
            let x = &state.history[(state.head + parts - part) % parts];
            if REAL_INPUT {
                self.real_mac.apply(&mut state.scratch, x, h)
            } else {
                (self.mac)(&mut state.scratch, x, h)
            };
        }
        planned_transform(
            &mut state.scratch,
            self.inverse.as_ref(),
            &mut state.work,
            true,
        );
        state.next = Some(start + count as i128);
        &mut state.scratch[BLOCK..]
    }
}
// Both private complex types are repr(C) pairs of f64, have equal layout,
// accept every initialized bit pattern and have no Drop. The exclusive bridge
// is scoped to a single FFT call; ownership and allocation remain with Vec<C>.
fn fft_slice(values: &mut [C]) -> &mut [F<f64>] {
    const {
        assert!(std::mem::size_of::<C>() == std::mem::size_of::<F<f64>>());
        assert!(std::mem::align_of::<C>() == std::mem::align_of::<F<f64>>());
    }
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<F<f64>>(), values.len()) }
}
fn planned_transform(values: &mut [C], plan: &dyn Fft<f64>, work: &mut [C], inverse: bool) {
    assert_eq!(values.len(), FFT);
    plan.process_with_scratch(fft_slice(values), fft_slice(work));
    if inverse {
        for z in values {
            z.re /= FFT as f64;
            z.im /= FFT as f64;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partitioned_complex_fir_matches_independent_dd_across_wraps_and_reset() {
        use crate::complex::WideComplex as WC;
        for (length, lag) in [(1, 0), (113, 17), (257, 99), (1801, 900), (3309, 1654)] {
            let coefficients: Vec<_> = (0..length)
                .map(|j| {
                    C::new(
                        ((j * 31 % 71) as f64 - 35.) / 71. * 1e-10,
                        ((j * 11 % 97) as f64 - 48.) / 97. * 1e-10,
                    )
                })
                .collect();
            let p = Plan::new(&coefficients, lag);
            let first = -256i128;
            let origin = first + lag as i128 - length as i128 + 1;
            let blocks = 2 * p.partition_count() + 5;
            let input: Vec<_> = (0..length + blocks * BLOCK + 2 * BLOCK)
                .map(|j| {
                    C::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let mut s = State::default();
            let mut maximum: f64 = 0.;
            let mut first_output = Vec::new();
            for block in 0..blocks {
                let at = first + (block * BLOCK) as i128;
                let actual = p.process(&mut s, at, BLOCK, &input, origin);
                if block == 0 {
                    first_output = actual.clone();
                }
                for (n, z) in actual.iter().enumerate() {
                    let q = at + n as i128 + lag as i128;
                    let mut expected = WC::real(0.);
                    for (j, c) in coefficients.iter().enumerate() {
                        expected = expected
                            + WC::from(*c) * WC::from(input[(q - j as i128 - origin) as usize]);
                    }
                    maximum = maximum.max((WC::from(*z) - expected).value().abs());
                }
            }
            assert!(maximum < 3e-21, "len{length} maximum{maximum:e}");
            assert!(maximum <= p.roundoff_screen(1.));
            s.reset();
            let again = p.process(&mut s, first, BLOCK, &input, origin);
            for (a, b) in again.iter().zip(&first_output) {
                assert_eq!(a.re.to_bits(), b.re.to_bits());
                assert_eq!(a.im.to_bits(), b.im.to_bits());
            }
            println!("COARSE_FFT length{length} lag{lag} parts{} blocks{blocks} maxDDerror{maximum:e} normwise_screen{:e}",p.partition_count(),p.roundoff_screen(1.));
        }
    }
}

#[cfg(test)]
mod real_input_tests {
    use super::*;
    #[test]
    fn real_input_spectrum_matches_independent_dense_dft_and_odd_admitted_origin() {
        let p = Plan::new(&[C::new(1., 0.)], 0);
        let start = 32i128;
        for omitted in [0usize, 1, 247] {
            for kind in 0..3 {
                let origin = start - BLOCK as i128 + omitted as i128;
                let input: Vec<_> = (0..FFT - omitted)
                    .map(|j| {
                        let at = origin + j as i128;
                        C::new(
                            match kind {
                                0 => 1.,
                                1 => {
                                    if at & 1 == 0 {
                                        1.
                                    } else {
                                        -1.
                                    }
                                }
                                _ => ((j * 37 % 127) as f64 - 63.) / 127.,
                            },
                            0.,
                        )
                    })
                    .collect();
                let mut s = State {
                    scratch: vec![C::new(f64::NAN, f64::NAN); FFT],
                    real_work: vec![C::new(f64::NAN, f64::NAN); BLOCK],
                    ..State::default()
                };
                p.input_spectrum::<true>(&mut s, start, &input, origin);
                assert!(s
                    .scratch
                    .iter()
                    .all(|z| z.re.is_finite() && z.im.is_finite()));
                let mut maximum: f64 = 0.;
                for k in 0..FFT {
                    let mut expected = C::default();
                    for (j, x) in input.iter().enumerate() {
                        let index = j + omitted;
                        let angle =
                            -2. * std::f64::consts::PI * k as f64 * index as f64 / FFT as f64;
                        expected = expected + C::new(angle.cos() * x.re, angle.sin() * x.re);
                    }
                    maximum = maximum.max((s.scratch[k] - expected).abs());
                }
                let bound = 8192. * f64::EPSILON * FFT as f64;
                println!("COARSE_REAL_DFT omitted{omitted} kind{kind} maximum{maximum:e} dense_trig_sum_screen{bound:e}");
                assert!(maximum < bound);
                assert_eq!(s.scratch[0].im.to_bits(), 0f64.to_bits());
                assert_eq!(s.scratch[BLOCK].im.to_bits(), 0f64.to_bits());
                for k in 1..BLOCK {
                    assert_eq!(s.scratch[FFT - k].re.to_bits(), s.scratch[k].re.to_bits());
                    assert_eq!(
                        s.scratch[FFT - k].im.to_bits(),
                        (-s.scratch[k].im).to_bits()
                    );
                }
                assert_eq!(s.real_work.len(), BLOCK);
                assert_eq!(s.real_work.capacity() * std::mem::size_of::<C>(), 8192);
            }
        }
    }
    #[test]
    fn real_input_partitioned_kernel_matches_independent_dd_wraps_and_reset() {
        use crate::complex::WideComplex as WC;
        for (length, lag) in [(1, 0), (113, 17), (257, 99), (1801, 900), (3309, 1654)] {
            let coefficients: Vec<_> = (0..length)
                .map(|j| {
                    C::new(
                        ((j * 31 % 71) as f64 - 35.) / 71. * 1e-10,
                        ((j * 11 % 97) as f64 - 48.) / 97. * 1e-10,
                    )
                })
                .collect();
            let p = Plan::new(&coefficients, lag);
            let first = -257i128;
            let origin = first + lag as i128 - length as i128 + 1;
            let blocks = 2 * p.partition_count() + 5;
            let input: Vec<_> = (0..length + blocks * BLOCK + 2 * BLOCK)
                .map(|j| C::new(((j * 37 % 127) as f64 - 63.) / 127., 0.))
                .collect();
            let mut state = State::default();
            let mut first_output = Vec::new();
            let mut maximum: f64 = 0.;
            for block in 0..blocks {
                let at = first + (block * BLOCK) as i128;
                let actual = p.process_real_input(&mut state, at, BLOCK, &input, origin);
                if block == 0 {
                    first_output = actual.clone();
                }
                for (n, z) in actual.iter().enumerate() {
                    let q = at + n as i128 + lag as i128;
                    let mut expected = WC::real(0.);
                    for (j, c) in coefficients.iter().enumerate() {
                        expected = expected
                            + WC::from(*c) * WC::from(input[(q - j as i128 - origin) as usize]);
                    }
                    maximum = maximum.max((WC::from(*z) - expected).value().abs());
                }
                assert_eq!(state.next, Some(at + BLOCK as i128));
            }
            println!("COARSE_REAL_KERNEL len{length} lag{lag} blocks{blocks} maximum_DD_error{maximum:e} normwise_screen{:e}",p.roundoff_screen(1.));
            assert!(maximum < 3e-21);
            assert!(maximum <= p.roundoff_screen(1.));
            let bytes = state.bytes();
            state.reset();
            assert!(state.next.is_none());
            let again = p.process_real_input(&mut state, first, BLOCK, &input, origin);
            assert_eq!(bytes, state.bytes());
            for (a, b) in again.iter().zip(&first_output) {
                assert_eq!(a.re.to_bits(), b.re.to_bits());
                assert_eq!(a.im.to_bits(), b.im.to_bits());
            }
        }
    }
}

#[cfg(test)]
#[path = "coarse_fft_full_history_reference.rs"]
mod full_history_reference;
#[cfg(test)]
mod compact_history_tests {
    use super::*;
    #[test]
    fn compact_real_history_preserves_full_reference_bits_wraps_reset_and_actual_capacity() {
        for (length, lag) in [(1, 0), (113, 17), (257, 99), (1801, 900), (3309, 1654)] {
            let coefficients: Vec<_> = (0..length)
                .map(|j| {
                    C::new(
                        ((j * 31 % 71) as f64 - 35.) / 71. * 1e-10,
                        ((j * 11 % 97) as f64 - 48.) / 97. * 1e-10,
                    )
                })
                .collect();
            let p = Plan::new(&coefficients, lag);
            let reference = full_history_reference::Plan::new(&coefficients, lag);
            let first = -257i128;
            let origin = first + lag as i128 - length as i128 + 1;
            let blocks = 2 * p.partition_count() + 5;
            let input: Vec<_> = (0..length + blocks * BLOCK + 2 * BLOCK)
                .map(|j| C::new(((j * 37 % 127) as f64 - 63.) / 127., 0.))
                .collect();
            let mut actual = State::default();
            let mut expected = full_history_reference::State::default();
            let mut first_output = Vec::new();
            for block in 0..blocks {
                let at = first + (block * BLOCK) as i128;
                let a = p.process_real_input(&mut actual, at, BLOCK, &input, origin);
                let b = reference.process_real_input(&mut expected, at, BLOCK, &input, origin);
                if block == 0 {
                    first_output = a.clone();
                }
                for (i, (a, b)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        a.re.to_bits(),
                        b.re.to_bits(),
                        "len{length},block{block},i{i},real"
                    );
                    assert_eq!(
                        a.im.to_bits(),
                        b.im.to_bits(),
                        "len{length},block{block},i{i},imag"
                    );
                }
                assert!(actual.history.iter().all(|h| h.len() == BLOCK + 1));
                assert_eq!(actual.next, Some(at + BLOCK as i128));
            }
            let saved = expected.bytes() - actual.bytes();
            let wanted = p.partition_count() * (FFT - BLOCK - 1) * std::mem::size_of::<C>();
            assert_eq!(saved, wanted);
            println!(
                "COMPACT_HISTORY len{length} parts{} saved{saved} actual_bytes{} full_bytes{}",
                p.partition_count(),
                actual.bytes(),
                expected.bytes()
            );
            actual.reset();
            let replay = p.process_real_input(&mut actual, first, BLOCK, &input, origin);
            for (a, b) in replay.iter().zip(&first_output) {
                assert_eq!(a.re.to_bits(), b.re.to_bits());
                assert_eq!(a.im.to_bits(), b.im.to_bits());
            }
        }
    }
    #[test]
    fn compact_history_keeps_full_complex_input_mode_unchanged() {
        let length = 1801;
        let lag = 900;
        let coefficients: Vec<_> = (0..length)
            .map(|j| {
                C::new(
                    ((j * 31 % 71) as f64 - 35.) / 71. * 1e-10,
                    ((j * 11 % 97) as f64 - 48.) / 97. * 1e-10,
                )
            })
            .collect();
        let p = Plan::new(&coefficients, lag);
        let reference = full_history_reference::Plan::new(&coefficients, lag);
        let first = -257i128;
        let origin = first + lag as i128 - length as i128 + 1;
        let blocks = 2 * p.partition_count() + 5;
        let input: Vec<_> = (0..length + blocks * BLOCK + 2 * BLOCK)
            .map(|j| {
                C::new(
                    ((j * 37 % 127) as f64 - 63.) / 127.,
                    ((j * 41 % 131) as f64 - 65.) / 131.,
                )
            })
            .collect();
        let mut actual = State::default();
        let mut expected = full_history_reference::State::default();
        for block in 0..blocks {
            let at = first + (block * BLOCK) as i128;
            let a = p.process(&mut actual, at, BLOCK, &input, origin);
            let b = reference.process(&mut expected, at, BLOCK, &input, origin);
            for (a, b) in a.iter().zip(&b) {
                assert_eq!(a.re.to_bits(), b.re.to_bits());
                assert_eq!(a.im.to_bits(), b.im.to_bits());
            }
        }
        assert!(actual.history.iter().all(|h| h.len() == FFT));
        assert_eq!(actual.bytes(), expected.bytes());
    }
}
