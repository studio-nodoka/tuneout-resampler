//! Uniformly partitioned overlap-save for the finite coarse correction.
//! The material IIR core remains separate. Input blocks start at output_start
//! +zero_lag; zero_lag need not be a multiple of the partition length.
use crate::complex::Complex as C;
use rustfft::{num_complex::Complex as F, Fft, FftPlanner};
use std::sync::Arc;
pub(crate) const BLOCK: usize = 512;
const FFT: usize = 2 * BLOCK;
pub(crate) struct Plan {
    mac: fn(&mut [C], &[C], &[C]),
    spectra: Vec<Vec<C>>,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    real_forward: Arc<dyn Fft<f64>>,
    real_roots: Vec<F<f64>>,
    scratch_len: usize,
    zero_lag: usize,
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
            mac: crate::simd::Kernel::coarse_detect().coarse_mac_function(),
            spectra,
            forward,
            inverse,
            real_forward,
            real_roots,
            scratch_len,
            zero_lag,
        }
    }
    fn input_spectrum<const REAL_INPUT: bool>(
        &self,
        state: &mut State,
        start: i128,
        input: &[C],
        origin: i128,
    ) {
        state.scratch.clear();
        state.scratch.resize(FFT, C::default());
        if REAL_INPUT {
            // Preserve exactly the old admitted read range, including an odd
            // first retained sample and the padded initial history partitions.
            state.real_work.clear();
            state.real_work.resize(BLOCK, C::default());
            for (i, z) in state.real_work.iter_mut().enumerate() {
                let at = start - BLOCK as i128 + 2 * i as i128;
                if at >= origin {
                    let j = usize::try_from(at - origin).expect("real coarse FFT even index");
                    assert!(j < input.len());
                    z.re = input[j].re;
                }
                if at + 1 >= origin {
                    let j = usize::try_from(at + 1 - origin).expect("real coarse FFT odd index");
                    assert!(j < input.len());
                    z.im = input[j].re;
                }
            }
            state.work.resize(self.scratch_len, C::default());
            self.real_forward
                .process_with_scratch(fft_slice(&mut state.real_work), fft_slice(&mut state.work));
            // Restore the full conjugate spectrum of real input. The finite
            // kernel spectra remain complex and the inverse is unchanged.
            crate::real_fft_post_f64::forward(
                fft_slice(&mut state.real_work),
                &self.real_roots,
                fft_slice(&mut state.scratch),
            );
            return;
        }
        for (i, z) in state.scratch.iter_mut().enumerate() {
            let at = start - BLOCK as i128 + i as i128;
            // The first saved partition may extend before the admitted input.
            // Those samples multiply exact zero padding in the final kernel
            // partition, so they cannot affect the linear valid output.
            if at >= origin {
                let j = usize::try_from(at - origin).expect("coarse FFT input index");
                assert!(j < input.len());
                *z = input[j];
            }
        }
        state.work.resize(self.scratch_len, C::default());
        planned_transform(
            &mut state.scratch,
            self.forward.as_ref(),
            &mut state.work,
            false,
        );
    }
    pub(crate) fn process(
        &self,
        state: &mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> Vec<C> {
        self.process_impl::<false>(state, start, count, input, origin)
    }
    /// The caller proves imaginary input is identically zero from clock geometry.
    pub(crate) fn process_real_input(
        &self,
        state: &mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> Vec<C> {
        self.process_impl::<true>(state, start, count, input, origin)
    }
    fn process_impl<const REAL_INPUT: bool>(
        &self,
        state: &mut State,
        start: i128,
        count: usize,
        input: &[C],
        origin: i128,
    ) -> Vec<C> {
        assert_eq!(
            count, BLOCK,
            "partitioned correction follows the fixed coarse block"
        );
        let parts = self.spectra.len();
        if parts == 0 {
            return vec![C::default(); count];
        }
        let input_start = start + self.zero_lag as i128;
        if state.next.is_none() {
            state.real_input = REAL_INPUT;
            state.history = (0..parts).map(|_| vec![C::default(); FFT]).collect();
            state.head = 0;
            for part in 0..parts {
                self.input_spectrum::<REAL_INPUT>(
                    state,
                    input_start - (part * BLOCK) as i128,
                    input,
                    origin,
                );
                let index = (parts - part) % parts;
                state.history[index].copy_from_slice(&state.scratch);
            }
        } else {
            assert_eq!(state.next, Some(start), "contiguous FFT correction clock");
            assert_eq!(
                state.real_input, REAL_INPUT,
                "fixed real/complex FFT history"
            );
            state.head = (state.head + 1) % parts;
            self.input_spectrum::<REAL_INPUT>(state, input_start, input, origin);
            state.history[state.head].copy_from_slice(&state.scratch);
        }
        state.scratch.fill(C::default());
        for (part, h) in self.spectra.iter().enumerate() {
            if h.is_empty() {
                continue;
            }
            let x = &state.history[(state.head + parts - part) % parts];
            (self.mac)(&mut state.scratch, x, h);
        }
        planned_transform(
            &mut state.scratch,
            self.inverse.as_ref(),
            &mut state.work,
            true,
        );
        state.next = Some(start + count as i128);
        state.scratch[BLOCK..].to_vec()
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
