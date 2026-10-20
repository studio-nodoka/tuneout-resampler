//! Align the timestamp of DC and very low frequencies without changing gain.
//!
//! A first-order all-pass adds the fraction needed to make the prototype's
//! exact DC group delay an integer. Dropping that many initial output frames
//! then compensates the declared delay. Higher frequencies retain nonlinear
//! phase. The owner supplies the additional final frames; this module never
//! fabricates an endpoint or decides the complete stream's length.

use crate::{complex::Complex, design::Design, startup};

pub(crate) struct Alignment {
    pub advance_frames: usize,
    pub dc_delay: f64,
    pub coefficient: f64,
    pub remaining_drop: usize,
    // Previous input/output for the one-multiply all-pass recurrence. An
    // equivalent transposed initial state s is represented by [s, 0].
    states: Vec<[f64; 2]>,
}

impl Alignment {
    pub fn new(design: &Design, channels: usize) -> Self {
        // At DC both parallel branches equal one, so their summed response
        // has their mean delay. Each second-order section contributes
        // 2*(1-c)/(1+c), one branch has an extra unit delay, and the warp
        // scales all delays by (1+warp)/(1-warp). Convert virtual-rate ticks
        // to output frames using the decimation factor.
        let dc_delay = design.dc_delay();
        // Keep the fractional delay in [0.1, 1.1), avoiding a nearly cancelled
        // pole at -1 and its long Nyquist transient when delay approaches zero.
        let advance_frames = (dc_delay + 0.1).ceil() as usize;
        let fraction = advance_frames as f64 - dc_delay;
        // A(z)=(a+z^-1)/(1+a*z^-1) has DC delay (1-a)/(1+a).
        // At fraction=0 use identity directly, avoiding a removable pole -1.
        let coefficient = (1.0 - fraction) / (1.0 + fraction);
        Self {
            advance_frames,
            dc_delay,
            coefficient,
            remaining_drop: advance_frames,
            states: vec![[0.0; 2]; channels],
        }
    }

    pub fn reset(&mut self) {
        self.states.fill([0.0; 2]);
        self.remaining_drop = self.advance_frames;
    }

    /// Initialize from a deterministic leading segment before process().
    /// Empty prefixes retain zero state; short records use the estimator's
    /// bounded constant continuation. Initialization has no effect on timing.
    pub fn initialize(&mut self, raw_prefix_outputs: &[f64]) {
        if raw_prefix_outputs.is_empty() || self.coefficient == 1.0 {
            return;
        }
        let channels = self.states.len();
        debug_assert!(raw_prefix_outputs.len().is_multiple_of(channels));
        let raw_prefix_outputs =
            &raw_prefix_outputs[..raw_prefix_outputs.len().min(startup::FRAMES * channels)];
        let mut previous = vec![Complex::default(); channels];
        startup::initialize_lagged(
            raw_prefix_outputs,
            channels,
            &[Complex::new(-self.coefficient, 0.0)],
            &mut previous,
        );
        // For y[n]=a*x[n]+s[n], s[n+1]=(1-a*a)*x[n]-a*s[n].
        // The missing-state sum is (1-a*a)*sum x[-k]*(-a)^(k-1).
        // The lagged helper also covers a=0 without dividing by a tiny pole.
        let scale = (1.0 - self.coefficient) * (1.0 + self.coefficient);
        for (state, previous) in self.states.iter_mut().zip(previous) {
            *state = [scale * previous.re, 0.0];
        }
    }

    /// Filter all supplied frames, retaining state even for discarded frames.
    /// Later calls drop only any remaining part of the fixed initial advance.
    pub fn process(&mut self, output: &mut Vec<f64>) {
        let channels = self.states.len();
        debug_assert!(output.len().is_multiple_of(channels));
        if self.coefficient != 1.0 {
            for frame in output.chunks_exact_mut(channels) {
                for (sample, state) in frame.iter_mut().zip(&mut self.states) {
                    let filtered = self.coefficient * (*sample - state[1]) + state[0];
                    *state = [*sample, filtered];
                    *sample = filtered;
                }
            }
        }
        let frames = self.remaining_drop.min(output.len() / channels);
        if frames != 0 {
            let samples = frames * channels;
            output.copy_within(samples.., 0);
            output.truncate(output.len() - samples);
            self.remaining_drop -= frames;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design;
    use std::f64::consts::PI;

    #[test]
    fn measured_allpass_impulse_has_unit_gain_and_compensates_dc_delay() {
        for (input, output) in [(96_000, 44_100), (44_100, 48_000), (48_000, 96_000)] {
            let design = design::make(input, output).unwrap();
            let mut alignment = Alignment::new(&design, 1);
            let mut impulse = vec![0.0; 256];
            impulse[0] = 1.0;
            alignment.remaining_drop = 0;
            alignment.process(&mut impulse);
            for index in 0..=128 {
                let omega = PI * index as f64 / 128.0;
                let response =
                    impulse
                        .iter()
                        .enumerate()
                        .fold(Complex::default(), |sum, (sample, &value)| {
                            let angle = -omega * sample as f64;
                            sum + Complex::new(value * angle.cos(), value * angle.sin())
                        });
                assert!((response.abs() - 1.0).abs() < 5e-14);
            }
            let omega = 1e-5;
            let relative = omega * output as f64 / (PI * input.min(output) as f64);
            let prototype = design.response(relative);
            let measured_delay = -prototype.im.atan2(prototype.re) / omega;
            assert!((measured_delay - alignment.dc_delay).abs() < 1e-8);
            let ap =
                impulse
                    .iter()
                    .enumerate()
                    .fold(Complex::default(), |sum, (sample, &value)| {
                        let angle = -omega * sample as f64;
                        sum + Complex::new(value * angle.cos(), value * angle.sin())
                    });
            let complete = prototype * ap;
            let delay = -complete.im.atan2(complete.re) / omega;
            assert!((delay - alignment.advance_frames as f64).abs() < 1e-8);
        }
        for rate in [1, 2, 3, 7, 11, 147, 160, 320, 4095, 4096] {
            for (input, output) in [(rate, 1), (1, rate)] {
                if input == output {
                    continue;
                }
                let design = design::make(input, output).unwrap();
                let alignment = Alignment::new(&design, 1);
                assert!(alignment.coefficient.abs() < 0.819);
            }
        }
    }

    #[test]
    fn chunking_reset_and_short_initialization_preserve_alignment() {
        let design = design::make(96_000, 44_100).unwrap();
        let input: Vec<_> = (0..400)
            .flat_map(|n| [(n as f64 * 0.37).sin(), (n as f64 * 0.71).cos()])
            .collect();
        let mut alignment = Alignment::new(&design, 2);
        let mut expected = input.clone();
        alignment.initialize(&input[..256]);
        alignment.process(&mut expected);
        alignment.reset();
        alignment.initialize(&input[..256]);
        let mut actual = Vec::new();
        for block in input.chunks(14) {
            let mut block = block.to_vec();
            alignment.process(&mut block);
            actual.extend(block);
        }
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), input.len() - 2 * alignment.advance_frames);
        assert_eq!(alignment.remaining_drop, 0);

        for coefficient in [0.0, 1e-15, 0.5, 1.0 - 1e-14, 1.0] {
            let mut alignment = Alignment {
                advance_frames: 0,
                dc_delay: 0.0,
                coefficient,
                remaining_drop: 0,
                states: vec![[0.0; 2]; 2],
            };
            alignment.initialize(&[]);
            alignment.initialize(&[0.75, -0.25]);
            let mut samples: Vec<_> = (0..16).flat_map(|_| [0.75, -0.25]).collect();
            alignment.process(&mut samples);
            for frame in samples.chunks_exact(2) {
                assert!((frame[0] - 0.75).abs() < 1e-12);
                assert!((frame[1] + 0.25).abs() < 1e-12);
            }
        }
    }
}
