//! Experimental exported joint-design bank, not a production converter.
//!
//! `design_iir_joint.py` exports poles at the INPUT rate and phase residues.
//! This runner exercises the actual pure-IIR rational recurrence without
//! inserted zeros, FIR fallback, startup extrapolation or delay correction.
//! A stable export may still fail the stringent phase/alias qualification.

use std::{error::Error, fs, str::SplitWhitespace};

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn times(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn plus_real(self, value: f64) -> Self {
        Self {
            re: self.re + value,
            im: self.im,
        }
    }
}

fn integer(tokens: &mut SplitWhitespace<'_>) -> Result<usize, Box<dyn Error>> {
    Ok(tokens.next().ok_or("incomplete bank")?.parse()?)
}

fn real(tokens: &mut SplitWhitespace<'_>) -> Result<f64, Box<dyn Error>> {
    let value: f64 = tokens.next().ok_or("incomplete bank")?.parse()?;
    if !value.is_finite() {
        return Err("nonfinite bank coefficient".into());
    }
    Ok(value)
}

fn complex(tokens: &mut SplitWhitespace<'_>) -> Result<Complex, Box<dyn Error>> {
    Ok(Complex {
        re: real(tokens)?,
        im: real(tokens)?,
    })
}

struct Bank {
    interpolation: usize,
    decimation: usize,
    poles: Vec<Complex>,
    phases: Vec<Complex>,
    direct: Vec<f64>,
}

impl Bank {
    fn read(text: &str) -> Result<Self, Box<dyn Error>> {
        let mut tokens = text.split_whitespace();
        let interpolation = integer(&mut tokens)?;
        let decimation = integer(&mut tokens)?;
        let modes = integer(&mut tokens)?;
        if !(1..=4096).contains(&interpolation)
            || !(1..=4096).contains(&decimation)
            || !(1..=512).contains(&modes)
        {
            return Err("bank dimension outside diagnostic limits".into());
        }
        let poles: Vec<_> = (0..modes)
            .map(|_| complex(&mut tokens))
            .collect::<Result<_, _>>()?;
        if poles.iter().any(|p| p.re.hypot(p.im) >= 1.0) {
            return Err("bank has an unstable pole".into());
        }
        let mut phases = Vec::with_capacity(interpolation * modes);
        let mut direct = Vec::with_capacity(interpolation);
        for _ in 0..interpolation {
            direct.push(real(&mut tokens)?);
            for _ in 0..modes {
                phases.push(complex(&mut tokens)?);
            }
        }
        if tokens.next().is_some() {
            return Err("trailing bank data".into());
        }
        Ok(Self {
            interpolation,
            decimation,
            poles,
            phases,
            direct,
        })
    }

    fn convert(&self, input: &[f64], channels: usize) -> Result<Vec<f64>, Box<dyn Error>> {
        if !(1..=64).contains(&channels) || !input.len().is_multiple_of(channels) {
            return Err("invalid channel count or incomplete input frame".into());
        }
        let modes = self.poles.len();
        let frames = input.len() / channels;
        let mut states = vec![Complex::default(); channels * modes];
        let total = frames
            .checked_mul(self.interpolation)
            .ok_or("sample count overflow")?
            .div_ceil(self.decimation);
        let mut output = Vec::with_capacity(total * channels);
        let mut next_input = 0;
        let mut phase = 0;
        for (index, frame) in input.chunks_exact(channels).enumerate() {
            while next_input == index {
                let coefficients = &self.phases[phase * modes..(phase + 1) * modes];
                for channel in 0..channels {
                    // Four partial sums reduce long cancellation chains.
                    let mut sum = [0.0; 4];
                    for (mode, &coefficient) in coefficients.iter().enumerate() {
                        sum[mode % 4] += coefficient.times(states[channel * modes + mode]).re;
                    }
                    let value = self.direct[phase] * frame[channel]
                        + ((sum[0] + sum[1]) + (sum[2] + sum[3]));
                    if !value.is_finite() {
                        return Err("candidate produced nonfinite output".into());
                    }
                    output.push(value);
                }
                let advanced = phase + self.decimation;
                next_input += advanced / self.interpolation;
                phase = advanced % self.interpolation;
            }
            for channel in 0..channels {
                for (mode, &pole) in self.poles.iter().enumerate() {
                    let state = &mut states[channel * modes + mode];
                    *state = pole.times(state.plus_real(frame[channel]));
                }
            }
        }
        Ok(output)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: raw_joint BANK.txt CHANNELS INPUT.f64le OUTPUT.f64le".into());
    }
    let bank = Bank::read(&fs::read_to_string(&args[1])?)?;
    let channels: usize = args[2].parse()?;
    let bytes = fs::read(&args[3])?;
    if !bytes.len().is_multiple_of(8) {
        return Err("incomplete little-endian f64 sample".into());
    }
    let input: Vec<_> = bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight bytes")))
        .collect();
    if input.iter().any(|sample| !sample.is_finite()) {
        return Err("nonfinite input sample".into());
    }
    let output = bank.convert(&input, channels)?;
    let mut bytes = Vec::with_capacity(output.len() * 8);
    for sample in output {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(&args[4], bytes)?;
    Ok(())
}
