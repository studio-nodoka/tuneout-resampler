//! Identical warmed block workloads for the experimental IIR and production FIR.
use std::{error::Error, hint::black_box, time::Instant};
use tuneout_resampler::{FilterLength, FilterLengthPolicy, FiniteResampler as Fir};
use tuneout_resampler_iir::Resampler as Iir;

fn main() -> Result<(), Box<dyn Error>> {
    println!("input_rate,output_rate,channels,block_frames,iir_multiplications,fir_taps_per_phase,fir_multiplications,iir_ms_per_second,fir_ms_per_second,iir_speedup,iir_setup_ms,fir_setup_ms,iir_first_block_ms,iir_startup_lookahead_frames");
    for (input_rate, output_rate) in [
        (96000, 44100),
        (44100, 48000),
        (48000, 96000),
        (192000, 44100),
    ] {
        let now = Instant::now();
        let mut iir = Iir::new(input_rate, output_rate, 2)?;
        let iir_setup = now.elapsed().as_secs_f64() * 1000.0;
        let now = Instant::now();
        let mut fir = Fir::with_filter_length_policy(
            input_rate,
            output_rate,
            2,
            FilterLength::Standard,
            FilterLengthPolicy::Fixed,
        )?;
        let fir_setup = now.elapsed().as_secs_f64() * 1000.0;
        for block in [256, 1024] {
            let input: Vec<_> = (0..2 * block)
                .map(|i| ((i * 731 % 997) as f64 - 498.0) / 1994.0)
                .collect();
            let mut output = Vec::new();
            iir.reset();
            let started = Instant::now();
            iir.process_f64_into(&input, false, &mut output)?;
            let iir_first_block_ms = started.elapsed().as_secs_f64() * 1000.0;
            iir.reset();
            fir.reset();
            let blocks = (8 * input_rate as usize / block).max(1);
            for _ in 0..blocks / 8 {
                iir.process_f64_into(&input, false, &mut output)?;
                fir.process_f64_into(&input, false, &mut output)?;
            }
            let mut timings = [Vec::new(), Vec::new()];
            for round in 0..5 {
                for engine in [round % 2, 1 - round % 2] {
                    let started = Instant::now();
                    for _ in 0..blocks {
                        if engine == 0 {
                            iir.process_f64_into(black_box(&input), false, &mut output)?;
                        } else {
                            fir.process_f64_into(black_box(&input), false, &mut output)?;
                        }
                        black_box(&output);
                    }
                    timings[engine].push(
                        started.elapsed().as_secs_f64() * 1000.0 * f64::from(input_rate)
                            / (blocks * block) as f64,
                    );
                }
            }
            for values in &mut timings {
                values.sort_by(f64::total_cmp);
            }
            let iir_ms = timings[0][2];
            let fir_ms = timings[1][2];
            let taps = fir.filter_info().taps_per_phase;
            let fir_mult = taps as f64 * f64::from(output_rate) / f64::from(input_rate);
            println!("{input_rate},{output_rate},2,{block},{:.4},{taps},{fir_mult:.4},{iir_ms:.6},{fir_ms:.6},{:.4},{iir_setup:.6},{fir_setup:.6},{iir_first_block_ms:.6},{}",iir.filter_info().multiplications_per_input_sample,fir_ms/iir_ms,iir.filter_info().startup_lookahead_frames);
        }
    }
    Ok(())
}
