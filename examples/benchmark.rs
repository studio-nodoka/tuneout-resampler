//! Measures software conversion cost. It does not measure a device, driver
//! or OS mixer.
mod support;

use std::{error::Error, hint::black_box, time::Instant};
use tuneout_resampler::Resampler;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if !(5..=7).contains(&args.len()) {
        return Err("usage: benchmark INPUT_HZ OUTPUT_HZ CHANNELS BLOCK_FRAMES [LENGTH] [POLICY]\nLENGTH: standard, long, extra-long, or an odd tap count\nPOLICY: generic (default) or fixed".into());
    }
    let from: u32 = args[1].parse()?;
    let to: u32 = args[2].parse()?;
    let channels: usize = args[3].parse()?;
    let block: usize = args[4].parse()?;
    let filter_length = support::filter_length(args.get(5))?;
    let policy = support::filter_length_policy(args.get(6))?;
    if block == 0 {
        return Err("block size must be positive".into());
    }
    let started = Instant::now();
    let converter =
        Resampler::with_filter_length_policy(from, to, channels, filter_length, policy)?;
    let first_setup = started.elapsed();
    let info = converter.filter_info();
    let started = Instant::now();
    let mut converter =
        Resampler::with_filter_length_policy(from, to, channels, filter_length, policy)?;
    let cached_setup = started.elapsed();
    let frames = from as usize;
    let samples = frames.checked_mul(channels).ok_or("input size overflow")?;
    let chunk_size = block.checked_mul(channels).ok_or("block size overflow")?;
    let input: Vec<f64> = (0..samples)
        .map(|i| (i as f64 * 0.017).sin() * 0.25)
        .collect();
    let mut output = Vec::new();
    let mut times = Vec::new();
    for round in 0..6 {
        converter.reset();
        let started = Instant::now();
        let mut made = 0;
        for chunk in input.chunks(chunk_size) {
            converter.process_f64_into(black_box(chunk), false, &mut output)?;
            made += black_box(&output).len();
        }
        converter.process_f64_into(&[], true, &mut output)?;
        made += black_box(&output).len();
        assert_eq!(made / channels, to as usize);
        if round != 0 {
            times.push(started.elapsed().as_secs_f64() * 1000.0);
        }
    }
    times.sort_by(f64::total_cmp);
    println!("{from} -> {to} Hz, {channels} channels, {block} input frames/call");
    println!(
        "filter: {filter_length:?}; policy: {policy:?}; taps per phase: {}",
        info.taps_per_phase
    );
    println!(
        "first setup: {:.6} ms; cached setup: {:.6} ms",
        first_setup.as_secs_f64() * 1000.0,
        cached_setup.as_secs_f64() * 1000.0
    );
    println!(
        "processing, median of 5 after warmup: {:.6} ms/audio second",
        times[2]
    );
    println!(
        "coefficients: {} bytes; filter lookahead: {:.3} ms; additional batching bound: {:.3} ms",
        info.coefficient_bytes,
        info.lookahead_input_frames as f64 * 1000.0 / f64::from(from),
        info.additional_batch_input_frames as f64 * 1000.0 / f64::from(from)
    );
    Ok(())
}
