//! Raw little-endian f64 transport for reproducible resampler measurements.
mod support;

use std::{error::Error, fs};
use tuneout_resampler::Resampler;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: raw coefficients IN_HZ OUT_HZ OUTPUT [LENGTH] [POLICY] | raw convert IN_HZ OUT_HZ CHANNELS INPUT OUTPUT BLOCK_FRAMES f64|f32 [LENGTH] [POLICY]\nLENGTH: standard, long, extra-long, or an odd tap count\nPOLICY: generic (default) or fixed".into());
    }
    let input_rate = args[2].parse()?;
    let output_rate = args[3].parse()?;
    if args[1] == "coefficients" && (5..=7).contains(&args.len()) {
        let converter = Resampler::with_filter_length_policy(
            input_rate,
            output_rate,
            1,
            support::filter_length(args.get(5))?,
            support::filter_length_policy(args.get(6))?,
        )?;
        let bytes: Vec<u8> = converter
            .coefficients()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        fs::write(&args[4], bytes)?;
        return Ok(());
    }
    if args[1] != "convert"
        || !(9..=11).contains(&args.len())
        || !["f64", "f32"].contains(&args[8].as_str())
    {
        return Err("invalid arguments".into());
    }
    let channels: usize = args[4].parse()?;
    let mut converter = Resampler::with_filter_length_policy(
        input_rate,
        output_rate,
        channels,
        support::filter_length(args.get(9))?,
        support::filter_length_policy(args.get(10))?,
    )?;
    let bytes = fs::read(&args[5])?;
    if !bytes.len().is_multiple_of(8) {
        return Err("raw input must contain complete little-endian f64 samples".into());
    }
    let input: Vec<f64> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&b| f64::from_le_bytes(b))
        .collect();
    let block: usize = args[7].parse()?;
    if block == 0 {
        return Err("block size must be positive".into());
    }
    let mut output = Vec::new();
    let block_samples = block.checked_mul(channels).ok_or("block size overflow")?;
    for chunk in input.chunks(block_samples).chain(std::iter::once(&[][..])) {
        let end_of_stream = chunk.is_empty();
        if args[8] == "f32" {
            let narrowed: Vec<f32> = chunk.iter().map(|&v| v as f32).collect();
            output.extend(
                converter
                    .process_f32(&narrowed, end_of_stream)?
                    .into_iter()
                    .map(f64::from),
            );
        } else {
            output.extend(converter.process_f64(chunk, end_of_stream)?);
        }
    }
    let bytes: Vec<u8> = output.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(&args[6], bytes)?;
    Ok(())
}
