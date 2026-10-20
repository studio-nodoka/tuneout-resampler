//! Measurement transport: little-endian f64 samples, no WAV/DSP dependency.
use std::{error::Error, fs};
use tuneout_resampler_iir::{InfiniteResampler, Resampler, Startup};

enum Converter {
    LowDelay(Box<Resampler>),
    Buffered(Box<InfiniteResampler>),
}

impl Converter {
    fn process(
        &mut self,
        input: &[f64],
        eos: bool,
        narrow: bool,
    ) -> Result<Vec<f64>, Box<dyn Error>> {
        if narrow {
            let input: Vec<_> = input.iter().map(|&x| x as f32).collect();
            let output = match self {
                Self::LowDelay(c) => c.process_f32(&input, eos)?,
                Self::Buffered(c) => c.process_f32(&input, eos)?,
            };
            Ok(output.into_iter().map(f64::from).collect())
        } else {
            Ok(match self {
                Self::LowDelay(c) => c.process_f64(input, eos)?,
                Self::Buffered(c) => c.process_f64(input, eos)?,
            })
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: raw info IN_HZ OUT_HZ [buffered] | raw convert IN_HZ OUT_HZ CHANNELS INPUT OUTPUT BLOCK_FRAMES f64|f32 [predict|zero|predict-causal|buffered|buffered-zero]".into());
    }
    let input_rate = args[2].parse()?;
    let output_rate = args[3].parse()?;
    if args[1] == "info" && args.len() == 5 && args[4] == "buffered" {
        let converter = InfiniteResampler::new(input_rate, output_rate, 1)?;
        let info = converter.filter_info();
        let (coefficients, warp) = converter.prototype();
        println!("{{\"mode\":\"buffered\",\"response_shape\":\"{}\",\"input_rate\":{},\"output_rate\":{},\"interpolation\":{},\"decimation\":{},\"order\":{},\"finite_correction_modes\":{},\"finite_correction_half_support_frames\":{},\"prototype_order\":{},\"recursive_modes\":{},\"causal_modes\":{},\"future_modes\":{},\"ideal_zero_phase\":{},\"coefficient_bytes\":{},\"multiplications_per_input_sample\":{},\"passband\":{},\"stopband\":{},\"elliptic_seed_attenuation_db\":{},\"flatness_moments\":{},\"max_flatness_coefficient_change\":{},\"max_dc_correction\":{},\"startup_lookahead_frames\":{},\"guard_input_frames\":{},\"guard_mode_frames\":{},\"min_guard_input_frames\":{},\"block_input_frames\":{},\"guard_error_bound\":{},\"warp\":{},\"coefficients\":{:?}}}",
            if info.bypass { "bypass" } else { "normalized_modal_sparse_tail" },
            info.input_rate, info.output_rate, info.interpolation, info.decimation, info.filter_order, info.finite_correction_modes, info.finite_correction_half_support_frames, info.prototype_order,
            info.causal_modes + info.future_modes, info.causal_modes, info.future_modes, info.ideal_zero_phase,
            info.coefficient_bytes, info.multiplications_per_input_sample,
            info.passband_fraction_of_lower_nyquist, info.stopband_fraction_of_lower_nyquist, info.design_attenuation_db,
            info.flatness_moments, info.max_flatness_coefficient_change,
            info.max_dc_correction, info.startup_lookahead_frames, info.guard_input_frames,
            info.guard_mode_frames, info.min_guard_input_frames,
            info.block_input_frames, info.guard_error_bound, warp, coefficients);
        return Ok(());
    }
    if args[1] == "info" && args.len() == 4 {
        let converter = Resampler::new(input_rate, output_rate, 1)?;
        let info = converter.filter_info();
        let (coefficients, warp) = converter.prototype();
        println!("{{\"input_rate\":{},\"output_rate\":{},\"interpolation\":{},\"decimation\":{},\"order\":{},\"recursive_modes\":{},\"coefficient_bytes\":{},\"multiplications_per_input_sample\":{},\"passband\":{},\"stopband\":{},\"attenuation_target_db\":{},\"max_dc_correction\":{},\"startup_lookahead_frames\":{},\"dc_group_delay_frames\":{},\"alignment_advance_frames\":{},\"warp\":{},\"coefficients\":{:?}}}",
            info.input_rate, info.output_rate, info.interpolation, info.decimation, info.filter_order,
            info.recursive_modes, info.coefficient_bytes, info.multiplications_per_input_sample,
            info.passband_fraction_of_lower_nyquist, info.stopband_fraction_of_lower_nyquist,
            info.design_attenuation_db, info.max_dc_correction, info.startup_lookahead_frames,
            info.dc_group_delay_frames, info.alignment_advance_frames, warp, coefficients);
        return Ok(());
    }
    if args[1] != "convert"
        || !(9..=10).contains(&args.len())
        || !["f64", "f32"].contains(&args[8].as_str())
    {
        return Err("invalid arguments".into());
    }
    let channels: usize = args[4].parse()?;
    let mut converter = match args.get(9).map(String::as_str).unwrap_or("predict") {
        "predict" => {
            Converter::LowDelay(Box::new(Resampler::new(input_rate, output_rate, channels)?))
        }
        "zero" => Converter::LowDelay(Box::new(Resampler::with_startup(
            input_rate,
            output_rate,
            channels,
            Startup::Zero,
        )?)),
        "predict-causal" => Converter::LowDelay(Box::new(Resampler::with_startup(
            input_rate,
            output_rate,
            channels,
            Startup::Predict,
        )?)),
        "buffered" => Converter::Buffered(Box::new(InfiniteResampler::new(
            input_rate,
            output_rate,
            channels,
        )?)),
        "buffered-zero" => Converter::Buffered(Box::new(InfiniteResampler::with_startup(
            input_rate,
            output_rate,
            channels,
            Startup::Zero,
        )?)),
        _ => return Err("invalid converter mode".into()),
    };
    let bytes = fs::read(&args[5])?;
    if !bytes.len().is_multiple_of(8) {
        return Err("incomplete f64 sample".into());
    }
    let input: Vec<_> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let block: usize = args[7].parse()?;
    if block == 0 || !input.len().is_multiple_of(channels) {
        return Err("block must be positive and input must contain complete frames".into());
    }
    let block = block.checked_mul(channels).ok_or("block size overflow")?;
    let mut output = Vec::new();
    for chunk in input.chunks(block).chain(std::iter::once(&[][..])) {
        output.extend(converter.process(chunk, chunk.is_empty(), args[8] == "f32")?);
    }
    let bytes: Vec<u8> = output.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(&args[6], bytes)?;
    Ok(())
}
