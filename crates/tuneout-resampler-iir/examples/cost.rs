//! Native setup, persistent allocation and warm streaming cost (no disk I/O).
use std::{hint::black_box, time::Instant};
use tuneout_resampler_iir::{BufferedResampler, Startup};
fn main() {
    for (input_rate, output_rate, frames) in [
        (44100, 48000, 441000),
        (48000, 44100, 480000),
        (96000, 44100, 960000),
        (48000, 96000, 480000),
        (384000, 8000, 3840000),
        (8000, 384000, 80000),
        (48000, 44099, 480000),
        (65537, 65539, 655370),
        (u32::MAX, u32::MAX - 1, 480000),
        (48000, 48000, 480000),
    ] {
        let channels = 2;
        let start = Instant::now();
        for _ in 0..100 {
            black_box(BufferedResampler::new(input_rate, output_rate, channels).unwrap());
        }
        let setup_us = start.elapsed().as_secs_f64() * 1e6 / 100.0;
        let mut r =
            BufferedResampler::with_startup(input_rate, output_rate, channels, Startup::Zero)
                .unwrap();
        let info = r.filter_info();
        let x: Vec<_> = (0..frames * channels)
            .map(|j| ((j * 1237 % 1013) as f64 - 506.0) / 1013.0)
            .collect();
        let mut out = Vec::new();
        let mut timings = Vec::new();
        let mut total = 0;
        for _ in 0..3 {
            r.reset();
            total = 0;
            let start = Instant::now();
            for chunk in x.chunks(1024 * channels) {
                r.process_f64_into(chunk, false, &mut out).unwrap();
                total += out.len();
                black_box(&out);
            }
            r.process_f64_into(&[], true, &mut out).unwrap();
            total += out.len();
            black_box(&out);
            timings.push(start.elapsed().as_secs_f64());
        }
        let mut sorted = timings.clone();
        sorted.sort_by(f64::total_cmp);
        let seconds = sorted[1];
        println!("{{\"input\":{input_rate},\"output\":{output_rate},\"channels\":{channels},\"input_frames\":{frames},\"output_frames\":{},\"setup_us\":{setup_us},\"process_seconds\":{seconds},\"process_runs_seconds\":{timings:?},\"million_input_samples_per_second\":{},\"coefficient_bytes\":{},\"cached_phases\":{},\"working_buffer_bytes\":{},\"retained_buffer_bytes\":{},\"output_capacity_bytes\":{},\"lookahead_frames\":{}}}",
            total/channels,frames as f64*channels as f64/seconds/1e6,info.coefficient_bytes,
            info.cached_phases,info.working_buffer_bytes,r.filter_info().working_buffer_bytes,
            out.capacity()*8,info.startup_lookahead_frames);
    }
}
