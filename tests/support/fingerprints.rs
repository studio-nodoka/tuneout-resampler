use tuneout_resampler::{FilterLength, FilterLengthPolicy, Resampler};
mod cases;

pub const HEADER: &str = "from,to,coefficients,frames,f64,f32";

fn fingerprint(samples: &[f64]) -> u64 {
    samples
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

pub fn rows(policy: FilterLengthPolicy) -> Vec<String> {
    let input: Vec<f64> = (0..257 * 2)
        .map(|i| {
            let bits = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (bits >> 32) as i32 as f64 / 2147483648.0 * 0.5
        })
        .collect();
    let input32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
    cases::pairs()
        .into_iter()
        .map(|(from, to)| {
            let mut r =
                Resampler::with_filter_length_policy(from, to, 2, FilterLength::Standard, policy)
                    .unwrap();
            let coefficients = fingerprint(r.coefficients());
            let mut output = Vec::new();
            for chunk in input.chunks(113 * 2) {
                output.extend(r.process_f64(chunk, false).unwrap());
            }
            output.extend(r.process_f64(&[], true).unwrap());
            let frames = output.len() / 2;
            let f64_hash = fingerprint(&output);
            r.reset();
            output.clear();
            for chunk in input32.chunks(67 * 2) {
                output.extend(
                    r.process_f32(chunk, false)
                        .unwrap()
                        .into_iter()
                        .map(f64::from),
                );
            }
            output.extend(r.process_f32(&[], true).unwrap().into_iter().map(f64::from));
            format!(
                "{from},{to},{coefficients:016x},{frames},{f64_hash:016x},{:016x}",
                fingerprint(&output)
            )
        })
        .collect()
}

pub const PRESET_HEADER: &str = "from,to,preset,taps,coefficients";

pub fn preset_rows(policy: FilterLengthPolicy) -> Vec<String> {
    cases::pairs()
        .into_iter()
        .flat_map(|(from, to)| {
            [
                ("standard", FilterLength::Standard),
                ("long", FilterLength::Long),
                ("extra-long", FilterLength::ExtraLong),
            ]
            .map(|(name, length)| {
                let r = Resampler::with_filter_length_policy(from, to, 1, length, policy).unwrap();
                format!(
                    "{from},{to},{name},{},{:016x}",
                    r.filter_info().taps_per_phase,
                    fingerprint(r.coefficients())
                )
            })
        })
        .collect()
}
