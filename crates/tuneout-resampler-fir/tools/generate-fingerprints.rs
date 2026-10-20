//! Print regression fingerprints for review. The checked-in fixture is unchanged.
#[path = "../tests/support/fingerprints.rs"]
mod fingerprints;
use tuneout_resampler::FilterLengthPolicy;

fn main() {
    let mut presets = false;
    let mut policy = FilterLengthPolicy::Generic;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--presets" => presets = true,
            "--fixed" => policy = FilterLengthPolicy::Fixed,
            _ => panic!("usage: fingerprints [--presets] [--fixed]"),
        }
    }
    let (header, rows) = if presets {
        (
            fingerprints::PRESET_HEADER,
            fingerprints::preset_rows(policy),
        )
    } else {
        (fingerprints::HEADER, fingerprints::rows(policy))
    };
    println!("{header}");
    for row in rows {
        println!("{row}");
    }
}
