use super::*;

#[test]
fn rational_phase_mirrors_match_independent_original_dd_rows() {
    for (input, output) in [
        (44100, 48000),
        (48000, 44100),
        (96000, 44100),
        (192000, 44100),
        (48000, 96000),
        (441, 479),
        (257, 263),
    ] {
        let pipeline = CorePipeline::new(input, output);
        let width = 2 * prefix::HALF;
        assert_eq!(pipeline.synthesis.len(), pipeline.period as usize * width);
        let denominator = u64::from(output) * pipeline.r as u64;
        let mut maximum: f64 = 0.;
        let mut changed = 0usize;
        for out in 1..pipeline.period {
            let rem = (u128::from(out) * u128::from(input)) % u128::from(denominator);
            let u = W::from(rem as f64) / W::from(denominator as f64);
            for j in 0..width {
                let x = u - W::from(j as f64 - (prefix::HALF as f64 - 1.));
                let expected = weight(x, prefix::HALF, prefix::BETA, pipeline.den);
                let actual = pipeline.synthesis[out as usize * width + j];
                maximum = maximum.max((W::from(actual) - expected).value().abs());
                if actual.to_bits() != expected.value().to_bits() {
                    changed += 1;
                }
                if pipeline.period - out != out {
                    assert_eq!(
                        actual.to_bits(),
                        pipeline.synthesis
                            [(pipeline.period - out) as usize * width + width - 1 - j]
                            .to_bits()
                    );
                }
            }
        }
        assert!(maximum <= 1.12e-16, "{input}/{output},maximum{maximum:e}");
        println!("SYNTHESIS_SYMMETRY,{input},{output},period={},rounded_changes={changed}/{},rounded_error={maximum:e}",pipeline.period,pipeline.synthesis.len()-width);
    }
}
