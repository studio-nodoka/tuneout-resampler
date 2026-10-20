use super::*;

#[test]
fn independent_half_generation_preserves_qualified_phase_bits() {
    // Compare both retained rows and the original direct-term storage rounding,
    // including the 512/513 cache boundary and the largest reduced clock.
    for (input, output) in [
        (96000, 44100),
        (44100, 96000),
        (511, 512),
        (512, 513),
        (44100, 192000),
        (48000, 44099),
        (u32::MAX, u32::MAX - 1),
    ] {
        let d = make(input, output).unwrap();
        let mut past = [Complex::default(); MODE_COUNT];
        let mut future = past;
        let mut high = past;
        let mut low = past;
        for j in 0..=32 {
            let r = (u64::from(d.interpolation - 1) * j / 32) as u32;
            let direct = d.phase(r, &mut past, &mut future);
            let single = d.past_phase(r, &mut high, &mut low);
            assert_eq!(direct.to_bits(), single.value().to_bits());
            for (expected, actual) in past.iter().zip(&high) {
                assert_eq!(expected.re.to_bits(), actual.re.to_bits());
                assert_eq!(expected.im.to_bits(), actual.im.to_bits());
            }
            d.future_phase(r, &mut high, &mut low);
            for (expected, actual) in future.iter().zip(&high) {
                assert_eq!(expected.re.to_bits(), actual.re.to_bits());
                assert_eq!(expected.im.to_bits(), actual.im.to_bits());
            }
        }
    }
}

#[test]
fn generic_dd_phase_fresh_cache_and_rational_scaling() {
    for (input, output) in [
        (3, 2),
        (320, 147),
        (147, 320),
        (511, 512),
        (512, 511),
        (513, 512),
    ] {
        let d = make(input, output).unwrap();
        let scaled = make(input * 1000, output * 1000).unwrap();
        let mut high = [Complex::default(); MODE_COUNT];
        let mut low = high;
        for r in 0..d.interpolation {
            let direct = d.past_phase(r, &mut high, &mut low);
            assert_eq!(direct.value().to_bits(), d.direct[r as usize].to_bits());
            assert_eq!(
                direct.value().to_bits(),
                scaled.direct[r as usize].to_bits()
            );
            for j in 0..MODE_COUNT {
                let index = r as usize * MODE_COUNT + j;
                assert_eq!(high[j].re.to_bits(), d.causal_phases[index].re.to_bits());
                assert_eq!(high[j].im.to_bits(), d.causal_phases[index].im.to_bits());
                assert_eq!(
                    high[j].re.to_bits(),
                    scaled.causal_phases[index].re.to_bits()
                );
                assert_eq!(
                    high[j].im.to_bits(),
                    scaled.causal_phases[index].im.to_bits()
                );
                assert!(low[j].abs() <= high[j].abs() * 2e-16);
            }
        }
    }
    for (input, output) in [(512, 513), (48000, 44099), (u32::MAX, u32::MAX - 1)] {
        let d = make(input, output).unwrap();
        assert!(d.causal_phases.is_empty() && d.future_phases.is_empty() && d.direct.is_empty());
        let mut a = [Complex::default(); MODE_COUNT];
        let mut b = a;
        for r in [0, 1, d.interpolation / 2, d.interpolation - 1] {
            let direct = d.phase(r, &mut a, &mut b);
            assert!(direct.is_finite());
            assert!(a.iter().chain(&b).all(|p| p.abs().is_finite()));
        }
    }
}

#[test]
fn generic_dd_phase_export_actual_setup_for_independent_audit() {
    let Ok(directory) = std::env::var("GENERIC_PHASE_AUDIT_DIR") else {
        return;
    };
    std::fs::create_dir_all(&directory).unwrap();
    for (input, output) in [
        (96000, 44100),
        (44100, 96000),
        (48000, 44099),
        (511, 512),
        (512, 513),
        (64, 1),
        (1, 96),
        (u32::MAX, u32::MAX - 1),
    ] {
        let d = make(input, output).unwrap();
        let phases: Vec<_> = if d.interpolation <= 512 {
            (0..d.interpolation).collect()
        } else {
            (0..=32)
                .map(|j| (u64::from(d.interpolation - 1) * j / 32) as u32)
                .collect()
        };
        let complex = |values: &[Complex]| {
            values
                .iter()
                .map(|p| format!("[{:.17e},{:.17e}]", p.re, p.im))
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut a = [Complex::default(); MODE_COUNT];
        let mut al = a;
        let mut b = a;
        let mut bl = a;
        let mut rows = Vec::new();
        for r in phases {
            let direct = d.past_phase(r, &mut a, &mut al);
            d.future_phase(r, &mut b, &mut bl);
            let (dh, dl) = direct.parts();
            rows.push(format!("{{\"phase\":{r},\"direct_high\":{dh:.17e},\"direct_low\":{dl:.17e},\"past_high\":[{}],\"past_low\":[{}],\"future_high\":[{}],\"future_low\":[{}]}}",complex(&a),complex(&al),complex(&b),complex(&bl)));
        }
        let text = format!("{{\"input_rate\":{input},\"output_rate\":{output},\"interpolation\":{},\"decimation\":{},\"poles\":[{}],\"rows\":[{}]}}",d.interpolation,d.decimation,complex(&d.causal_poles),rows.join(","));
        std::fs::write(
            std::path::Path::new(&directory).join(format!("phase-bank-{input}-{output}.json")),
            text,
        )
        .unwrap();
    }
}
