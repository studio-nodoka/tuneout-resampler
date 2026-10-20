use super::*;
#[test]
fn translated_sinc_phase_rows_match_original_dd_weight() {
    for (half, beta, period) in [
        (20, 32., 17),
        (24, 32., 147),
        (24, 32., 640),
        (40, 32., 127),
    ] {
        let den = crate::core_pipeline::i0(W::from(beta));
        let p = Kaiser::new(beta, den).unwrap();
        let mut arguments = Vec::new();
        let mut sines = Vec::new();
        for phase in 1..period {
            let u = W::from(phase as f64) / W::from(period as f64);
            sines.push(
                crate::prefix::exp(WC {
                    re: W::from(0.),
                    im: pi() * u,
                })
                .im,
            );
            for j in -(half as i64) + 1..=half as i64 {
                arguments.push(u - W::from(j as f64));
            }
        }
        let actual = p.weights_phased(&arguments, half, &sines);
        let mut maximum: f64 = 0.;
        let mut changed = 0;
        for (i, (&x, &v)) in arguments.iter().zip(&actual).enumerate() {
            let mut sine = sines[i / (2 * half)];
            if (half - 1 + i) & 1 != 0 {
                sine = W::from(0.) - sine;
            }
            let value = p.weight_with_sine(x, half, Some(sine));
            let expected = crate::core_pipeline::weight(x, half, beta, den);
            maximum = maximum.max((value - expected).value().abs());
            if v.to_bits() != expected.value().to_bits() {
                changed += 1;
            }
            assert_eq!(v.to_bits(), value.value().to_bits());
        }
        assert!(
            maximum < 5e-27,
            "half{half},period{period},error{maximum:e}"
        );
        println!("KAISER_PHASE,half={half},period={period},DDerror={maximum:e},rounded_changes={changed}/{}",arguments.len());
    }
}
