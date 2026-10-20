use super::*;

fn full_pair(c: Complex, p: Complex, q: Complex, past: bool) -> Complex {
    let one = Complex::new(1.0, 0.0);
    let v = if past { p * q } else { one / q };
    let vc = if past { p.conj() * q } else { one / q };
    let den = if past { one - v } else { one - p / q };
    let denc = if past { one - vc } else { one - p.conj() / q };
    (c * v / den + c.conj() * vc / denc) * Complex::new(0.5, 0.0)
}

/// Exact infinite transfer of the actual rounded phase bank for e^(iw n).
fn transfer(d: &TwoSidedDesign, phase: u32, relative: f64) -> Complex {
    let w = PI * relative * d.alpha;
    let q = Complex::new(w.cos(), -w.sin());
    let mut a = [Complex::default(); MODE_COUNT];
    let mut b = a;
    let mut h = Complex::new(d.phase(phase, &mut a, &mut b), 0.0);
    for j in 0..MODE_COUNT {
        h = h
            + full_pair(a[j], d.causal_poles[j], q, true)
            + full_pair(b[j], d.future_poles[j], q, false);
    }
    let angle = w * f64::from(phase) / f64::from(d.interpolation);
    h * Complex::new(angle.cos(), -angle.sin())
}

pub(crate) fn pairs() -> Vec<(u32, u32)> {
    let rates = [44_100, 48_000, 88_200, 96_000, 176_400, 192_000];
    let mut pairs: Vec<_> = rates
        .iter()
        .flat_map(|&a| rates.iter().map(move |&b| (a, b)))
        .collect();
    pairs.extend([
        (8_000, 16_000),
        (16_000, 8_000),
        (32_000, 48_000),
        (48_000, 32_000),
        (384_000, 8_000),
        (8_000, 384_000),
        (384_000, 44_100),
        (44_100, 384_000),
        (44_099, 48_000),
        (48_000, 44_099),
        (65_537, 65_539),
        (65_539, 65_537),
        (1_000_000_007, 1_000_000_009),
        (1_000_000_009, 1_000_000_007),
        (u32::MAX, u32::MAX - 1),
        (u32::MAX - 1, u32::MAX),
        (1, 2),
        (2, 1),
        (1, 4096),
        (64, 1),
        (1, 64),
        (511, 512),
        (512, 511),
        (512, 513),
        (513, 512),
    ]);
    pairs
}

#[test]
fn rounded_phase_matrix_dc_gain_phase_and_stopband() {
    for (input, output) in pairs().into_iter().filter(|(i, o)| i != o) {
        let d = make(input, output).unwrap();
        let l = d.interpolation;
        let phases: Vec<_> = if l <= 512 {
            (0..l).collect()
        } else {
            (0..65)
                .map(|k| ((u64::from(l) - 1) * k / 64) as u32)
                .collect()
        };
        let mut dc: f64 = 0.0;
        let mut gain: f64 = 0.0;
        let mut angle: f64 = 0.0;
        let mut rejection: f64 = 0.0;
        let mut images: f64 = 0.0;
        for r in phases {
            dc = dc.max((transfer(&d, r, 0.0).re - 1.0).abs());
            for k in 0..=96 {
                let f = PASSBAND * k as f64 / 96.0;
                let h = transfer(&d, r, f);
                // Supremum of periodic phase error also bounds every unwanted
                // Fourier coefficient (image) of this modulation, by triangle
                // inequality. Dense phase sampling is evidence, not a proof.
                images = images.max((h - d.response(f)).abs());
                gain = gain.max((20.0 * h.abs().log10()).abs());
                angle = angle.max(h.im.atan2(h.re).abs());
            }
            let highest_input = 1.0 / d.alpha;
            if highest_input >= STOPBAND {
                for k in 0..=96 {
                    let f = STOPBAND + (highest_input - STOPBAND) * k as f64 / 96.0;
                    rejection = rejection.max(transfer(&d, r, f).abs());
                }
            }
        }
        println!("MATRIX {{\"input\":{input},\"output\":{output},\"dc_error\":{dc:e},\"gain_db\":{gain:e},\"phase_rad\":{angle:e},\"image_envelope\":{images:e},\"stop_peak\":{rejection:e},\"guard\":{},\"tail_bound\":{},\"cached_phases\":{}}}",d.guard_frames,d.guard_bound,d.direct.len());
        assert!(dc < 1e-10, "{input}->{output}: DC {dc:e}");
        assert!(gain < 1e-6, "{input}->{output}: gain {gain:e}");
        assert!(angle < 1e-8, "{input}->{output}: phase {angle:e}");
        assert!(rejection <= 1e-9, "{input}->{output}: alias {rejection:e}");
        assert!(images <= 1e-9, "{input}->{output}: images {images:e}");
        assert!(d.guard_bound <= 1e-14);
        assert!(d.direct.len() <= MAX_TABLE_PHASES as usize);
    }
}

#[test]
fn normalized_model_dense_edges_moments_and_exp_precision() {
    let d = make(48_000, 44_100).unwrap();
    let mut pass: f64 = 0.0;
    let mut stop: f64 = 0.0;
    for k in 0..=32_768 {
        pass = pass.max((20.0 * d.response(PASSBAND * k as f64 / 32768.0).re.log10()).abs());
        let f = STOPBAND + (2.0 - STOPBAND) * k as f64 / 32768.0;
        stop = stop.max(d.response(f).re.abs());
    }
    assert!(pass <= 1e-6, "{pass:e}");
    assert!(stop <= 1e-9, "{stop:e}");
    for m in &d.modes {
        assert!(m.s.re.value() < 0.0 && m.s.value().abs() < 4.0);
        for scale in [1e-10, 0.01, 0.5, 1.0] {
            let h = exp_wide(m.s * WideComplex::real(scale)).value();
            let expected = exponential(m.s.value(), scale);
            assert!((h - expected).abs() < 1e-15);
        }
    }
    // Continuous DC = 1 and first eight even DC derivatives vanish.
    for n in 0..9 {
        let mut moment = Wide::from(0.0);
        for m in &d.modes {
            let mut p = m.s;
            for _ in 0..2 * n {
                p = p * m.s;
            }
            moment = moment + (WideComplex::real(-2.0) * m.c / p).re;
        }
        assert!((moment.value() - if n == 0 { 1.0 } else { 0.0 }).abs() < 1e-24);
    }
    // The first eight odd infinity moments suppress far-away sampling images.
    for n in 0..8 {
        let mut moment = Wide::from(0.0);
        for m in &d.modes {
            let mut power = m.s;
            for _ in 0..2 * n {
                power = power * m.s;
            }
            moment = moment + (m.c * power).re;
        }
        assert!(
            moment.value().abs() < 1e-20,
            "infinity moment {n}: {}",
            moment.value()
        );
    }
    println!("MODEL {{\"max_gain_db\":{pass:e},\"max_stop_amplitude\":{stop:e}}}");
}

#[test]
fn continuous_phase_envelope_bounds_omitted_terms_and_stable_poles() {
    for (input, output) in [
        (2, 1),
        (1, 2),
        (384000, 8000),
        (44100, 48000),
        (48000, 44099),
        (u32::MAX, u32::MAX - 1),
    ] {
        let d = make(input, output).unwrap();
        let mut a = [Complex::default(); MODE_COUNT];
        let mut b = a;
        for k in 0..257 {
            let r = (u64::from(d.interpolation - 1) * k / 256) as u32;
            d.phase(r, &mut a, &mut b);
            let mut bound = 0.0;
            for (j, coefficient) in b.iter().enumerate() {
                let p = d.future_poles[j].abs();
                assert!(p < 1.0 && p > 0.0);
                let weight = coefficient.abs() / (1.0 - p);
                assert!(weight <= d.weights[j].1);
                bound += weight * p.powf(d.future_guards[j] as f64);
            }
            assert!(bound <= d.guard_bound && bound < 1e-14);
        }
        assert!(tail_bound(&d.weights, d.guard_frames) <= TAIL_ERROR);
        assert!(tail_bound(&d.weights, d.guard_frames - 1) > TAIL_ERROR);
    }
    assert!(matches!(make(u32::MAX, 1), Err(Error::ResourceLimit)));
    assert!(matches!(make(4096, 1), Err(Error::ResourceLimit)));
}

#[test]
fn arbitrary_phase_caching_is_bit_identical_and_rate_scale_invariant() {
    for (input, output) in [(3, 2), (511, 512), (512, 511), (320, 147), (147, 320)] {
        let a = make(input, output).unwrap();
        let b = make(input * 1000, output * 1000).unwrap();
        assert_eq!(a.guard_frames, b.guard_frames);
        for phase in 0..a.interpolation {
            let mut c = [Complex::default(); MODE_COUNT];
            let mut f = c;
            assert_eq!(
                a.phase(phase, &mut c, &mut f).to_bits(),
                a.direct[phase as usize].to_bits()
            );
            for (j, coefficient) in c.iter().enumerate() {
                let stored = a.causal_phases[phase as usize * MODE_COUNT + j];
                let scaled = b.causal_phases[phase as usize * MODE_COUNT + j];
                assert_eq!(coefficient.re.to_bits(), stored.re.to_bits());
                assert_eq!(coefficient.im.to_bits(), stored.im.to_bits());
                assert_eq!(coefficient.re.to_bits(), scaled.re.to_bits());
                assert_eq!(coefficient.im.to_bits(), scaled.im.to_bits());
            }
        }
    }
}
