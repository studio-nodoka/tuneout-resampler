use crate::{complex::Wide, BufferedResampler, Startup};
#[test]
fn whole_multirate_core_impulse_matches_analytic_selected_points() {
    let from = 96000;
    let to = 44100;
    let center = 64000usize;
    let mut input = vec![0.; 128001];
    input[center] = 1.;
    let mut r = BufferedResampler::with_startup(from, to, 1, Startup::Zero).unwrap();
    let info = r.filter_info();
    let y = r.process_f64(&input, true).unwrap();
    let mut maximum: f64 = 0.;
    let mut where_at = 0;
    let mut selected: Vec<_> = (0..y.len()).step_by(269).collect();
    selected.extend(29300..29501);
    for out in selected {
        let signed_t = out as f64 * from.min(to) as f64 / to as f64
            - center as f64 * from.min(to) as f64 / from as f64;
        let t = signed_t.abs();
        let mut h = crate::prefix::modes()
            .into_iter()
            .fold(Wide::from(0.), |sum, (s, c)| {
                sum + (c * crate::prefix::exp(s.scale(Wide::from(t)))).re
            })
            .value();
        for &(a, b, sr, si, cr, ci) in crate::late_tail_coefficients::MODES {
            if t >= a as f64 && t < b as f64 {
                let u = t - a as f64;
                let (sn, cs) = (si * u).sin_cos();
                h += (sr * u).exp() * (cr * cs - ci * sn);
            }
        }
        let correction = crate::compact_stopband::oracle_delta(Wide::from(t))
            * (Wide::from(1.) - crate::prefix::turnon(Wide::from(t)));
        let pi = Wide::from(std::f64::consts::PI) + Wide::from(1.2246467991473532e-16);
        let carrier = crate::prefix::exp(crate::complex::WideComplex {
            re: Wide::from(0.),
            im: pi * Wide::from(signed_t),
        });
        let edge = (carrier * crate::joint_edge::oracle_envelope(t, signed_t < 0.))
            .re
            .value();
        let want =
            (h * crate::cosine_shape::factor(Wide::from(t)).value() + correction.value() + edge)
                * from.min(to) as f64
                / from as f64;
        let error = (y[out] - want).abs();
        if error > maximum {
            maximum = error;
            where_at = out;
        }
    }
    println!("whole core: max selected analytic impulse error {maximum:.17e} at {where_at}; startup {} frames",info.startup_lookahead_frames);
    assert!(maximum < 2e-14);
}
