use super::*;

#[test]
fn cardinal_exact_clock_identities_and_coefficient_support() {
    for (input, output) in [
        (44100, 48000),
        (48000, 96000),
        (32000, 48000),
        (48000, 44100),
        (96000, 44100),
        (192000, 44100),
        (48000, 24000),
    ] {
        let p = CorePipeline::new(input, output);
        if input <= output {
            assert!(p.analysis_im.is_empty());
            assert!(p.input_rotation.is_empty());
            if p.r == 1 {
                assert_eq!(p.analysis_re, vec![1.]);
                let origin = -41;
                let x: Vec<_> = (0..97).map(|i| (i as f64 - 41.) / 97.).collect();
                for m in -40..=40 {
                    let y = p.low_sample(m, &x, origin);
                    assert_eq!(y.re, parity(x[(m - origin) as usize], m));
                    assert_eq!(y.im, 0.);
                }
            }
            for offset in [-1153, -2, -1, 0, 1, 2, 1153] {
                let z = rotation(offset, p.l, p.k);
                assert_eq!(z.re.value(), parity(1., offset));
                assert_eq!(z.im.value(), 0.);
            }
        }
        if input >= output {
            assert!(p.output_rotation.is_empty());
            for out in [0, 1, 2, 3, 101, u64::MAX - 1, u64::MAX] {
                let z = output_rotation(out, input, output);
                assert_eq!(z.re.value(), parity(1., i128::from(out)));
                assert_eq!(z.im.value(), 0.);
            }
        }
        if !p.synthesis.is_empty() {
            for (i, &w) in p.synthesis[..2 * prefix::HALF].iter().enumerate() {
                assert_eq!(w, if i == prefix::HALF - 1 { 1. } else { 0. });
            }
        }
        let legacy = cardinal_reference::CorePipeline::new(input, output);
        assert!(p.omitted_future_bound().is_finite());
        assert!(p.omitted_future_bound() <= legacy.omitted_future_bound() * (1. + 1e-12) + 1e-30);
    }
}

#[test]
fn cardinal_runtime_matches_clock_reference_and_reset() {
    for (input, output, count) in [
        (44100, 48000, 4153),
        (48000, 44100, 4153),
        (96000, 44100, 4153),
        (192000, 44100, 4153),
        (48000, 96000, 4153),
        (48000, 24000, 4153),
        (32000, 48000, 4153),
        (48000, 48001, 31),
    ] {
        let mut p = CorePipeline::new(input, output);
        p.projected = None;
        let reference = cardinal_reference::CorePipeline::new(input, output);
        let end = CorePipeline::lookahead_for(input, output);
        let origin = -(end as i128);
        let input_span = (count as u64 * u64::from(input)).div_ceil(u64::from(output)) as usize;
        let x: Vec<_> = (0..2 * end + input_span + 64)
            .map(|i| {
                let centered = i as i128 + origin;
                let noise = ((i * 31 % 71) as f64 - 35.) / 71.;
                noise * 0.5 + if centered == 0 { 0.5 } else { 0. }
            })
            .collect();
        let mut a = PipelineState::default();
        let mut b = cardinal_reference::PipelineState::default();
        let mut actual = vec![W::from(0.); count];
        let mut expected = actual.clone();
        let mut first = 0;
        let chunks = [1, 17, 1024, 57, 2049, 511];
        let mut j = 0;
        while first < count {
            let n = chunks[j % chunks.len()].min(count - first);
            p.output_block(
                &mut a,
                first as u64,
                &mut actual[first..first + n],
                &x,
                origin,
            );
            reference.output_block(
                &mut b,
                first as u64,
                &mut expected[first..first + n],
                &x,
                origin,
            );
            first += n;
            j += 1;
        }
        let maximum = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (*a - *b).value().abs())
            .fold(0., f64::max);
        // Unit-bounded PCM. This threshold is far below the existing 1e-19
        // independent model qualification and includes f64 reassociation.
        assert!(
            maximum < 1e-22,
            "{input}->{output} legacy difference={maximum:e}"
        );
        println!("CARDINAL,{input}->{output},maximum_clock_difference={maximum:e},period={},analysis_taps={},analysis_im_taps={}",p.period,p.analysis_re.len(),p.analysis_im.len());
        if input >= output {
            assert!(a.q_im.is_empty());
        }
        a.reset();
        assert!(a.output_clock.is_none());
        assert!(a.low.is_empty());
        assert!(a.q_re.is_empty());
        assert!(a.q_im.is_empty());
        let mut replay = vec![W::from(0.); count];
        p.output_block(&mut a, 0, &mut replay, &x, origin);
        for (i, (a, b)) in actual.iter().zip(&replay).enumerate() {
            assert_eq!(a.parts(), b.parts(), "{input}->{output} reset output {i}");
        }
    }
}

#[test]
fn cardinal_buffered_stereo_chunks_match_one_shot_and_reset() {
    use crate::{BufferedResampler, Startup};
    for (input, output) in [
        (44100, 48000),
        (48000, 44100),
        (96000, 44100),
        (192000, 44100),
        (48000, 96000),
        (48000, 24000),
    ] {
        let x: Vec<_> = (0..4097)
            .flat_map(|i| {
                [
                    (((i * 31 % 71) as f64) - 35.) / 71.,
                    if i == 31 { 0.75 } else { 0. },
                ]
            })
            .collect();
        let mut r = BufferedResampler::with_startup(input, output, 2, Startup::Zero).unwrap();
        let whole = r.process_f64(&x, true).unwrap();
        r.reset();
        let mut stream = Vec::new();
        let mut first = 0;
        let chunks = [1, 17, 257, 2049, 511];
        let mut j = 0;
        while first < 4097 {
            let count = chunks[j % chunks.len()].min(4097 - first);
            stream.extend(
                r.process_f64(&x[2 * first..2 * (first + count)], false)
                    .unwrap(),
            );
            first += count;
            j += 1;
        }
        stream.extend(r.process_f64(&[], true).unwrap());
        assert_eq!(whole.len(), stream.len());
        for (i, (a, b)) in whole.iter().zip(stream).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{input}->{output} stream sample {i}"
            );
        }
        r.reset();
        let replay = r.process_f64(&x, true).unwrap();
        assert_eq!(whole, replay);
    }
}

#[test]
fn exact_clock_carry_matches_u128_for_extreme_rates_and_chunks() {
    for (input, output) in [
        (1, 1),
        (1, u32::MAX),
        (u32::MAX, 1),
        (u32::MAX, u32::MAX - 1),
        (u32::MAX - 1, u32::MAX),
        (44100, 48000),
        (48000, 44100),
        (96000, 44100),
        (192000, 44100),
        (48000, 96000),
        (48000, 48001),
    ] {
        let factor = CorePipeline::factor(input, output);
        let denominator = u64::from(output) * factor as u64;
        let step = u64::from(input);
        let whole = step / denominator;
        let fraction = step % denominator;
        let outputs = denominator / step;
        let remainder = denominator % step;
        for first in [0, 1, 131, (1_u64 << 60) + 77, u64::MAX - 1024] {
            let ticks = u128::from(first) * u128::from(input);
            let mut center = (ticks / u128::from(denominator)) as i128;
            let mut rem = (ticks % u128::from(denominator)) as u64;
            let mut consumed = 0;
            for size in [1, 17, 31, 257, 511] {
                let mut done = 0;
                while done < size {
                    let actual_same = if whole != 0 {
                        1
                    } else if rem < step {
                        outputs + u64::from(rem < remainder)
                    } else {
                        (denominator - 1 - rem) / step + 1
                    };
                    let original_same = (denominator - 1 - rem) / step + 1;
                    assert_eq!(actual_same, original_same);
                    let bounded = usize::try_from(actual_same).unwrap_or(usize::MAX);
                    let count = (size - done).min(bounded);
                    let next = rem + count as u64 * fraction;
                    assert!(next < 2 * denominator);
                    center += i128::from(count as u64 * whole);
                    if next >= denominator {
                        center += 1;
                        rem = next - denominator;
                    } else {
                        rem = next;
                    }
                    done += count;
                    consumed += count;
                    let ticks = u128::from(first + consumed as u64) * u128::from(input);
                    assert_eq!(center, (ticks / u128::from(denominator)) as i128);
                    assert_eq!(rem, (ticks % u128::from(denominator)) as u64);
                }
            }
        }
    }
}

#[test]
fn cardinal_quarter_turn_projection_uses_one_exact_plane() {
    for (input, output) in [
        (48000, 96000),
        (32000, 48000),
        (44100, 48000),
        (96000, 48000),
    ] {
        let p = CorePipeline::new(input, output);
        let n = 2 * prefix::HALF;
        let real: Vec<_> = (0..n).map(|i| ((i * 31 % 71) as f64 - 35.) / 71.).collect();
        let imag: Vec<_> = (0..n).map(|i| ((i * 11 % 97) as f64 - 48.) / 97.).collect();
        let weights: Vec<_> = (0..n)
            .map(|i| ((i * 17 % 103) as f64 - 51.) / 103.)
            .collect();
        let re = crate::f64_dot::dot(&real, &weights);
        let im = crate::f64_dot::dot(&imag, &weights);
        let mut axis_count = 0;
        for out in 0..320 {
            let rem = u128::from(out) * u128::from(input.min(output)) % (2 * u128::from(output));
            let twice = 2 * rem;
            let expected = if rem == 0 {
                Some((false, re))
            } else if rem == u128::from(output) {
                Some((false, -re))
            } else if twice == u128::from(output) {
                Some((true, -im))
            } else if twice == 3 * u128::from(output) {
                Some((true, im))
            } else {
                None
            };
            if let Some((imaginary, value)) = expected {
                let r = output_rotation(out, input, output);
                assert!(r.re.parts() == (0., 0.) || r.im.parts() == (0., 0.));
                let actual = if imaginary {
                    p.synthesis_value(out, &[], &imag, &weights, crate::real_carrier::select())
                } else {
                    p.synthesis_value(out, &real, &[], &weights, crate::real_carrier::select())
                };
                assert_eq!(actual.parts(), W::from(value).parts());
                axis_count += 1;
            }
        }
        assert!(axis_count > 0);
        if output == 2 * input {
            assert_eq!(axis_count, 320);
        }
        for out in [0, 1, 2, 3, u64::MAX - 1, u64::MAX] {
            if output == 2 * input {
                let r = output_rotation(out, input, output);
                let expected = match out % 4 {
                    0 => (1., 0.),
                    1 => (0., 1.),
                    2 => (-1., 0.),
                    _ => (0., -1.),
                };
                assert_eq!((r.re.value(), r.im.value()), expected);
            }
        }
    }
}
