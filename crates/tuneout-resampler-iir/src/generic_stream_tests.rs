use crate::{BufferedResampler, Error, Startup};

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn f32_convenience_path_narrows_only_final_output() {
    for (i, o) in [
        (48000, 44100),
        (44100, 48000),
        (48000, 44099),
        (48000, 48000),
    ] {
        let x: Vec<f32> = input(600).into_iter().map(|x| x as f32).collect();
        let promoted: Vec<f64> = x.iter().map(|&x| f64::from(x)).collect();
        let mut a = BufferedResampler::new(i, o, 2).unwrap();
        let mut b = BufferedResampler::new(i, o, 2).unwrap();
        let actual = a.process_f32(&x, true).unwrap();
        let expected: Vec<f32> = b
            .process_f64(&promoted, true)
            .unwrap()
            .into_iter()
            .map(|x| x as f32)
            .collect();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
fn count(n: usize, i: u32, o: u32) -> usize {
    (n as u128 * o as u128).div_ceil(i as u128) as usize
}
fn input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| ((j * 1237 % 1013) as f64 - 506.0) / 1013.0)
        .collect()
}

#[test]
fn matrix_stream_clock_chunks_gapless_reset_and_short_eos() {
    for (i, o) in crate::bidirectional_design::tests::pairs() {
        if (i, o) == (1, 4096) {
            assert!(matches!(
                BufferedResampler::new(i, o, 1),
                Err(Error::ResourceLimit)
            ));
            continue;
        }
        for startup in [Startup::Zero, Startup::Predict] {
            let mut r = BufferedResampler::with_startup(i, o, 1, startup).unwrap();
            for n in [0, 1, 2, 7, 129] {
                r.reset();
                let x = input(n);
                let expected = r.process_f64(&x, true).unwrap();
                assert_eq!(expected.len(), count(n, i, o));
                assert!(expected.iter().all(|x| x.is_finite()));
                r.reset();
                let mut actual = Vec::new();
                for chunk in x.chunks(17) {
                    actual.extend(r.process_f64(chunk, false).unwrap());
                }
                actual.extend(r.process_f64(&[], true).unwrap());
                assert_eq!(bits(&actual), bits(&expected), "{i}->{o} {n} {startup:?}");
                assert!(r.process_f64(&[], true).unwrap().is_empty());
                assert_eq!(
                    r.process_f64(&[0.0], false).unwrap_err(),
                    Error::StreamFinished
                );
            }
            // Large expansions are covered above without allocating their
            // entire lookahead window's output at once in the test harness.
            if o as u64 > i as u64 * 8 {
                continue;
            }
            r.reset();
            let info = r.filter_info();
            let n = info.startup_lookahead_frames + 2 * info.block_input_frames + 271;
            let x = input(n);
            let expected = r.process_f64(&x, true).unwrap();
            assert_eq!(expected.len(), count(n, i, o));
            r.reset();
            let mut actual = Vec::new();
            // Continuous tracks use the very same object and do not send EOS.
            for chunk in x.chunks(997) {
                actual.extend(r.process_f64(chunk, false).unwrap());
            }
            assert!(info.bypass || !actual.is_empty());
            actual.extend(r.process_f64(&[], true).unwrap());
            assert_eq!(
                bits(&actual),
                bits(&expected),
                "stream {i}->{o} {startup:?}"
            );
        }
    }
}

#[test]
fn resource_rejection_nonfinite_and_frame_errors_are_transactional() {
    assert!(matches!(
        BufferedResampler::new(4096, 1, 1),
        Err(Error::ResourceLimit)
    ));
    assert!(matches!(
        BufferedResampler::new(384000, 8000, 64),
        Err(Error::ResourceLimit)
    ));
    assert!(matches!(
        BufferedResampler::new(1, u32::MAX, 1),
        Err(Error::ResourceLimit)
    ));
    assert!(matches!(
        BufferedResampler::new(8000, 384000, 7),
        Err(Error::ResourceLimit)
    ));
    let mut r = BufferedResampler::new(1, 64, 2).unwrap();
    // The generic finite-tail wrapper returns ordinary f64 PCM, capped at
    // 64 MiB per call. Its internal core Wide buffers are separately bounded;
    // they no longer halve the final PCM cap. Exceed the actual returned cap
    // by one stereo input frame and verify that rejection consumes nothing.
    assert_eq!(
        r.process_f64(&vec![0.25; 2 * 65537], true).unwrap_err(),
        Error::ResourceLimit
    );
    assert!(r.process_f64(&[], true).unwrap().is_empty());
    for startup in [Startup::Predict, Startup::Zero] {
        let mut r = BufferedResampler::with_startup(48000, 44099, 2, startup).unwrap();
        let mut fresh = BufferedResampler::with_startup(48000, 44099, 2, startup).unwrap();
        assert_eq!(
            r.process_f64(&[0.0], false).unwrap_err(),
            Error::IncompleteFrame
        );
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                r.process_f64(&[0.0, bad], true).unwrap_err(),
                Error::NonFiniteInput
            );
        }
        let x = input(600);
        assert_eq!(
            bits(&r.process_f64(&x, true).unwrap()),
            bits(&fresh.process_f64(&x, true).unwrap())
        );
    }
}

#[test]
fn matrix_dc_channel_isolation_and_actual_lookahead() {
    for (i, o) in [
        (44100, 48000),
        (48000, 44100),
        (8000, 16000),
        (32000, 8000),
        (384000, 44100),
        (44100, 384000),
        (44099, 48000),
        (48000, 44099),
        (u32::MAX, u32::MAX - 1),
    ] {
        let mut r = BufferedResampler::new(i, o, 4).unwrap();
        let info = r.filter_info();
        let before = vec![0.0; 4 * (info.startup_lookahead_frames - 1)];
        assert!(r.process_f64(&before, false).unwrap().is_empty());
        assert_eq!(
            r.process_f64(&[0.0; 4], false).unwrap().len(),
            4 * count(4096, i, o)
        );
        r.reset();
        let x: Vec<_> = (0..(info.startup_lookahead_frames + 4097))
            .flat_map(|_| [0.75, 0.0, -0.25, 0.0])
            .collect();
        let out = r.process_f64(&x, true).unwrap();
        for f in out.chunks_exact(4) {
            assert!((f[0] - 0.75).abs() < 1e-10, "{i}->{o} {}", f[0]);
            assert!((f[2] + 0.25).abs() < 1e-10);
            assert_eq!(f[1], 0.0);
            assert_eq!(f[3], 0.0);
        }
    }
}
