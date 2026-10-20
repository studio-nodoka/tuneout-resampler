use super::*;
fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}
#[test]
fn mirrored_stereo_integer_and_mono_windows_match_original_rolling_bits() {
    for (from, to, channels) in [(96000, 44100, 2), (48000, 96000, 2), (44100, 48000, 1)] {
        for startup in [Startup::Zero, Startup::Predict] {
            let mut actual = InfiniteResampler::with_startup(from, to, channels, startup).unwrap();
            let mut reference =
                InfiniteResampler::with_startup(from, to, channels, startup).unwrap();
            assert!(actual.mirrored_windows);
            reference.mirrored_windows = false;
            let length = 2 * actual.end + BLOCK;
            let frames = length * 3 + actual.lookahead + BLOCK + 137;
            let input: Vec<_> = (0..frames * channels)
                .map(|i| {
                    if i % 911 == 0 {
                        -0.0
                    } else if i % 919 == 0 {
                        f64::from_bits(1)
                    } else {
                        ((i * 37 + i / channels * 11) % 997) as f64 / 1994. - 0.25
                    }
                })
                .collect();
            let mut pos = 0;
            let mut calls = 0;
            let mut emitted = 0;
            let mut wraps = 0;
            let mut heads = vec![0; channels];
            let chunks = [733, 4096, 17, 5003, 255, 1024];
            while pos < frames {
                let end = (pos + chunks[calls % chunks.len()]).min(frames);
                let x = &input[pos * channels..end * channels];
                let a = actual.process_f64(x, false).unwrap();
                let b = reference.process_f64(x, false).unwrap();
                assert_eq!(
                    bits(&a),
                    bits(&b),
                    "{from}->{to} {channels}ch {startup:?} call{calls}"
                );
                emitted += a.len();
                for (ch, cache) in actual.rolling_windows.iter().enumerate() {
                    if cache.head < heads[ch] {
                        wraps += 1;
                    }
                    heads[ch] = cache.head;
                }
                pos = end;
                calls += 1;
            }
            let a = actual.process_f64(&[], true).unwrap();
            let b = reference.process_f64(&[], true).unwrap();
            assert_eq!(bits(&a), bits(&b));
            emitted += a.len();
            assert_eq!(
                emitted,
                (frames as u128 * to as u128).div_ceil(from as u128) as usize * channels
            );
            assert!(wraps >= channels * 2);
            actual.reset();
            reference.reset();
            assert!(actual
                .rolling_windows
                .iter()
                .all(|c| c.head == 0 && c.origin.is_none() && c.samples.is_empty()));
            assert_eq!(
                bits(&actual.process_f64(&input[..137 * channels], true).unwrap()),
                bits(
                    &reference
                        .process_f64(&input[..137 * channels], true)
                        .unwrap()
                )
            );
            println!("{from}->{to} {channels}ch {startup:?}: {wraps}mirrorwraps,{frames}frames;exactoriginalrollingPCM");
        }
    }
}
#[test]
fn mirrored_memory_budget_retains_single_rolling_and_bulk_fallbacks() {
    // These ratios and channel counts exercise geometry, not a rate classifier.
    let single = InfiniteResampler::with_startup(96000, 44100, 64, Startup::Zero).unwrap();
    assert!(!single.mirrored_windows);
    assert_eq!(single.rolling_windows.len(), 64);
    assert!(
        single
            .rolling_windows
            .iter()
            .map(|c| c.samples.capacity())
            .sum::<usize>()
            <= MAX_AUDIO_SAMPLES
    );
    let bulk = InfiniteResampler::with_startup(384000, 8000, 8, Startup::Zero).unwrap();
    assert!(bulk.rolling_windows.is_empty());
    assert!(!bulk.mirrored_windows);
}
