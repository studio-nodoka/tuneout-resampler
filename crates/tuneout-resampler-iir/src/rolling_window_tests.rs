use super::*;

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}
fn force_bulk(r: &mut InfiniteResampler) {
    r.mirrored_windows = false;
    r.window_head = 0;
    if !r.rolling_windows.is_empty() {
        r.window = std::mem::take(&mut r.rolling_windows[0].samples);
        r.rolling_windows = Vec::new();
    }
}

#[test]
fn rolling_cache_matches_bulk_with_wrapped_three_and_eight_channels() {
    for (from, to, channels) in [(96000, 44100, 3), (192000, 44100, 8)] {
        for startup in [Startup::Zero, Startup::Predict] {
            let mut actual = InfiniteResampler::with_startup(from, to, channels, startup).unwrap();
            let mut reference =
                InfiniteResampler::with_startup(from, to, channels, startup).unwrap();
            assert_eq!(actual.rolling_windows.len(), channels);
            force_bulk(&mut reference);
            let capacity = actual.samples.capacity();
            let frames = (capacity / channels * 3 + BLOCK + 137) / BLOCK * BLOCK + 137;
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
            let mut position = 0;
            let mut calls = 0;
            let mut wrapped = 0;
            let mut unaligned = 0;
            let mut emitted = 0;
            let mut mirrored_wraps = 0;
            let mut heads = vec![0; channels];
            let chunks = [733, 4096, 17, 5003, 255, 1024];
            while position < frames {
                let end = (position + chunks[calls % chunks.len()]).min(frames);
                let x = &input[position * channels..end * channels];
                let a = actual.process_f64(x, false).unwrap();
                let b = reference.process_f64(x, false).unwrap();
                assert_eq!(
                    bits(&a),
                    bits(&b),
                    "{from}->{to} {channels}ch {startup:?} call{calls}"
                );
                assert!(a.iter().all(|v| v.is_finite()));
                emitted += a.len();
                let (a, b) = actual.samples.as_slices();
                if !b.is_empty() {
                    wrapped += 1;
                    if a.len() % channels != 0 {
                        unaligned += 1;
                    }
                }
                if actual.mirrored_windows {
                    for (ch, cache) in actual.rolling_windows.iter().enumerate() {
                        if cache.head < heads[ch] {
                            mirrored_wraps += 1;
                        }
                        heads[ch] = cache.head;
                        if cache.origin.is_some() {
                            let half = cache.samples.len() / 2;
                            assert_eq!(bits(&cache.samples[..half]), bits(&cache.samples[half..]));
                        }
                    }
                }
                position = end;
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
            assert!(wrapped > 0, "the actual retained ring never wrapped");
            assert!(actual.rolling_reuses > 0);
            if actual.mirrored_windows {
                assert!(mirrored_wraps > 0, "physical mirrored rings never wrapped");
            }
            assert!(actual.rolling_rebuilds >= channels);
            assert!(
                actual
                    .rolling_windows
                    .iter()
                    .all(|cache| cache.origin.is_none()),
                "short final block must invalidate overlap"
            );
            println!("{from}->{to} {channels}ch {startup:?}: {frames}frames {wrapped} wrapped calls ({unaligned} unaligned), {}reuse/{}rebuild; exact bulk PCM",actual.rolling_reuses,actual.rolling_rebuilds);
            actual.reset();
            reference.reset();
            assert_eq!(actual.rolling_reuses, 0);
            assert_eq!(actual.rolling_rebuilds, 0);
            assert!(actual
                .rolling_windows
                .iter()
                .all(|cache| cache.origin.is_none() && cache.samples.is_empty()));
            assert_eq!(
                bits(&actual.process_f64(&input[..137 * channels], true).unwrap()),
                bits(
                    &reference
                        .process_f64(&input[..137 * channels], true)
                        .unwrap()
                )
            );
        }
    }
}

#[test]
fn rolling_cache_budget_falls_back_without_new_constructor_rejection() {
    let r = InfiniteResampler::with_startup(96000, 44100, 3, Startup::Zero).unwrap();
    let cached = r
        .rolling_windows
        .iter()
        .map(|cache| cache.samples.capacity())
        .sum::<usize>();
    assert!(cached > 0 && cached <= MAX_AUDIO_SAMPLES);
    assert_eq!(r.window.capacity(), 0);
    let mut bulk = InfiniteResampler::with_startup(96000, 44100, 3, Startup::Zero).unwrap();
    force_bulk(&mut bulk);
    let extra = r.filter_info().working_buffer_bytes - bulk.filter_info().working_buffer_bytes;
    assert_eq!(
        extra,
        (cached - bulk.window.capacity()) * 8
            + r.rolling_windows.capacity() * std::mem::size_of::<MonoWindow>()
    );
    let fallback = InfiniteResampler::with_startup(384000, 8000, 8, Startup::Zero).unwrap();
    assert!((2 * fallback.end + BLOCK) * fallback.channels > MAX_AUDIO_SAMPLES);
    assert!(fallback.rolling_windows.is_empty());
    assert!(fallback.window.capacity() > 0);
    println!("cache budget {cached}samples; high-channel/large-support constructor preserves bulk fallback");
}

#[test]
fn rolling_cache_nonoverlap_uses_bulk_reference() {
    let mut r = InfiniteResampler::with_startup(44100, 48000, 1, Startup::Zero).unwrap();
    let mut bulk = InfiniteResampler::with_startup(44100, 48000, 1, Startup::Zero).unwrap();
    force_bulk(&mut bulk);
    let length = 2 * r.end + BLOCK;
    // Immutable startup-zero prehistory gives valid arbitrary disjoint windows
    // without mutating the logical audio clock or fabricating an input record.
    for origin in [-(length as i128) * 3, -(length as i128)] {
        r.prepare_window(origin, length, 0);
        bulk.fill_window(origin, length, 0);
        assert_eq!(
            bits(&r.window[r.window_head..r.window_head + length]),
            bits(&bulk.window)
        );
        r.release_window(0);
    }
    assert_eq!(r.rolling_reuses, 0);
    assert_eq!(r.rolling_rebuilds, 2);
}
