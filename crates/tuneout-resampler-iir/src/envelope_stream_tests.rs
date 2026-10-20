use crate::{BufferedResampler, Startup};

#[test]
fn envelope_stream_chunks_reset_and_endpoints_remain_exact() {
    for (from, to, channels) in [
        (96000, 44100, 2),
        (192000, 44100, 1),
        (44100, 48000, 2),
        (48000, 44117, 1),
    ] {
        for startup in [Startup::Zero, Startup::Predict] {
            let frames = if to == 44117 { 17 } else { 100001 };
            let input: Vec<_> = (0..frames * channels)
                .map(|n| ((n * 37 % 997) as f64 - 498.) / 1994.)
                .collect();
            let mut one = BufferedResampler::with_startup(from, to, channels, startup).unwrap();
            let expected = one.process_f64(&input, true).unwrap();
            let mut r = BufferedResampler::with_startup(from, to, channels, startup).unwrap();
            let mut actual = Vec::new();
            for part in input.chunks(733 * channels) {
                actual.extend(r.process_f64(part, false).unwrap());
            }
            actual.extend(r.process_f64(&[], true).unwrap());
            assert_eq!(
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{from}->{to} {startup:?}"
            );
            assert_eq!(
                actual.len(),
                (frames as u128 * to as u128).div_ceil(from as u128) as usize * channels
            );
            r.reset();
            let again = r.process_f64(&input, true).unwrap();
            assert_eq!(
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                again.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            println!("stream {from}->{to} {channels} channels {startup:?} chunks and reset passed");
        }
    }
}
