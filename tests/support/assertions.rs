// Compare stored bits, including signed zero, without allocating bit vectors.
macro_rules! assert_bits_eq {
    ($actual:expr, $expected:expr $(,)?) => {
        assert_bits_eq!($actual, $expected, "")
    };
    ($actual:expr, $expected:expr, $($context:tt)+) => {{
        let (actual, expected): (&[f64], &[f64]) = ($actual, $expected);
        assert_eq!(actual.len(), expected.len(), $($context)+);
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "sample {index}: {a} != {b}; {}",
                format_args!($($context)+));
        }
    }};
}

pub(crate) use assert_bits_eq;
