//! Portable extended-precision coefficient generation, rounded once to f64.
//!
//! Each value keeps a leading f64 and its small residual. This is confined to
//! filter construction. Streaming convolution continues to use ordinary f64.

use crate::design::Design;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
struct Wide {
    hi: f64,
    lo: f64,
}

impl Wide {
    const ZERO: Self = Self::from_f64(0.0);
    const ONE: Self = Self::from_f64(1.0);
    const PI: Self = Self {
        hi: std::f64::consts::PI,
        lo: 1.224_646_799_147_353_2e-16,
    };

    const fn from_f64(hi: f64) -> Self {
        Self { hi, lo: 0.0 }
    }

    #[inline]
    fn sum(a: f64, b: f64) -> Self {
        let hi = a + b;
        let part = hi - a;
        let lo = (a - (hi - part)) + (b - part);
        Self { hi, lo }
    }

    #[inline]
    fn product(a: f64, b: f64) -> Self {
        // Dekker splitting recovers multiplication's residual without requiring
        // FMA or platform long double. Our bounded filter inputs avoid overflow.
        let split = |x: f64| {
            let scaled = 134_217_729.0 * x;
            let high = scaled - (scaled - x);
            (high, x - high)
        };
        let (ah, al) = split(a);
        let (bh, bl) = split(b);
        let hi = a * b;
        let lo = (((ah * bh - hi) + ah * bl) + al * bh) + al * bl;
        Self { hi, lo }
    }
}

impl Add for Wide {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        let leading = Self::sum(self.hi, other.hi);
        let trailing = Self::sum(self.lo, other.lo);
        let combined = Self::sum(leading.hi, leading.lo + trailing.hi);
        Self::sum(combined.hi, combined.lo + trailing.lo)
    }
}

impl Neg for Wide {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Sub for Wide {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        self + -other
    }
}

impl Mul for Wide {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        let product = Self::product(self.hi, other.hi);
        let residual = product.lo + (self.hi * other.lo + self.lo * other.hi);
        Self::sum(product.hi, residual + self.lo * other.lo)
    }
}

impl Div for Wide {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        let first = Self::from_f64(self.hi / other.hi);
        let residual = self - other * first;
        let second = Self::from_f64(residual.hi / other.hi);
        let residual = residual - other * second;
        first + second + Self::from_f64(residual.hi / other.hi)
    }
}

struct Series {
    squares: [Wide; 64],
    sine: [Wide; 20],
}

impl Series {
    fn new() -> Self {
        Self {
            squares: std::array::from_fn(|i| {
                Wide::ONE / Wide::from_f64(((i + 1) * (i + 1)) as f64)
            }),
            sine: std::array::from_fn(|i| {
                let k = i + 1;
                Wide::ONE / Wide::from_f64((2 * k * (2 * k + 1)) as f64)
            }),
        }
    }

    fn tail(&self, y: Wide) -> Wide {
        let mut term = Wide::ONE;
        let mut sum = Wide::ZERO;
        for (i, &reciprocal) in self.squares.iter().enumerate() {
            term = term * y * reciprocal;
            if i >= 1 {
                sum = sum + term;
            }
            if i > 1 && term.hi.abs() < sum.hi.abs() * 1e-33 {
                break;
            }
        }
        sum
    }

    // Argument reduction before multiplication by pi avoids losing precision
    // for distant sinc taps. The reduced argument stays within [-pi/2, pi/2].
    fn sin_pi(&self, value: Wide) -> Wide {
        let whole = value.hi.round();
        let x = (value - Wide::from_f64(whole)) * Wide::PI;
        let negative_square = -(x * x);
        let mut term = x;
        let mut sum = x;
        for &reciprocal in &self.sine {
            term = term * negative_square * reciprocal;
            sum = sum + term;
        }
        if (whole as i64) & 1 == 0 {
            sum
        } else {
            -sum
        }
    }

    fn sin_pi_ratio(&self, numerator: i64, denominator: i64) -> Wide {
        let whole = numerator.div_euclid(denominator);
        let remainder = numerator.rem_euclid(denominator);
        let sine =
            self.sin_pi(Wide::from_f64(remainder as f64) / Wide::from_f64(denominator as f64));
        if whole & 1 == 0 {
            sine
        } else {
            -sine
        }
    }
}

pub(crate) fn coefficients(
    from_rate: u32,
    to_rate: u32,
    Design {
        phases,
        taps,
        beta,
        rolloff,
    }: Design,
) -> Vec<f64> {
    let series = Series::new();
    let half = (taps / 2) as i64;
    let phase_count = Wide::from_f64(phases as f64);
    let input_rate = Wide::from_f64(f64::from(from_rate));
    let lower_rate = from_rate.min(to_rate);
    let scale = Wide::from_f64(f64::from(lower_rate)) / input_rate * Wide::from_f64(rolloff);
    let beta = Wide::from_f64(beta);
    let window_scale = beta * beta * Wide::from_f64(0.25);
    let window_normalization = Wide::ONE / series.tail(window_scale);
    let inverse_support = Wide::ONE / Wide::from_f64((half * phases as i64) as f64);
    let inverse_phases = Wide::ONE / phase_count;
    let sine_denominator = i64::from(from_rate) * phases as i64;
    let mut bank = vec![0.0; (phases + 1) * taps];
    let mut row = vec![Wide::ZERO; taps];

    // Opposite fractional phases are reflected copies. Generate each pair
    // once. The zero edge tap falls outside the reflected row's support.
    for phase in 0..=phases / 2 {
        let mut sum = Wide::ZERO;
        for (tap, value) in row.iter_mut().enumerate() {
            let numerator = (tap as i64 - half) * phases as i64 - phase as i64;
            *value = if numerator.abs() >= half * phases as i64 {
                Wide::ZERO
            } else {
                let coordinate = Wide::from_f64(numerator as f64);
                let x = coordinate * inverse_phases;
                let u = coordinate * inverse_support;
                let window = series.tail(window_scale * (Wide::ONE - u * u)) * window_normalization;
                let kernel = if numerator == 0 {
                    scale
                } else {
                    let sine = if rolloff == 1.0 {
                        series.sin_pi_ratio(numerator * i64::from(lower_rate), sine_denominator)
                    } else {
                        series.sin_pi(scale * x)
                    };
                    sine / (Wide::PI * x)
                };
                kernel * window
            };
            sum = sum + *value;
        }
        let normalization = Wide::ONE / sum;
        for (tap, value) in row.iter().enumerate() {
            bank[phase * taps + tap] = (*value * normalization).hi;
        }
        if phase > 0 && phases - phase != phase {
            for tap in 1..taps {
                bank[(phases - phase) * taps + tap] = bank[phase * taps + taps - tap];
            }
        }
    }
    // The interpolation endpoint is phase zero delayed by exactly one tap.
    for tap in 1..taps {
        bank[phases * taps + tap] = bank[tap - 1];
    }
    bank
}
