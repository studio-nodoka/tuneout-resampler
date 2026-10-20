//! Small private complex arithmetic; no DSP dependency is needed at runtime.

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(crate) struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    #[inline(always)]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Div for Complex {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let norm = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / norm,
            (self.im * rhs.re - self.re * rhs.im) / norm,
        )
    }
}

// Extended precision is used only during construction. Products such as
// 1 - conj(p) * p lose significant digits when a pole is close to the unit
// circle. Computing the residues and phase bank here avoids that cancellation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Wide {
    hi: f64,
    lo: f64,
}

impl From<f64> for Wide {
    #[inline(always)]
    fn from(hi: f64) -> Self {
        Self { hi, lo: 0.0 }
    }
}

impl Wide {
    #[inline(always)]
    pub fn parts(self) -> (f64, f64) {
        (self.hi, self.lo)
    }

    /// Error-free binary64 product split (assuming a finite, normal product).
    #[inline(always)]
    pub fn product(a: f64, b: f64) -> Self {
        let hi = a * b;
        Self::normalized(hi, a.mul_add(b, -hi))
    }

    /// Product normalization was already performed inside a guarded ISA loop.
    #[inline(always)]
    pub(crate) fn from_product_parts(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    fn normalized(hi: f64, lo: f64) -> Self {
        let sum = hi + lo;
        Self {
            hi: sum,
            lo: lo - (sum - hi),
        }
    }

    #[inline(always)]
    pub fn value(self) -> f64 {
        self.hi + self.lo
    }
}

impl std::ops::Add for Wide {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let sum = self.hi + rhs.hi;
        let v = sum - self.hi;
        let error = (self.hi - (sum - v)) + (rhs.hi - v);
        Self::normalized(sum, error + self.lo + rhs.lo)
    }
}

impl std::ops::Sub for Wide {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self + Self {
            hi: -rhs.hi,
            lo: -rhs.lo,
        }
    }
}

impl std::ops::Mul for Wide {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let product = self.hi * rhs.hi;
        let error = self.hi.mul_add(rhs.hi, -product);
        Self::normalized(
            product,
            error + self.hi * rhs.lo + self.lo * rhs.hi + self.lo * rhs.lo,
        )
    }
}

impl std::ops::Div for Wide {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let first = Self::from(self.hi / rhs.hi);
        let remainder = self - rhs * first;
        first + Self::from(remainder.value() / rhs.hi)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WideComplex {
    pub re: Wide,
    pub im: Wide,
}

impl From<Complex> for WideComplex {
    #[inline(always)]
    fn from(value: Complex) -> Self {
        Self {
            re: value.re.into(),
            im: value.im.into(),
        }
    }
}

impl WideComplex {
    #[inline(always)]
    pub fn scale(self, value: Wide) -> Self {
        Self {
            re: self.re * value,
            im: self.im * value,
        }
    }

    #[inline(always)]
    pub fn split(self) -> (Complex, Complex) {
        let (re, re_low) = self.re.parts();
        let (im, im_low) = self.im.parts();
        (Complex::new(re, im), Complex::new(re_low, im_low))
    }

    #[inline(always)]
    pub fn real(value: f64) -> Self {
        Complex::new(value, 0.0).into()
    }

    #[inline(always)]
    pub fn value(self) -> Complex {
        Complex::new(self.re.value(), self.im.value())
    }

    #[inline(always)]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: Wide::from(0.0) - self.im,
        }
    }
}

impl std::ops::Add for WideComplex {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for WideComplex {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for WideComplex {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Div for WideComplex {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let norm = rhs.re * rhs.re + rhs.im * rhs.im;
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / norm,
            im: (self.im * rhs.re - self.re * rhs.im) / norm,
        }
    }
}
