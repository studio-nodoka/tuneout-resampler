//! Grouped complex-input recurrences, reusing the qualified state-update ISA
//! dispatch. SIMD lanes are modes, and every multiply/add stays unfused.
use super::*;

impl Kernel {
    pub(crate) fn coarse_detect() -> Self {
        let kernel = Self::detect();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if kernel.0 == Backend::Scalar && std::is_x86_feature_detected!("sse2") {
            return Self(Backend::Sse2);
        }
        kernel
    }
    pub(crate) fn coarse_update<const FINITE: bool>(
        self,
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
        leaves: &[Complex],
        outgoing: Complex,
    ) {
        assert_eq!(poles.len(), states.len());
        if FINITE {
            assert_eq!(leaves.len(), states.len());
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { wasm::update::<FINITE>(poles, states, incoming, leaves, outgoing) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => update_scalar::<FINITE>(poles, states, incoming, leaves, outgoing),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe {
                x86::sse_update::<FINITE>(poles, states, incoming, leaves, outgoing)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe {
                x86::avx_update::<FINITE>(poles, states, incoming, leaves, outgoing)
            },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe {
                x86::avx512_update::<FINITE>(poles, states, incoming, leaves, outgoing)
            },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe {
                neon::update::<FINITE>(poles, states, incoming, leaves, outgoing)
            },
        }
    }
    /// Independent complex frequency products use the same unfused multiply
    /// ordering as Complex. This reuses the already audited state-update ISA.
    pub(crate) fn coarse_mac(self, sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        assert_eq!(sum.len(), x.len());
        assert_eq!(sum.len(), h.len());
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { wasm::mac(sum, x, h) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => mac_scalar(sum, x, h),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::sse_mac(sum, x, h) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe { x86::avx_mac(sum, x, h) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe { x86::avx512_mac(sum, x, h) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { neon::mac(sum, x, h) },
        }
    }
    /// Hold a vector of modal states and its poles in registers for the whole
    /// seed interval. REVERSE selects the anti-causal order of input samples.
    pub(crate) fn coarse_seed<const REVERSE: bool>(
        self,
        poles: &[Complex],
        states: &mut [Complex],
        samples: &[Complex],
    ) {
        assert_eq!(poles.len(), states.len());
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { wasm::seed::<REVERSE>(poles, states, samples) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => seed_scalar::<REVERSE>(poles, states, samples),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::sse_seed::<REVERSE>(poles, states, samples) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe { x86::avx_seed::<REVERSE>(poles, states, samples) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe { x86::avx512_seed::<REVERSE>(poles, states, samples) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { neon::seed::<REVERSE>(poles, states, samples) },
        }
    }
}

fn mac_scalar(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
    for ((s, &x), &h) in sum.iter_mut().zip(x).zip(h) {
        *s = *s + x * h;
    }
}
fn update_scalar<const FINITE: bool>(
    poles: &[Complex],
    states: &mut [Complex],
    incoming: Complex,
    leaves: &[Complex],
    outgoing: Complex,
) {
    for i in 0..states.len() {
        states[i] = poles[i] * states[i] + incoming;
        if FINITE {
            states[i] = states[i] - leaves[i] * outgoing;
        }
    }
}
fn seed_scalar<const REVERSE: bool>(
    poles: &[Complex],
    states: &mut [Complex],
    samples: &[Complex],
) {
    for (state, &pole) in states.iter_mut().zip(poles) {
        let mut value = Complex::default();
        if REVERSE {
            for &x in samples.iter().rev() {
                value = pole * value + x;
            }
        } else {
            for &x in samples {
                value = pole * value + x;
            }
        }
        *state = value;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[target_feature(enable = "sse2")]
    unsafe fn sse_broadcast(x: Complex) -> __m128d {
        _mm_set_pd(x.im, x.re)
    }
    #[target_feature(enable = "sse2")]
    unsafe fn sse_multiply(p: __m128d, s: __m128d) -> __m128d {
        let re = _mm_mul_pd(_mm_unpacklo_pd(p, p), s);
        let im = _mm_mul_pd(_mm_unpackhi_pd(p, p), _mm_shuffle_pd::<1>(s, s));
        _mm_add_pd(re, _mm_xor_pd(im, _mm_set_pd(0., -0.)))
    }
    #[target_feature(enable = "avx")]
    unsafe fn avx_broadcast(x: Complex) -> __m256d {
        _mm256_set_pd(x.im, x.re, x.im, x.re)
    }
    #[target_feature(enable = "avx")]
    unsafe fn avx_multiply(p: __m256d, s: __m256d) -> __m256d {
        _mm256_addsub_pd(
            _mm256_mul_pd(_mm256_movedup_pd(p), s),
            _mm256_mul_pd(_mm256_permute_pd::<15>(p), _mm256_permute_pd::<5>(s)),
        )
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_broadcast(x: Complex) -> __m512d {
        _mm512_set_pd(x.im, x.re, x.im, x.re, x.im, x.re, x.im, x.re)
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_multiply(p: __m512d, s: __m512d) -> __m512d {
        let re = _mm512_mul_pd(_mm512_movedup_pd(p), s);
        let im = _mm512_mul_pd(_mm512_permute_pd::<255>(p), _mm512_permute_pd::<85>(s));
        _mm512_mask_sub_pd(_mm512_add_pd(re, im), 0x55, re, im)
    }
    macro_rules! complex_dot_implementation {
        ($name:ident,$features:literal,$modes:expr,$zero:ident,$load:ident,$store:ident,$add:ident,$multiply:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(row: &[Complex], states: &[Complex]) -> Complex {
                unsafe {
                    let mut accum = [$zero(); 4 / $modes];
                    let bulk = row.len() / 4 * 4;
                    for i in (0..bulk).step_by(4) {
                        for lane in 0..4 / $modes {
                            let c = $load(row.as_ptr().add(i + lane * $modes).cast());
                            let s = $load(states.as_ptr().add(i + lane * $modes).cast());
                            accum[lane] = $add(accum[lane], $multiply(c, s));
                        }
                    }
                    let mut sums = [Complex::default(); 4];
                    for lane in 0..4 / $modes {
                        $store(sums.as_mut_ptr().add(lane * $modes).cast(), accum[lane]);
                    }
                    finish_complex_dot(sums, &row[bulk..], &states[bulk..])
                }
            }
        };
    }
    complex_dot_implementation!(
        sse_complex_dot,
        "sse2",
        1,
        _mm_setzero_pd,
        _mm_loadu_pd,
        _mm_storeu_pd,
        _mm_add_pd,
        sse_multiply
    );
    complex_dot_implementation!(
        avx_complex_dot,
        "avx",
        2,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        avx_multiply
    );
    complex_dot_implementation!(
        avx512_complex_dot,
        "avx512f",
        4,
        _mm512_setzero_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        avx512_multiply
    );

    macro_rules! weight_update_implementation {
        ($name:ident,$features:literal,$modes:expr,$zero:ident,$load:ident,$store:ident,$add:ident,$broadcast:ident,$multiply:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(
                row: &[Complex],
                poles: &[Complex],
                states: &mut [Complex],
                incoming: Complex,
            ) -> Complex {
                unsafe {
                    let x = $broadcast(incoming);
                    let mut accum = [$zero(); 4 / $modes];
                    let bulk = row.len() / 4 * 4;
                    for i in (0..bulk).step_by(4) {
                        for lane in 0..4 / $modes {
                            let s = $load(states.as_ptr().add(i + lane * $modes).cast());
                            let c = $load(row.as_ptr().add(i + lane * $modes).cast());
                            let p = $load(poles.as_ptr().add(i + lane * $modes).cast());
                            accum[lane] = $add(accum[lane], $multiply(c, s));
                            $store(
                                states.as_mut_ptr().add(i + lane * $modes).cast(),
                                $add($multiply(p, s), x),
                            );
                        }
                    }
                    let mut sums = [Complex::default(); 4];
                    for lane in 0..4 / $modes {
                        $store(sums.as_mut_ptr().add(lane * $modes).cast(), accum[lane]);
                    }
                    weight_update_finish(
                        sums,
                        &row[bulk..],
                        &poles[bulk..],
                        &mut states[bulk..],
                        incoming,
                    )
                }
            }
        };
    }
    weight_update_implementation!(
        sse_weight_update,
        "sse2",
        1,
        _mm_setzero_pd,
        _mm_loadu_pd,
        _mm_storeu_pd,
        _mm_add_pd,
        sse_broadcast,
        sse_multiply
    );
    weight_update_implementation!(
        avx_weight_update,
        "avx",
        2,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        avx_broadcast,
        avx_multiply
    );
    weight_update_implementation!(
        avx512_weight_update,
        "avx512f",
        4,
        _mm512_setzero_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        avx512_broadcast,
        avx512_multiply
    );

    #[target_feature(enable = "sse2")]
    unsafe fn sse_real_product(c: __m128d, s: __m128d) -> __m128d {
        let products = _mm_mul_pd(c, s);
        _mm_sub_pd(products, _mm_shuffle_pd::<1>(products, products))
    }
    #[target_feature(enable = "avx")]
    unsafe fn avx_real_product(c: __m256d, s: __m256d) -> __m256d {
        let products = _mm256_mul_pd(c, s);
        _mm256_sub_pd(products, _mm256_permute_pd::<5>(products))
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_real_product(c: __m512d, s: __m512d) -> __m512d {
        let products = _mm512_mul_pd(c, s);
        _mm512_sub_pd(products, _mm512_permute_pd::<85>(products))
    }
    macro_rules! real_weight_update_implementation {
        ($name:ident,$features:literal,$modes:expr,$zero:ident,$load:ident,$store:ident,$add:ident,$broadcast:ident,$multiply:ident,$real_product:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(
                row: &[Complex],
                poles: &[Complex],
                states: &mut [Complex],
                incoming: Complex,
            ) -> f64 {
                unsafe {
                    let x = $broadcast(incoming);
                    let mut accum = [$zero(); 4 / $modes];
                    let bulk = row.len() / 4 * 4;
                    for i in (0..bulk).step_by(4) {
                        for lane in 0..4 / $modes {
                            let s = $load(states.as_ptr().add(i + lane * $modes).cast());
                            let c = $load(row.as_ptr().add(i + lane * $modes).cast());
                            let p = $load(poles.as_ptr().add(i + lane * $modes).cast());
                            accum[lane] = $add(accum[lane], $real_product(c, s));
                            $store(
                                states.as_mut_ptr().add(i + lane * $modes).cast(),
                                $add($multiply(p, s), x),
                            );
                        }
                    }
                    let mut pairs = [Complex::default(); 4];
                    for lane in 0..4 / $modes {
                        $store(pairs.as_mut_ptr().add(lane * $modes).cast(), accum[lane]);
                    }
                    weight_update_real_finish(
                        pairs.map(|z| z.re),
                        &row[bulk..],
                        &poles[bulk..],
                        &mut states[bulk..],
                        incoming,
                    )
                }
            }
        };
    }
    real_weight_update_implementation!(
        sse_weight_update_real,
        "sse2",
        1,
        _mm_setzero_pd,
        _mm_loadu_pd,
        _mm_storeu_pd,
        _mm_add_pd,
        sse_broadcast,
        sse_multiply,
        sse_real_product
    );
    real_weight_update_implementation!(
        avx_weight_update_real,
        "avx",
        2,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        avx_broadcast,
        avx_multiply,
        avx_real_product
    );
    real_weight_update_implementation!(
        avx512_weight_update_real,
        "avx512f",
        4,
        _mm512_setzero_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        avx512_broadcast,
        avx512_multiply,
        avx512_real_product
    );

    #[target_feature(enable = "avx,fma")]
    pub(super) unsafe fn fma_mac_avx(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        unsafe {
            let bulk = sum.len() / 2 * 2;
            let sign = _mm256_set_pd(0., -0., 0., -0.);
            for i in (0..bulk).step_by(2) {
                let s = _mm256_loadu_pd(sum.as_ptr().add(i).cast());
                let x = _mm256_loadu_pd(x.as_ptr().add(i).cast());
                let h = _mm256_loadu_pd(h.as_ptr().add(i).cast());
                let signed = _mm256_xor_pd(_mm256_permute_pd::<5>(x), sign);
                let s = _mm256_fmadd_pd(_mm256_movedup_pd(h), x, s);
                let s = _mm256_fmadd_pd(_mm256_permute_pd::<15>(h), signed, s);
                _mm256_storeu_pd(sum.as_mut_ptr().add(i).cast(), s);
            }
            mac_fused_scalar(&mut sum[bulk..], &x[bulk..], &h[bulk..]);
        }
    }
    #[target_feature(enable = "avx512f,fma")]
    pub(super) unsafe fn fma_mac_avx512(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        unsafe {
            let bulk = sum.len() / 4 * 4;
            let sign = _mm512_set_epi64(0, i64::MIN, 0, i64::MIN, 0, i64::MIN, 0, i64::MIN);
            for i in (0..bulk).step_by(4) {
                let s = _mm512_loadu_pd(sum.as_ptr().add(i).cast());
                let x = _mm512_loadu_pd(x.as_ptr().add(i).cast());
                let h = _mm512_loadu_pd(h.as_ptr().add(i).cast());
                let signed = _mm512_castsi512_pd(_mm512_xor_si512(
                    _mm512_castpd_si512(_mm512_permute_pd::<85>(x)),
                    sign,
                ));
                let s = _mm512_fmadd_pd(_mm512_movedup_pd(h), x, s);
                let s = _mm512_fmadd_pd(_mm512_permute_pd::<255>(h), signed, s);
                _mm512_storeu_pd(sum.as_mut_ptr().add(i).cast(), s);
            }
            mac_fused_scalar(&mut sum[bulk..], &x[bulk..], &h[bulk..]);
        }
    }

    macro_rules! implementation {
        ($update:ident,$seed:ident,$features:literal,$modes:expr,$zero:ident,$load:ident,$store:ident,$add:ident,$sub:ident,$broadcast:ident,$multiply:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $update<const FINITE: bool>(
                poles: &[Complex],
                states: &mut [Complex],
                incoming: Complex,
                leaves: &[Complex],
                outgoing: Complex,
            ) {
                // Complex is repr(C); complete vectors remain slice-bounded.
                unsafe {
                    let incoming_scalar = incoming;
                    let outgoing_scalar = outgoing;
                    let incoming = $broadcast(incoming);
                    let outgoing = $broadcast(outgoing);
                    let bulk = states.len() / $modes * $modes;
                    for i in (0..bulk).step_by($modes) {
                        let p = $load(poles.as_ptr().add(i).cast());
                        let s = $load(states.as_ptr().add(i).cast());
                        let mut value = $add($multiply(p, s), incoming);
                        if FINITE {
                            value = $sub(
                                value,
                                $multiply($load(leaves.as_ptr().add(i).cast()), outgoing),
                            );
                        }
                        $store(states.as_mut_ptr().add(i).cast(), value);
                    }
                    let tail = if FINITE { &leaves[bulk..] } else { &[] };
                    update_scalar::<FINITE>(
                        &poles[bulk..],
                        &mut states[bulk..],
                        incoming_scalar,
                        tail,
                        outgoing_scalar,
                    );
                }
            }
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $seed<const REVERSE: bool>(
                poles: &[Complex],
                states: &mut [Complex],
                samples: &[Complex],
            ) {
                unsafe {
                    let bulk = states.len() / $modes * $modes;
                    for i in (0..bulk).step_by($modes) {
                        let pole = $load(poles.as_ptr().add(i).cast());
                        let mut value = $zero();
                        if REVERSE {
                            for &x in samples.iter().rev() {
                                value = $add($multiply(pole, value), $broadcast(x));
                            }
                        } else {
                            for &x in samples {
                                value = $add($multiply(pole, value), $broadcast(x));
                            }
                        }
                        $store(states.as_mut_ptr().add(i).cast(), value);
                    }
                    seed_scalar::<REVERSE>(&poles[bulk..], &mut states[bulk..], samples);
                }
            }
        };
    }
    macro_rules! mac_implementation {
        ($name:ident,$features:literal,$modes:expr,$load:ident,$store:ident,$add:ident,$multiply:ident) => {
            #[target_feature(enable=$features)]
            pub(super) unsafe fn $name(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
                unsafe {
                    let bulk = sum.len() / $modes * $modes;
                    for i in (0..bulk).step_by($modes) {
                        let s = $load(sum.as_ptr().add(i).cast());
                        let a = $load(x.as_ptr().add(i).cast());
                        let b = $load(h.as_ptr().add(i).cast());
                        $store(sum.as_mut_ptr().add(i).cast(), $add(s, $multiply(a, b)));
                    }
                    mac_scalar(&mut sum[bulk..], &x[bulk..], &h[bulk..]);
                }
            }
        };
    }
    mac_implementation!(
        sse_mac,
        "sse2",
        1,
        _mm_loadu_pd,
        _mm_storeu_pd,
        _mm_add_pd,
        sse_multiply
    );
    mac_implementation!(
        avx_mac,
        "avx",
        2,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        avx_multiply
    );
    mac_implementation!(
        avx512_mac,
        "avx512f",
        4,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        avx512_multiply
    );
    implementation!(
        sse_update,
        sse_seed,
        "sse2",
        1,
        _mm_setzero_pd,
        _mm_loadu_pd,
        _mm_storeu_pd,
        _mm_add_pd,
        _mm_sub_pd,
        sse_broadcast,
        sse_multiply
    );
    implementation!(
        avx_update,
        avx_seed,
        "avx",
        2,
        _mm256_setzero_pd,
        _mm256_loadu_pd,
        _mm256_storeu_pd,
        _mm256_add_pd,
        _mm256_sub_pd,
        avx_broadcast,
        avx_multiply
    );
    implementation!(
        avx512_update,
        avx512_seed,
        "avx512f",
        4,
        _mm512_setzero_pd,
        _mm512_loadu_pd,
        _mm512_storeu_pd,
        _mm512_add_pd,
        _mm512_sub_pd,
        avx512_broadcast,
        avx512_multiply
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    unsafe fn multiply(p: float64x2_t, s: float64x2_t) -> float64x2_t {
        let re = vmulq_f64(vdupq_laneq_f64::<0>(p), s);
        let im = vmulq_f64(vdupq_laneq_f64::<1>(p), vextq_f64::<1>(s, s));
        let signed = veorq_u64(
            vreinterpretq_u64_f64(im),
            vsetq_lane_u64::<0>(1u64 << 63, vdupq_n_u64(0)),
        );
        vaddq_f64(re, vreinterpretq_f64_u64(signed))
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mac(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        unsafe {
            for i in 0..sum.len() {
                let s = vld1q_f64(sum.as_ptr().add(i).cast());
                let a = vld1q_f64(x.as_ptr().add(i).cast());
                let b = vld1q_f64(h.as_ptr().add(i).cast());
                vst1q_f64(sum.as_mut_ptr().add(i).cast(), vaddq_f64(s, multiply(a, b)));
            }
        }
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn update<const FINITE: bool>(
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
        leaves: &[Complex],
        outgoing: Complex,
    ) {
        unsafe {
            let incoming = vld1q_f64([incoming.re, incoming.im].as_ptr());
            let outgoing = vld1q_f64([outgoing.re, outgoing.im].as_ptr());
            for i in 0..states.len() {
                let mut value = vaddq_f64(
                    multiply(
                        vld1q_f64(poles.as_ptr().add(i).cast()),
                        vld1q_f64(states.as_ptr().add(i).cast()),
                    ),
                    incoming,
                );
                if FINITE {
                    value = vsubq_f64(
                        value,
                        multiply(vld1q_f64(leaves.as_ptr().add(i).cast()), outgoing),
                    );
                }
                vst1q_f64(states.as_mut_ptr().add(i).cast(), value);
            }
        }
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn seed<const REVERSE: bool>(
        poles: &[Complex],
        states: &mut [Complex],
        samples: &[Complex],
    ) {
        unsafe {
            for i in 0..states.len() {
                let pole = vld1q_f64(poles.as_ptr().add(i).cast());
                let mut value = vdupq_n_f64(0.);
                if REVERSE {
                    for &x in samples.iter().rev() {
                        value = vaddq_f64(multiply(pole, value), vld1q_f64([x.re, x.im].as_ptr()));
                    }
                } else {
                    for &x in samples {
                        value = vaddq_f64(multiply(pole, value), vld1q_f64([x.re, x.im].as_ptr()));
                    }
                }
                vst1q_f64(states.as_mut_ptr().add(i).cast(), value);
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use super::*;
    use std::arch::wasm32::*;
    fn multiply(p: v128, s: v128) -> v128 {
        let re = f64x2_mul(i64x2_shuffle::<0, 0>(p, p), s);
        let im = f64x2_mul(i64x2_shuffle::<1, 1>(p, p), i64x2_shuffle::<1, 0>(s, s));
        f64x2_add(re, v128_xor(im, i64x2(i64::MIN, 0)))
    }
    pub(super) unsafe fn mac(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        unsafe {
            for i in 0..sum.len() {
                let s = v128_load(sum.as_ptr().add(i).cast());
                let a = v128_load(x.as_ptr().add(i).cast());
                let b = v128_load(h.as_ptr().add(i).cast());
                v128_store(sum.as_mut_ptr().add(i).cast(), f64x2_add(s, multiply(a, b)));
            }
        }
    }
    pub(super) unsafe fn update<const FINITE: bool>(
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
        leaves: &[Complex],
        outgoing: Complex,
    ) {
        unsafe {
            let incoming = f64x2(incoming.re, incoming.im);
            let outgoing = f64x2(outgoing.re, outgoing.im);
            for i in 0..states.len() {
                let mut value = f64x2_add(
                    multiply(
                        v128_load(poles.as_ptr().add(i).cast()),
                        v128_load(states.as_ptr().add(i).cast()),
                    ),
                    incoming,
                );
                if FINITE {
                    value = f64x2_sub(
                        value,
                        multiply(v128_load(leaves.as_ptr().add(i).cast()), outgoing),
                    );
                }
                v128_store(states.as_mut_ptr().add(i).cast(), value);
            }
        }
    }
    pub(super) unsafe fn seed<const REVERSE: bool>(
        poles: &[Complex],
        states: &mut [Complex],
        samples: &[Complex],
    ) {
        unsafe {
            for i in 0..states.len() {
                let pole = v128_load(poles.as_ptr().add(i).cast());
                let mut value = f64x2_splat(0.);
                if REVERSE {
                    for &x in samples.iter().rev() {
                        value = f64x2_add(multiply(pole, value), f64x2(x.re, x.im));
                    }
                } else {
                    for &x in samples {
                        value = f64x2_add(multiply(pole, value), f64x2(x.re, x.im));
                    }
                }
                v128_store(states.as_mut_ptr().add(i).cast(), value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_bits(a: &[Complex], b: &[Complex]) {
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.re.to_bits(), b.re.to_bits(), "mode {i} real");
            assert_eq!(a.im.to_bits(), b.im.to_bits(), "mode {i} imaginary");
        }
    }
    #[test]
    fn coarse_group_kernels_preserve_all_scalar_rounding() {
        let poles: Vec<_> = (0..199)
            .map(|i| Complex::new(0.5 + (i % 7) as f64 / 17., ((i * 7 % 19) as f64 - 9.) / 29.))
            .collect();
        let leaves: Vec<_> = (0..199)
            .map(|i| Complex::new((i as f64 - 91.) / 401., ((i * 3 % 17) as f64 - 8.) / 89.))
            .collect();
        let initial: Vec<_> = (0..199)
            .map(|i| {
                Complex::new(
                    ((i * 31 % 97) as f64 - 48.) / 109.,
                    ((i * 17 % 83) as f64 - 41.) / 101.,
                )
            })
            .collect();
        let samples: Vec<_> = (0..73)
            .map(|i| {
                Complex::new(
                    ((i * 23 % 53) as f64 - 26.) / 59.,
                    ((i * 13 % 71) as f64 - 35.) / 79.,
                )
            })
            .collect();
        for kernel in crate::simd::tests::coarse_available() {
            for n in 0..=199 {
                let mut expected = initial[..n].to_vec();
                let mut actual = expected.clone();
                update_scalar::<false>(
                    &poles[..n],
                    &mut expected,
                    samples[0],
                    &[],
                    Complex::default(),
                );
                kernel.coarse_update::<false>(
                    &poles[..n],
                    &mut actual,
                    samples[0],
                    &[],
                    Complex::default(),
                );
                assert_bits(&actual, &expected);
                update_scalar::<true>(
                    &poles[..n],
                    &mut expected,
                    samples[1],
                    &leaves[..n],
                    samples[2],
                );
                kernel.coarse_update::<true>(
                    &poles[..n],
                    &mut actual,
                    samples[1],
                    &leaves[..n],
                    samples[2],
                );
                assert_bits(&actual, &expected);
                seed_scalar::<false>(&poles[..n], &mut expected, &samples);
                kernel.coarse_seed::<false>(&poles[..n], &mut actual, &samples);
                assert_bits(&actual, &expected);
                seed_scalar::<true>(&poles[..n], &mut expected, &samples);
                kernel.coarse_seed::<true>(&poles[..n], &mut actual, &samples);
                assert_bits(&actual, &expected);
            }
        }
    }
}

#[cfg(test)]
mod mac_tests {
    use super::*;
    #[test]
    fn frequency_products_match_independent_scalar_for_every_native_backend() {
        let kernels = crate::simd::tests::coarse_available();
        for len in 0..1032 {
            let x: Vec<_> = (0..len)
                .map(|j| {
                    Complex::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let h: Vec<_> = x.iter().rev().copied().collect();
            let initial: Vec<_> = x.iter().map(|c| *c * Complex::new(0.4, 0.7)).collect();
            let mut expected = initial.clone();
            mac_scalar(&mut expected, &x, &h);
            for &kernel in &kernels {
                let mut actual = initial.clone();
                kernel.coarse_mac(&mut actual, &x, &h);
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits(), "n{len} {kernel:?}");
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
            }
        }
    }
}

// One complex coefficient/state pass, with the same four mode accumulators
// and component reduction as two independent qualified real dot products.
impl Kernel {
    pub(crate) fn coarse_complex_dot(self, row: &[Complex], states: &[Complex]) -> Complex {
        assert_eq!(row.len(), states.len());
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { complex_dot_wasm::run(row, states) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => complex_dot_scalar(row, states),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::sse_complex_dot(row, states) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe { x86::avx_complex_dot(row, states) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe { x86::avx512_complex_dot(row, states) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { complex_dot_neon::run(row, states) },
        }
    }
}
fn finish_complex_dot(mut sums: [Complex; 4], row: &[Complex], states: &[Complex]) -> Complex {
    for (i, (&c, &s)) in row.iter().zip(states).enumerate() {
        sums[i] = sums[i] + c * s;
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}
fn complex_dot_scalar(row: &[Complex], states: &[Complex]) -> Complex {
    let mut sums = [Complex::default(); 4];
    let bulk = row.len() / 4 * 4;
    for i in (0..bulk).step_by(4) {
        for lane in 0..4 {
            sums[lane] = sums[lane] + row[i + lane] * states[i + lane];
        }
    }
    finish_complex_dot(sums, &row[bulk..], &states[bulk..])
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)] // Fixed independent accumulation lanes.
mod complex_dot_neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run(row: &[Complex], states: &[Complex]) -> Complex {
        unsafe {
            let mut accum = [vdupq_n_f64(0.); 4];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for lane in 0..4 {
                    let c = vld1q_f64(row.as_ptr().add(i + lane).cast());
                    let s = vld1q_f64(states.as_ptr().add(i + lane).cast());
                    let re = vmulq_f64(vdupq_laneq_f64::<0>(c), s);
                    let im = vmulq_f64(vdupq_laneq_f64::<1>(c), vextq_f64::<1>(s, s));
                    let signed = veorq_u64(
                        vreinterpretq_u64_f64(im),
                        vsetq_lane_u64::<0>(1u64 << 63, vdupq_n_u64(0)),
                    );
                    accum[lane] =
                        vaddq_f64(accum[lane], vaddq_f64(re, vreinterpretq_f64_u64(signed)));
                }
            }
            let mut sums = [Complex::default(); 4];
            for lane in 0..4 {
                vst1q_f64(sums.as_mut_ptr().add(lane).cast(), accum[lane]);
            }
            finish_complex_dot(sums, &row[bulk..], &states[bulk..])
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::needless_range_loop)] // Fixed independent accumulation lanes.
mod complex_dot_wasm {
    use super::*;
    use std::arch::wasm32::*;
    pub(super) unsafe fn run(row: &[Complex], states: &[Complex]) -> Complex {
        unsafe {
            let mut accum = [f64x2_splat(0.); 4];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for lane in 0..4 {
                    let c = v128_load(row.as_ptr().add(i + lane).cast());
                    let s = v128_load(states.as_ptr().add(i + lane).cast());
                    let re = f64x2_mul(i64x2_shuffle::<0, 0>(c, c), s);
                    let im = f64x2_mul(i64x2_shuffle::<1, 1>(c, c), i64x2_shuffle::<1, 0>(s, s));
                    accum[lane] =
                        f64x2_add(accum[lane], f64x2_add(re, v128_xor(im, i64x2(i64::MIN, 0))));
                }
            }
            let mut sums = [Complex::default(); 4];
            for lane in 0..4 {
                v128_store(sums.as_mut_ptr().add(lane).cast(), accum[lane]);
            }
            finish_complex_dot(sums, &row[bulk..], &states[bulk..])
        }
    }
}
#[cfg(test)]
mod complex_dot_tests {
    use super::*;
    #[test]
    fn fused_complex_weighting_matches_two_independent_real_dots() {
        let kernels = crate::simd::tests::coarse_available();
        for len in 0..1032 {
            let row: Vec<_> = (0..len)
                .map(|j| {
                    Complex::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let states: Vec<_> = row.iter().rev().copied().collect();
            let imag: Vec<_> = row.iter().map(|c| Complex::new(c.im, -c.re)).collect();
            for &kernel in &kernels {
                let expected = Complex::new(kernel.dot(&row, &states), kernel.dot(&imag, &states));
                let actual = kernel.coarse_complex_dot(&row, &states);
                assert_eq!(
                    actual.re.to_bits(),
                    expected.re.to_bits(),
                    "len{len} {kernel:?} real"
                );
                assert_eq!(
                    actual.im.to_bits(),
                    expected.im.to_bits(),
                    "len{len} {kernel:?} imaginary"
                );
            }
        }
    }
}

impl Kernel {
    /// Weight the old state, then advance it, reusing the same load. The four
    /// independent accumulators and every modal arithmetic operation remain.
    pub(crate) fn coarse_weight_update(
        self,
        row: &[Complex],
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
    ) -> Complex {
        assert_eq!(row.len(), states.len());
        assert_eq!(row.len(), poles.len());
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { weight_update_wasm::run(row, poles, states, incoming) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => weight_update_scalar(row, poles, states, incoming),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::sse_weight_update(row, poles, states, incoming) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe { x86::avx_weight_update(row, poles, states, incoming) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe { x86::avx512_weight_update(row, poles, states, incoming) },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { weight_update_neon::run(row, poles, states, incoming) },
        }
    }
}
fn weight_update_finish(
    mut sums: [Complex; 4],
    row: &[Complex],
    poles: &[Complex],
    states: &mut [Complex],
    incoming: Complex,
) -> Complex {
    for (i, ((&c, &p), s)) in row.iter().zip(poles).zip(states).enumerate() {
        let old = *s;
        sums[i] = sums[i] + c * old;
        *s = p * old + incoming;
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}
fn weight_update_scalar(
    row: &[Complex],
    poles: &[Complex],
    states: &mut [Complex],
    incoming: Complex,
) -> Complex {
    let mut sums = [Complex::default(); 4];
    let bulk = row.len() / 4 * 4;
    for i in (0..bulk).step_by(4) {
        for lane in 0..4 {
            let old = states[i + lane];
            sums[lane] = sums[lane] + row[i + lane] * old;
            states[i + lane] = poles[i + lane] * old + incoming;
        }
    }
    weight_update_finish(
        sums,
        &row[bulk..],
        &poles[bulk..],
        &mut states[bulk..],
        incoming,
    )
}
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)] // Fixed independent accumulation lanes.
mod weight_update_neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    unsafe fn multiply(c: float64x2_t, s: float64x2_t) -> float64x2_t {
        let re = vmulq_f64(vdupq_laneq_f64::<0>(c), s);
        let im = vmulq_f64(vdupq_laneq_f64::<1>(c), vextq_f64::<1>(s, s));
        let signed = veorq_u64(
            vreinterpretq_u64_f64(im),
            vsetq_lane_u64::<0>(1u64 << 63, vdupq_n_u64(0)),
        );
        vaddq_f64(re, vreinterpretq_f64_u64(signed))
    }
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run(
        row: &[Complex],
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
    ) -> Complex {
        unsafe {
            let x = vld1q_f64([incoming.re, incoming.im].as_ptr());
            let mut accum = [vdupq_n_f64(0.); 4];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for lane in 0..4 {
                    let s = vld1q_f64(states.as_ptr().add(i + lane).cast());
                    let c = vld1q_f64(row.as_ptr().add(i + lane).cast());
                    let p = vld1q_f64(poles.as_ptr().add(i + lane).cast());
                    accum[lane] = vaddq_f64(accum[lane], multiply(c, s));
                    vst1q_f64(
                        states.as_mut_ptr().add(i + lane).cast(),
                        vaddq_f64(multiply(p, s), x),
                    );
                }
            }
            let mut sums = [Complex::default(); 4];
            for lane in 0..4 {
                vst1q_f64(sums.as_mut_ptr().add(lane).cast(), accum[lane]);
            }
            weight_update_finish(
                sums,
                &row[bulk..],
                &poles[bulk..],
                &mut states[bulk..],
                incoming,
            )
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::needless_range_loop)] // Fixed independent accumulation lanes.
mod weight_update_wasm {
    use super::*;
    use std::arch::wasm32::*;
    fn multiply(c: v128, s: v128) -> v128 {
        let re = f64x2_mul(i64x2_shuffle::<0, 0>(c, c), s);
        let im = f64x2_mul(i64x2_shuffle::<1, 1>(c, c), i64x2_shuffle::<1, 0>(s, s));
        f64x2_add(re, v128_xor(im, i64x2(i64::MIN, 0)))
    }
    pub(super) unsafe fn run(
        row: &[Complex],
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
    ) -> Complex {
        unsafe {
            let x = f64x2(incoming.re, incoming.im);
            let mut accum = [f64x2_splat(0.); 4];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for lane in 0..4 {
                    let s = v128_load(states.as_ptr().add(i + lane).cast());
                    let c = v128_load(row.as_ptr().add(i + lane).cast());
                    let p = v128_load(poles.as_ptr().add(i + lane).cast());
                    accum[lane] = f64x2_add(accum[lane], multiply(c, s));
                    v128_store(
                        states.as_mut_ptr().add(i + lane).cast(),
                        f64x2_add(multiply(p, s), x),
                    );
                }
            }
            let mut sums = [Complex::default(); 4];
            for lane in 0..4 {
                v128_store(sums.as_mut_ptr().add(lane).cast(), accum[lane]);
            }
            weight_update_finish(
                sums,
                &row[bulk..],
                &poles[bulk..],
                &mut states[bulk..],
                incoming,
            )
        }
    }
}
#[cfg(test)]
mod weight_update_tests {
    use super::*;
    #[test]
    fn weighted_updates_preserve_separate_dot_and_update_for_all_native_backends() {
        let kernels = crate::simd::tests::coarse_available();
        for len in 0..1032 {
            let row: Vec<_> = (0..len)
                .map(|j| {
                    Complex::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let poles: Vec<_> = row.iter().map(|z| *z * Complex::new(0.8, 0.2)).collect();
            let initial: Vec<_> = row.iter().rev().copied().collect();
            let incoming = Complex::new(0.37, -0.29);
            for &kernel in &kernels {
                let mut expected = initial.clone();
                let dot = kernel.coarse_complex_dot(&row, &expected);
                kernel.coarse_update::<false>(
                    &poles,
                    &mut expected,
                    incoming,
                    &[],
                    Complex::default(),
                );
                let mut actual = initial.clone();
                let got = kernel.coarse_weight_update(&row, &poles, &mut actual, incoming);
                assert_eq!(dot.re.to_bits(), got.re.to_bits());
                assert_eq!(dot.im.to_bits(), got.im.to_bits());
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits());
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
            }
        }
    }
}

// Downsample synthesis uses only the real demodulated output. State updates
// remain complex; each real product and four-way reduction keeps its ordering.
impl Kernel {
    pub(crate) fn coarse_weight_update_real(
        self,
        row: &[Complex],
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
    ) -> f64 {
        assert_eq!(row.len(), states.len());
        assert_eq!(row.len(), poles.len());
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return unsafe { weight_update_real_wasm::run(row, poles, states, incoming) };
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => weight_update_real_scalar(row, poles, states, incoming),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => unsafe { x86::sse_weight_update_real(row, poles, states, incoming) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => unsafe { x86::avx_weight_update_real(row, poles, states, incoming) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => unsafe {
                x86::avx512_weight_update_real(row, poles, states, incoming)
            },
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => unsafe { weight_update_real_neon::run(row, poles, states, incoming) },
        }
    }
}
fn weight_update_real_finish(
    mut sums: [f64; 4],
    row: &[Complex],
    poles: &[Complex],
    states: &mut [Complex],
    incoming: Complex,
) -> f64 {
    for (i, ((&c, &p), s)) in row.iter().zip(poles).zip(states).enumerate() {
        let old = *s;
        sums[i] += c.re * old.re - c.im * old.im;
        *s = p * old + incoming;
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}
fn weight_update_real_scalar(
    row: &[Complex],
    poles: &[Complex],
    states: &mut [Complex],
    incoming: Complex,
) -> f64 {
    let mut sums = [0.; 4];
    let bulk = row.len() / 4 * 4;
    for i in (0..bulk).step_by(4) {
        for lane in 0..4 {
            let old = states[i + lane];
            let c = row[i + lane];
            sums[lane] += c.re * old.re - c.im * old.im;
            states[i + lane] = poles[i + lane] * old + incoming;
        }
    }
    weight_update_real_finish(
        sums,
        &row[bulk..],
        &poles[bulk..],
        &mut states[bulk..],
        incoming,
    )
}
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)] // Fixed independent accumulation lanes.
mod weight_update_real_neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run(
        row: &[Complex],
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
    ) -> f64 {
        unsafe {
            let x = vld1q_f64([incoming.re, incoming.im].as_ptr());
            let mut accum = [vdupq_n_f64(0.); 4];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for lane in 0..4 {
                    let s = vld1q_f64(states.as_ptr().add(i + lane).cast());
                    let c = vld1q_f64(row.as_ptr().add(i + lane).cast());
                    let p = vld1q_f64(poles.as_ptr().add(i + lane).cast());
                    let products = vmulq_f64(c, s);
                    let re = vsubq_f64(products, vextq_f64::<1>(products, products));
                    accum[lane] = vaddq_f64(accum[lane], re);
                    let pr = vmulq_f64(vdupq_laneq_f64::<0>(p), s);
                    let pi = vmulq_f64(vdupq_laneq_f64::<1>(p), vextq_f64::<1>(s, s));
                    let signed = veorq_u64(
                        vreinterpretq_u64_f64(pi),
                        vsetq_lane_u64::<0>(1u64 << 63, vdupq_n_u64(0)),
                    );
                    vst1q_f64(
                        states.as_mut_ptr().add(i + lane).cast(),
                        vaddq_f64(vaddq_f64(pr, vreinterpretq_f64_u64(signed)), x),
                    );
                }
            }
            let sums = std::array::from_fn(|lane| vgetq_lane_f64::<0>(accum[lane]));
            weight_update_real_finish(
                sums,
                &row[bulk..],
                &poles[bulk..],
                &mut states[bulk..],
                incoming,
            )
        }
    }
}
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::needless_range_loop)] // Fixed independent accumulation lanes.
mod weight_update_real_wasm {
    use super::*;
    use std::arch::wasm32::*;
    pub(super) unsafe fn run(
        row: &[Complex],
        poles: &[Complex],
        states: &mut [Complex],
        incoming: Complex,
    ) -> f64 {
        unsafe {
            let x = f64x2(incoming.re, incoming.im);
            let mut accum = [f64x2_splat(0.); 4];
            let bulk = row.len() / 4 * 4;
            for i in (0..bulk).step_by(4) {
                for lane in 0..4 {
                    let s = v128_load(states.as_ptr().add(i + lane).cast());
                    let c = v128_load(row.as_ptr().add(i + lane).cast());
                    let p = v128_load(poles.as_ptr().add(i + lane).cast());
                    let products = f64x2_mul(c, s);
                    let re = f64x2_sub(products, i64x2_shuffle::<1, 0>(products, products));
                    accum[lane] = f64x2_add(accum[lane], re);
                    let pr = f64x2_mul(i64x2_shuffle::<0, 0>(p, p), s);
                    let pi = f64x2_mul(i64x2_shuffle::<1, 1>(p, p), i64x2_shuffle::<1, 0>(s, s));
                    v128_store(
                        states.as_mut_ptr().add(i + lane).cast(),
                        f64x2_add(f64x2_add(pr, v128_xor(pi, i64x2(i64::MIN, 0))), x),
                    );
                }
            }
            let sums = std::array::from_fn(|lane| f64x2_extract_lane::<0>(accum[lane]));
            weight_update_real_finish(
                sums,
                &row[bulk..],
                &poles[bulk..],
                &mut states[bulk..],
                incoming,
            )
        }
    }
}
#[cfg(test)]
mod weight_update_real_tests {
    use super::*;
    #[test]
    fn real_weighted_updates_preserve_complex_states_and_real_output_on_every_native_backend() {
        let kernels = crate::simd::tests::coarse_available();
        for len in 0..1032 {
            let row: Vec<_> = (0..len)
                .map(|j| {
                    Complex::new(
                        ((j * 37 % 127) as f64 - 63.) / 127.,
                        ((j * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let poles: Vec<_> = row.iter().map(|z| *z * Complex::new(0.8, 0.2)).collect();
            let initial: Vec<_> = row.iter().rev().copied().collect();
            let incoming = Complex::new(0.37, -0.29);
            for &kernel in &kernels {
                let mut expected = initial.clone();
                let dot = kernel.coarse_weight_update(&row, &poles, &mut expected, incoming);
                let mut actual = initial.clone();
                let got = kernel.coarse_weight_update_real(&row, &poles, &mut actual, incoming);
                assert_eq!(dot.re.to_bits(), got.to_bits(), "len{len} {kernel:?}");
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits());
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
            }
        }
    }
}

impl Kernel {
    /// Select once when constructing the finite FFT plan. FMA changes its
    /// accumulation rounding deliberately; non-FMA targets retain the old ISA.
    pub(crate) fn coarse_mac_function(self) -> fn(&mut [Complex], &[Complex], &[Complex]) {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        return mac_wasm_checked;
        #[allow(unreachable_code)]
        match self.0 {
            Backend::Scalar => mac_scalar_checked,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2 => mac_sse_checked,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx => {
                if std::is_x86_feature_detected!("fma") {
                    mac_avx_fma_checked
                } else {
                    mac_avx_checked
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => {
                if std::is_x86_feature_detected!("fma") {
                    mac_avx512_fma_checked
                } else {
                    mac_avx512_checked
                }
            }
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => mac_neon_fma_checked,
        }
    }
}
#[inline(always)]
fn mac_fused_scalar(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
    for ((s, &x), &h) in sum.iter_mut().zip(x).zip(h) {
        let re = h.re.mul_add(x.re, s.re);
        let im = h.re.mul_add(x.im, s.im);
        s.re = h.im.mul_add(-x.im, re);
        s.im = h.im.mul_add(x.re, im);
    }
}
fn mac_scalar_checked(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
    assert_eq!(sum.len(), x.len());
    assert_eq!(sum.len(), h.len());
    mac_scalar(sum, x, h);
}
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! checked_mac {
    ($name:ident,$implementation:path) => {
        fn $name(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
            assert_eq!(sum.len(), x.len());
            assert_eq!(sum.len(), h.len());
            // The private selector above verifies every required CPU feature.
            unsafe { $implementation(sum, x, h) }
        }
    };
}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked_mac!(mac_sse_checked, x86::sse_mac);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked_mac!(mac_avx_checked, x86::avx_mac);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked_mac!(mac_avx512_checked, x86::avx512_mac);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked_mac!(mac_avx_fma_checked, x86::fma_mac_avx);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
checked_mac!(mac_avx512_fma_checked, x86::fma_mac_avx512);
#[cfg(target_arch = "aarch64")]
checked_mac!(mac_neon_fma_checked, mac_fma_neon::run);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
checked_mac!(mac_wasm_checked, wasm::mac);
#[cfg(target_arch = "aarch64")]
mod mac_fma_neon {
    use super::*;
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn run(sum: &mut [Complex], x: &[Complex], h: &[Complex]) {
        unsafe {
            for i in 0..sum.len() {
                let s = vld1q_f64(sum.as_ptr().add(i).cast());
                let x = vld1q_f64(x.as_ptr().add(i).cast());
                let h = vld1q_f64(h.as_ptr().add(i).cast());
                let swapped = vextq_f64::<1>(x, x);
                let signed = vreinterpretq_f64_u64(veorq_u64(
                    vreinterpretq_u64_f64(swapped),
                    vsetq_lane_u64::<0>(1u64 << 63, vdupq_n_u64(0)),
                ));
                let s = vfmaq_f64(s, vdupq_laneq_f64::<0>(h), x);
                let s = vfmaq_f64(s, vdupq_laneq_f64::<1>(h), signed);
                vst1q_f64(sum.as_mut_ptr().add(i).cast(), s);
            }
        }
    }
}
#[cfg(test)]
mod fma_mac_tests {
    use super::*;
    #[test]
    fn guarded_fma_frequency_products_match_their_independent_rounding_model() {
        let kernels: Vec<_> = crate::simd::tests::available()
            .into_iter()
            .map(|kernel| {
                let fused = match kernel.0 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    Backend::Avx | Backend::Avx512 => std::is_x86_feature_detected!("fma"),
                    #[cfg(target_arch = "aarch64")]
                    Backend::Neon => true,
                    _ => false,
                };
                (kernel, fused)
            })
            .collect();
        for n in 0..1032 {
            let x: Vec<_> = (0..n)
                .map(|i| {
                    Complex::new(
                        ((i * 37 % 127) as f64 - 63.) / 127.,
                        ((i * 41 % 131) as f64 - 65.) / 131.,
                    )
                })
                .collect();
            let h: Vec<_> = x.iter().rev().copied().collect();
            let initial: Vec<_> = x.iter().map(|z| *z * Complex::new(0.7, 0.3)).collect();
            for &(kernel, fused) in &kernels {
                let mut actual = initial.clone();
                let mut expected = initial.clone();
                let mac = kernel.coarse_mac_function();
                for _ in 0..3 {
                    mac(&mut actual, &x, &h);
                    if fused {
                        mac_fused_scalar(&mut expected, &x, &h)
                    } else {
                        mac_scalar(&mut expected, &x, &h)
                    }
                }
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.re.to_bits(), b.re.to_bits(), "n{n} {kernel:?}");
                    assert_eq!(a.im.to_bits(), b.im.to_bits());
                }
            }
        }
    }
}

#[path = "coarse_real_history.rs"]
mod real_history;
pub(crate) use real_history::RealHistoryMac;
