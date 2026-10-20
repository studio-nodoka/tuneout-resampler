//! Full-block recurrences in real/imaginary registers. Modes and channels stay
//! independent; each output retains the original four-lane reduction order.
use super::*;

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) struct Block<'a> {
    pub poles: &'a [Complex],
    pub direct: &'a [f64],
    pub first_output: usize,
    pub packed_rows: &'a [[f64; 40]],
    pub rows: &'a [Complex],
    pub states: &'a mut [Complex],
    pub input: &'a [f64],
    pub channels: usize,
    pub phase: i64,
    pub interpolation: u32,
    pub decimation: u32,
    pub outputs: usize,
}

impl Kernel {
    /// Cache only a lane permutation of the existing f64 bit patterns. The
    /// 512KiB per-bank cap bounds additional heap use to1MiB per converter.
    /// Unsupported shapes, excessive size and allocation failure use the
    /// existing direct-output block with ordinary coefficient rows instead.
    pub(crate) fn pack_phase_rows(
        self,
        rows: &[Complex],
        modes: usize,
        channels: usize,
    ) -> Vec<[f64; 40]> {
        #[cfg(target_arch = "x86_64")]
        if self.0 == Backend::Avx512
            && std::is_x86_feature_detected!("avx512vl")
            && channels == 2
            && modes == 20
            && rows.len().is_multiple_of(20)
        {
            const MAX_CACHE_BYTES: usize = 512 * 1024;
            let phases = rows.len() / 20;
            let Some(bytes) = phases.checked_mul(std::mem::size_of::<[f64; 40]>()) else {
                return Vec::new();
            };
            if bytes > MAX_CACHE_BYTES {
                return Vec::new();
            }
            let mut packed = Vec::new();
            if packed.try_reserve_exact(phases).is_err() {
                return Vec::new();
            }
            for row in rows.chunks_exact(20) {
                let mut result = [0.0; 40];
                for group in 0..2 {
                    for (lane, mode) in [0, 2, 1, 3, 4, 6, 5, 7].into_iter().enumerate() {
                        result[group * 16 + lane] = row[group * 8 + mode].re;
                        result[group * 16 + 8 + lane] = row[group * 8 + mode].im;
                    }
                }
                for (lane, mode) in [16, 18, 17, 19].into_iter().enumerate() {
                    result[32 + lane] = row[mode].re;
                    result[36 + lane] = row[mode].im;
                }
                packed.push(result);
            }
            return packed;
        }
        let _ = (rows, modes, channels);
        Vec::new()
    }

    pub(crate) fn block<const CAUSAL: bool>(
        self,
        pass: Block<'_>,
        output: &mut Vec<f64>,
    ) -> Option<i64> {
        #[cfg(target_arch = "x86_64")]
        if self.0 == Backend::Avx512
            && std::is_x86_feature_detected!("avx512vl")
            && pass.channels == 2
            && pass.poles.len() == 20
        {
            let mut pass = pass;
            assert!(pass.interpolation != 0 && pass.decimation != 0);
            assert_eq!(pass.states.len(), 40);
            assert_eq!(
                pass.rows.len(),
                (pass.interpolation as usize).checked_mul(20).unwrap()
            );
            assert_eq!(pass.input.len() % 2, 0);
            let frames = (pass.input.len() / 2) as i128;
            let l = i128::from(pass.interpolation);
            let m = i128::from(pass.decimation);
            let phase = i128::from(pass.phase);
            let expected_outputs = if CAUSAL {
                assert!((0..m).contains(&phase));
                let ticks = (frames * l - phase).max(0);
                (ticks + m - 1) / m
            } else {
                assert!((l - m..l).contains(&phase));
                let last = phase + (frames - 1) * l;
                if frames == 0 || last < 0 {
                    0
                } else {
                    last / m + 1
                }
            };
            assert_eq!(pass.outputs as i128, expected_outputs);
            let end = pass
                .first_output
                .checked_add(pass.outputs.checked_mul(2)?)?;
            if CAUSAL {
                assert_eq!(pass.direct.len(), pass.interpolation as usize);
                assert_eq!(pass.first_output, output.len());
                // The public caller reserved ready-block output before any
                // filtering. This creates the same final output slots and no
                // separate scratch buffer or additional audio lookahead.
                output.resize(end, 0.0);
            } else {
                assert!(end <= output.len());
            }
            // SAFETY: both CPU features, every slice dimension, clock and
            // initialized output range are checked before entering the kernel.
            if pass.packed_rows.is_empty() {
                return Some(unsafe { x86_blocks::run::<CAUSAL, false>(&mut pass, output) });
            }
            assert_eq!(pass.packed_rows.len(), pass.interpolation as usize);
            return Some(unsafe { x86_blocks::run::<CAUSAL, true>(&mut pass, output) });
        }
        let _ = (pass, output);
        None
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_blocks {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f,avx512vl")]
    pub(super) unsafe fn run<const CAUSAL: bool, const PACKED: bool>(
        pass: &mut Block<'_>,
        output_values: &mut [f64],
    ) -> i64 {
        // SAFETY: the parent checked both required CPU features, all sizes,
        // phases and output count. Exactly twenty modes form two groups of
        // eight and one of four. Loads/stores cover only these checked modes.
        unsafe {
            // Within each half-vector, physical lanes are modes0,2,1,3. This
            // matches the existing AVX dot, so no sum reassociation is needed.
            let real_indices = _mm512_set_epi64(14, 10, 12, 8, 6, 2, 4, 0);
            let imag_indices = _mm512_set_epi64(15, 11, 13, 9, 7, 3, 5, 1);
            let mut pr = [_mm512_setzero_pd(); 2];
            let mut pi = [_mm512_setzero_pd(); 2];
            let mut lr = [_mm512_setzero_pd(); 2];
            let mut li = [_mm512_setzero_pd(); 2];
            let mut rr = [_mm512_setzero_pd(); 2];
            let mut ri = [_mm512_setzero_pd(); 2];
            for group in 0..2 {
                let base = group * 8;
                let pa = _mm512_loadu_pd(pass.poles.as_ptr().add(base).cast());
                let pb = _mm512_loadu_pd(pass.poles.as_ptr().add(base + 4).cast());
                pr[group] = _mm512_permutex2var_pd(pa, real_indices, pb);
                pi[group] = _mm512_permutex2var_pd(pa, imag_indices, pb);
                let la = _mm512_loadu_pd(pass.states.as_ptr().add(base).cast());
                let lb = _mm512_loadu_pd(pass.states.as_ptr().add(base + 4).cast());
                lr[group] = _mm512_permutex2var_pd(la, real_indices, lb);
                li[group] = _mm512_permutex2var_pd(la, imag_indices, lb);
                let ra = _mm512_loadu_pd(pass.states.as_ptr().add(20 + base).cast());
                let rb = _mm512_loadu_pd(pass.states.as_ptr().add(24 + base).cast());
                rr[group] = _mm512_permutex2var_pd(ra, real_indices, rb);
                ri[group] = _mm512_permutex2var_pd(ra, imag_indices, rb);
            }
            let pa = _mm256_loadu_pd(pass.poles.as_ptr().add(16).cast());
            let pb = _mm256_loadu_pd(pass.poles.as_ptr().add(18).cast());
            let pr2 = _mm256_unpacklo_pd(pa, pb);
            let pi2 = _mm256_unpackhi_pd(pa, pb);
            let la = _mm256_loadu_pd(pass.states.as_ptr().add(16).cast());
            let lb = _mm256_loadu_pd(pass.states.as_ptr().add(18).cast());
            let mut lr2 = _mm256_unpacklo_pd(la, lb);
            let mut li2 = _mm256_unpackhi_pd(la, lb);
            let ra = _mm256_loadu_pd(pass.states.as_ptr().add(36).cast());
            let rb = _mm256_loadu_pd(pass.states.as_ptr().add(38).cast());
            let mut rr2 = _mm256_unpacklo_pd(ra, rb);
            let mut ri2 = _mm256_unpackhi_pd(ra, rb);
            let frames = pass.input.len() / 2;
            let mut phase = pass.phase;
            let mut output = if CAUSAL { 0 } else { pass.outputs };
            for step in 0..frames {
                let n = if CAUSAL { step } else { frames - 1 - step };
                while if CAUSAL {
                    phase < i64::from(pass.interpolation)
                } else {
                    output != 0 && phase >= 0
                } {
                    if !CAUSAL {
                        output -= 1;
                    }
                    let row = pass.rows.as_ptr().add(phase as usize * 20);
                    let mut sl = _mm256_setzero_pd();
                    let mut sr = _mm256_setzero_pd();
                    for group in 0..2 {
                        let (cr, ci) = if PACKED {
                            let cached = pass.packed_rows[phase as usize].as_ptr().add(group * 16);
                            (_mm512_loadu_pd(cached), _mm512_loadu_pd(cached.add(8)))
                        } else {
                            let ca = _mm512_loadu_pd(row.add(group * 8).cast());
                            let cb = _mm512_loadu_pd(row.add(group * 8 + 4).cast());
                            (
                                _mm512_permutex2var_pd(ca, real_indices, cb),
                                _mm512_permutex2var_pd(ca, imag_indices, cb),
                            )
                        };
                        let dl = _mm512_sub_pd(
                            _mm512_mul_pd(cr, lr[group]),
                            _mm512_mul_pd(ci, li[group]),
                        );
                        let dr = _mm512_sub_pd(
                            _mm512_mul_pd(cr, rr[group]),
                            _mm512_mul_pd(ci, ri[group]),
                        );
                        // Each half is the next four modes. Add halves one at
                        // a time to retain each original accumulator's order.
                        sl = _mm256_add_pd(sl, _mm512_castpd512_pd256(dl));
                        sl = _mm256_add_pd(sl, _mm512_extractf64x4_pd::<1>(dl));
                        sr = _mm256_add_pd(sr, _mm512_castpd512_pd256(dr));
                        sr = _mm256_add_pd(sr, _mm512_extractf64x4_pd::<1>(dr));
                    }
                    let (cr, ci) = if PACKED {
                        let cached = pass.packed_rows[phase as usize].as_ptr();
                        (
                            _mm256_loadu_pd(cached.add(32)),
                            _mm256_loadu_pd(cached.add(36)),
                        )
                    } else {
                        let ca = _mm256_loadu_pd(row.add(16).cast());
                        let cb = _mm256_loadu_pd(row.add(18).cast());
                        (_mm256_unpacklo_pd(ca, cb), _mm256_unpackhi_pd(ca, cb))
                    };
                    let dl = _mm256_sub_pd(_mm256_mul_pd(cr, lr2), _mm256_mul_pd(ci, li2));
                    let dr = _mm256_sub_pd(_mm256_mul_pd(cr, rr2), _mm256_mul_pd(ci, ri2));
                    sl = _mm256_add_pd(sl, dl);
                    sr = _mm256_add_pd(sr, dr);
                    let mut lane_l = [0.0; 4];
                    let mut lane_r = [0.0; 4];
                    _mm256_storeu_pd(lane_l.as_mut_ptr(), sl);
                    _mm256_storeu_pd(lane_r.as_mut_ptr(), sr);
                    let value_l = (lane_l[0] + lane_l[2]) + (lane_l[1] + lane_l[3]);
                    let value_r = (lane_r[0] + lane_r[2]) + (lane_r[1] + lane_r[3]);
                    let index = pass.first_output + output * 2;
                    if CAUSAL {
                        let direct = pass.direct[phase as usize];
                        output_values[index] = value_l + direct * pass.input[n * 2];
                        output_values[index + 1] = value_r + direct * pass.input[n * 2 + 1];
                    } else {
                        output_values[index] += value_l;
                        output_values[index + 1] += value_r;
                    }
                    if CAUSAL {
                        output += 1;
                        phase += i64::from(pass.decimation);
                    } else {
                        phase -= i64::from(pass.decimation);
                    }
                }
                if !CAUSAL && output == 0 {
                    break;
                }
                let il = _mm512_set1_pd(pass.input[n * 2]);
                let ir = _mm512_set1_pd(pass.input[n * 2 + 1]);
                for group in 0..2 {
                    if CAUSAL {
                        lr[group] = _mm512_add_pd(lr[group], il);
                        rr[group] = _mm512_add_pd(rr[group], ir);
                    }
                    let nlr = _mm512_sub_pd(
                        _mm512_mul_pd(pr[group], lr[group]),
                        _mm512_mul_pd(pi[group], li[group]),
                    );
                    let nli = _mm512_add_pd(
                        _mm512_mul_pd(pr[group], li[group]),
                        _mm512_mul_pd(pi[group], lr[group]),
                    );
                    let nrr = _mm512_sub_pd(
                        _mm512_mul_pd(pr[group], rr[group]),
                        _mm512_mul_pd(pi[group], ri[group]),
                    );
                    let nri = _mm512_add_pd(
                        _mm512_mul_pd(pr[group], ri[group]),
                        _mm512_mul_pd(pi[group], rr[group]),
                    );
                    lr[group] = nlr;
                    li[group] = nli;
                    rr[group] = nrr;
                    ri[group] = nri;
                    if !CAUSAL {
                        lr[group] = _mm512_add_pd(lr[group], il);
                        li[group] = _mm512_add_pd(li[group], _mm512_setzero_pd());
                        rr[group] = _mm512_add_pd(rr[group], ir);
                        ri[group] = _mm512_add_pd(ri[group], _mm512_setzero_pd());
                    }
                }
                let il2 = _mm512_castpd512_pd256(il);
                let ir2 = _mm512_castpd512_pd256(ir);
                if CAUSAL {
                    lr2 = _mm256_add_pd(lr2, il2);
                    rr2 = _mm256_add_pd(rr2, ir2);
                }
                let nlr2 = _mm256_sub_pd(_mm256_mul_pd(pr2, lr2), _mm256_mul_pd(pi2, li2));
                let nli2 = _mm256_add_pd(_mm256_mul_pd(pr2, li2), _mm256_mul_pd(pi2, lr2));
                let nrr2 = _mm256_sub_pd(_mm256_mul_pd(pr2, rr2), _mm256_mul_pd(pi2, ri2));
                let nri2 = _mm256_add_pd(_mm256_mul_pd(pr2, ri2), _mm256_mul_pd(pi2, rr2));
                lr2 = nlr2;
                li2 = nli2;
                rr2 = nrr2;
                ri2 = nri2;
                if !CAUSAL {
                    lr2 = _mm256_add_pd(lr2, il2);
                    li2 = _mm256_add_pd(li2, _mm256_setzero_pd());
                    rr2 = _mm256_add_pd(rr2, ir2);
                    ri2 = _mm256_add_pd(ri2, _mm256_setzero_pd());
                }
                if CAUSAL {
                    phase -= i64::from(pass.interpolation);
                } else {
                    phase += i64::from(pass.interpolation);
                }
            }
            let low_indices = _mm512_set_epi64(11, 3, 9, 1, 10, 2, 8, 0);
            let high_indices = _mm512_set_epi64(15, 7, 13, 5, 14, 6, 12, 4);
            for group in 0..2 {
                let base = group * 8;
                let la = _mm512_permutex2var_pd(lr[group], low_indices, li[group]);
                let lb = _mm512_permutex2var_pd(lr[group], high_indices, li[group]);
                let ra = _mm512_permutex2var_pd(rr[group], low_indices, ri[group]);
                let rb = _mm512_permutex2var_pd(rr[group], high_indices, ri[group]);
                _mm512_storeu_pd(pass.states.as_mut_ptr().add(base).cast(), la);
                _mm512_storeu_pd(pass.states.as_mut_ptr().add(base + 4).cast(), lb);
                _mm512_storeu_pd(pass.states.as_mut_ptr().add(20 + base).cast(), ra);
                _mm512_storeu_pd(pass.states.as_mut_ptr().add(24 + base).cast(), rb);
            }
            _mm256_storeu_pd(
                pass.states.as_mut_ptr().add(16).cast(),
                _mm256_unpacklo_pd(lr2, li2),
            );
            _mm256_storeu_pd(
                pass.states.as_mut_ptr().add(18).cast(),
                _mm256_unpackhi_pd(lr2, li2),
            );
            _mm256_storeu_pd(
                pass.states.as_mut_ptr().add(36).cast(),
                _mm256_unpacklo_pd(rr2, ri2),
            );
            _mm256_storeu_pd(
                pass.states.as_mut_ptr().add(38).cast(),
                _mm256_unpackhi_pd(rr2, ri2),
            );
            phase
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare<const CAUSAL: bool>(
        kernel: Kernel,
        modes: usize,
        frames: usize,
        l: u32,
        m: u32,
        initial_phase: u32,
        settings: (bool, bool),
    ) {
        let (adverse, cached) = settings;
        let values = [
            -0.0,
            0.0,
            f64::from_bits(1),
            -f64::from_bits(1),
            1e-310,
            -1e-310,
            1.0,
            -1.0,
            1.0 + 2.0f64.powi(-27),
        ];
        let input: Vec<_> = (0..frames * 2)
            .map(|i| {
                if adverse {
                    values[i % values.len()]
                } else {
                    ((i * 731 % 997) as f64 - 498.0) / 1994.0
                }
            })
            .collect();
        let poles: Vec<_> = (0..modes)
            .map(|i| {
                Complex::new(
                    (i as f64 * 0.071).cos() * 0.73,
                    (i as f64 * 0.013).sin() * 0.17,
                )
            })
            .collect();
        let rows: Vec<_> = (0..modes * l as usize)
            .map(|i| {
                Complex::new(
                    if adverse {
                        values[i % values.len()]
                    } else {
                        (i as f64 * 0.21).sin()
                    },
                    if adverse {
                        values[(i * 7 + 3) % values.len()]
                    } else {
                        (i as f64 * 0.71).cos()
                    },
                )
            })
            .collect();
        let initial: Vec<_> = (0..modes * 2)
            .map(|i| {
                Complex::new(
                    if adverse {
                        values[i % values.len()]
                    } else {
                        (i as f64 * 0.13).sin()
                    },
                    if adverse {
                        values[(i * 3 + 1) % values.len()]
                    } else {
                        (i as f64 * 0.27).cos()
                    },
                )
            })
            .collect();
        let count = (frames as i128 * i128::from(l) - i128::from(initial_phase)).max(0);
        let outputs = ((count + i128::from(m) - 1) / i128::from(m)) as usize;
        let final_phase =
            i64::from(initial_phase) + outputs as i64 * i64::from(m) - frames as i64 * i64::from(l);
        let phase = if CAUSAL {
            i64::from(initial_phase)
        } else {
            final_phase + i64::from(l) - i64::from(m)
        };
        let mut expected_states = initial.clone();
        let mut expected = vec![[0.0; 4]; outputs * 2];
        let mut tick = phase;
        let mut out = if CAUSAL { 0 } else { outputs };
        for step in 0..frames {
            let n = if CAUSAL { step } else { frames - step - 1 };
            while if CAUSAL {
                tick < i64::from(l)
            } else {
                out != 0 && tick >= 0
            } {
                if !CAUSAL {
                    out -= 1;
                }
                for channel in 0..2 {
                    for mode in 0..modes {
                        let c = rows[tick as usize * modes + mode];
                        let s = expected_states[channel * modes + mode];
                        expected[out * 2 + channel][mode % 4] += c.re * s.re - c.im * s.im;
                    }
                }
                if CAUSAL {
                    out += 1;
                    tick += i64::from(m);
                } else {
                    tick -= i64::from(m);
                }
            }
            if !CAUSAL && out == 0 {
                break;
            }
            for channel in 0..2 {
                for mode in 0..modes {
                    let state = &mut expected_states[channel * modes + mode];
                    *state = if CAUSAL {
                        poles[mode] * Complex::new(state.re + input[n * 2 + channel], state.im)
                    } else {
                        poles[mode] * *state + Complex::new(input[n * 2 + channel], 0.0)
                    };
                }
            }
            if CAUSAL {
                tick -= i64::from(l);
            } else {
                tick += i64::from(l);
            }
        }
        let mut actual_states = initial;
        let mut actual: Vec<f64> = if CAUSAL {
            vec![0.314159]
        } else {
            vec![0.314159; outputs * 2 + 1]
        };
        let direct: Vec<_> = (0..l).map(|i| (i as f64 * 0.33).cos()).collect();
        let original_state_bits: Vec<_> = actual_states
            .iter()
            .map(|s| [s.re.to_bits(), s.im.to_bits()])
            .collect();
        let original_output_bits: Vec<_> = actual.iter().map(|s| s.to_bits()).collect();
        let packed = if cached {
            kernel.pack_phase_rows(&rows, modes, 2)
        } else {
            Vec::new()
        };
        let used = kernel.block::<CAUSAL>(
            Block {
                poles: &poles,
                packed_rows: &packed,
                direct: &direct,
                first_output: 1,
                rows: &rows,
                states: &mut actual_states,
                input: &input,
                channels: 2,
                phase,
                interpolation: l,
                decimation: m,
                outputs,
            },
            &mut actual,
        );
        if used.is_none() {
            assert_eq!(
                actual_states
                    .iter()
                    .map(|s| [s.re.to_bits(), s.im.to_bits()])
                    .collect::<Vec<_>>(),
                original_state_bits,
                "fallback state {kernel:?}, {modes} modes"
            );
            assert_eq!(
                actual.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
                original_output_bits,
                "fallback output {kernel:?}, {modes} modes"
            );
            return;
        }
        assert_eq!(used, Some(tick), "returned clock {kernel:?}, causal {CAUSAL}, {modes} modes, {frames} frames, {l}/{m}, initial {initial_phase}");
        for (i, (a, e)) in actual_states.iter().zip(&expected_states).enumerate() {
            assert_eq!([a.re.to_bits(),a.im.to_bits()],[e.re.to_bits(),e.im.to_bits()], "state {i}, causal {CAUSAL}, {modes} modes, {frames} frames, {l}/{m}, phase {initial_phase}");
        }
        assert_eq!(actual.len(), expected.len() + 1);
        assert_eq!(actual[0].to_bits(), 0.314159f64.to_bits());
        let mut tick = i64::from(initial_phase);
        let mut output_index = 0;
        for n in 0..frames {
            while tick < i64::from(l) {
                for channel in 0..2 {
                    let lane = expected[output_index * 2 + channel];
                    let dot = (lane[0] + lane[1]) + (lane[2] + lane[3]);
                    let value = if CAUSAL {
                        dot + direct[tick as usize] * input[n * 2 + channel]
                    } else {
                        0.314159 + dot
                    };
                    assert_eq!(actual[1 + output_index*2+channel].to_bits(),value.to_bits(), "output {output_index}, ch {channel}, causal {CAUSAL}, frames {frames}, {l}/{m}, phase {initial_phase}");
                }
                output_index += 1;
                tick += i64::from(m);
            }
            tick -= i64::from(l);
        }
    }

    #[test]
    fn register_blocks_preserve_states_outputs_clock_and_fallback() {
        for kernel in crate::simd::test_kernels() {
            for modes in [4, 8, 12, 16, 20, 24, 28, 32, 40, 44] {
                for (l, m) in [(147, 640), (1, 4095), (3, 2), (2, 3), (13, 19), (17, 1)] {
                    for phase in [0, m - 1] {
                        for frames in [0, 1, 5, 63, 257] {
                            for adverse in [false, true] {
                                for cached in [false, true] {
                                    compare::<true>(
                                        kernel,
                                        modes,
                                        frames,
                                        l,
                                        m,
                                        phase,
                                        (adverse, cached),
                                    );
                                    compare::<false>(
                                        kernel,
                                        modes,
                                        frames,
                                        l,
                                        m,
                                        phase,
                                        (adverse, cached),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packed_phases_preserve_every_bit_and_bound_allocation() {
        let rows: Vec<_> = (0..3 * 20)
            .map(|i| {
                Complex::new(
                    f64::from_bits(
                        [
                            0x8000000000000000,
                            0x7ff8123456789abc,
                            1,
                            0x3ff0000000000000,
                        ][i % 4],
                    ),
                    f64::from_bits(
                        [
                            0,
                            0x7ff8abc123456789,
                            0x8000000000000001,
                            0xbff0000000000000,
                        ][i % 4],
                    ),
                )
            })
            .collect();
        for kernel in crate::simd::test_kernels() {
            let packed = kernel.pack_phase_rows(&rows, 20, 2);
            if !packed.is_empty() {
                assert_eq!(packed.len(), 3);
                for phase in 0..3 {
                    for group in 0..2 {
                        for (lane, mode) in [0, 2, 1, 3, 4, 6, 5, 7].into_iter().enumerate() {
                            assert_eq!(
                                packed[phase][group * 16 + lane].to_bits(),
                                rows[phase * 20 + group * 8 + mode].re.to_bits()
                            );
                            assert_eq!(
                                packed[phase][group * 16 + 8 + lane].to_bits(),
                                rows[phase * 20 + group * 8 + mode].im.to_bits()
                            );
                        }
                    }
                    for (lane, mode) in [16, 18, 17, 19].into_iter().enumerate() {
                        assert_eq!(
                            packed[phase][32 + lane].to_bits(),
                            rows[phase * 20 + mode].re.to_bits()
                        );
                        assert_eq!(
                            packed[phase][36 + lane].to_bits(),
                            rows[phase * 20 + mode].im.to_bits()
                        );
                    }
                }
            }
            assert!(kernel.pack_phase_rows(&rows, 19, 2).is_empty());
            assert!(kernel.pack_phase_rows(&rows, 20, 1).is_empty());
            assert!(kernel.pack_phase_rows(&rows, 20, 3).is_empty());
            assert!(kernel.pack_phase_rows(&rows[..59], 20, 2).is_empty());
            assert!(kernel
                .pack_phase_rows(&vec![Complex::default(); 1640 * 20], 20, 2)
                .is_empty());
        }
    }
}
