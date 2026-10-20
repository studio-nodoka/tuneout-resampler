fn tail_bound(weights: &[(f64, f64)], guard: usize) -> f64 {
    weights
        .iter()
        .map(|&(radius, weight)| weight * radius.powf(guard as f64))
        .sum()
}

fn allocated_tail_bound(weights: &[(f64, f64)], guards: &[usize]) -> f64 {
    weights
        .iter()
        .zip(guards)
        .map(|(&(radius, weight), &guard)| weight * radius.powf(guard as f64))
        .sum()
}

/// Remove cheap mode updates first, subject to one combined error budget.
///
/// Reducing a mode's horizon from g to g-1 raises its bound by
/// w*(1-r)*r^(g-1). These increments increase monotonically as more frames are
/// removed. A shared marginal threshold therefore allocates almost all of the
/// budget in a fixed number of design-time iterations, even at extreme ratios.
/// A final greedy step uses any leftover budget after integer rounding. Every
/// horizon is capped at the original common guard, preserving its latency.
fn allocate_guards(weights: &[(f64, f64)], maximum: usize, epsilon: f64) -> Vec<usize> {
    let mut guards = vec![maximum; weights.len()];
    let mut lower = 0.0;
    let mut upper = weights
        .iter()
        .map(|&(radius, weight)| weight * (1.0 - radius))
        .fold(0.0, f64::max);
    for _ in 0..128 {
        let threshold = lower + (upper - lower) * 0.5;
        if threshold == lower || threshold == upper {
            break;
        }
        let candidate: Vec<_> = weights
            .iter()
            .map(|&(radius, weight)| {
                if weight == 0.0 || threshold >= weight * (1.0 - radius) {
                    0
                } else if radius == 0.0 {
                    1.min(maximum)
                } else {
                    ((threshold / (weight * (1.0 - radius))).ln() / radius.ln())
                        .ceil()
                        .max(0.0)
                        .min(maximum as f64) as usize
                }
            })
            .collect();
        if allocated_tail_bound(weights, &candidate) <= epsilon {
            lower = threshold;
            guards = candidate;
        } else {
            upper = threshold;
        }
    }
    loop {
        let next = weights
            .iter()
            .zip(&guards)
            .enumerate()
            .filter(|(_, (_, guard))| **guard != 0)
            .map(|(index, (&(radius, weight), &guard))| {
                (
                    index,
                    weight * (1.0 - radius) * radius.powf((guard - 1) as f64),
                )
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((index, _)) = next else { break };
        guards[index] -= 1;
        if allocated_tail_bound(weights, &guards) > epsilon {
            guards[index] += 1;
            break;
        }
    }
    guards
}

fn minimum_guard(weights: &[(f64, f64)], epsilon: f64) -> Result<usize, Error> {
    if tail_bound(weights, 0) <= epsilon {
        return Ok(0);
    }
    let mut lower = 0;
    let mut upper = 1_usize;
    while tail_bound(weights, upper) > epsilon {
        lower = upper;
        if upper >= MAX_GUARD_FRAMES { return Err(Error::ResourceLimit); }
        upper = upper.saturating_mul(2).min(MAX_GUARD_FRAMES);
    }
    while lower + 1 < upper {
        let middle = lower + (upper - lower) / 2;
        if tail_bound(weights, middle) <= epsilon {
            upper = middle;
        } else {
            lower = middle;
        }
    }
    Ok(upper)
}
