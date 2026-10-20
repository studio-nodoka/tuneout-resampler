//! Every listed rate receives the same ordered-pair and neighboring-rate checks.

pub fn pairs() -> Vec<(u32, u32)> {
    let rates: Vec<u32> = include_str!("../fixtures/rates.txt")
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    let mut pairs: Vec<_> = rates
        .iter()
        .flat_map(|&from| rates.iter().map(move |&to| (from, to)))
        .collect();
    for rate in rates {
        for neighbor in [rate - 1, rate + 1] {
            pairs.extend([(rate, neighbor), (neighbor, rate)]);
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}
