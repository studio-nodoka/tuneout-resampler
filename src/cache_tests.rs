use super::*;

fn key(from: u32) -> BankKey {
    BankKey::new(
        from,
        32001,
        Design {
            phases: 1,
            taps: 3,
            beta: 20.,
            rolloff: 1.,
        },
    )
}

#[test]
fn bank_cache_enforces_bytes_lru_entries_and_live_bank_lifetime() {
    let mut cache = BankCache::new(96, 2);
    let a = cache.reserve(key(32000));
    let b = cache.reserve(key(48000));
    let live = b.get_or_init(|| std::sync::Arc::new(vec![0.25; 6])).clone();
    assert!(std::sync::Arc::ptr_eq(&a, &cache.reserve(key(32000))));
    let _c = cache.reserve(key(64000));
    assert_eq!(cache.reserved_bytes, 96);
    assert_eq!(cache.entries.len(), 2);
    assert!(!cache.entries.iter().any(|entry| entry.key == key(48000)));
    assert_eq!(live.as_slice(), &[0.25; 6]);
    assert!(!std::sync::Arc::ptr_eq(&b, &cache.reserve(key(48000))));
    let mut huge = key(10000);
    huge.taps = usize::MAX;
    let _uncached = cache.reserve(huge);
    assert_eq!(cache.reserved_bytes, 96);
    assert_eq!(cache.entries.len(), 2);
    let mut one = BankCache::new(1024, 1);
    one.reserve(key(32000));
    one.reserve(key(48000));
    assert_eq!(one.entries.len(), 1);
    assert_eq!(one.reserved_bytes, 48);
}

#[test]
fn cache_key_includes_actual_rates_and_every_design_parameter() {
    let base = key(32000);
    let mut variants = [base; 6];
    variants[0].from_rate += 1;
    variants[1].to_rate += 1;
    variants[2].phases += 1;
    variants[3].taps += 2;
    variants[4].beta_bits += 1;
    variants[5].rolloff_bits -= 1;
    let mut cache = BankCache::new(4096, 8);
    let original = cache.reserve(base);
    for changed in variants {
        assert!(!std::sync::Arc::ptr_eq(&original, &cache.reserve(changed)));
    }
}

#[test]
fn length_presets_share_cache_entries_only_when_the_effective_design_matches() {
    use crate::{design::select, FilterLength, FilterLengthPolicy};
    let mut cache = BankCache::new(64 * 1024 * 1024, 8);
    for (from, to) in crate::test_cases::pairs().into_iter().chain([(1, 2)]) {
        let key = |length| {
            BankKey::new(
                from,
                to,
                select(from, to, length, FilterLengthPolicy::Generic).unwrap(),
            )
        };
        let standard = cache.reserve(key(FilterLength::Standard));
        let long_key = key(FilterLength::Long);
        let long = cache.reserve(long_key);
        let custom = cache.reserve(key(FilterLength::Custom(long_key.taps)));
        assert!(std::sync::Arc::ptr_eq(&long, &custom));
        // Resource limits can make different presets select the same length.
        assert_eq!(
            std::sync::Arc::ptr_eq(&standard, &long),
            key(FilterLength::Standard).taps == long_key.taps
        );
    }
}

#[test]
fn concurrent_cache_hits_share_one_initialization_without_holding_cache_lock() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier, Mutex,
    };
    let cache = Arc::new(Mutex::new(BankCache::new(4096, 8)));
    let barrier = Arc::new(Barrier::new(8));
    let builds = Arc::new(AtomicUsize::new(0));
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let (cache, barrier, builds) = (cache.clone(), barrier.clone(), builds.clone());
            std::thread::spawn(move || {
                barrier.wait();
                let bank = cache.lock().unwrap().reserve(key(32000));
                bank.get_or_init(|| {
                    // Another key can be requested while this bank is constructed.
                    cache.lock().unwrap().reserve(key(48000));
                    builds.fetch_add(1, Ordering::SeqCst);
                    Arc::new(vec![0.25; 6])
                })
                .clone()
            })
        })
        .collect();
    let results: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    assert_eq!(builds.load(Ordering::SeqCst), 1);
    assert!(results.iter().all(|bank| Arc::ptr_eq(bank, &results[0])));
}
