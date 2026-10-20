//! Share immutable filter banks while keeping stream history and clocks separate.

use crate::{design::Design, precision::coefficients};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

const MAX_CACHED_BYTES: usize = 64 * 1024 * 1024;
const MAX_CACHED_BANKS: usize = 8;

// Keep the Vec allocation: converting a large bank to Arc<[f64]> would copy it.
// Only immutable playback banks enter the cache. Stream state stays per instance.
pub(crate) type Coefficients = Arc<Vec<f64>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BankKey {
    from_rate: u32,
    to_rate: u32,
    phases: usize,
    taps: usize,
    beta_bits: u64,
    rolloff_bits: u64,
}

impl BankKey {
    fn new(from_rate: u32, to_rate: u32, design: Design) -> Self {
        Self {
            from_rate,
            to_rate,
            phases: design.phases,
            taps: design.taps,
            beta_bits: design.beta.to_bits(),
            rolloff_bits: design.rolloff.to_bits(),
        }
    }
}

struct BankEntry {
    key: BankKey,
    bytes: usize,
    // Callers share this initialization slot before the coefficient bank exists.
    // The bank has its own Arc so streams can retain it after cache eviction.
    bank: Arc<OnceLock<Coefficients>>,
}

struct BankCache {
    // The least recently used entry is at the front. Cache hits move to the back.
    entries: VecDeque<BankEntry>,
    reserved_bytes: usize,
    byte_limit: usize,
    entry_limit: usize,
}

impl BankCache {
    fn new(byte_limit: usize, entry_limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            reserved_bytes: 0,
            byte_limit,
            entry_limit,
        }
    }

    // Called under the cache mutex. Initialization happens after releasing it,
    // so constructing one rate never blocks a hit for another rate. OnceLock
    // also shares an in-flight construction while its entry remains resident.
    fn reserve(&mut self, key: BankKey) -> Arc<OnceLock<Coefficients>> {
        if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
            let entry = self.entries.remove(index).expect("existing bank entry");
            let bank = entry.bank.clone();
            self.entries.push_back(entry);
            return bank;
        }
        let bank = Arc::new(OnceLock::new());
        let bytes = key
            .phases
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(key.taps))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()));
        let Some(bytes) = bytes.filter(|&bytes| bytes <= self.byte_limit) else {
            return bank;
        };
        if self.entry_limit == 0 {
            return bank;
        }
        while self.entries.len() >= self.entry_limit
            || self.reserved_bytes > self.byte_limit - bytes
        {
            let entry = self.entries.pop_front().expect("nonempty bounded cache");
            self.reserved_bytes -= entry.bytes;
        }
        self.reserved_bytes += bytes;
        self.entries.push_back(BankEntry {
            key,
            bytes,
            bank: bank.clone(),
        });
        bank
    }
}

pub(crate) fn shared_coefficients(from_rate: u32, to_rate: u32, design: Design) -> Coefficients {
    static CACHE: OnceLock<Mutex<BankCache>> = OnceLock::new();
    // This bounds coefficients retained by the cache, including reservations
    // for builds in progress. Active streams can keep an evicted bank alive.
    let cache =
        CACHE.get_or_init(|| Mutex::new(BankCache::new(MAX_CACHED_BYTES, MAX_CACHED_BANKS)));
    let bank = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .reserve(BankKey::new(from_rate, to_rate, design));
    bank.get_or_init(|| Arc::new(coefficients(from_rate, to_rate, design)))
        .clone()
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
