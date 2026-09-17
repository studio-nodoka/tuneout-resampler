"""Independent C1 reference: requires mpmath (development only).

Prints selected correctly rounded coefficients from complete normalized rows.
The 80- and 100-decimal-digit calculations must agree before printing a value.
This does not import or invoke the Rust implementation.
"""
import struct
import math
import sys
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
import mpmath as mp


def cases():
    rates = [int(line) for line in
             (Path(__file__).parents[1] / 'tests/fixtures/rates.txt').read_text().splitlines()]
    pairs = {(source, target) for source in rates for target in rates}
    for rate in rates:
        for neighbor in (rate - 1, rate + 1):
            pairs.update(((rate, neighbor), (neighbor, rate)))
    pairs.update(((1, 768000), (4294967295, 4294967291)))
    for source, target in sorted(pairs):
        if source == target:
            continue
        denominator = target // math.gcd(source, target)
        phases = denominator if denominator <= 2048 else 1024
        limit = (max(3, min(8388608 // (phases + 1), 65535, source * 160 // 1000)) - 1) | 1
        lower = min(source, target)
        support = (1664 * source + lower - 1) // lower
        taps = min(support, limit) | 1
        yield source, target, phases, taps


@lru_cache(maxsize=128)
def row(scale_fraction, phase_fraction, taps, digits):
    with mp.workdps(digits):
        half = taps // 2
        scale = mp.mpf(scale_fraction.numerator) / scale_fraction.denominator
        values = []
        for tap in range(taps):
            coordinate = Fraction(tap - half) - phase_fraction
            x = mp.mpf(coordinate.numerator) / coordinate.denominator
            if abs(x) >= half:
                value = mp.mpf(0)
            else:
                y = 100 * (1 - (x / half) ** 2)
                window = mp.besseli(0, 2 * mp.sqrt(y)) - 1 - y
                if coordinate == 0:
                    kernel = scale
                elif (coordinate * scale_fraction).denominator == 1:
                    kernel = mp.mpf(0)
                else:
                    kernel = mp.sin(mp.pi * scale * x) / (mp.pi * x)
                value = kernel * window
            values.append(value)
        total = mp.fsum(values)
        return tuple(float(value / total) for value in values)


if __name__ == "__main__":
    print("from,to,phases,taps,phase,tap,f64_bits")
    for index, (source, target, phases, taps) in enumerate(cases(), 1):
        scale = Fraction(min(source, target), source)
        for phase in sorted({0, phases // 2}):
            fraction = Fraction(phase, phases)
            expected = row(scale, fraction, taps, 80)
            assert expected == row(scale, fraction, taps, 100)
            indices = {0, 1, taps // 4, taps // 2 - 2, taps // 2 - 1, taps // 2,
                       taps // 2 + 1, taps // 2 + 2, taps - 2, taps - 1}
            indices.update((i * 7919 + phase * 101) % taps for i in range(16))
            for tap in sorted(i for i in indices if 0 <= i < taps):
                bits = struct.unpack("<Q", struct.pack("<d", expected[tap]))[0]
                print(f"{source},{target},{phases},{taps},{phase},{tap},{bits:016x}")
        if index % 10 == 0:
            print(f"Verified {index} rate pairs at both reference precisions", file=sys.stderr, flush=True)
