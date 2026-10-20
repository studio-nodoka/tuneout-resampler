# Third-party notices

## Runtime dependencies

The buffered implementation uses RustFFT 6.4.1 and its locked dependencies:
`num-complex`, `num-integer`, `num-traits`, `primal-check`, `strength_reduce`
and `transpose`. Each is distributed under MIT OR Apache-2.0. This package
uses the Apache-2.0 option; retain the dependencies' copyright and license
notices when distributing their source or binaries. `autocfg`, also dual
licensed, is a build dependency. The production `tuneout-resampler` is used
only by development benchmarks.

## Legacy all-IIR design reference

The half-band prototype coefficient equations in `src/design.rs` are adapted
from Laurent de Soras's HIIR `PolyphaseIir2Designer` (2005), distributed under
the Do What The Fuck You Want To Public License, version 2. No HIIR processing
code or binary dependency is included.

Source inspected: https://github.com/unevens/hiir/blob/master/PolyphaseIir2Designer.h

    DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                       Version 2, December 2004

    Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

    Everyone is permitted to copy and distribute verbatim or modified
    copies of this license document, and changing it is allowed as long
    as the name is changed.

               DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
      TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

     0. You just DO WHAT THE FUCK YOU WANT TO.

## Startup algorithm references

The original Rust startup implementation uses the conventional Burg and Levinson
equations described by John P. Burg (1975) and Koen Vos, *A Fast Implementation of
Burg's Method* (August 2013), https://opus-codec.org/docs/vos_fastburg.pdf.
Vos's paper is published under Creative Commons Attribution 3.0. No source code
from Opus is copied or linked. The analytic modal initialization and local
order-selection safeguards are described in [RESEARCH.md](RESEARCH.md).
