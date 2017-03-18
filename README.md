**WORK IN PROGRESS**: This branch is meant to be built on top of SIMD in
`std`, which currently uses https://github.com/BurntSushi/stdsimd. The goal
is to get this crate building on Rust stable. Currently, the crate builds,
but won't work (yet) for supporting non-Intel or newer things than SSE2.
Getting it to work completely on Rust stable is blocked on actually stabilizing
SIMD and doing a bit more work to make everything this crate needs available
in `stdsimd`.

# `simd`

[![Build Status](https://travis-ci.org/rust-lang-nursery/simd.png)](https://travis-ci.org/rust-lang-nursery/simd)

`simd` offers a basic interface to the SIMD functionality of CPUs.

[Documentation](https://docs.rs/simd)
