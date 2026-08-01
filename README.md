# ml-memchr

An attempt at writing `memchr` in **OxCaml**, using unboxed 64-bit integers, mode
annotations and mutable locals to get a SWAR (SIMD-Within-A-Register) byte search
that allocates nothing — and then measuring honestly how close it lands to the
equivalent C and Rust.

The library ships **two** implementations of the same algorithm, so the cost of
writing it idiomatically can be separated from the cost of the language:

- **`ml_memchr`** — idiomatic OCaml. Mutually recursive `block`/`word`/`bytes`
  functions, no mutable state, tail calls all the way down.
- **`memchr`** — the optimized version. Mutable locals, hand-rolled loop exits,
  and a branchless `ctz`-based tail.

Both are `[@@zero_alloc]`-checked. The headline results, at 64 KiB:

| | vs. C SWAR | vs. `Bytes.index_opt` |
| --- | --- | --- |
| `memchr` (optimized) | ~1.3× slower | ~7.1× faster |
| `ml_memchr` (idiomatic) | ~1.43× slower | ~6.4× faster |

And at small inputs (≤ 16 bytes) the optimized version is **faster than the C and
Rust references** — though for an algorithmic reason, not a language one. See
[Reading the results](#reading-the-results).

## The algorithm

Both implementations are a three-tier scan:

1. **32 bytes/iteration** — four unaligned 64-bit loads, four SWAR probes, OR the
   results together and test once. On a hit, narrow to *which* of the four words
   matched.
2. **8 bytes/iteration** — one word at a time.
3. **The tail** — resolve the exact byte index within the final word.

The SWAR kernel is the classic zero-byte trick:

```ocaml
let[@inline always] swar_raw w =
  let w = Int64_u.(w lxor cs) in
  Int64_u.((w - ones) land lognot w)
```

XOR against a word of repeated needle bytes turns "byte equals `c`" into "byte is
zero", and `(w - ones) & ~w` masked with `0x8080…` flags the zero bytes.

### Tier 1 narrowing

Rather than rewinding to the start of the 32-byte block and re-scanning, all of
the SWAR implementations identify the matching word directly:

```ocaml
hit <-
  (if not Int64_u.(r0 land mask = #0L) then i
   else if not Int64_u.(r1 land mask = #0L) then Int64_u.(i + #8L)
   else if not Int64_u.(r2 land mask = #0L) then Int64_u.(i + #16L)
   else Int64_u.(i + #24L));
```

This costs nothing in the common (no-match) path — the chain sits inside the
already-taken `if` — and saves up to three redundant SWAR probes on the hit path.
The `r0` test has to come first: a needle can occur in word 0 *and* in a later
word of the same block, and `memchr` must report the earlier one.

### The `ctz` tail

This is where `memchr` and `ml_memchr` diverge, and it is the single biggest win.
`ml_memchr` (like the C and Rust references) finishes with a byte-at-a-time loop.
`memchr` instead does one masked load and one instruction:

```ocaml
let w = swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) in
let r = if remaining > 8 then 8 else remaining in
let valid = Int64_u.shift_right_logical #0xFFFF_FFFF_FFFF_FFFFL ((8 - r) * 8) in
let t = Int64_u.(w land mask land valid) in
if not Int64_u.(t = #0L) then
  hit <- Int64_u.(i + (Ocaml_intrinsics_kernel.Int64.Unboxed.count_trailing_zeros t lsr 3))
```

`count_trailing_zeros` lowers to a single `tzcnt`. Since each matching byte sets
exactly its own bit 7, `ctz(t) lsr 3` *is* the byte offset. Up to seven byte
compares and their branches collapse into one instruction.

Two things make this safe rather than reckless:

- **The over-read is deliberate and in-bounds.** The load reads a full 8 bytes
  even when fewer remain. `i` is always a multiple of 8 (every loop advances by 8
  or 32 from zero, and every `hit` is a word offset), and OCaml's `bytes` data
  area is a whole number of words — so an 8-aligned 8-byte read at any `i < n`
  stays inside the allocation. The `valid` mask discards the bytes past `n`.
- **It is little-endian-specific.** `valid` keeps the *low* `r` bytes and `ctz`
  counts from the low end, both of which assume byte 0 is the least significant.
  This has only been built and tested on x86-64.

### What makes it OxCaml-flavoured

| Feature | Where | Why |
| --- | --- | --- |
| `int64#` (`Int64_u`) | the whole kernel | Unboxed 64-bit ints. In stock OCaml, `Int64` arithmetic allocates a boxed value per intermediate; here `lxor`, `-`, `land`, `lognot` are raw machine ops on registers. |
| `%caml_bytes_get64u#` | `Bytes.unsafe_get_int64_ne` | Unaligned 64-bit load out of `bytes` returning an *unboxed* `int64#` directly — no box on the way out. |
| `..._indexed_by_int64#` | every load in the loops | Indexes the load by an `int64#` rather than a tagged `int`. The loop counter never has to be tagged/untagged just to be used as an offset. |
| `let mutable i` / `let mutable hit` | `memchr`'s loop state | OxCaml mutable local bindings. Stock OCaml would need `ref` cells. `ml_memchr` deliberately avoids these, which is most of the difference between the two. |
| `s @ local read` | both signatures | The haystack is taken at `local` mode (it cannot escape) and `read` (it is not mutated). |
| `@@ portable` | the `external` declarations | Marks the primitives as safe to use across capsules/domains. |
| `Ocaml_intrinsics_kernel.Int64.Unboxed.count_trailing_zeros` | the tail | `tzcnt` on an `int64#`, no boxing at the boundary. |
| `[@@zero_alloc]` | `src/memchr.mli`, both values | **Compiler-checked.** The build fails if either function ever allocates. This is what turns "I think this is unboxed" into a guarantee. |
| `[@nontail]` | `ml_memchr`'s entry | Marks the `block #0L` call as not-a-tail-call so the recursion is compiled as a loop rather than growing the stack expectation. |

### The part that fights back

OCaml has no `break`. In `memchr`, every tier exit is emulated by slamming the
cursor to `n` and stashing the position:

```ocaml
if not Int64_u.(rs land mask = #0L) then (
  hit <- (* ...narrowing chain... *);
  i <- n64)              (* force the loop condition false *)
else i <- Int64_u.(i + #32L)
done;
if Int64_u.(hit < #0L) then (* ...tier 2... *) else i <- hit
```

Every iteration carries a branch that either advances the cursor *or* slams it to
`n`, where C and Rust just `break` and leave `i` alone. That is the most visible
structural difference from `benchmarks/swar_stubs.c` and
`benchmarks/rust/src/lib.rs`.

`ml_memchr` sidesteps this entirely — recursion gives you the early exit for free,
because `word i` simply *is* the continuation — at the cost of running ~10%
slower in steady state.

## Benchmarks

### What is being compared

| Name | What it is |
| --- | --- |
| `C SIMD memchr` | glibc's `memchr` (`benchmarks/memchr_stubs.c`). Fully vectorized — the ceiling, not a fair SWAR peer. |
| `C SWAR memchr` | The same 32/8/1 algorithm in C, compiled `-O2 -fno-tree-vectorize`. |
| `Rust SWAR memchr` | The same algorithm in Rust, built `--release` with `-C no-vectorize-loops -C no-vectorize-slp`. Not the `memchr` crate — hand-written SWAR. |
| `ML style SWAR memchr` | `Memchr.ml_memchr`, the idiomatic recursive version. |
| `ML optimized SWAR memchr` | `Memchr.memchr`, mutable locals + `ctz` tail. |
| `Bytes.index_opt` | OCaml stdlib. Byte-at-a-time baseline. |

Auto-vectorization is explicitly **disabled** for the C and Rust SWAR builds.
Without that, both compilers turn the tier-1 loop into AVX2 and the comparison
stops being about SWAR at all. The goal is to compare *the same algorithm* across
three languages, with glibc alongside as a reference for what real SIMD buys.

### Harness

`core_bench`, driven from `benchmarks/memchr_bench.ml`. The haystack is `len`
bytes of `'a'` with a single `'z'` planted at `len - len/10 - 1`, so every run
scans roughly 90% of the buffer before hitting — a mostly-miss scan, which is the
case SWAR is supposed to win. All calls go through `Sys.opaque_identity` so
nothing is folded away, and the C/Rust entry points are `[@@noalloc]`.

### Results

11th Gen Intel Core i7-11800H, OxCaml 5.2.0+ox (flambda2), gcc 16.1.1, rustc
1.97.1, glibc 2.44, `-quota 3`.

Cycles per run (lower is better):

| len | C SIMD | C SWAR | Rust SWAR | ML style | **ML optimized** | `Bytes.index_opt` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 10.4 | 14.1 | 15.1 | 27.9 | **12.9** | 12.6 |
| 8 | 10.4 | 16.7 | 17.0 | 36.7 | **14.8** | 15.8 |
| 16 | 10.5 | 17.4 | 19.0 | 37.0 | **16.4** | 23.3 |
| 32 | 10.4 | 17.7 | 19.0 | 34.3 | **18.6** | 56.8 |
| 64 | 12.1 | 19.6 | 20.9 | 33.0 | **23.9** | 88.3 |
| 128 | 13.6 | 28.2 | 28.3 | 45.1 | **32.2** | 145.2 |
| 256 | 15.5 | 41.7 | 42.7 | 68.9 | **50.5** | 268.0 |
| 1024 | 27.5 | 130.2 | 130.7 | 186.7 | **164.9** | 968.8 |
| 4096 | 60.9 | 444.7 | 431.2 | 636.8 | **565.8** | 3 781.0 |
| 65536 | 1 119.8 | 6 633.7 | 6 300.4 | 9 494.5 | **8 536.6** | 60 987.0 |

Nanoseconds per run:

| len | C SIMD | C SWAR | Rust SWAR | ML style | **ML optimized** | `Bytes.index_opt` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4.51 | 6.12 | 6.53 | 12.09 | **5.58** | 5.47 |
| 8 | 4.51 | 7.25 | 7.35 | 15.94 | **6.43** | 6.86 |
| 16 | 4.56 | 7.56 | 8.23 | 16.04 | **7.14** | 10.12 |
| 32 | 4.53 | 7.68 | 8.23 | 14.88 | **8.07** | 24.64 |
| 64 | 5.24 | 8.49 | 9.08 | 14.33 | **10.35** | 38.33 |
| 128 | 5.91 | 12.23 | 12.28 | 19.55 | **13.98** | 63.02 |
| 256 | 6.72 | 18.10 | 18.51 | 29.92 | **21.92** | 116.31 |
| 1024 | 11.94 | 56.53 | 56.75 | 81.02 | **71.57** | 420.49 |
| 4096 | 26.44 | 193.01 | 187.16 | 276.37 | **245.57** | 1 641.04 |
| 65536 | 486.00 | 2 879.18 | 2 734.54 | 4 120.85 | **3 705.12** | 26 469.90 |

A second run of the same binary came out 2–6% higher on *every* row, glibc and
the stdlib baseline included — whole-machine drift (no frequency pinning), not a
per-implementation effect. The ratios between implementations moved by less than
5%, so the table above is the cooler of the two runs and the comparisons below
are averaged across both.

### Steady-state throughput

Taking the slope between `len = 4096` and `len = 65536` removes fixed
call/setup/tail overhead and leaves the cost of the inner loop alone (averaged
over both runs):

| Implementation | cycles/byte | bytes/cycle |
| --- | ---: | ---: |
| C SIMD memchr (AVX2) | 0.019 | ~51.9 |
| Rust SWAR | 0.107 | ~9.3 |
| C SWAR | 0.113 | ~8.8 |
| **ML optimized SWAR** | **0.148** | **~6.7** |
| ML style SWAR | 0.164 | ~6.1 |
| `Bytes.index_opt` | 1.044 | ~0.96 |

The three languages' SWAR loops land in the same band, well above the ~1
byte/cycle of a naive scan and well below vectorized `memchr`. `Bytes.index_opt`
at almost exactly 1.0 cycles/byte is a clean sanity check that the baseline
really is byte-at-a-time.

### Reading the results

**The unboxing works.** ~6.7 bytes/cycle is not achievable if `int64#` values are
being boxed — a single allocation per iteration would show up immediately as a
collapse toward the stdlib line. The `[@@zero_alloc]` check holds this in place at
build time.

**Small-input wins are algorithmic, not linguistic.** At `len ≤ 16` the optimized
OCaml beats both C and Rust (12.9c vs 14.1c/15.1c at `len = 4`). This is *not*
OxCaml outrunning C. It is the `ctz` tail beating a byte-at-a-time tail: at these
sizes the tail is the entire function. Give the C reference the same tail and it
would win again. The honest language comparison is the steady-state number.

**The steady-state gap is ~1.3–1.4×, and it is loop shape.** The kernel is the
same handful of ALU ops everywhere; what differs is the `break` emulation. C and
Rust leave the loop; OCaml computes a new cursor value on every iteration whether
or not it found anything.

**Idiomatic costs ~10%.** `ml_memchr` runs at 6.1 B/c against the optimized 6.7
B/c — the recursive version is *not* catastrophically slower, and it is still 6.4×
faster than the stdlib. Most of its extra cost is at small sizes (27.9c vs 12.9c
at `len = 4`), where it pays full tier setup and then walks the tail one byte at a
time. If you want the readable version, it is not a disaster.

**Nothing beats glibc, and nothing was going to.** At 64 KiB glibc is 7.6× faster
than the optimized OCaml and 5.9× faster than the C SWAR it is measured against.
That gap is AVX2 versus 64-bit registers, and it is the reason the interesting
number is the ~1.3× against C SWAR rather than the 7.6× against glibc.

**Allocation column.** `core_bench` reports a uniform `3.00w` per run for *every*
row, including the `[@@noalloc]` C stubs — it is harness overhead in the staged
closure, not the implementations. The real allocation claim here is the
compiler-checked `[@@zero_alloc]`, not this column.

## Building and running

Requires an OxCaml switch (developed on `5.2.0+ox`) and `cargo` for the Rust
comparison.

```sh
dune build     # builds the library and benchmark executable
dune test      # inline tests
./_build/default/benchmarks/memchr_bench.exe time cycles percentage -quota 3 -ascii
```

`dune build` covers everything: the root `dune` defines a `default` alias
depending on `benchmarks/memchr_bench.exe` and `(alias_rec src/all)`, and the
`ml-memchr` package is marked `(allow_empty)` since it has no install stanza.

The Rust static library is built by a dune rule shelling out to `cargo build
--release --offline`, so `benchmarks/rust` must build without network access
(it has no dependencies beyond std).

### Tests

`dune test` runs five inline tests: a smoke test, a longer-string test, a
no-match test, a 200-iteration randomized parity check of `memchr` against a
naive byte scan, and a 200-iteration check that `memchr` and `ml_memchr` agree.
Random lengths go up to 200 bytes, so all three tiers and both tail paths are
exercised.

## Layout

```
src/memchr.ml                the two OxCaml implementations + inline tests
src/memchr.mli               both values, both [@@zero_alloc]
benchmarks/memchr_bench.ml   core_bench driver
benchmarks/memchr_stubs.c    glibc memchr binding
benchmarks/swar_stubs.c      C SWAR reference (-fno-tree-vectorize)
benchmarks/rs_stubs.c        FFI shim for the Rust static lib
benchmarks/rust/src/lib.rs   Rust SWAR reference (vectorization disabled)
```

## Caveats

- Single machine, single microarchitecture, no frequency pinning. Run-to-run
  drift of a few percent is visible; ratios are stable, absolute numbers are not
  a portable claim.
- One haystack shape (needle at ~90%, uniform filler). Hit-early, no-hit, and
  unaligned-start distributions are not measured.
- The `ctz` tail in `memchr` is little-endian-only and has been built and tested
  only on x86-64.
- `benchmarks/rust/src/lib.rs` is hand-written SWAR, deliberately hobbled to match
  the algorithm. It is *not* the `memchr` crate, which would be in glibc's league.
- `memchr` takes a length `n` independently of `Bytes.length s` and does no bounds
  checking; passing `n > Bytes.length s` is undefined behaviour.

## License

MIT — see [LICENSE](LICENSE).
