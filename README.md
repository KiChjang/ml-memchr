# ml-memchr

An attempt at writing `memchr` in [**OxCaml**](https://oxcaml.org), using unboxed 64-bit integers, mode
annotations and mutable locals to get a SWAR (SIMD-Within-A-Register) byte search
that allocates nothing, and then measuring honestly how close it lands to the
equivalent C and Rust.

The library ships **two** implementations of the same algorithm, so the cost of
writing it idiomatically can be separated from the cost of the language:

- **`ml_memchr`**: idiomatic OCaml. Mutually recursive `block`/`word`/`bytes`
  functions, no mutable state, tail calls all the way down.
- **`memchr`**: the imperative version. Mutable locals, hand-rolled loop exits,
  and a branchless `ctz`-based tail.

The Rust side mirrors this split, with a raw-pointer version and a
mostly-safe iterator version, so the "idiomatic tax" can be compared across both
languages rather than just asserted about one. See
[The two Rust versions](#the-two-rust-versions).

Both OCaml versions are `[@@zero_alloc]`-checked. The headline results, at 64 KiB:

| | vs. C SWAR | vs. safe Rust | vs. `Bytes.index_opt` |
| --- | --- | --- | --- |
| `memchr` (imperative) | ~1.28x slower | ~1.12x slower | ~7.1x faster |
| `ml_memchr` (idiomatic) | ~1.29x slower | ~1.13x slower | ~7.1x faster |

That middle column is arguably the fairest one in the whole document. Measured
against C, OCaml gives up ~28%; measured against Rust code written at a
comparable level of abstraction, it gives up ~12%.

The idiomatic OCaml version used to trail the imperative one by ~10% in steady
state. It no longer does, and the reason it did turned out to be more interesting
than the gap itself. See [The `[@nontail]` trap](#the-nontail-trap).

At small inputs (≤ 16 bytes) the imperative version is **faster than the C and
Rust references**, though for an algorithmic reason, not a language one. See
[Reading the results](#reading-the-results).

## The algorithm

Both implementations are a three-tier scan:

1. **32 bytes/iteration**: four unaligned 64-bit loads, four SWAR probes, OR the
   results together and test once. On a hit, narrow to *which* of the four words
   matched.
2. **8 bytes/iteration**: one word at a time.
3. **The tail**: resolve the exact byte index within the final word.

The SWAR kernel is the classic zero-byte trick:

```ocaml
let[@inline always] swar_raw w cs ones =
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

This costs nothing in the common (no-match) path: the chain sits inside the
already-taken `if` and saves up to three redundant SWAR probes on the hit path.
The `r0` test has to come first: a needle can occur in word 0 *and* in a later
word of the same block, and `memchr` must report the earlier one.

### The `ctz` tail

This is where `memchr` and `ml_memchr` diverge, and it is what buys the
imperative version its small-input win. `ml_memchr` (like the C and Rust
references) finishes with a byte-at-a-time loop. `memchr` instead does one masked
load and one instruction:

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
  area is a whole number of words, so an 8-aligned 8-byte read at any `i < n`
  stays inside the allocation. The `valid` mask discards the bytes past `n`.
- **It is little-endian-specific.** `valid` keeps the *low* `r` bytes and `ctz`
  counts from the low end, both of which assume byte 0 is the least significant.
  This has only been built and tested on x86-64.

## The `[@nontail]` trap

This is the most instructive thing the project turned up, and it is specific to
having modes in the language.

`ml_memchr` originally defined its helpers as closures over the enclosing scope,
which is the obvious way to write it. `s`, `c`, `n`, `cs`, `ones` and `mask` are
fixed for the entire call, so why thread them through every recursive call?

```ocaml
let ml_memchr (s @ local read) c n =
  let ones = ... in
  let mask = ... in
  let cs = ... in
  let rec bytes i = ... in
  let rec word i  = ... in
  let rec block i = ... in
  block #0L [@nontail]
```

The reason to thread them through is that `s` is `local`. A closure that captures
a local value is itself local, so each helper had to be built as a **region-local
allocation on every call**. And a local value cannot be the target of a tail
call, because the region has to be torn down once the call returns. Drop the
annotation and the compiler says exactly that:

```
Error: This value is "local"
       because it is allocated at ... containing data
       which is "local" to the parent region
       because it closes over the value "s" ...
       However, the highlighted expression is expected to be "local" to the parent
       region or "global" because it is the function in a tail call.
```

`[@nontail]` makes the error go away. It does not make the allocation go away.
It says "fine, do not tail-call it", the region stays, the closures keep getting
built on every call, and the compiler stops objecting. It reads like a tail-call
hint. It was really an acknowledgement that something was being allocated.

`[@@zero_alloc]` does not catch this either. It polices *heap* allocation, and
these closures live in the local region on the stack. So the function carried a
compiler-checked "allocates nothing" annotation while allocating a closure per
helper on every single call, and `Gc.minor_words` reports 0.0 words/call for both
the broken and the fixed version. Neither of the two guardrails that look like
they cover this actually do. The only thing that pointed at it was the
`[@nontail]` that had to be there for the code to compile at all.

The fix is to make the helpers closed by passing the state explicitly:

```ocaml
let rec block i s cs ones mask c n =
  ...
  else block Int64_u.(i + #32L) s cs ones mask c n
in
block #0L s cs ones mask c n
```

Nothing is captured, so no closures are built, so no region is needed, so the
call is an ordinary tail call and `[@nontail]` is gone. The signature noise is
the price of the mode system being honest with you.

### What it cost

Two effects, both visible in the benchmarks:

- **A fixed ~10 to 13 cycles per call**, from building the closures and opening
  and closing the region. At `len = 4` the idiomatic version went from 27.9c to
  14.9c, and at `len = 32` from 34.3c to 22.7c. At small inputs that overhead was
  most of the function.
- **~10% of steady-state throughput**, from 0.164 to 0.147 cycles/byte, which
  closed the gap to the imperative version entirely. A once-per-call allocation
  cannot explain a per-byte cost; the likely cause is that every recursive call
  carried an environment pointer and reloaded `cs`, `ones`, `mask` and `n`
  through it instead of keeping them in registers. That is inference from the
  numbers, not from reading the emitted code.

One confound worth stating: `-O3` was added to `src/dune` in the same change, so
the two are not perfectly separated. It does not appear to account for much,
since the imperative `memchr`, which got the same flag and no structural change,
moved by under 2% (within run-to-run drift).

## The two Rust versions

`benchmarks/rust/src/lib.rs` carries two implementations of the same three-tier
algorithm, for the same reason the OCaml side does.

**`rs_memchr`** is the imperative one: `while offset + 32 <= len`, raw pointer
arithmetic, `read_unaligned` on every load, `break` out of each tier. It is a
transliteration of the C, and every load sits in an `unsafe` block.

**`rs_safe_memchr`** is written the way you would actually reach for if you were
handed the problem and the optimization strategy, and it contains exactly **one**
`unsafe` block, at the FFI boundary, to turn the incoming pointer into a slice:

```rust
let slice = unsafe { std::slice::from_raw_parts(p, len) };
```

Everything after that line is safe Rust. The tier structure falls out of the
iterator API rather than being spelled out:

- Tier 1 is `slice.chunks_exact(32)` driven by `find_map`, with the 32-byte chunk
  destructured by fixed ranges (`chunk[0..8]`, `chunk[8..16]`, ...) and
  `u64::from_ne_bytes(...try_into().unwrap())` doing the loads.
- Tier 2 is `iter.remainder().as_chunks::<8>()`, which splits the leftover into
  `&[[u8; 8]]` plus a final short slice in one call.
- Tier 3 is `.iter().enumerate().find_map(|(i, b)| (*b == needle).then_some(...))`
  over that final slice.

The narrowing step reuses the same shape: on a tier-1 hit it runs the byte-level
`find_map` over just the matching 8-byte range. The `r0` case is still checked
first, for the same reason as everywhere else.

Two things are worth calling out about writing it this way. The `try_into().unwrap()`
on every load looks like a runtime check but is not one: the range is a compile-time
constant length, so the `TryInto<[u8; 8]>` impl and its panic path both fold away.
And the running `offset` has to be a `let mut` captured by the `find_map` closure,
which is the one place where the safe version is visibly fighting the iterator
abstraction rather than being helped by it.

### What it costs

| | small inputs (≤ 32 B) | steady state |
| --- | --- | --- |
| `rs_safe_memchr` vs `rs_memchr` | ~5 to 9% **faster** | ~1.20x slower |

The steady-state penalty is real and consistent: 0.131 cycles/byte against 0.108,
across three runs. At small sizes the safe version is not merely competitive, it
wins, beating the pointer version at every length up to 32 bytes and beating the C
SWAR reference too. That is the iterator formulation reaching the byte-level
resolution with less setup, not vectorization sneaking back in.

I have not read the generated assembly for either, so I am not going to assert a
cause for the steady-state gap. The captured-`offset` mutation inside the tier-1
closure is the obvious suspect, since it is a loop-carried dependency the pointer
version does not have, but that is a hypothesis and not a measurement.

The honest summary is that "mostly safe" cost about 20% of inner-loop throughput
here and nothing at all at small sizes, in exchange for deleting every `unsafe`
block except the one the FFI signature forces.

### Correctness

Both Rust versions were checked against a naive byte scan over 200,000 randomized
haystacks (random lengths to 300, random needle placement including no-needle
cases) plus an exhaustive sweep placing the needle at every position for every
length from 0 to 200, which covers all three tier boundaries and both narrowing
paths. Zero mismatches.

### What makes it OxCaml-flavoured

| Feature | Where | Why |
| --- | --- | --- |
| `int64#` (`Int64_u`) | the whole kernel | Unboxed 64-bit ints. In stock OCaml, `Int64` arithmetic allocates a boxed value per intermediate; here `lxor`, `-`, `land`, `lognot` are raw machine ops on registers. |
| `%caml_bytes_get64u#` | `Bytes.unsafe_get_int64_ne` | Unaligned 64-bit load out of `bytes` returning an *unboxed* `int64#` directly, no box on the way out. |
| `..._indexed_by_int64#` | every load in the loops | Indexes the load by an `int64#` rather than a tagged `int`. The loop counter never has to be tagged/untagged just to be used as an offset. |
| `let mutable i` / `let mutable hit` | `memchr`'s loop state | OxCaml mutable local bindings. Stock OCaml would need `ref` cells. `ml_memchr` deliberately avoids these, which is now the *only* difference between the two apart from the tail. |
| `s @ local read` | both signatures | The haystack is taken at `local` mode (it cannot escape) and `read` (it is not mutated). The `local` half is also what made the closure capture in `ml_memchr` expensive. |
| `@@ portable` | the `external` declarations | Marks the primitives as safe to use across capsules/domains. |
| `Int64.Unboxed.count_trailing_zeros` | the tail | `tzcnt` on an `int64#`, no boxing at the boundary. |
| `[@@zero_alloc]` | `src/memchr.mli`, both values | **Compiler-checked.** The build fails if either function ever allocates on the heap. That is a real guarantee, but note the qualifier: it says nothing about the local region. |

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

`ml_memchr` sidesteps this entirely: recursion gives you the early exit for free,
because `word i` simply *is* the continuation. Now that the closures are gone,
it does so at no measurable cost.

## Benchmarks

### What is being compared

| Name | What it is |
| --- | --- |
| `C SIMD memchr` | glibc's `memchr` (`benchmarks/memchr_stubs.c`). Fully vectorized; the ceiling, not a fair SWAR peer. |
| `C SWAR memchr` | The same 32/8/1 algorithm in C, compiled `-O2 -fno-tree-vectorize`. |
| `Rust imperative SWAR memchr` | `rs_memchr`. Raw pointers, `read_unaligned`, `unsafe` on every load. |
| `Rust safe SWAR memchr` | `rs_safe_memchr`. Iterators and slices, one `unsafe` block at the FFI boundary. |
| `ML style SWAR memchr` | `Memchr.ml_memchr`, the idiomatic recursive version. |
| `ML imperative SWAR memchr` | `Memchr.memchr`, mutable locals + `ctz` tail. |
| `Bytes.index_opt` | OCaml stdlib. Byte-at-a-time baseline. |

Auto-vectorization is explicitly **disabled** for the C and Rust SWAR builds
(`RUSTFLAGS` is set in the dune rule that shells out to cargo, so it applies to
both Rust functions). Without that, both compilers turn the tier-1 loop into AVX2
and the comparison stops being about SWAR at all. The goal is to compare *the same
algorithm* across three languages, with glibc alongside as a reference for what
real SIMD buys.

### Harness

`core_bench`, driven from `benchmarks/memchr_bench.ml`. The haystack is `len`
bytes of `'a'` with a single `'z'` planted at `len - len/10 - 1`, so every run
scans roughly 90% of the buffer before hitting, making it a mostly-miss scan, which
is the case SWAR is supposed to win. All calls go through `Sys.opaque_identity` so
nothing is folded away, and the C/Rust entry points are `[@@noalloc]`.

### Results

11th Gen Intel Core i7-11800H, OxCaml 5.2.0+ox (flambda2), gcc 16.1.1, rustc
1.97.1, glibc 2.44, `dune build --profile release`, `-quota 3`.

Cycles per run (lower is better):

| len | C SIMD | C SWAR | Rust imp. | Rust safe | ML style | ML imperative | `Bytes.index_opt` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 10.8 | 14.5 | 14.7 | 14.6 | 14.9 | **13.0** | 11.8 |
| 8 | 11.0 | 18.0 | 16.2 | 16.4 | 26.9 | **14.8** | 16.1 |
| 16 | 10.8 | 18.2 | 17.8 | 16.9 | 28.5 | **16.9** | 23.9 |
| 32 | 10.9 | 18.9 | 18.7 | 17.1 | 22.7 | **19.7** | 57.5 |
| 64 | 12.4 | 20.1 | 20.1 | 20.7 | 22.9 | **25.4** | 89.6 |
| 128 | 14.3 | 29.1 | 29.1 | 27.5 | 35.8 | **33.5** | 151.6 |
| 256 | 17.4 | 44.0 | 42.3 | 41.9 | 59.6 | **55.1** | 268.7 |
| 1024 | 28.6 | 132.9 | 134.5 | 145.2 | 172.5 | **169.3** | 993.8 |
| 4096 | 62.8 | 454.5 | 436.5 | 506.5 | 583.9 | **587.6** | 3 857.7 |
| 65536 | 1 162.7 | 6 743.5 | 6 423.8 | 7 849.2 | 8 691.0 | **8 693.5** | 61 856.6 |

Nanoseconds per run:

| len | C SIMD | C SWAR | Rust imp. | Rust safe | ML style | ML imperative | `Bytes.index_opt` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4.71 | 6.31 | 6.37 | 6.33 | 6.47 | **5.62** | 5.14 |
| 8 | 4.76 | 7.81 | 7.03 | 7.12 | 11.67 | **6.44** | 6.97 |
| 16 | 4.68 | 7.90 | 7.73 | 7.32 | 12.37 | **7.33** | 10.35 |
| 32 | 4.72 | 8.19 | 8.13 | 7.43 | 9.83 | **8.56** | 24.95 |
| 64 | 5.38 | 8.73 | 8.72 | 9.00 | 9.96 | **11.01** | 38.87 |
| 128 | 6.19 | 12.65 | 12.63 | 11.93 | 15.55 | **14.54** | 65.82 |
| 256 | 7.55 | 19.08 | 18.38 | 18.17 | 25.87 | **23.93** | 116.61 |
| 1024 | 12.41 | 57.70 | 58.36 | 63.02 | 74.88 | **73.47** | 431.33 |
| 4096 | 27.23 | 197.25 | 189.46 | 219.83 | 253.45 | **255.03** | 1 674.34 |
| 65536 | 504.64 | 2 926.88 | 2 788.09 | 3 406.79 | 3 772.14 | **3 773.20** | 26 847.35 |

A second run of the same binary agreed within ~1% on every row, so the ratios
below are averaged across those two. A third run came in 5 to 8% hotter across the
board (`C SIMD` at `len = 64` nearly doubled, which is drift and not a real
effect), so it is excluded from the numbers; it preserved the ordering of every
row and put the safe Rust steady state at 0.132 cycles/byte against the 0.131
reported below. There is no frequency pinning here, so treat absolute numbers as
machine-specific and the ratios as the durable part.

### Steady-state throughput

Taking the slope between `len = 4096` and `len = 65536` removes fixed
call/setup/tail overhead and leaves the cost of the inner loop alone (averaged
over both runs):

| Implementation | cycles/byte | bytes/cycle |
| --- | ---: | ---: |
| C SIMD memchr (AVX2) | 0.020 | ~50.4 |
| Rust imperative SWAR | 0.108 | ~9.3 |
| C SWAR | 0.114 | ~8.8 |
| Rust safe SWAR | 0.131 | ~7.7 |
| **ML imperative SWAR** | **0.146** | **~6.8** |
| ML style SWAR | 0.147 | ~6.8 |
| `Bytes.index_opt` | 1.044 | ~0.96 |

The three languages' SWAR loops land in the same band, well above the ~1
byte/cycle of a naive scan and well below vectorized `memchr`. `Bytes.index_opt`
at almost exactly 1.0 cycles/byte is a clean sanity check that the baseline
really is byte-at-a-time.

Note where safe Rust sits: between the pointer-chasing implementations and the
OCaml ones. It is the closest neighbour either OCaml version has, and the gap to
it is less than half the gap to C.

### Reading the results

**The unboxing works.** ~6.8 bytes/cycle is not achievable if `int64#` values are
being boxed: a single heap allocation per iteration would show up immediately as a
collapse toward the stdlib line. The `[@@zero_alloc]` check holds this in place at
build time.

**Idiomatic now costs nothing in steady state.** `ml_memchr` and `memchr` are
0.147 and 0.146 cycles/byte, a gap well inside run-to-run noise. Before the
closure fix the idiomatic version was ~10% behind. The recursive, immutable,
tail-call version of this algorithm is not slower than the mutable-loop version;
it was only slower because it was quietly allocating.

**Small-input wins are algorithmic, not linguistic.** At `len ≤ 16` the imperative
OCaml beats C and both Rust versions (13.0c vs 14.5c/14.7c/14.6c at `len = 4`).
This is *not* OxCaml outrunning C. It is the `ctz` tail beating a byte-at-a-time
tail: at these sizes the tail is the entire function. Give the C reference the same
tail and it would win again. The honest language comparison is the steady-state
number.

**The idiomatic tax is not an OCaml-specific phenomenon.** Writing the algorithm
at a higher level of abstraction cost Rust ~20% of inner-loop throughput
(0.108 to 0.131 cycles/byte) while costing OCaml nothing measurable. That is the
opposite of the result I expected, and it is worth stating plainly: after the
`[@nontail]` fix, the idiomatic-versus-imperative gap in this codebase is larger
in Rust than in OCaml. One benchmark on one algorithm is not a general claim about
either language, but it does undercut the assumption that the recursive OCaml
formulation is the expensive one here.

**Where the two OCaml versions still differ is the tail, not the loop.**
`ml_memchr` is 26.9c at `len = 8` against `memchr`'s 14.8c, because it walks the
final partial word one byte at a time while `memchr` resolves it with one `tzcnt`.
That difference is deliberate: `ml_memchr` is there to show what the idiomatic
formulation costs, and a `ctz` tail is not the idiomatic formulation.

**The steady-state gap to C is ~1.28x, and it is loop shape.** The kernel is the
same handful of ALU ops everywhere; what differs is the `break` emulation. C and
imperative Rust leave the loop; OCaml computes a new cursor value on every
iteration whether or not it found anything. Against safe Rust, which reaches its
tier exits through `find_map` rather than a `break`, the gap narrows to ~1.12x.

**Nothing beats glibc, and nothing was going to.** At 64 KiB glibc is 7.5x faster
than the imperative OCaml and 5.8x faster than the C SWAR it is measured against.
That gap is AVX2 versus 64-bit registers, and it is the reason the interesting
number is the ~1.28x against C SWAR rather than the 7.5x against glibc.

**Allocation column.** `core_bench` reports a uniform `3.00w` per run for *every*
row, including the `[@@noalloc]` C stubs. This is harness overhead in the staged
closure, not the implementations. It is also, as the `[@nontail]` section shows,
not a column that can prove much: it never moved while `ml_memchr` was allocating
in the local region.

## Building and running

Requires an OxCaml switch (developed on `5.2.0+ox`) and `cargo` for the Rust
comparison.

```sh
dune build                      # builds the library and benchmark executable
dune build --profile release    # builds an optimized executable for benchmarks
dune test                       # inline tests
./_build/default/benchmarks/memchr_bench.exe time cycles percentage -quota 3 -ascii
```

`dune build` covers everything: the root `dune` defines a `default` alias
depending on `benchmarks/memchr_bench.exe` and `(alias_rec src/all)`, and the
`ml-memchr` package is marked `(allow_empty)` since it has no install stanza.
The library itself is built with `-O3` via `ocamlopt_flags` in `src/dune`.
If you're doing profiling, ensure that you run `dune build --profile release`
for accurate numbers, otherwise dune may simply produce the debug build.

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
benchmarks/rs_stubs.c        FFI shim for both Rust entry points
benchmarks/rust/src/lib.rs   Rust SWAR references, imperative + safe
```

## Caveats

- Single machine, single microarchitecture, no frequency pinning. Ratios are
  stable across runs; absolute numbers are not a portable claim.
- One haystack shape (needle at ~90%, uniform filler). Hit-early, no-hit, and
  unaligned-start distributions are not measured.
- The `ctz` tail in `memchr` is little-endian-only and has been built and tested
  only on x86-64.
- Both Rust functions are hand-written SWAR, deliberately hobbled to match the
  algorithm. Neither is the `memchr` crate, which would be in glibc's league.
- "Safe" in `rs_safe_memchr` means one `unsafe` block instead of one per load. The
  `from_raw_parts` call is still a real obligation: it trusts the caller's `len`,
  exactly as the OCaml `memchr` trusts its `n`.
- `memchr` takes a length `n` independently of `Bytes.length s` and does no bounds
  checking; passing `n > Bytes.length s` is undefined behaviour.

## License

MIT; see [LICENSE](LICENSE).
