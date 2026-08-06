# ml-memchr

An attempt at writing `memchr` in [**OxCaml**](https://oxcaml.org), using unboxed
64-bit integers, mode annotations, mutable locals and now explicit 256-bit vector
types, and then measuring honestly how close it lands to the equivalent C and
Rust.

The library ships **three** implementations, so that two different questions can
be asked separately: what does writing it *idiomatically* cost, and what does
staying in OCaml cost once you are allowed real SIMD?

- **`ml_memchr`**: idiomatic OCaml SWAR. Recursive `scan_block`/`scan_word`/`scan_bytes`,
  no mutable state, tail calls all the way down.
- **`memchr`**: the imperative SWAR version. Mutable locals, exception-driven loop
  exits, and a branchless `ctz`-based tail.
- **`simd_memchr`**: actual AVX2. 32-byte vector compares, four at a time, with an
  alignment prologue and a SWAR tail.

The Rust side mirrors the first split, with a raw-pointer version and a
mostly-safe iterator version, so the "idiomatic tax" can be compared across both
languages rather than just asserted about one. It also carries the
[`memchr` crate](https://crates.io/crates/memchr) as a second production SIMD
reference alongside glibc. See [The two Rust versions](#the-two-rust-versions)
and [The Rust `memchr` crate](#the-rust-memchr-crate).

All three OCaml versions are `[@@zero_alloc]`-checked. The headline results, at 64 KiB:

| | vs. C SWAR | vs. safe Rust | vs. `Bytes.index_opt` |
| --- | --- | --- | --- |
| `memchr` (imperative SWAR) | ~1.23x slower | ~1.13x slower | ~8.4x faster |
| `ml_memchr` (idiomatic SWAR) | ~1.23x slower | ~1.13x slower | ~8.4x faster |

That middle column is arguably the fairest one in the whole document. Measured
against C, OCaml gives up ~23%; measured against Rust code written at a
comparable level of abstraction, it gives up ~13%.

And then the SIMD version, which is measured against two production
implementations rather than against the scalar-only references. These are
steady-state ratios, so they describe the inner loop rather than the whole call:

| | vs. glibc | vs. Rust `memchr` crate | vs. `memchr` (SWAR) | vs. `Bytes.index_opt` |
| --- | --- | --- | --- | --- |
| `simd_memchr` | ~1.06x slower | ~1.05x **faster** | ~7.1x faster | ~60x faster |

Hand-written OxCaml lands **between the two production SIMD implementations**:
~6% off glibc's inner loop and ~5% ahead of the `memchr` crate's, in steady state.
See [The SIMD version](#the-simd-version) for why the crate comparison is the
weaker of the two claims.

The idiomatic OCaml version used to trail the imperative one by ~10% in steady
state. It no longer does, and the reason it did turned out to be more interesting
than the gap itself. See [The `[@nontail]` trap](#the-nontail-trap).

## The SWAR algorithm

`memchr` and `ml_memchr` are both a three-tier scan (`simd_memchr` is described
[below](#the-simd-version)):

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
i <-
  (if Int64_u.(r0 land mask <> #0L) then i
   else if Int64_u.(r1 land mask <> #0L) then Int64_u.(i + #8L)
   else if Int64_u.(r2 land mask <> #0L) then Int64_u.(i + #16L)
   else Int64_u.(i + #24L));
raise_notrace Exit
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
if Int64_u.(t <> #0L) then
  hit <- Int64_u.(i + (count_trailing_zeros t lsr 3))
```

`count_trailing_zeros` is declared directly as a `[@@builtin]` external on
`int64#` rather than pulled from `ocaml_intrinsics_kernel`:

```ocaml
external count_trailing_zeros : int64# -> int64#
  = "caml_int64_ctz" "caml_int64_ctz_unboxed_to_untagged"
[@@noalloc] [@@builtin] [@@no_effects] [@@no_coeffects]
```

`[@@builtin]` is what lets the compiler recognise it and emit `tzcnt` inline
instead of a call, and `[@@no_effects] [@@no_coeffects]` let it be moved and
eliminated like any other pure operation. Since each matching byte sets exactly
its own bit 7, `ctz(t) lsr 3` *is* the byte offset. Up to seven byte compares and
their branches collapse into one instruction.

Two things make this safe rather than reckless:

- **The over-read is deliberate and in-bounds.** The load reads a full 8 bytes
  even when fewer remain. `i` is always a multiple of 8 (every loop advances by 8
  or 32 from zero, and the tier-1 exit sets it to a word offset), and OCaml's `bytes` data
  area is a whole number of words, so an 8-aligned 8-byte read at any `i < n`
  stays inside the allocation. The `valid` mask discards the bytes past `n`.
- **It is little-endian-specific.** `valid` keeps the *low* `r` bytes and `ctz`
  counts from the low end, both of which assume byte 0 is the least significant.
  This has only been built and tested on x86-64.

## The SIMD version

`simd_memchr` drops SWAR and uses `ocaml_simd`'s vector types directly. It is the
one implementation here written to be as fast as the machine allows rather than
constrained to a fixed algorithm, so it is measured against glibc rather than
against the C SWAR reference.

The needle is broadcast into a 32-byte vector once, and the whole search is
vector compare plus `movemask` plus `ctz`:

```ocaml
let v = Int8x32.Bytes.Int64_u.unsafe_get ~byte:i s in
let bitmask = Int8x32.(movemask (cs = v)) in
if Int64_u.(bitmask <> #0L) then
  Int64_u.(to_int (count_trailing_zeros bitmask + i))
```

`Int8x32.(cs = v)` is a lanewise byte compare producing a mask vector, `movemask`
collapses one bit per lane into an integer, and `ctz` of that integer *is* the
lane index. The same `count_trailing_zeros` trick as the SWAR tail, one tier up.

### Five tiers, not three

| Tier | Width | What it does |
| --- | ---: | --- |
| `scan_matrix` | 128 B/iter | Four 32-byte loads, four compares, OR the masks together and test once |
| `scan_vec` | 32 B/iter | One 32-byte compare |
| `scan_remainder` | 16 B, once | A single SSE 16-byte compare, not a loop |
| `scan_word` | 8 B/iter | The SWAR kernel |
| `scan_bytes` | 1 B/iter | Byte at a time |

The bottom two tiers are literally the same functions `ml_memchr` uses. Lifting
`scan_word` and `scan_bytes` out of `ml_memchr` to the top level is what lets both
implementations share them.

`scan_matrix` uses the same narrowing idea as the SWAR tier 1, and for the same
reason: OR the four comparison vectors, take one `movemask` of the result, and
only on a hit re-derive the individual masks in order.

```ocaml
let combined = Int8x32.(e0 lor e1 lor (e2 lor e3)) in
if Int64_u.(Int8x32.movemask combined <> #0L) then ...
```

`e0` is still tested first, because a needle can appear in more than one of the
four vectors and the earliest index has to win.

### The alignment prologue

This is the part with no SWAR equivalent. Unaligned 32-byte loads are cheap but
not free, and the matrix tier does four of them per iteration, so the entry point
aligns the cursor first:

```ocaml
let misalign = Nativeint_u.(Nativeint_u.unsafe_of_value s land #31n) in
if Nativeint_u.(misalign = #0n) then scan_matrix #0L s c cs n64
else
  let vec = Int8x32.Bytes.unsafe_get ~byte:0 s in
  let bitmask = Int8x32.(movemask (cs = vec)) in
  if Int64_u.(bitmask <> #0L) then Int64_u.(to_int (count_trailing_zeros bitmask))
  else scan_matrix Int64_u.(#32L - of_nativeint_u misalign) s c cs n64
```

`unsafe_of_value` takes the address of the `bytes` payload as a `nativeint#`, and
the low five bits are the misalignment. If it is already 32-aligned, go straight
to the matrix tier. Otherwise do one unaligned 32-byte probe covering bytes 0 to
31, and if it misses, resume at `32 - misalign`, which is 32-aligned by
construction.

The overlap is intentional and safe: bytes `32 - misalign` through 31 get scanned
twice, but the prologue probe already proved there is no match anywhere in 0 to
31, so the second pass cannot produce an earlier index than the first pass missed.

Because OCaml heap blocks are 8-byte aligned, `misalign` is always one of 0, 8, 16
or 24. All four occur in practice, so both prologue branches are genuinely
exercised; see [Tests](#tests).

### What it buys

| len | `simd_memchr` | glibc | Rust `memchr` crate | `memchr` (SWAR) |
| ---: | ---: | ---: | ---: | ---: |
| 16 | **10.8c** | 10.5c | 14.8c | 16.2c |
| 256 | **17.1c** | 16.6c | 21.9c | 50.8c |
| 4096 | **96.5c** | 62.6c | 80.8c | 568.0c |
| 65536 | **1 229.4c** | 1 134.1c | 1 269.2c | 8 602.4c |

Steady state, taking the slope between 4096 and 65536:

| | cycles/byte | bytes/cycle | vs. glibc |
| --- | ---: | ---: | ---: |
| glibc `__memchr_evex` | 0.0194 | ~51.6 | 1.00x |
| **`simd_memchr`** | **0.0205** | **~48.8** | **1.06x** |
| Rust `memchr` crate 2.8.3 | 0.0215 | ~46.5 | 1.11x |

So the OCaml sits between the two production implementations. Both comparisons
deserve separate treatment, because they are not equally strong.

**Against glibc, ~6% behind is a real gap.** The two separate cleanly, by far more
than the within-run confidence interval on either. Unpinned, the measurement is
not good enough to see it: run the same benchmark without `taskset` and the two
rows land on top of each other at 0.0194, which is an artifact of the noise floor
rather than a tie. Pin the core and the gap is stable.

**Against the Rust crate, ~5% ahead is a weak claim,** and it should not be read
as "OxCaml beats Rust". The `memchr` crate is solving a much harder problem than
`simd_memchr` is: runtime CPU feature detection and dispatch, a portable fallback,
correct behaviour on every input length, and no assumption whatsoever about
alignment or architecture. `simd_memchr` assumes AVX2 exists, assumes x86-64,
assumes little-endian, and is tuned against exactly one haystack shape, which
happens to be the one being measured. A 5% win under those conditions says the
inner loop is competitive. It does not say the function is better.

Worth being precise about the rest, too. The steady-state number describes the
*inner loop*, which is the part written in OCaml. It does not describe the whole
function, which is meaningfully worse than glibc's: at 4096 bytes `simd_memchr`
is 1.54x behind, and extrapolating the two-point fit back to zero length puts
~26 cycles of fixed overhead on it against roughly none for glibc. The Rust crate
beats it at 4096 too (80.8c against 96.5c), for the same reason. glibc and the
crate both spend real effort on short inputs; `simd_memchr` falls off a cliff into
a byte-at-a-time tail instead, and at `len = 8` it is one of the slowest rows in
the whole table.

Note also that glibc is not running AVX2 here. On this machine `memchr` resolves
to `__memchr_evex`, which uses EVEX encoding, the upper vector registers and `k`
mask registers. Both are 256 bits wide per compare, so the throughput comparison
is fair, but the vendor implementation is reaching for a newer instruction set to
get its ~6%.

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
let rec scan_block i s cs ones mask c n =
  ...
  else scan_block Int64_u.(i + #32L) s cs ones mask c n
in
scan_block #0L s cs ones mask c n
```

Nothing is captured, so no closures are built, so no region is needed, so the
call is an ordinary tail call and `[@nontail]` is gone. The signature noise is
the price of the mode system being honest with you.

`scan_word` and `scan_bytes` were subsequently lifted all the way out to the top
level, which is what allows `simd_memchr` to reuse them as its bottom two tiers.
Passing state explicitly stopped being a tax the moment a second caller existed.

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

## What makes it OxCaml-flavoured

| Feature | Where | Why |
| --- | --- | --- |
| `int64#` (`Int64_u`) | the whole kernel | Unboxed 64-bit ints. In stock OCaml, `Int64` arithmetic allocates a boxed value per intermediate; here `lxor`, `-`, `land`, `lognot` are raw machine ops on registers. |
| `%caml_bytes_get64u#` | `Bytes.unsafe_get_int64_ne` | Unaligned 64-bit load out of `bytes` returning an *unboxed* `int64#` directly, no box on the way out. |
| `..._indexed_by_int64#` | every load in the loops | Indexes the load by an `int64#` rather than a tagged `int`. The loop counter never has to be tagged/untagged just to be used as an offset. |
| `let mutable i` / `let mutable hit` | `memchr`'s loop state | OxCaml mutable local bindings. Stock OCaml would need `ref` cells. `ml_memchr` deliberately avoids these, which is now the *only* difference between the two apart from the tail. |
| `s @ local read` | all three signatures | The haystack is taken at `local` mode (it cannot escape) and `read` (it is not mutated). The `local` half is also what made the closure capture in `ml_memchr` expensive. |
| `@@ portable` | the `external` declarations | Marks the primitives as safe to use across capsules/domains. |
| `[@@builtin]` `count_trailing_zeros` | both tails | `tzcnt` on an `int64#`, declared as a local external rather than imported, so it inlines instead of becoming a call. |
| `Int8x32` / `Int8x16` | `simd_memchr` | 256-bit and 128-bit vector types from `ocaml_simd`. `set1` broadcasts, `=` is a lanewise compare, `movemask` collapses the result to an integer. Real SIMD as a first-class type, not an intrinsic escape hatch. |
| `Int8x32.Bytes.Int64_u.unsafe_get ~byte:i` | `simd_memchr`'s tiers | Vector load straight out of `bytes`, indexed by an `int64#`, with no intermediate copy. |
| `nativeint#` + `unsafe_of_value` | the alignment prologue | Takes the raw address of a `bytes` payload as an unboxed `nativeint#` so the low bits can be tested. This is the one genuinely low-level thing in the file. |
| `[@@zero_alloc]` | `src/memchr.mli`, all three values | **Compiler-checked.** The build fails if any of them ever allocates on the heap. That is a real guarantee, but note the qualifier: it says nothing about the local region. |

## The part that fights back

OCaml has no `break`. `memchr` originally emulated one by slamming the cursor to
`n` to force the loop condition false, which meant every iteration carried a
branch that either advanced the cursor *or* jumped it to the end, where C and
Rust just `break` and leave `i` alone.

That is now done with exceptions instead. Each tier is wrapped in a `try`, and the
exit is a `raise_notrace`:

```ocaml
(try
   while Int64_u.(i + #32L <= n64) do
     ...
     if Int64_u.(rs land mask <> #0L) then (
       i <- (* ...narrowing chain... *);
       raise_notrace Exit)
     else i <- Int64_u.(i + #32L)
   done
 with Exit -> ());
```

`raise_notrace` skips backtrace construction, and the handler is in the same
function, so this is a local control transfer rather than a real unwind. The
payoff is that the loop body no longer has to keep the cursor and the exit
condition consistent with each other: `i` is only ever advanced or set to the
answer, never overloaded to mean "stop".

It also removed a bug. The cursor-slamming version had to thread a separate `hit`
variable through all three tiers, and getting the tier-1 exit to write the word
offset into the right variable is exactly the kind of thing that is easy to get
wrong. The exception version has one cursor with one meaning.

Two things this is not. It is not free: `Exit` is still an exception, and the
`try` blocks are real. And it is not the shape a C programmer would write, which
is rather the point of keeping `ml_memchr` around as a comparison.

`ml_memchr` sidesteps the whole question: recursion gives you the early exit for
free, because `scan_word i` simply *is* the continuation. Now that the closures
are gone, it does so at no measurable cost, and it is the only one of the three
that needs neither a sentinel nor an exception.

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

All three Rust entry points were checked against a naive byte scan over
**536,703 cases** with zero mismatches: every needle position for every length
from 0 to 400, dense random haystacks over a two-character alphabet up to 3000
bytes, no-needle cases, and high-bit needles (`\xff` against `\xfe` filler,
`\x80` against NUL filler) to rule out signed-versus-unsigned confusion.

## The Rust `memchr` crate

`rs_simd_memchr` is not another hand-written implementation. It is the
[`memchr` crate](https://crates.io/crates/memchr), version 2.8.3, behind a
three-line shim:

```rust
#[unsafe(no_mangle)]
pub extern "C" fn rs_simd_memchr(p: *const u8, needle: u8, len: usize) -> isize {
    let slice = unsafe { std::slice::from_raw_parts(p, len) };
    memchr::memchr(needle, slice).map(|found| found as isize).unwrap_or(-1)
}
```

It is here as a **second production reference** next to glibc. Two independently
tuned, widely deployed SIMD `memchr` implementations bracket the OxCaml one much
more usefully than either does alone, and they disagree with each other by ~11%,
which is itself a useful calibration for how much any single number here is worth.

The crate is vendored into `benchmarks/rust/vendor` with a `.cargo/config.toml`
source replacement, so the dune rule's `cargo build --offline` still works without
network access.

### Do the no-vectorize flags affect the crate?

Worth checking, because the dune rule sets
`RUSTFLAGS="-C no-vectorize-loops -C no-vectorize-slp"` for the whole cargo
invocation, and that now includes a dependency whose entire purpose is
vectorization. If those flags applied, the comparison would be worthless.

They do not. The crate reaches SIMD through explicit `core::arch` intrinsics
guarded by runtime feature detection, not through the loop vectorizer, and LLVM's
vectorizer flags have no effect on intrinsics. Building the same crate twice and
timing a 64 KiB scan confirms it:

| Build | ns/call | bytes/ns |
| --- | ---: | ---: |
| with `-C no-vectorize-loops -C no-vectorize-slp` | 547.0 | ~107.8 |
| without | 568.4 | ~103.8 |

The flagged build is marginally *faster*, which is run-to-run noise. There is no
penalty. The sustained ~48 bytes/cycle it reaches in the benchmark proper is also
far outside anything a scalar loop could do, which is a second confirmation.

Auto-vectorization is suppressed for the SWAR rows and only for them. That is
deliberate: those exist to compare one algorithm across three languages, and
letting the compiler silently substitute a different algorithm would defeat the
point.

## Benchmarks

### What is being compared

| Name | What it is |
| --- | --- |
| `C SIMD memchr` | glibc's `memchr` (`benchmarks/memchr_stubs.c`). Resolves to `__memchr_evex` on this machine. Production SIMD reference. |
| `Rust SIMD memchr` | The `memchr` crate 2.8.3 behind an FFI shim. Second production SIMD reference. Not written for this project. |
| `ML SIMD memchr` | `Memchr.simd_memchr`. AVX2 via `ocaml_simd`, 128/32/16/8/1 tiers. No compiler restrictions. |
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
algorithm* across three languages.

That leaves two comparisons in one table, and they should not be mixed up. The
five SWAR rows are a like-for-like language comparison. `ML SIMD` and `C SIMD` are
a separate like-for-like comparison at a different point on the abstraction
ladder, where neither side is holding anything back. Reading `ML SIMD` against
`C SWAR` is not meaningful in either direction.

### Harness

`core_bench`, driven from `benchmarks/memchr_bench.ml`. The haystack is `len`
bytes of `'a'` with a single `'z'` planted at `len - len/10 - 1`, so every run
scans roughly 90% of the buffer before hitting, making it a mostly-miss scan, which
is the case SWAR is supposed to win. All calls go through `Sys.opaque_identity` so
nothing is folded away, and the C/Rust entry points are `[@@noalloc]`.

### Methodology

11th Gen Intel Core i7-11800H, OxCaml 5.2.0+ox (flambda2), gcc 16.1.1, rustc
1.97.1, glibc 2.44, `memchr` crate 2.8.3, `dune build --profile release`.

```sh
taskset -c 9 ./_build/default/benchmarks/memchr_bench.exe \
  +time +cycles percentage -quota 3 -ascii -ci-absolute
```

This is a **single run pinned to one core**. Pinning is doing the work that
repetition would otherwise have to: unpinned, this benchmark drifts by up to ~25%
between runs at small lengths, and averaging several noisy runs is a much weaker
tool than removing the noise source. Most of that variance is the process being
migrated between cores, and `taskset` deletes it outright.

The `+cycles` and `+time` prefixes ask `core_bench` for 95% confidence intervals
on each estimate, so uncertainty is reported per row from within the run instead
of being inferred from repeated runs. Those intervals are tight: at `len = 65536`
they are ±0.3% for glibc, ±0.9% for `ML SIMD`, and ±0.2% for `ML imperative`. Where
`core_bench` prints no interval at all, it is below display resolution.

One caveat this does not fix. A within-run confidence interval measures sampling
noise inside one process. It does not capture whatever differs *between*
processes: heap layout, haystack alignment, and frequency state. Two rows sitting
within a percent of each other should therefore be read as "too close to call"
rather than as a ranking, so treat the intervals as a lower bound on real
uncertainty and the cross-row ratios within this table as the durable part.

Cycles per run (lower is better):

| len | glibc | Rust crate | **ML SIMD** | C SWAR | Rust imp. | Rust safe | ML style | ML imp. | `index_opt` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 10.1 | 15.9 | **14.6** | 14.9 | 15.7 | 13.9 | 13.9 | 13.2 | 12.4 |
| 8 | 10.2 | 19.8 | **19.0** | 17.8 | 17.5 | 14.6 | 17.5 | 14.3 | 18.8 |
| 16 | 10.5 | 14.8 | **10.8** | 18.9 | 18.5 | 15.8 | 19.1 | 16.2 | 29.9 |
| 32 | 10.8 | 16.2 | **11.2** | 19.0 | 18.8 | 15.9 | 20.0 | 19.1 | 63.8 |
| 64 | 12.0 | 19.5 | **17.8** | 20.5 | 20.4 | 18.8 | 21.8 | 25.2 | 103.3 |
| 128 | 14.1 | 22.0 | **19.6** | 30.1 | 28.0 | 26.1 | 32.9 | 33.7 | 177.7 |
| 256 | 16.6 | 21.9 | **17.1** | 45.1 | 41.7 | 41.5 | 53.1 | 50.8 | 329.9 |
| 1024 | 27.7 | 33.7 | **30.9** | 136.3 | 131.8 | 143.4 | 168.5 | 167.1 | 1 222.2 |
| 4096 | 62.6 | 80.8 | **96.5** | 463.6 | 433.2 | 502.4 | 575.4 | 568.0 | 4 856.1 |
| 65536 | 1 134.1 | 1 269.2 | **1 229.4** | 7 011.5 | 6 370.1 | 7 631.0 | 8 620.4 | 8 602.4 | 72 332.7 |

Nanoseconds per run:

| len | glibc | Rust crate | **ML SIMD** | C SWAR | Rust imp. | Rust safe | ML style | ML imp. | `index_opt` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4.37 | 6.89 | **6.36** | 6.47 | 6.82 | 6.05 | 6.03 | 5.73 | 5.38 |
| 8 | 4.42 | 8.57 | **8.23** | 7.73 | 7.60 | 6.34 | 7.58 | 6.22 | 8.14 |
| 16 | 4.57 | 6.42 | **4.67** | 8.19 | 8.02 | 6.86 | 8.27 | 7.02 | 13.00 |
| 32 | 4.71 | 7.01 | **4.87** | 8.25 | 8.14 | 6.89 | 8.68 | 8.28 | 27.70 |
| 64 | 5.22 | 8.45 | **7.74** | 8.88 | 8.87 | 8.17 | 9.45 | 10.94 | 44.86 |
| 128 | 6.11 | 9.57 | **8.50** | 13.05 | 12.14 | 11.34 | 14.27 | 14.64 | 77.12 |
| 256 | 7.21 | 9.50 | **7.43** | 19.57 | 18.10 | 18.01 | 23.06 | 22.06 | 143.17 |
| 1024 | 12.01 | 14.64 | **13.41** | 59.15 | 57.22 | 62.26 | 73.15 | 72.53 | 530.48 |
| 4096 | 27.16 | 35.06 | **41.90** | 201.21 | 188.02 | 218.05 | 249.72 | 246.52 | 2 107.76 |
| 65536 | 492.25 | 550.90 | **533.62** | 3 043.24 | 2 764.82 | 3 312.09 | 3 741.51 | 3 733.68 | 31 394.95 |

There is no frequency pinning, only core pinning, so absolute numbers remain
machine-specific and the ratios are the portable part.

### Steady-state throughput

Taking the slope between `len = 4096` and `len = 65536` removes fixed
call/setup/tail overhead and leaves the cost of the inner loop alone:

| Implementation | cycles/byte | bytes/cycle | vs. glibc |
| --- | ---: | ---: | ---: |
| glibc `__memchr_evex` | 0.0194 | ~51.6 | 1.00x |
| **`simd_memchr`** (OxCaml AVX2) | **0.0205** | **~48.8** | **1.06x** |
| Rust `memchr` crate 2.8.3 | 0.0215 | ~46.5 | 1.11x |
| Rust imperative SWAR | 0.1074 | ~9.3 | 5.54x |
| C SWAR | 0.1184 | ~8.4 | 6.11x |
| Rust safe SWAR | 0.1289 | ~7.8 | 6.65x |
| `memchr` (ML imperative SWAR) | 0.1453 | ~6.9 | 7.50x |
| `ml_memchr` (ML style SWAR) | 0.1455 | ~6.9 | 7.51x |
| `Bytes.index_opt` | 1.2203 | ~0.8 | 62.97x |

There are three clear bands, and it is worth reading them as bands rather than as
one ranking. The three SIMD implementations sit within 11% of each other. The five
SWAR implementations sit within 35% of each other, roughly 6x below. And the
byte-at-a-time stdlib baseline is another 8x below that, at 1.22 cycles per byte,
which is the right order for a scalar loop with a bounds check and a branch per
byte, and a clean sanity check that the baseline really is doing what it looks
like.

The gap that matters for the language question is *within* the SWAR band. The gap
that matters for the "can OxCaml do SIMD" question is within the SIMD band. Rows
compared across bands are measuring the algorithm, not the language.

Note where safe Rust sits: between the pointer-chasing implementations and the
OCaml ones. It is the closest neighbour either OCaml version has.

### Reading the results

**OxCaml gets within ~6% of vendor `memchr` throughput, and ahead of one of the
two references.** `simd_memchr` sustains 0.0205 cycles/byte against glibc's
0.0194 and the Rust crate's 0.0215. There is no asterisk on the OCaml side: it is
`[@@zero_alloc]`-checked, it goes through the same harness as everything else, and
glibc is reaching for a *newer* instruction set (EVEX) than the AVX2 the OCaml
uses. Whatever tax OCaml is paying in the SWAR rows, very little of it is being
paid in this loop.

**The crate result is the weaker half of that.** Beating `memchr` 2.8.3 by ~5% is
not "OxCaml beats Rust"; the crate carries runtime dispatch, a portable fallback,
and correctness on inputs `simd_memchr` never has to think about, while
`simd_memchr` is tuned against precisely the haystack shape being measured. Two
production implementations disagreeing with each other by 11% is the more useful
takeaway: it sets the scale on which the OCaml's ~6% should be read.

**Throughput is not the whole function.** `simd_memchr` is 1.08x behind glibc at
64 KiB, 1.12x at 1024, and 1.54x at 4096. Extrapolating the two-point fit back to
zero length puts ~26 cycles of fixed overhead on it and roughly none on glibc,
which mostly says a two-point fit is crude, though the sign and rough size are
consistent across sizes. At `len = 8` it is 19.0c, beaten by every row in the
table except the Rust crate, and beaten by `Bytes.index_opt`, because it gives up
and walks bytes. Both production implementations spend real effort on short inputs
and on page-cross safety. That is the half that is still missing here.

**The unboxing works.** ~6.9 bytes/cycle is not achievable if `int64#` values are
being boxed: a single heap allocation per iteration would show up immediately as a
collapse toward the stdlib line. The `[@@zero_alloc]` check holds this in place at
build time.

**Idiomatic costs nothing in steady state.** `ml_memchr` and `memchr` are 0.1455
and 0.1453 cycles/byte, a 0.13% gap, which is as close to identical as this
harness can resolve. Before the closure fix the idiomatic version was ~10% behind.
The recursive, immutable, tail-call version of this algorithm is not slower than
the mutable-loop version; it was only slower because it was quietly allocating.

**Small-input wins are algorithmic, not linguistic.** At `len = 4` and `len = 8`
the imperative OCaml is the fastest SWAR row in the table (13.2c and 14.3c,
against 14.9c and 17.8c for C). At 16 and 32 safe Rust takes it back. Where the
win is real it is the `ctz` tail beating a byte-at-a-time tail, not OxCaml
outrunning C: at these sizes the tail is the entire function, and giving the C
reference the same tail would take it back. The honest language comparison is the
steady-state number.

**The idiomatic tax is not an OCaml-specific phenomenon.** Writing the algorithm
at a higher level of abstraction cost Rust ~20% of inner-loop throughput
(0.1074 to 0.1289 cycles/byte) while costing OCaml nothing measurable. That is the
opposite of the result I expected, and it is worth stating plainly: after the
`[@nontail]` fix, the idiomatic-versus-imperative gap in this codebase is larger
in Rust than in OCaml. One benchmark on one algorithm is not a general claim about
either language, but it does undercut the assumption that the recursive OCaml
formulation is the expensive one here.

**Where the two OCaml SWAR versions still differ is the tail, not the loop.**
`ml_memchr` is 17.5c at `len = 8` against `memchr`'s 14.3c, because it walks the
final partial word one byte at a time while `memchr` resolves it with one `tzcnt`.
That difference is deliberate: `ml_memchr` is there to show what the idiomatic
formulation costs, and a `ctz` tail is not the idiomatic formulation.

**The steady-state SWAR gap to C is ~1.23x.** The kernel is the same handful of
ALU ops everywhere, so what is left is loop shape and register pressure. Against
safe Rust, which reaches its tier exits through `find_map` rather than a `break`,
the gap narrows to ~1.13x. I have not read the emitted code for any of these, so
I am not going to attribute the remaining difference to a specific cause.

**The SIMD band beats the SWAR band by 5x to 8x, as it should.** At 64 KiB glibc
is 7.6x faster than the imperative OCaml SWAR and 6.2x faster than the C SWAR it
is measured against. That gap is 256-bit vectors versus 64-bit registers, and it
is very nearly the gap `simd_memchr` closes by using the same weapon: 7.0x over
its own SWAR sibling, in the same language, with the same allocation guarantee.

**Allocation column.** `core_bench` reports a uniform `3.00w` per run for *every*
row, including the `[@@noalloc]` C stubs. This is harness overhead in the staged
closure, not the implementations. It is also, as the `[@nontail]` section shows,
not a column that can prove much: it never moved while `ml_memchr` was allocating
in the local region.

## Building and running

Requires an OxCaml switch (developed on `5.2.0+ox`) and `cargo` for the Rust
comparison. `simd_memchr` pulls in `ocaml_simd.avx` and `ocaml_simd.sse`, and
needs a CPU with AVX2.

```sh
dune build                      # builds the library and benchmark executable
dune build --profile release    # builds an optimized executable for benchmarks
dune test                       # inline tests

taskset -c 9 ./_build/default/benchmarks/memchr_bench.exe \
  +time +cycles percentage -quota 3 -ascii -ci-absolute
```

Pin the benchmark to a core. Without `taskset` the numbers move by double-digit
percentages between runs at small lengths, and no amount of averaging recovers
that cleanly. The `+` prefixes turn on 95% confidence intervals so a single run
reports its own uncertainty.

`dune build` covers everything: the root `dune` defines a `default` alias
depending on `benchmarks/memchr_bench.exe` and `(alias_rec src/all)`, and the
`ml-memchr` package is marked `(allow_empty)` since it has no install stanza.
The library itself is built with `-O3` via `ocamlopt_flags` in `src/dune`.
If you're doing profiling, ensure that you run `dune build --profile release`
for accurate numbers, otherwise dune may simply produce the debug build.

The Rust static library is built by a dune rule shelling out to `cargo build
--release --offline`, so `benchmarks/rust` must build without network access. Its
one dependency, the `memchr` crate, is vendored into `benchmarks/rust/vendor` with
a source replacement in `benchmarks/rust/.cargo/config.toml`, which is what keeps
`--offline` working. Refresh it with `cargo vendor` from `benchmarks/rust` if the
dependency ever changes.

### Tests

`dune test` runs six inline tests: a smoke test, a longer-string test, a
no-match test, a 200-iteration randomized parity check of `memchr` against a
naive byte scan, and 200-iteration checks that `ml_memchr` and `simd_memchr` each
agree with `memchr`. Random lengths go up to 200 bytes.

That is enough for the SWAR versions, whose tiers are 32/8/1, but not for
`simd_memchr`, whose first tier is 128 bytes wide and whose behaviour depends on
the *address* of the haystack as well as its length. It was additionally checked
against a naive scan over **253,945 cases** with zero mismatches:

- every needle position for every length from 0 to 320, which crosses the
  128-byte, 32-byte, 16-byte and 8-byte tier boundaries;
- two-needle cases across lengths 2 to 160, so the lane-narrowing order has to
  report the earlier match rather than any match;
- dense random haystacks over a two-character alphabet, where nearly every vector
  compare hits and the narrowing path runs constantly;
- single needles swept across lengths 127, 128, 129, 159, 160, 161, 255, 256,
  257, 511 and 4096, sitting directly on the tier boundaries;
- high-bit needles (`\xff` against `\xfe` filler, `\x80` against NUL filler) to
  rule out signed-versus-unsigned confusion in the byte comparison.

Because the alignment prologue branches on the haystack's address, each case
allocates a fresh buffer behind a randomly sized spacer. Instrumenting that
allocation pattern confirms all four reachable misalignments (0, 8, 16, 24) occur
in roughly equal proportion, so neither prologue branch is going untested.

The three Rust entry points got the same treatment on their own side: **536,703
cases** against a naive scan, zero mismatches, covering every needle position for
every length from 0 to 400, dense random haystacks up to 3000 bytes, no-needle
cases, and high-bit needles.

## Layout

```
src/memchr.ml                the three OxCaml implementations + inline tests
src/memchr.mli               all three values, all three [@@zero_alloc]
benchmarks/memchr_bench.ml   core_bench driver
benchmarks/memchr_stubs.c    glibc memchr binding
benchmarks/swar_stubs.c      C SWAR reference (-fno-tree-vectorize)
benchmarks/rs_stubs.c        FFI shim for all three Rust entry points
benchmarks/rust/src/lib.rs   Rust SWAR references + memchr crate shim
benchmarks/rust/vendor/      vendored memchr crate, for offline builds
```

## Caveats

- Single machine, single microarchitecture. The benchmark is pinned to one core
  but the frequency is not pinned, so absolute numbers are not a portable claim.
- **Single run.** Uncertainty is reported as within-run 95% confidence intervals,
  which do not capture between-process variation in heap layout, haystack
  alignment or frequency state. Differences of a percent or two between rows are
  below what this setup can resolve, even when the intervals suggest otherwise.
- One haystack shape (needle at ~90%, uniform filler). Hit-early, no-hit, and
  unaligned-start distributions are not measured. The last of these matters more
  for `simd_memchr` than for the others, since its prologue exists precisely to
  handle unaligned starts.
- `simd_memchr` requires AVX2 and does not check for it at runtime. There is no
  scalar fallback and no `cpuid` dispatch, which is the main thing separating it
  from a `memchr` you could ship, and the main reason its ~5% edge over the
  `memchr` crate should not be read as a like-for-like win.
- The `ctz` tail in `memchr` and the vector narrowing in `simd_memchr` are both
  little-endian-only, and everything here has been built and tested only on
  x86-64.
- The glibc comparison is against whatever ifunc this machine selects, which is
  `__memchr_evex`. A machine without AVX-512 would select a different and probably
  slower variant, which would flatter `simd_memchr`.
- The two Rust SWAR functions are hand-written and compiled with
  auto-vectorization off so they run the SWAR algorithm rather than a
  compiler-generated vector one. The `memchr` crate is unaffected by those flags,
  which is measured rather than assumed. See
  [Do the no-vectorize flags affect the crate?](#do-the-no-vectorize-flags-affect-the-crate).
- "Safe" in `rs_safe_memchr` means one `unsafe` block instead of one per load. The
  `from_raw_parts` call is still a real obligation: it trusts the caller's `len`,
  exactly as the OCaml `memchr` trusts its `n`.
- All three OCaml functions take a length `n` independently of `Bytes.length s`
  and do no bounds checking; passing `n > Bytes.length s` is undefined behaviour.

## License

MIT; see [LICENSE](LICENSE).
