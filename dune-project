(lang dune 3.22)

(name ml-memchr)

(generate_opam_files true)

(source
 (github KiChjang/ml-memchr))

(authors "Keith Yeung <kungfukeith11@gmail.com>")

(maintainers "Keith Yeung <kungfukeith11@gmail.com>")

(license MIT)

(package
 (name ml-memchr)
 (allow_empty)
 (synopsis "A SWAR memchr written in OxCaml, benchmarked against C and Rust")
 (description
  "Two implementations of memchr in OxCaml -- one idiomatic and recursive, one \
   optimized with mutable locals and a ctz-based tail -- both using unboxed \
   int64# arithmetic and checked zero-alloc. Includes a core_bench suite \
   comparing them against glibc's vectorized memchr, hand-written SWAR in C and \
   Rust, and Bytes.index_opt from the stdlib. Requires an OxCaml switch; this \
   will not build on stock OCaml.")
 (depends
  (ocaml
   (>= 5.2))
  base
  core
  core_bench
  core_unix
  ocaml_intrinsics_kernel
  ppx_assert
  ppx_inline_test)
 (tags
  (memchr swar simd string-search benchmark oxcaml)))

; See the complete stanza docs at https://dune.readthedocs.io/en/stable/reference/dune-project/index.html
