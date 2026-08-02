open Stdlib_upstream_compatible
open Core_bench

external c_memchr : bytes -> char -> int -> int = "ml_bytes_memchr" [@@noalloc]

external c_swar_memchr : bytes -> char -> int -> int = "ml_bytes_swar_memchr"
[@@noalloc]

external rs_memchr : bytes -> char -> int -> int = "ml_rs_memchr" [@@noalloc]

let haystack len =
  let bytes = Bytes.make len 'a' in
  let i = len - (len / 10) - 1 in
  Bytes.set bytes i 'z';
  bytes

let tests =
  [
    Bench.Test.create_indexed ~name:"C SIMD memchr"
      ~args:[ 4; 8; 16; 32; 64; 128; 256; 1024; 4096; 65536 ] (fun len ->
        let s = haystack len in
        Core.Staged.stage (fun () -> Sys.opaque_identity (c_memchr s 'z' len)));
    Bench.Test.create_indexed ~name:"C SWAR memchr"
      ~args:[ 4; 8; 16; 32; 64; 128; 256; 1024; 4096; 65536 ] (fun len ->
        let s = haystack len in
        Core.Staged.stage (fun () ->
            Sys.opaque_identity (c_swar_memchr s 'z' len)));
    Bench.Test.create_indexed ~name:"Rust SWAR memchr"
      ~args:[ 4; 8; 16; 32; 64; 128; 256; 1024; 4096; 65536 ] (fun len ->
        let s = haystack len in
        Core.Staged.stage (fun () -> Sys.opaque_identity (rs_memchr s 'z' len)));
    Bench.Test.create_indexed ~name:"ML style SWAR memchr"
      ~args:[ 4; 8; 16; 32; 64; 128; 256; 1024; 4096; 65536 ] (fun len ->
        let s = haystack len in
        Core.Staged.stage (fun () ->
            Sys.opaque_identity (Memchr.ml_memchr s 'z' len)));
    Bench.Test.create_indexed ~name:"ML imperative SWAR memchr"
      ~args:[ 4; 8; 16; 32; 64; 128; 256; 1024; 4096; 65536 ] (fun len ->
        let s = haystack len in
        Core.Staged.stage (fun () ->
            Sys.opaque_identity (Memchr.memchr s 'z' len)));
    Bench.Test.create_indexed ~name:"Bytes.index_opt"
      ~args:[ 4; 8; 16; 32; 64; 128; 256; 1024; 4096; 65536 ] (fun len ->
        let s = haystack len in
        Core.Staged.stage (fun () ->
            Sys.opaque_identity
              (match Bytes.index_opt s 'z' with Some i -> i | None -> -1)));
  ]

let () = Command_unix.run (Bench.make_command tests)
