open Stdlib_upstream_compatible
open Ocaml_simd_sse
open Ocaml_simd_avx

module Bytes = struct
  external unsafe_get : (bytes[@local_opt]) @ read -> int -> int @@ portable
    = "%bytes_unsafe_get"

  external unsafe_get_int8_indexed_by_int64 :
    (bytes[@local_opt]) @ read -> int64# -> int @@ portable
    = "%caml_bytes_geti8u_indexed_by_int64#"

  external unsafe_get_int64_ne : (bytes[@local_opt]) @ read -> int -> int64#
    @@ portable = "%caml_bytes_get64u#"

  external unsafe_get_int64_ne_indexed_by_int64 :
    (bytes[@local_opt]) @ read -> int64# -> int64# @@ portable
    = "%caml_bytes_get64u#_indexed_by_int64#"
end

module Int64_u = struct
  include Int64_u

  let ( = ) a b = equal a b
  let ( <> ) a b = not (equal a b)
  let ( + ) a b = add a b
  let ( - ) a b = sub a b
  let ( < ) a b = compare a b < 0
  let ( > ) a b = compare a b > 0
  let ( <= ) a b = compare a b <= 0
  let ( >= ) a b = compare a b >= 0
  let ( land ) a b = logand a b
  let ( lor ) a b = logor a b
  let ( lxor ) a b = logxor a b
  let ( lsl ) a b = shift_left a b
  let ( lsr ) a b = shift_right_logical a b

  external count_trailing_zeros : int64# -> int64#
    = "caml_int64_ctz" "caml_int64_ctz_unboxed_to_untagged"
  [@@noalloc] [@@builtin] [@@no_effects] [@@no_coeffects]
end

module Nativeint_u = struct
  include Nativeint_u

  external unsafe_of_value : ('a : value_or_null). 'a @ local read -> nativeint#
    @@ portable
    = "caml_native_pointer_of_value_bytecode" "caml_native_pointer_of_value"
  [@@noalloc] [@@no_effects] [@@no_coeffects]

  let ( = ) a b = equal a b
  let ( land ) a b = logand a b
end

let memchr (s @ local read) c n =
  let n64 = Int64_u.of_int n in
  let ones = Int64_u.of_int (Sys.opaque_identity 0x0101_0101_0101_0101) in
  let mask = Int64_u.(ones lsl 7) in
  let cs = Int64_u.mul ones (Int64_u.of_int (Char.code c)) in
  let[@inline always] swar_raw w =
    let w = Int64_u.(w lxor cs) in
    Int64_u.((w - ones) land lognot w)
  in
  let mutable i = #0L in
  let mutable hit = -#1L in
  (try
     while Int64_u.(i + #32L <= n64) do
       let r0 = swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) in
       let r1 =
         swar_raw
           (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #8L))
       in
       let r2 =
         swar_raw
           (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #16L))
       in
       let r3 =
         swar_raw
           (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #24L))
       in
       let rs = Int64_u.(r0 lor r1 lor (r2 lor r3)) in
       if Int64_u.(rs land mask <> #0L) then (
         i <-
           (if Int64_u.(r0 land mask <> #0L) then i
            else if Int64_u.(r1 land mask <> #0L) then Int64_u.(i + #8L)
            else if Int64_u.(r2 land mask <> #0L) then Int64_u.(i + #16L)
            else Int64_u.(i + #24L));
         raise_notrace Exit)
       else i <- Int64_u.(i + #32L)
     done
   with Exit -> ());
  (try
     while Int64_u.(i + #8L <= n64) do
       let r = swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) in
       if Int64_u.(r land mask <> #0L) then raise_notrace Exit
       else i <- Int64_u.(i + #8L)
     done
   with Exit -> ());
  let remaining = n - Int64_u.to_int i in
  (if remaining > 0 then
     let w = swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) in
     let r = if remaining > 8 then 8 else remaining in
     let valid =
       Int64_u.shift_right_logical #0xFFFF_FFFF_FFFF_FFFFL ((8 - r) * 8)
     in
     let t = Int64_u.(w land mask land valid) in
     if Int64_u.(t <> #0L) then
       hit <- Int64_u.(i + (count_trailing_zeros t lsr 3)));
  Int64_u.to_int hit

let[@inline always] swar_raw w cs ones =
  let w = Int64_u.(w lxor cs) in
  Int64_u.((w - ones) land lognot w)

let[@inline always] rec scan_bytes i s c n =
  if Int64_u.(i >= n) then -1
  else if ((Bytes.unsafe_get_int8_indexed_by_int64 s i) land 0xFF) = Char.code c then (Int64_u.to_int i)
  else scan_bytes Int64_u.(i + #1L) s c n

let[@inline always] rec scan_word i s cs ones mask c n =
  if Int64_u.((i + #8L) > n) then scan_bytes i s c n
  else if
    Int64_u.(
      swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) cs ones
      land mask
      <> #0L)
  then scan_bytes i s c n
  else scan_word Int64_u.(i + #8L) s cs ones mask c n

let ml_memchr (s @ local read) c n =
  let rec scan_block i s cs ones mask c n =
    if Int64_u.to_int i + 32 > n then scan_word i s cs ones mask c (Int64_u.of_int n)
    else
      let r0 =
        swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) cs ones
      in
      let r1 =
        swar_raw
          (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #8L))
          cs ones
      in
      let r2 =
        swar_raw
          (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #16L))
          cs ones
      in
      let r3 =
        swar_raw
          (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #24L))
          cs ones
      in
      let rs = Int64_u.(r0 lor r1 lor (r2 lor r3)) in
      if Int64_u.(rs land mask <> #0L) then
        if Int64_u.(r0 land mask <> #0L) then scan_word i s cs ones mask c (Int64_u.of_int n)
        else if Int64_u.(r1 land mask <> #0L) then
          scan_word Int64_u.(i + #8L) s cs ones mask c (Int64_u.of_int n)
        else if Int64_u.(r2 land mask <> #0L) then
          scan_word Int64_u.(i + #16L) s cs ones mask c (Int64_u.of_int n)
        else scan_word Int64_u.(i + #24L) s cs ones mask c (Int64_u.of_int n)
      else scan_block Int64_u.(i + #32L) s cs ones mask c n
  in
  let ones = Int64_u.of_int (Sys.opaque_identity 0x0101_0101_0101_0101) in
  let mask = Int64_u.(ones lsl 7) in
  let cs = Int64_u.mul ones (Int64_u.of_int (Char.code c)) in
  scan_block #0L s cs ones mask c n

let simd_memchr (s @ local read) c n =
  let scan_remainder i s c cs n =
    if Int64_u.(i + #16L > n) then
      let cs = Int64x4.extract0 (Int64x4.of_int8x32_bits cs) in
      let ones = #0x0101_0101_0101_0101L in
      let mask = Int64_u.(ones lsl 7) in
      scan_word i s cs ones mask c n
    else
      let half_cs = Int8x32.extract_lane0 cs in
      let haystack = Int8x16.Bytes.Int64_u.unsafe_get ~byte:i s in
      let bitmask = Int8x16.(movemask (half_cs = haystack)) in
      if Int64_u.(bitmask <> #0L) then
        Int64_u.(to_int (count_trailing_zeros bitmask + i))
      else
        let cs = Int64x4.extract0 (Int64x4.of_int8x32_bits cs) in
        let ones = #0x0101_0101_0101_0101L in
        let mask = Int64_u.(ones lsl 7) in
        scan_word Int64_u.(i + #16L) s cs ones mask c n
  in
  let rec scan_vec i s c cs n =
    if Int64_u.(i + #32L > n) then scan_remainder i s c cs n
    else
      let v = Int8x32.Bytes.Int64_u.unsafe_get ~byte:i s in
      let bitmask = Int8x32.(movemask (cs = v)) in
      if Int64_u.(bitmask <> #0L) then
        Int64_u.(to_int (count_trailing_zeros bitmask + i))
      else scan_vec Int64_u.(i + #32L) s c cs n
  in
  let rec scan_matrix i s c cs n =
    if Int64_u.(i + #128L > n) then scan_vec i s c cs n
    else
      let v0 = Int8x32.Bytes.Int64_u.unsafe_get ~byte:i s in
      let v1 = Int8x32.Bytes.Int64_u.unsafe_get ~byte:Int64_u.(i + #32L) s in
      let v2 = Int8x32.Bytes.Int64_u.unsafe_get ~byte:Int64_u.(i + #64L) s in
      let v3 = Int8x32.Bytes.Int64_u.unsafe_get ~byte:Int64_u.(i + #96L) s in
      let e0 = Int8x32.(cs = v0) and e1 = Int8x32.(cs = v1) in
      let e2 = Int8x32.(cs = v2) and e3 = Int8x32.(cs = v3) in
      let combined = Int8x32.(e0 lor e1 lor (e2 lor e3)) in
      if Int64_u.(Int8x32.movemask combined <> #0L) then
        let bitmask = Int8x32.movemask e0 in
        if Int64_u.(bitmask <> #0L) then
          Int64_u.(to_int (count_trailing_zeros bitmask + i))
        else
          let bitmask = Int8x32.movemask e1 in
          if Int64_u.(bitmask <> #0L) then
            Int64_u.(to_int (count_trailing_zeros bitmask + i + #32L))
          else
            let bitmask = Int8x32.movemask e2 in
            if Int64_u.(bitmask <> #0L) then
              Int64_u.(to_int (count_trailing_zeros bitmask + i + #64L))
            else
              Int64_u.(
                to_int (count_trailing_zeros (Int8x32.movemask e3) + i + #96L))
      else scan_matrix Int64_u.(i + #128L) s c cs n
  in
  let cs = Int8x32.set1 (Stdlib_stable.Int8_u.of_int (Char.code c)) in
  let n64 = Int64_u.of_int n in
  if Int64_u.(n64 < #32L) then scan_remainder #0L s c cs n64
  else
    let misalign = Nativeint_u.(Nativeint_u.unsafe_of_value s land #31n) in
    if Nativeint_u.(misalign = #0n) then scan_matrix #0L s c cs n64
    else
      let vec = Int8x32.Bytes.unsafe_get ~byte:0 s in
      let bitmask = Int8x32.(movemask (cs = vec)) in
      if Int64_u.(bitmask <> #0L) then
        Int64_u.(to_int (count_trailing_zeros bitmask))
      else scan_matrix Int64_u.(#32L - of_nativeint_u misalign) s c cs n64

let () = Random.self_init ()

let%test_unit "smoke" =
  let open Base in
  let str = "hello" in
  [%test_eq: int] (memchr (Bytes.of_string str) 'l' (String.length str)) 2

let%test_unit "longer strings" =
  let open Base in
  let str = "this is harder" in
  [%test_eq: int] (memchr (Bytes.of_string str) 'r' (String.length str)) 10

let%test_unit "negative" =
  let open Base in
  let str = "some long string of words that does not contain the character" in
  [%test_eq: int] (memchr (Bytes.of_string str) 'z' (String.length str)) (-1)

let%test_unit "parity" =
  let open Base in
  let naive s c n =
    let rec go i =
      if i >= n then -1
      else if Char.equal (Bytes.get s i) c then i
      else go (i + 1)
    in
    go 0
  in
  for i = 1 to 200 do
    let alphabet = "abcdefgh" in
    let alphabet_len = String.length alphabet in
    let random_len = Random.int 200 in
    let str =
      String.init random_len ~f:(fun _ -> alphabet.[Random.int alphabet_len])
    in
    let s = Bytes.of_string str in
    [%test_eq: int] (memchr s 'e' random_len) (naive s 'e' random_len)
  done

let%test_unit "equal output" =
  let open Base in
  for i = 1 to 200 do
    let alphabet = "abcdefgh" in
    let alphabet_len = String.length alphabet in
    let random_len = Random.int 200 in
    let str =
      String.init random_len ~f:(fun _ -> alphabet.[Random.int alphabet_len])
    in
    let s = Bytes.of_string str in
    [%test_eq: int] (memchr s 'e' random_len) (ml_memchr s 'e' random_len)
  done

let%test_unit "simd equal output" =
  let open Base in
  for i = 1 to 200 do
    let alphabet = "abcdefgh" in
    let alphabet_len = String.length alphabet in
    let random_len = Random.int 200 in
    let str =
      String.init random_len ~f:(fun _ -> alphabet.[Random.int alphabet_len])
    in
    let s = Bytes.of_string str in
    [%test_eq: int] (memchr s 'e' random_len) (simd_memchr s 'e' random_len)
  done
