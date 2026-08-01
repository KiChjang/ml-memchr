open Stdlib_upstream_compatible

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
  let ( + ) a b = add a b
  let ( - ) a b = sub a b
  let ( < ) a b = compare a b < 0
  let ( <= ) a b = compare a b <= 0
  let ( >= ) a b = compare a b >= 0
  let ( land ) a b = logand a b
  let ( lor ) a b = logor a b
  let ( lxor ) a b = logxor a b
  let ( lsl ) a b = shift_left a b
  let ( lsr ) a b = shift_right_logical a b
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
  while Int64_u.to_int i + 32 <= n do
    let r0 = swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) in
    let r1 =
      swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #8L))
    in
    let r2 =
      swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #16L))
    in
    let r3 =
      swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s Int64_u.(i + #24L))
    in
    let rs = Int64_u.(r0 lor r1 lor (r2 lor r3)) in
    if not Int64_u.(rs land mask = #0L) then (
      hit <-
        (if not Int64_u.(r0 land mask = #0L) then i
         else if not Int64_u.(r1 land mask = #0L) then Int64_u.(i + #8L)
         else if not Int64_u.(r2 land mask = #0L) then Int64_u.(i + #16L)
         else Int64_u.(i + #24L));
      i <- n64)
    else i <- Int64_u.(i + #32L)
  done;
  if Int64_u.(hit < #0L) then
    while Int64_u.to_int i + 8 <= n do
      let r = swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) in
      if not Int64_u.(r land mask = #0L) then (
        hit <- i;
        i <- n64)
      else i <- Int64_u.(i + #8L)
    done
  else i <- hit;
  if Int64_u.(hit >= #0L) then i <- hit;
  let remaining = n - Int64_u.to_int i in
  (if remaining > 0 then
     let w = swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) in
     let r = if remaining > 8 then 8 else remaining in
     let valid =
       Int64_u.shift_right_logical #0xFFFF_FFFF_FFFF_FFFFL ((8 - r) * 8)
     in
     let t = Int64_u.(w land mask land valid) in
     if not Int64_u.(t = #0L) then
       hit <-
         Int64_u.(
           i
           + (Ocaml_intrinsics_kernel.Int64.Unboxed.count_trailing_zeros t lsr 3)));
  Int64_u.to_int hit

let ml_memchr (s @ local read) c n =
  let ones = Int64_u.of_int (Sys.opaque_identity 0x0101_0101_0101_0101) in
  let mask = Int64_u.(ones lsl 7) in
  let cs = Int64_u.mul ones (Int64_u.of_int (Char.code c)) in
  let[@inline always] swar_raw w =
    let w = Int64_u.(w lxor cs) in
    Int64_u.((w - ones) land lognot w)
  in
  let rec bytes i =
    if Int64_u.to_int i >= n then -1
    else if Bytes.unsafe_get_int8_indexed_by_int64 s i land 0xFF = Char.code c
    then Int64_u.to_int i
    else bytes Int64_u.(i + #1L)
  in
  let rec word i =
    if Int64_u.to_int i + 8 > n then bytes i
    else if
      not
        Int64_u.(
          swar_raw (Bytes.unsafe_get_int64_ne_indexed_by_int64 s i) land mask
          = #0L)
    then bytes i
    else word Int64_u.(i + #8L)
  in
  let rec block i =
    if Int64_u.to_int i + 32 > n then word i
    else
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
      if not Int64_u.(r0 lor r1 lor r2 lor r3 land mask = #0L) then
        if not Int64_u.(r0 land mask = #0L) then word i
        else if not Int64_u.(r1 land mask = #0L) then word Int64_u.(i + #8L)
        else if not Int64_u.(r2 land mask = #0L) then word Int64_u.(i + #16L)
        else word Int64_u.(i + #24L)
      else block Int64_u.(i + #32L)
  in
  block #0L [@nontail]

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
