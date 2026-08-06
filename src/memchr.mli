val memchr : bytes @ local read -> char -> int -> int [@@zero_alloc]

val ml_memchr : bytes @ local read -> char -> int -> int [@@zero_alloc]

val simd_memchr : bytes @ local read -> char -> int -> int [@@zero_alloc]
