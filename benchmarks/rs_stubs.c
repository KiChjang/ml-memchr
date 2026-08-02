#include <caml/mlvalues.h>
#include <stddef.h>

extern long rs_memchr(const unsigned char *, unsigned char, size_t);
extern long rs_safe_memchr(const unsigned char *, unsigned char, size_t);

CAMLprim value ml_rs_memchr(value s, value c, value n) {
    return Val_long(rs_memchr(Bytes_val(s), (unsigned char) Int_val(c), Long_val(n)));
}

CAMLprim value ml_rs_safe_memchr(value s, value c, value n) {
    return Val_long(rs_safe_memchr(Bytes_val(s), (unsigned char) Int_val(c), Long_val(n)));
}
