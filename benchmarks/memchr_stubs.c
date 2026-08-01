#include <caml/mlvalues.h>
#include <string.h>

CAMLprim value ml_bytes_memchr(value s, value c, value n)
{
  const char *base = (const char *) Bytes_val(s);
  const char *r = (const char *) memchr(base, Int_val(c), Long_val(n));
  return Val_long(r == NULL ? -1 : (r - base));
}
