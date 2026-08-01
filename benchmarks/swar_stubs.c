#include <caml/mlvalues.h>
#include <stdint.h>
#include <string.h>

#define ONES  0x0101010101010101ULL
#define HIGHS 0x8080808080808080ULL

static inline uint64_t load64(const unsigned char *p) {
  uint64_t w; memcpy(&w, p, 8); return w;   /* compiles to one mov */
}

static inline uint64_t swar(uint64_t w, uint64_t cs) {
  w ^= cs;
  return (w - ONES) & ~w;
}

CAMLprim value ml_bytes_swar_memchr(value s, value c, value n)
{
  const unsigned char *p = (const unsigned char *) Bytes_val(s);
  long len = Long_val(n);
  unsigned char needle = (unsigned char) Int_val(c);
  uint64_t cs = ONES * (uint64_t) needle;
  long i = 0;

  while (i + 32 <= len) {
    uint64_t r0 = swar(load64(p + i),      cs);
    uint64_t r1 = swar(load64(p + i + 8),  cs);
    uint64_t r2 = swar(load64(p + i + 16), cs);
    uint64_t r3 = swar(load64(p + i + 24), cs);
    if (((r0 | r1 | r2 | r3) & HIGHS) != 0) {
        if ((r0 & HIGHS) != 0) {}
        else if ((r1 & HIGHS) != 0) {
            i += 8;
        } else if ((r2 & HIGHS) != 0) {
            i += 16;
        } else {
            i += 24;
        }
        break;
    }
    i += 32;
  }
  while (i + 8 <= len) {
      if ((swar(load64(p + i), cs) & HIGHS) != 0) break;
      i += 8;
  }
  for (; i < len; i++)
    if (p[i] == needle) return Val_long(i);
  return Val_long(-1);
}
