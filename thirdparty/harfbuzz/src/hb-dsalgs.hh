/*
 * Copyright © 2017  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_DSALGS_HH
#define HB_DSALGS_HH

#include "hb-private.hh"


/* Void! For when we need a expression-type of void. */
typedef const struct _hb_void_t *hb_void_t;
#define HB_VOID ((const _hb_void_t *) nullptr)


/*
 * Bithacks.
 */

/* Return the number of 1 bits in v. */
template <typename T>
static inline HB_CONST_FUNC unsigned int
hb_popcount (T v)
{
#if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)) && defined(__OPTIMIZE__)
  if (sizeof (T) <= sizeof (unsigned int))
    return __builtin_popcount (v);

  if (sizeof (T) <= sizeof (unsigned long))
    return __builtin_popcountl (v);

  if (sizeof (T) <= sizeof (unsigned long long))
    return __builtin_popcountll (v);
#endif

  if (sizeof (T) <= 4)
  {
    /* "HACKMEM 169" */
    uint32_t y;
    y = (v >> 1) &033333333333;
    y = v - y - ((y >>1) & 033333333333);
    return (((y + (y >> 3)) & 030707070707) % 077);
  }

  if (sizeof (T) == 8)
  {
    unsigned int shift = 32;
    return hb_popcount<uint32_t> ((uint32_t) v) + hb_popcount ((uint32_t) (v >> shift));
  }

  if (sizeof (T) == 16)
  {
    unsigned int shift = 64;
    return hb_popcount<uint64_t> ((uint64_t) v) + hb_popcount ((uint64_t) (v >> shift));
  }

  assert (0);
  return 0; /* Shut up stupid compiler. */
}

/* Returns the number of bits needed to store number */
template <typename T>
static inline HB_CONST_FUNC unsigned int
hb_bit_storage (T v)
{
  if (unlikely (!v)) return 0;

#if defined(__GNUC__) && (__GNUC__ >= 4) && defined(__OPTIMIZE__)
  if (sizeof (T) <= sizeof (unsigned int))
    return sizeof (unsigned int) * 8 - __builtin_clz (v);

  if (sizeof (T) <= sizeof (unsigned long))
    return sizeof (unsigned long) * 8 - __builtin_clzl (v);

  if (sizeof (T) <= sizeof (unsigned long long))
    return sizeof (unsigned long long) * 8 - __builtin_clzll (v);
#endif

#if (defined(_MSC_VER) && _MSC_VER >= 1500) || defined(__MINGW32__)
  if (sizeof (T) <= sizeof (unsigned int))
  {
    unsigned long where;
    _BitScanReverse (&where, v);
    return 1 + where;
  }
# if _WIN64
  if (sizeof (T) <= 8)
  {
    unsigned long where;
    _BitScanReverse64 (&where, v);
    return 1 + where;
  }
# endif
#endif

  if (sizeof (T) <= 4)
  {
    /* "bithacks" */
    const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    const unsigned int S[] = {1, 2, 4, 8, 16};
    unsigned int r = 0;
    for (int i = 4; i >= 0; i--)
      if (v & b[i])
      {
	v >>= S[i];
	r |= S[i];
      }
    return r + 1;
  }
  if (sizeof (T) <= 8)
  {
    /* "bithacks" */
    const uint64_t b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
    const unsigned int S[] = {1, 2, 4, 8, 16, 32};
    unsigned int r = 0;
    for (int i = 5; i >= 0; i--)
      if (v & b[i])
      {
	v >>= S[i];
	r |= S[i];
      }
    return r + 1;
  }
  if (sizeof (T) == 16)
  {
    unsigned int shift = 64;
    return (v >> shift) ? hb_bit_storage<uint64_t> ((uint64_t) (v >> shift)) + shift :
			  hb_bit_storage<uint64_t> ((uint64_t) v);
  }

  assert (0);
  return 0; /* Shut up stupid compiler. */
}

/* Returns the number of zero bits in the least significant side of v */
template <typename T>
static inline HB_CONST_FUNC unsigned int
hb_ctz (T v)
{
  if (unlikely (!v)) return 0;

#if defined(__GNUC__) && (__GNUC__ >= 4) && defined(__OPTIMIZE__)
  if (sizeof (T) <= sizeof (unsigned int))
    return __builtin_ctz (v);

  if (sizeof (T) <= sizeof (unsigned long))
    return __builtin_ctzl (v);

  if (sizeof (T) <= sizeof (unsigned long long))
    return __builtin_ctzll (v);
#endif

#if (defined(_MSC_VER) && _MSC_VER >= 1500) || defined(__MINGW32__)
  if (sizeof (T) <= sizeof (unsigned int))
  {
    unsigned long where;
    _BitScanForward (&where, v);
    return where;
  }
# if _WIN64
  if (sizeof (T) <= 8)
  {
    unsigned long where;
    _BitScanForward64 (&where, v);
    return where;
  }
# endif
#endif

  if (sizeof (T) <= 4)
  {
    /* "bithacks" */
    unsigned int c = 32;
    v &= - (int32_t) v;
    if (v) c--;
    if (v & 0x0000FFFF) c -= 16;
    if (v & 0x00FF00FF) c -= 8;
    if (v & 0x0F0F0F0F) c -= 4;
    if (v & 0x33333333) c -= 2;
    if (v & 0x55555555) c -= 1;
    return c;
  }
  if (sizeof (T) <= 8)
  {
    /* "bithacks" */
    unsigned int c = 64;
    v &= - (int64_t) (v);
    if (v) c--;
    if (v & 0x00000000FFFFFFFFULL) c -= 32;
    if (v & 0x0000FFFF0000FFFFULL) c -= 16;
    if (v & 0x00FF00FF00FF00FFULL) c -= 8;
    if (v & 0x0F0F0F0F0F0F0F0FULL) c -= 4;
    if (v & 0x3333333333333333ULL) c -= 2;
    if (v & 0x5555555555555555ULL) c -= 1;
    return c;
  }
  if (sizeof (T) == 16)
  {
    unsigned int shift = 64;
    return (uint64_t) v ? hb_bit_storage<uint64_t> ((uint64_t) v) :
			  hb_bit_storage<uint64_t> ((uint64_t) (v >> shift)) + shift;
  }

  assert (0);
  return 0; /* Shut up stupid compiler. */
}


/*
 * Tiny stuff.
 */

/* ASCII tag/character handling */
static inline bool ISALPHA (unsigned char c)
{ return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static inline bool ISALNUM (unsigned char c)
{ return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); }
static inline bool ISSPACE (unsigned char c)
{ return c == ' ' || c =='\f'|| c =='\n'|| c =='\r'|| c =='\t'|| c =='\v'; }
static inline unsigned char TOUPPER (unsigned char c)
{ return (c >= 'a' && c <= 'z') ? c - 'a' + 'A' : c; }
static inline unsigned char TOLOWER (unsigned char c)
{ return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c; }

#undef MIN
template <typename Type>
static inline Type MIN (const Type &a, const Type &b) { return a < b ? a : b; }

#undef MAX
template <typename Type>
static inline Type MAX (const Type &a, const Type &b) { return a > b ? a : b; }

static inline unsigned int DIV_CEIL (const unsigned int a, unsigned int b)
{ return (a + (b - 1)) / b; }


#undef  ARRAY_LENGTH
template <typename Type, unsigned int n>
static inline unsigned int ARRAY_LENGTH (const Type (&)[n]) { return n; }
/* A const version, but does not detect erratically being called on pointers. */
#define ARRAY_LENGTH_CONST(__array) ((signed int) (sizeof (__array) / sizeof (__array[0])))

static inline bool
hb_unsigned_mul_overflows (unsigned int count, unsigned int size)
{
  return (size > 0) && (count >= ((unsigned int) -1) / size);
}

static inline unsigned int
hb_ceil_to_4 (unsigned int v)
{
  return ((v - 1) | 3) + 1;
}

template <typename T> class hb_assert_unsigned_t;
template <> class hb_assert_unsigned_t<unsigned char> {};
template <> class hb_assert_unsigned_t<unsigned short> {};
template <> class hb_assert_unsigned_t<unsigned int> {};
template <> class hb_assert_unsigned_t<unsigned long> {};

template <typename T> static inline bool
hb_in_range (T u, T lo, T hi)
{
  /* The sizeof() is here to force template instantiation.
   * I'm sure there are better ways to do this but can't think of
   * one right now.  Declaring a variable won't work as HB_UNUSED
   * is unusable on some platforms and unused types are less likely
   * to generate a warning than unused variables. */
  static_assert ((sizeof (hb_assert_unsigned_t<T>) >= 0), "");

  /* The casts below are important as if T is smaller than int,
   * the subtract results will become a signed int! */
  return (T)(u - lo) <= (T)(hi - lo);
}
template <typename T> static inline bool
hb_in_ranges (T u, T lo1, T hi1, T lo2, T hi2)
{
  return hb_in_range (u, lo1, hi1) || hb_in_range (u, lo2, hi2);
}
template <typename T> static inline bool
hb_in_ranges (T u, T lo1, T hi1, T lo2, T hi2, T lo3, T hi3)
{
  return hb_in_range (u, lo1, hi1) || hb_in_range (u, lo2, hi2) || hb_in_range (u, lo3, hi3);
}


/*
 * Sort and search.
 */

static inline void *
hb_bsearch_r (const void *key, const void *base,
	      size_t nmemb, size_t size,
	      int (*compar)(const void *_key, const void *_item, void *_arg),
	      void *arg)
{
  int min = 0, max = (int) nmemb - 1;
  while (min <= max)
  {
    int mid = (min + max) / 2;
    const void *p = (const void *) (((const char *) base) + (mid * size));
    int c = compar (key, p, arg);
    if (c < 0)
      max = mid - 1;
    else if (c > 0)
      min = mid + 1;
    else
      return (void *) p;
  }
  return nullptr;
}


/* From https://github.com/noporpoise/sort_r */

/* Isaac Turner 29 April 2014 Public Domain */

/*

hb_sort_r function to be exported.

Parameters:
  base is the array to be sorted
  nel is the number of elements in the array
  width is the size in bytes of each element of the array
  compar is the comparison function
  arg is a pointer to be passed to the comparison function

void hb_sort_r(void *base, size_t nel, size_t width,
               int (*compar)(const void *_a, const void *_b, void *_arg),
               void *arg);
*/


/* swap a, b iff a>b */
/* __restrict is same as restrict but better support on old machines */
static int sort_r_cmpswap(char *__restrict a, char *__restrict b, size_t w,
			  int (*compar)(const void *_a, const void *_b,
					void *_arg),
			  void *arg)
{
  char tmp, *end = a+w;
  if(compar(a, b, arg) > 0) {
    for(; a < end; a++, b++) { tmp = *a; *a = *b; *b = tmp; }
    return 1;
  }
  return 0;
}

/* Note: quicksort is not stable, equivalent values may be swapped */
static inline void sort_r_simple(void *base, size_t nel, size_t w,
				 int (*compar)(const void *_a, const void *_b,
					       void *_arg),
				 void *arg)
{
  char *b = (char *)base, *end = b + nel*w;
  if(nel < 7) {
    /* Insertion sort for arbitrarily small inputs */
    char *pi, *pj;
    for(pi = b+w; pi < end; pi += w) {
      for(pj = pi; pj > b && sort_r_cmpswap(pj-w,pj,w,compar,arg); pj -= w) {}
    }
  }
  else
  {
    /* nel > 6; Quicksort */

    /* Use median of first, middle and last items as pivot */
    char *x, *y, *xend, ch;
    char *pl, *pr;
    char *last = b+w*(nel-1), *tmp;
    char *l[3];
    l[0] = b;
    l[1] = b+w*(nel/2);
    l[2] = last;

    if(compar(l[0],l[1],arg) > 0) { tmp=l[0]; l[0]=l[1]; l[1]=tmp; }
    if(compar(l[1],l[2],arg) > 0) {
      tmp=l[1]; l[1]=l[2]; l[2]=tmp; /* swap(l[1],l[2]) */
      if(compar(l[0],l[1],arg) > 0) { tmp=l[0]; l[0]=l[1]; l[1]=tmp; }
    }

    /* swap l[id], l[2] to put pivot as last element */
    for(x = l[1], y = last, xend = x+w; x<xend; x++, y++) {
      ch = *x; *x = *y; *y = ch;
    }

    pl = b;
    pr = last;

    while(pl < pr) {
      for(; pl < pr; pl += w) {
        if(sort_r_cmpswap(pl, pr, w, compar, arg)) {
          pr -= w; /* pivot now at pl */
          break;
        }
      }
      for(; pl < pr; pr -= w) {
        if(sort_r_cmpswap(pl, pr, w, compar, arg)) {
          pl += w; /* pivot now at pr */
          break;
        }
      }
    }

    sort_r_simple(b, (pl-b)/w, w, compar, arg);
    sort_r_simple(pl+w, (end-(pl+w))/w, w, compar, arg);
  }
}

static inline void hb_sort_r(void *base, size_t nel, size_t width,
			     int (*compar)(const void *_a, const void *_b, void *_arg),
			     void *arg)
{
    sort_r_simple(base, nel, width, compar, arg);
}


template <typename T, typename T2> static inline void
hb_stable_sort (T *array, unsigned int len, int(*compar)(const T *, const T *), T2 *array2)
{
  for (unsigned int i = 1; i < len; i++)
  {
    unsigned int j = i;
    while (j && compar (&array[j - 1], &array[i]) > 0)
      j--;
    if (i == j)
      continue;
    /* Move item i to occupy place for item j, shift what's in between. */
    {
      T t = array[i];
      memmove (&array[j + 1], &array[j], (i - j) * sizeof (T));
      array[j] = t;
    }
    if (array2)
    {
      T2 t = array2[i];
      memmove (&array2[j + 1], &array2[j], (i - j) * sizeof (T2));
      array2[j] = t;
    }
  }
}

template <typename T> static inline void
hb_stable_sort (T *array, unsigned int len, int(*compar)(const T *, const T *))
{
  hb_stable_sort (array, len, compar, (int *) nullptr);
}

static inline hb_bool_t
hb_codepoint_parse (const char *s, unsigned int len, int base, hb_codepoint_t *out)
{
  /* Pain because we don't know whether s is nul-terminated. */
  char buf[64];
  len = MIN (ARRAY_LENGTH (buf) - 1, len);
  strncpy (buf, s, len);
  buf[len] = '\0';

  char *end;
  errno = 0;
  unsigned long v = strtoul (buf, &end, base);
  if (errno) return false;
  if (*end) return false;
  *out = v;
  return true;
}


template <typename Type>
struct hb_auto_t : Type
{
  hb_auto_t (void) { Type::init (); }
  ~hb_auto_t (void) { Type::fini (); }
  private: /* Hide */
  void init (void) {}
  void fini (void) {}
};

struct hb_bytes_t
{
  inline hb_bytes_t (void) : bytes (nullptr), len (0) {}
  inline hb_bytes_t (const char *bytes_, unsigned int len_) : bytes (bytes_), len (len_) {}

  inline int cmp (const hb_bytes_t &a) const
  {
    if (len != a.len)
      return (int) a.len - (int) len;

    return memcmp (a.bytes, bytes, len);
  }
  static inline int cmp (const void *pa, const void *pb)
  {
    hb_bytes_t *a = (hb_bytes_t *) pa;
    hb_bytes_t *b = (hb_bytes_t *) pb;
    return b->cmp (*a);
  }

  const char *bytes;
  unsigned int len;
};


struct HbOpOr
{
  static const bool passthru_left = true;
  static const bool passthru_right = true;
  template <typename T> static void process (T &o, const T &a, const T &b) { o = a | b; }
};
struct HbOpAnd
{
  static const bool passthru_left = false;
  static const bool passthru_right = false;
  template <typename T> static void process (T &o, const T &a, const T &b) { o = a & b; }
};
struct HbOpMinus
{
  static const bool passthru_left = true;
  static const bool passthru_right = false;
  template <typename T> static void process (T &o, const T &a, const T &b) { o = a & ~b; }
};
struct HbOpXor
{
  static const bool passthru_left = true;
  static const bool passthru_right = true;
  template <typename T> static void process (T &o, const T &a, const T &b) { o = a ^ b; }
};


/* Compiler-assisted vectorization. */

/* Type behaving similar to vectorized vars defined using __attribute__((vector_size(...))),
 * using vectorized operations if HB_VECTOR_SIZE is set to **bit** numbers (eg 128).
 * Define that to 0 to disable. */
template <typename elt_t, unsigned int byte_size>
struct hb_vector_size_t
{
  elt_t& operator [] (unsigned int i) { return u.v[i]; }
  const elt_t& operator [] (unsigned int i) const { return u.v[i]; }

  template <class Op>
  inline hb_vector_size_t process (const hb_vector_size_t &o) const
  {
    hb_vector_size_t r;
#if HB_VECTOR_SIZE
    if (HB_VECTOR_SIZE && 0 == (byte_size * 8) % HB_VECTOR_SIZE)
      for (unsigned int i = 0; i < ARRAY_LENGTH (u.vec); i++)
	Op::process (r.u.vec[i], u.vec[i], o.u.vec[i]);
    else
#endif
      for (unsigned int i = 0; i < ARRAY_LENGTH (u.v); i++)
	Op::process (r.u.v[i], u.v[i], o.u.v[i]);
    return r;
  }
  inline hb_vector_size_t operator | (const hb_vector_size_t &o) const
  { return process<HbOpOr> (o); }
  inline hb_vector_size_t operator & (const hb_vector_size_t &o) const
  { return process<HbOpAnd> (o); }
  inline hb_vector_size_t operator ^ (const hb_vector_size_t &o) const
  { return process<HbOpXor> (o); }
  inline hb_vector_size_t operator ~ () const
  {
    hb_vector_size_t r;
#if HB_VECTOR_SIZE && 0
    if (HB_VECTOR_SIZE && 0 == (byte_size * 8) % HB_VECTOR_SIZE)
      for (unsigned int i = 0; i < ARRAY_LENGTH (u.vec); i++)
	r.u.vec[i] = ~u.vec[i];
    else
#endif
    for (unsigned int i = 0; i < ARRAY_LENGTH (u.v); i++)
      r.u.v[i] = ~u.v[i];
    return r;
  }

  private:
  static_assert (byte_size / sizeof (elt_t) * sizeof (elt_t) == byte_size, "");
  union {
    elt_t v[byte_size / sizeof (elt_t)];
#if HB_VECTOR_SIZE
    hb_vector_size_impl_t vec[byte_size / sizeof (hb_vector_size_impl_t)];
#endif
  } u;
};


#endif /* HB_DSALGS_HH */
