/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2006  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * \file bitset.h
 * \brief Bitset of arbitrary size definitions.
 * \author Michal Krol
 */

#ifndef BITSET_H
#define BITSET_H

#include "util/bitscan.h"
#include "util/macros.h"

/****************************************************************************
 * generic bitset implementation
 */

#define BITSET_WORD unsigned int
#define BITSET_WORDBITS (sizeof (BITSET_WORD) * 8)

/* bitset declarations
 */
#define BITSET_WORDS(bits) (((bits) + BITSET_WORDBITS - 1) / BITSET_WORDBITS)
#define BITSET_DECLARE(name, bits) BITSET_WORD name[BITSET_WORDS(bits)]

/* bitset operations
 */
#define BITSET_COPY(x, y) memcpy( (x), (y), sizeof (x) )
#define BITSET_EQUAL(x, y) (memcmp( (x), (y), sizeof (x) ) == 0)
#define BITSET_ZERO(x) memset( (x), 0, sizeof (x) )
#define BITSET_ONES(x) memset( (x), 0xff, sizeof (x) )
#define BITSET_SIZE(x) (8 * sizeof(x))  // bitset size in bits

#define BITSET_BITWORD(b) ((b) / BITSET_WORDBITS)
#define BITSET_BIT(b) (1u << ((b) % BITSET_WORDBITS))

/* single bit operations
 */
#define BITSET_TEST(x, b) (((x)[BITSET_BITWORD(b)] & BITSET_BIT(b)) != 0)
#define BITSET_SET(x, b) ((x)[BITSET_BITWORD(b)] |= BITSET_BIT(b))
#define BITSET_CLEAR(x, b) ((x)[BITSET_BITWORD(b)] &= ~BITSET_BIT(b))

#define BITSET_MASK(b) (((b) % BITSET_WORDBITS == 0) ? ~0 : BITSET_BIT(b) - 1)
#define BITSET_RANGE(b, e) ((BITSET_MASK((e) + 1)) & ~(BITSET_BIT(b) - 1))

/* logic bit operations
 */
static inline void
__bitset_and(BITSET_WORD *r, const BITSET_WORD *x, const BITSET_WORD *y, unsigned n)
{
   for (unsigned i = 0; i < n; i++)
      r[i] = x[i] & y[i];
}

static inline void
__bitset_or(BITSET_WORD *r, const BITSET_WORD *x, const BITSET_WORD *y, unsigned n)
{
   for (unsigned i = 0; i < n; i++)
      r[i] = x[i] | y[i];
}

static inline void
__bitset_not(BITSET_WORD *x, unsigned n)
{
   for (unsigned i = 0; i < n; i++)
      x[i] = ~x[i];
}

#define BITSET_AND(r, x, y)   \
   do { \
      assert(ARRAY_SIZE(r) == ARRAY_SIZE(x)); \
      assert(ARRAY_SIZE(r) == ARRAY_SIZE(y)); \
      __bitset_and(r, x, y, ARRAY_SIZE(r)); \
   } while (0)

#define BITSET_OR(r, x, y)   \
   do { \
      assert(ARRAY_SIZE(r) == ARRAY_SIZE(x)); \
      assert(ARRAY_SIZE(r) == ARRAY_SIZE(y)); \
      __bitset_or(r, x, y, ARRAY_SIZE(r)); \
   } while (0)

#define BITSET_NOT(x)   \
   __bitset_not(x, ARRAY_SIZE(x))

static inline void
__bitset_rotate_right(BITSET_WORD *x, unsigned amount, unsigned n)
{
   assert(amount < BITSET_WORDBITS);

   if (amount == 0)
      return;

   for (unsigned i = 0; i < n - 1; i++) {
      x[i] = (x[i] >> amount) | (x[i + 1] << (BITSET_WORDBITS - amount));
   }

   x[n - 1] = x[n - 1] >> amount;
}

static inline void
__bitset_rotate_left(BITSET_WORD *x, unsigned amount, unsigned n)
{
   assert(amount < BITSET_WORDBITS);

   if (amount == 0)
      return;

   for (int i = n - 1; i > 0; i--) {
      x[i] = (x[i] << amount) | (x[i - 1] >> (BITSET_WORDBITS - amount));
   }

   x[0] = x[0] << amount;
}

static inline void
__bitset_shr(BITSET_WORD *x, unsigned amount, unsigned n)
{
   const unsigned int words = amount / BITSET_WORDBITS;

   if (amount == 0)
      return;

   if (words) {
      unsigned i;

      for (i = 0; i < n - words; i++)
         x[i] = x[i + words];

      while (i < n)
         x[i++] = 0;

      amount %= BITSET_WORDBITS;
   }

   __bitset_rotate_right(x, amount, n);
}


static inline void
__bitset_shl(BITSET_WORD *x, unsigned amount, unsigned n)
{
   const int words = amount / BITSET_WORDBITS;

   if (amount == 0)
      return;

   if (words) {
      int i;

      for (i = n - 1; i >= words; i--) {
         x[i] = x[i - words];
      }

      while (i >= 0) {
         x[i--] = 0;
      }

      amount %= BITSET_WORDBITS;
   }

   __bitset_rotate_left(x, amount, n);
}

#define BITSET_SHR(x, n)   \
   __bitset_shr(x, n, ARRAY_SIZE(x));

#define BITSET_SHL(x, n)   \
   __bitset_shl(x, n, ARRAY_SIZE(x));

/* bit range operations
 */
#define BITSET_TEST_RANGE_INSIDE_WORD(x, b, e) \
   (BITSET_BITWORD(b) == BITSET_BITWORD(e) ? \
   (((x)[BITSET_BITWORD(b)] & BITSET_RANGE(b, e)) != 0) : \
   (assert (!"BITSET_TEST_RANGE: bit range crosses word boundary"), 0))
#define BITSET_SET_RANGE_INSIDE_WORD(x, b, e) \
   (BITSET_BITWORD(b) == BITSET_BITWORD(e) ? \
   ((x)[BITSET_BITWORD(b)] |= BITSET_RANGE(b, e)) : \
   (assert (!"BITSET_SET_RANGE_INSIDE_WORD: bit range crosses word boundary"), 0))
#define BITSET_CLEAR_RANGE_INSIDE_WORD(x, b, e) \
   (BITSET_BITWORD(b) == BITSET_BITWORD(e) ? \
   ((x)[BITSET_BITWORD(b)] &= ~BITSET_RANGE(b, e)) : \
   (assert (!"BITSET_CLEAR_RANGE: bit range crosses word boundary"), 0))

static inline bool
__bitset_test_range(const BITSET_WORD *r, unsigned start, unsigned end)
{
   const unsigned size = end - start + 1;
   const unsigned start_mod = start % BITSET_WORDBITS;

   if (start_mod + size <= BITSET_WORDBITS) {
      return BITSET_TEST_RANGE_INSIDE_WORD(r, start, end);
   } else {
      const unsigned first_size = BITSET_WORDBITS - start_mod;

      return __bitset_test_range(r, start, start + first_size - 1) ||
             __bitset_test_range(r, start + first_size, end);
   }
}

#define BITSET_TEST_RANGE(x, b, e) \
   __bitset_test_range(x, b, e)

static inline void
__bitset_set_range(BITSET_WORD *r, unsigned start, unsigned end)
{
   const unsigned size = end - start + 1;
   const unsigned start_mod = start % BITSET_WORDBITS;

   if (start_mod + size <= BITSET_WORDBITS) {
      BITSET_SET_RANGE_INSIDE_WORD(r, start, end);
   } else {
      const unsigned first_size = BITSET_WORDBITS - start_mod;

      __bitset_set_range(r, start, start + first_size - 1);
      __bitset_set_range(r, start + first_size, end);
   }
}

#define BITSET_SET_RANGE(x, b, e) \
   __bitset_set_range(x, b, e)

static inline void
__bitclear_clear_range(BITSET_WORD *r, unsigned start, unsigned end)
{
   const unsigned size = end - start + 1;
   const unsigned start_mod = start % BITSET_WORDBITS;

   if (start_mod + size <= BITSET_WORDBITS) {
      BITSET_CLEAR_RANGE_INSIDE_WORD(r, start, end);
   } else {
      const unsigned first_size = BITSET_WORDBITS - start_mod;

      __bitclear_clear_range(r, start, start + first_size - 1);
      __bitclear_clear_range(r, start + first_size, end);
   }
}

#define BITSET_CLEAR_RANGE(x, b, e) \
   __bitclear_clear_range(x, b, e)

static inline unsigned
__bitset_prefix_sum(const BITSET_WORD *x, unsigned b, unsigned n)
{
   unsigned prefix = 0;

   for (unsigned i = 0; i < n; i++) {
      if ((i + 1) * BITSET_WORDBITS <= b) {
         prefix += util_bitcount(x[i]);
      } else {
         prefix += util_bitcount(x[i] & BITFIELD_MASK(b - i * BITSET_WORDBITS));
         break;
      }
   }
   return prefix;
}

/* Count set bits in the bitset (compute the size/cardinality of the bitset).
 * This is a special case of prefix sum, but this convenience method is more
 * natural when applicable.
 */

static inline unsigned
__bitset_count(const BITSET_WORD *x, unsigned n)
{
   return __bitset_prefix_sum(x, ~0, n);
}

#define BITSET_PREFIX_SUM(x, b) \
   __bitset_prefix_sum(x, b, ARRAY_SIZE(x))

#define BITSET_COUNT(x) \
   __bitset_count(x, ARRAY_SIZE(x))

/* Get first bit set in a bitset.
 */
static inline int
__bitset_ffs(const BITSET_WORD *x, int n)
{
   for (int i = 0; i < n; i++) {
      if (x[i])
         return ffs(x[i]) + BITSET_WORDBITS * i;
   }

   return 0;
}

/* Get the last bit set in a bitset.
 */
static inline int
__bitset_last_bit(const BITSET_WORD *x, int n)
{
   for (int i = n - 1; i >= 0; i--) {
      if (x[i])
         return util_last_bit(x[i]) + BITSET_WORDBITS * i;
   }

   return 0;
}

#define BITSET_FFS(x) __bitset_ffs(x, ARRAY_SIZE(x))
#define BITSET_LAST_BIT(x) __bitset_last_bit(x, ARRAY_SIZE(x))
#define BITSET_LAST_BIT_SIZED(x, size) __bitset_last_bit(x, size)

static inline unsigned
__bitset_next_set(unsigned i, BITSET_WORD *tmp,
                  const BITSET_WORD *set, unsigned size)
{
   unsigned bit, word;

   /* NOTE: The initial conditions for this function are very specific.  At
    * the start of the loop, the tmp variable must be set to *set and the
    * initial i value set to 0.  This way, if there is a bit set in the first
    * word, we ignore the i-value and just grab that bit (so 0 is ok, even
    * though 0 may be returned).  If the first word is 0, then the value of
    * `word` will be 0 and we will go on to look at the second word.
    */
   word = BITSET_BITWORD(i);
   while (*tmp == 0) {
      word++;

      if (word >= BITSET_WORDS(size))
         return size;

      *tmp = set[word];
   }

   /* Find the next set bit in the non-zero word */
   bit = ffs(*tmp) - 1;

   /* Unset the bit */
   *tmp &= ~(1ull << bit);

   return word * BITSET_WORDBITS + bit;
}

/**
 * Iterates over each set bit in a set
 *
 * @param __i    iteration variable, bit number
 * @param __set  the bitset to iterate (will not be modified)
 * @param __size number of bits in the set to consider
 */
#define BITSET_FOREACH_SET(__i, __set, __size) \
   for (BITSET_WORD __tmp = (__size) == 0 ? 0 : *(__set), *__foo = &__tmp; __foo != NULL; __foo = NULL) \
      for (__i = 0; \
           (__i = __bitset_next_set(__i, &__tmp, __set, __size)) < __size;)

static inline void
__bitset_next_range(unsigned *start, unsigned *end, const BITSET_WORD *set,
                    unsigned size)
{
   /* To find the next start, start searching from end. In the first iteration
    * it will be at 0, in every subsequent iteration it will be at the first
    * 0-bit after the range.
    */
   unsigned word = BITSET_BITWORD(*end);
   if (word >= BITSET_WORDS(size)) {
      *start = *end = size;
      return;
   }
   BITSET_WORD tmp = set[word] & ~(BITSET_BIT(*end) - 1);
   while (!tmp) {
      word++;
      if (word >= BITSET_WORDS(size)) {
         *start = *end = size;
         return;
      }
      tmp = set[word];
   }

   *start = word * BITSET_WORDBITS + ffs(tmp) - 1;

   /* Now do the opposite to find end. Here we can start at start + 1, because
    * we know that the bit at start is 1 and we're searching for the first
    * 0-bit.
    */
   word = BITSET_BITWORD(*start + 1);
   if (word >= BITSET_WORDS(size)) {
      *end = size;
      return;
   }
   tmp = set[word] | (BITSET_BIT(*start + 1) - 1);
   while (~tmp == 0) {
      word++;
      if (word >= BITSET_WORDS(size)) {
         *end = size;
         return;
      }
      tmp = set[word];
   }

   /* Cap "end" at "size" in case there are extra bits past "size" set in the
    * word. This is only necessary for "end" because we terminate the loop if
    * "start" goes past "size".
    */
   *end = MIN2(word * BITSET_WORDBITS + ffs(~tmp) - 1, size);
}

/**
 * Iterates over each contiguous range of set bits in a set
 *
 * @param __start the first 1 bit of the current range
 * @param __end   the bit after the last 1 bit of the current range
 * @param __set   the bitset to iterate (will not be modified)
 * @param __size  number of bits in the set to consider
 */
#define BITSET_FOREACH_RANGE(__start, __end, __set, __size) \
   for (__start = 0, __end = 0, \
        __bitset_next_range(&__start, &__end, __set, __size); \
        __start < __size; \
        __bitset_next_range(&__start, &__end, __set, __size))


#ifdef __cplusplus

/**
 * Simple C++ wrapper of a bitset type of static size, with value semantics
 * and basic bitwise arithmetic operators.  The operators defined below are
 * expected to have the same semantics as the same operator applied to other
 * fundamental integer types.  T is the name of the struct to instantiate
 * it as, and N is the number of bits in the bitset.
 */
#define DECLARE_BITSET_T(T, N) struct T {                       \
      explicit                                                  \
      operator bool() const                                     \
      {                                                         \
         for (unsigned i = 0; i < BITSET_WORDS(N); i++)         \
            if (words[i])                                       \
               return true;                                     \
         return false;                                          \
      }                                                         \
                                                                \
      T &                                                       \
      operator=(int x)                                          \
      {                                                         \
         const T c = {{ (BITSET_WORD)x }};                      \
         return *this = c;                                      \
      }                                                         \
                                                                \
      friend bool                                               \
      operator==(const T &b, const T &c)                        \
      {                                                         \
         return BITSET_EQUAL(b.words, c.words);                 \
      }                                                         \
                                                                \
      friend bool                                               \
      operator!=(const T &b, const T &c)                        \
      {                                                         \
         return !(b == c);                                      \
      }                                                         \
                                                                \
      friend bool                                               \
      operator==(const T &b, int x)                             \
      {                                                         \
         const T c = {{ (BITSET_WORD)x }};                      \
         return b == c;                                         \
      }                                                         \
                                                                \
      friend bool                                               \
      operator!=(const T &b, int x)                             \
      {                                                         \
         return !(b == x);                                      \
      }                                                         \
                                                                \
      friend T                                                  \
      operator~(const T &b)                                     \
      {                                                         \
         T c;                                                   \
         for (unsigned i = 0; i < BITSET_WORDS(N); i++)         \
            c.words[i] = ~b.words[i];                           \
         return c;                                              \
      }                                                         \
                                                                \
      T &                                                       \
      operator|=(const T &b)                                    \
      {                                                         \
         for (unsigned i = 0; i < BITSET_WORDS(N); i++)         \
            words[i] |= b.words[i];                             \
         return *this;                                          \
      }                                                         \
                                                                \
      friend T                                                  \
      operator|(const T &b, const T &c)                         \
      {                                                         \
         T d = b;                                               \
         d |= c;                                                \
         return d;                                              \
      }                                                         \
                                                                \
      T &                                                       \
      operator&=(const T &b)                                    \
      {                                                         \
         for (unsigned i = 0; i < BITSET_WORDS(N); i++)         \
            words[i] &= b.words[i];                             \
         return *this;                                          \
      }                                                         \
                                                                \
      friend T                                                  \
      operator&(const T &b, const T &c)                         \
      {                                                         \
         T d = b;                                               \
         d &= c;                                                \
         return d;                                              \
      }                                                         \
                                                                \
      bool                                                      \
      test(unsigned i) const                                    \
      {                                                         \
         return BITSET_TEST(words, i);                          \
      }                                                         \
                                                                \
      T &                                                       \
      set(unsigned i)                                           \
      {                                                         \
         BITSET_SET(words, i);                                  \
         return *this;                                          \
      }                                                         \
                                                                \
      T &                                                       \
      clear(unsigned i)                                         \
      {                                                         \
         BITSET_CLEAR(words, i);                                \
         return *this;                                          \
      }                                                         \
                                                                \
      BITSET_WORD words[BITSET_WORDS(N)];                       \
   }

#endif

#endif
